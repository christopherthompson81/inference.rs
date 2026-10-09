//! Qwen3-VL's text stack: the shared decoder with interleaved M-RoPE, fused q/k norm and optional experts.

use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Result, Tensor};

use super::config::TextConfig;
use crate::{
    decoder::{CausalLm, DecoderSpec, MLP, MoeRouting, MoeSpec, NormKind, QkNorm, RopeKind},
    layers,
    model::NormalLoadingMetadata,
    moe::ExpertProjNames,
    paged_attention::AttentionImplementation,
};

// MLX checkpoints put the text stack under `language_model.model.` rather than Hugging Face's `model.language_model.`
const MLX_EMBED_TOKENS: &str = "language_model.model.embed_tokens.weight";

impl TextConfig {
    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn decoder_spec(&self, tie_word_embeddings: bool) -> DecoderSpec {
        let moe = (self.num_experts > 0).then(|| MoeSpec {
            num_experts: self.num_experts,
            intermediate_size: self.moe_intermediate_size,
            routing: MoeRouting::TopK {
                k: self.num_experts_per_tok,
                renormalize: self.norm_topk_prob,
            },
            quantized_router: false,
            expert_names: ExpertProjNames::DEFAULT,
            name: MLP,
            dense_layers: (0..self.num_hidden_layers)
                .filter(|&layer_idx| !self.is_moe_layer(layer_idx))
                .collect(),
            cuda_decode_graphs: true,
        });
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::MRope {
                theta: self.rope_theta as f32,
                sections: self.rope_scaling.mrope_section.clone(),
                interleaved: true,
            },
            max_position_embeddings: self.max_position_embeddings,
            qk_norm: Some(QkNorm::BEFORE_ROPE),
            // every text layer is full attention, whatever window the config names
            layer_windows: vec![None; self.num_hidden_layers],
            tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            // transformers normalises in F32 and scales after casting back, dense and MoE alike
            norm: NormKind::F32Rms,
            qk_norm_kind: Some(NormKind::Rms),
            moe,
            ..Default::default()
        }
    }
}

/// The text model, under Hugging Face's `model.language_model.` or MLX's `language_model.model.`.
pub fn text_model(
    cfg: &TextConfig,
    vb: ShardedVarBuilder,
    tie_word_embeddings: bool,
    is_gptx: bool,
    normal_loading_metadata: NormalLoadingMetadata,
    attention_mechanism: AttentionImplementation,
) -> Result<CausalLm> {
    let vb_m = if layers::contains_tensor_or_uqff(&vb, MLX_EMBED_TOKENS) {
        vb.pp("language_model").pp("model")
    } else {
        vb.pp("model").pp("language_model")
    };
    CausalLm::new_inner(
        &cfg.decoder_spec(tie_word_embeddings),
        vb_m,
        vb.pp("lm_head"),
        is_gptx,
        normal_loading_metadata,
        attention_mechanism,
    )
}

/// transformers' `_deepstack_process`: `hidden_states[visual_pos_masks, :] += visual_embeds` on a copy.
pub fn deepstack_process(
    hidden_states: Tensor,
    visual_pos_masks: &Tensor,
    visual_embeds: &Tensor,
) -> Result<Tensor> {
    let device = hidden_states.device();
    let dtype = hidden_states.dtype();
    let visual_embeds = visual_embeds.to_device(device)?.to_dtype(dtype)?;

    let (batch, seq, hidden) = hidden_states.dims3()?;
    let total = batch * seq;
    let hidden_flat = hidden_states.reshape((total, hidden))?;

    let mask_flat: Vec<f32> = visual_pos_masks
        .to_device(device)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let indices: Vec<u32> = mask_flat
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > 0.0)
        .map(|(i, _)| u32::try_from(i).expect("token index fits u32"))
        .collect();

    if indices.is_empty() {
        return Ok(hidden_states);
    }
    if indices.len() != visual_embeds.dim(0)? {
        inference_tensor::bail!(
            "Mismatch between DeepStack visual embeds ({}) and mask positions ({})",
            visual_embeds.dim(0)?,
            indices.len()
        );
    }

    let idx = Tensor::from_vec(indices, (visual_embeds.dim(0)?,), device)?;
    let idx_expanded = idx.unsqueeze(1)?.repeat((1, hidden))?;
    let result = hidden_flat.scatter_add(&idx_expanded, &visual_embeds, 0)?;
    result.reshape((batch, seq, hidden))
}
