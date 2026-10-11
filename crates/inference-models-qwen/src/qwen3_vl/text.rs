//! Qwen3-VL's text stack: the shared decoder with interleaved M-RoPE, fused q/k norm and optional experts.

use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Result, Tensor};

use super::config::TextConfig;
use crate::{
    decoder::CausalLm, layers, model::NormalLoadingMetadata,
    paged_attention::AttentionImplementation,
};

// MLX checkpoints put the text stack under `language_model.model.` rather than Hugging Face's `model.language_model.`
const MLX_EMBED_TOKENS: &str = "language_model.model.embed_tokens.weight";

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
