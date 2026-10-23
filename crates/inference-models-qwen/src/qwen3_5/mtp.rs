//! Multi-token prediction head shipped inside Qwen3.5 / Qwen3.8 checkpoints (`mtp.*`).
//!
//! Mirrors vLLM's `Qwen3_5MultiTokenPredictor`: one full-attention decoder block fed with
//! `fc(concat(norm(embed(next_token)), norm(target_hidden)))`, sharing the target's embeddings and
//! `lm_head`. Unlike the Gemma 4 assistant it keeps its own paged KV cache, addressed through the
//! sequence's block table as the layer right after the main stack.
//!
//! Qwen3.8-Flash-Next's (vLLM's `Qwen4ExpMultiTokenPredictor`) reads the target's pre-mixer hyper-connection
//! streams: `fc_embedding(norm(embed))` is added to every stream of `fc_hidden(norm(streams))`, one QSA + MoE
//! block with hyper-connections runs, and the head's own mixer collapses the streams for `lm_head` while the
//! streams feed the next chained draft.

use crate::attention::FlashParams;
use crate::paged_attention::PagedAttentionInputMetadata;
use std::sync::Arc;

use inference_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::{D, DType, Device, Module, Result, Tensor};

use crate::{
    attention::AttentionMask,
    device_map::DeviceMapper,
    layers::{GemmaRmsNorm, Qwen3VLRotaryEmbedding},
    model::NormalLoadingMetadata,
    paged_attention::{AttentionImplementation, PagedAttention, load_fp8_attention_scales},
    utils::unvarbuilder::UnVarBuilder,
};

use super::{
    config::TextConfig,
    feed_forward::FeedForward,
    hyper::GatedResidual,
    qsa::QsaLayerInput,
    text::{DecoderLayer, OutputNorm},
};

pub const MTP_FC_WEIGHT: &str = "mtp.fc.weight";
// Flash-Next's split input projections, from a Hugging Face checkpoint
const MTP_FC_EMBEDDING_WEIGHT: &str = "mtp.fc_embedding.weight";

pub struct MtpAttentionInputs<'a> {
    pub kv_cache: (Tensor, Tensor),
    pub metadata: &'a PagedAttentionInputMetadata,
    pub attention_mask: &'a AttentionMask,
    pub flash_params: &'a FlashParams,
}

enum MtpInput {
    // one projection over concat(norm(embedding), norm(hidden)): Qwen3.5's, or a Flash-Next GGUF's `eh_proj`
    // applied per stream
    Concat(Arc<dyn QuantMethod>),
    // Flash-Next's: `fc_embedding(norm(embedding))` added to `fc_hidden(norm(hidden))` of every stream
    Split {
        embedding: Arc<dyn QuantMethod>,
        hidden: Arc<dyn QuantMethod>,
    },
}

/// One MTP forward's outputs: what `lm_head` reads, and what the next chained draft takes as its hidden state.
pub struct MtpDraft {
    pub logits_hidden: Tensor,
    pub chain_hidden: Tensor,
}

pub struct Qwen3_5MtpHead {
    pre_fc_norm_embedding: GemmaRmsNorm,
    pre_fc_norm_hidden: GemmaRmsNorm,
    input: MtpInput,
    layer: DecoderLayer,
    // the final norm, or with hyper-connections the head's own mixer
    output: OutputNorm,
    // 1 without hyper-connections
    streams: usize,
    qsa: Option<super::config::QsaConfig>,
    hidden_size: usize,
    kv_layer_idx: usize,
    device: Device,
    dtype: DType,
}

impl Qwen3_5MtpHead {
    /// `vb` is the checkpoint root; the head lives on the non-mapped device with the final norm and
    /// `lm_head` it feeds.
    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        normal_loading_metadata: &NormalLoadingMetadata,
        attention_mechanism: &AttentionImplementation,
    ) -> Result<Self> {
        let concat = crate::layers::contains_tensor_or_uqff(&vb, MTP_FC_WEIGHT);
        let hc = cfg.hyper_connections();
        if !concat
            && !(hc.is_some()
                && crate::layers::contains_tensor_or_uqff(&vb, MTP_FC_EMBEDDING_WEIGHT))
        {
            inference_tensor::bail!(
                "`--mtp` requested but the checkpoint has no built-in MTP head (`{MTP_FC_WEIGHT}`)."
            );
        }
        if cfg.mtp_num_hidden_layers != 1 {
            inference_tensor::bail!(
                "Qwen3.5 MTP supports exactly one MTP layer, config has {}",
                cfg.mtp_num_hidden_layers
            );
        }
        if cfg.mtp_use_dedicated_embeddings {
            inference_tensor::bail!("Qwen3.5 MTP with dedicated embeddings is not supported");
        }
        let loading_isq = normal_loading_metadata.loading_isq;
        let device = normal_loading_metadata.real_device.clone();
        let vb_mtp = vb.pp("mtp");
        let vb_quant = mapper.set_nm_device(vb_mtp.clone(), loading_isq);
        let vb_plain = mapper.set_nm_device(vb_mtp, false);
        let comm = mapper.get_comm_for(cfg.mtp_layer_idx())?;

        let pre_fc_norm_embedding = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_plain.pp("pre_fc_norm_embedding"),
        )?;
        let streams = hc.map_or(1, |hc| hc.streams);
        // over every stream at once
        let pre_fc_norm_hidden = GemmaRmsNorm::new(
            streams * cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_plain.pp("pre_fc_norm_hidden"),
        )?;
        let linear = |inputs: usize, name: &str| {
            ReplicatedLayer::new(
                inputs,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                vb_quant.pp(name),
            )
        };
        let input = if concat {
            MtpInput::Concat(linear(2 * cfg.hidden_size, "fc")?)
        } else {
            MtpInput::Split {
                embedding: linear(cfg.hidden_size, "fc_embedding")?,
                hidden: linear(cfg.hidden_size, "fc_hidden")?,
            }
        };
        let rotary_emb = Arc::new(match cfg.yarn_rope_config()? {
            Some(yarn) => {
                Qwen3VLRotaryEmbedding::new_yarn(&yarn, &device, cfg.mrope_section().to_vec())?
            }
            None => Qwen3VLRotaryEmbedding::new(
                cfg.rope_theta() as f32,
                cfg.rot_dim(),
                &device,
                cfg.mrope_section().to_vec(),
            )?,
        });
        let vb_layer = vb_plain.pp("layers").pp(0);
        let paged_attn = match attention_mechanism {
            AttentionImplementation::Eager => None,
            AttentionImplementation::PagedAttention => {
                let attention_vb = vb_layer.pp("self_attn");
                Some(PagedAttention::new_with_fp8_attention_scales(
                    cfg.head_dim,
                    &device,
                    None,
                    load_fp8_attention_scales(&attention_vb)?,
                )?)
            }
        };
        let vb_layer_quant = vb_quant.pp("layers").pp(0);
        let mlp = FeedForward::load(
            cfg,
            vb_layer_quant.pp("mlp"),
            device.clone(),
            loading_isq,
            &comm,
        )?;
        let (layer, output) = match (hc, cfg.qsa()) {
            (Some(hc), Some(qsa)) => (
                DecoderLayer::load_hyper_qsa(
                    vb_layer_quant,
                    vb_layer,
                    cfg,
                    hc,
                    qsa,
                    cfg.mtp_qsa_cache_layer(),
                    rotary_emb,
                    paged_attn,
                    mlp,
                    &comm,
                )?,
                OutputNorm::Mixer(GatedResidual::load(
                    cfg,
                    hc,
                    vb_quant.pp("hyper_connection_mixer"),
                    vb_plain.pp("hyper_connection_mixer"),
                    false,
                )?),
            ),
            (Some(_), None) => {
                inference_tensor::bail!("a hyper-connection MTP head without QSA is not supported")
            }
            (None, _) => (
                DecoderLayer::load_full_attention(
                    vb_layer_quant,
                    vb_layer,
                    cfg,
                    rotary_emb,
                    paged_attn,
                    mlp,
                    &comm,
                )?,
                OutputNorm::Rms(GemmaRmsNorm::new(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb_plain.pp("norm"),
                )?),
            ),
        };

        Ok(Self {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            input,
            layer,
            output,
            streams,
            qsa: cfg.qsa().filter(|_| streams > 1),
            hidden_size: cfg.hidden_size,
            kv_layer_idx: cfg.mtp_layer_idx(),
            device,
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Absolute paged-KV layer index the head writes to and reads from.
    pub fn kv_layer_idx(&self) -> usize {
        self.kv_layer_idx
    }

    /// The head's QSA config and the paged layer of its indexer keys, on a Flash-Next head.
    pub fn qsa(&self) -> Option<(super::config::QsaConfig, usize)> {
        self.qsa.zip(self.layer.qsa_cache_layer())
    }

    // The block's input: one stream, or every hyper-connection stream side by side
    fn input(&self, embeds: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        if self.streams == 1 {
            let MtpInput::Concat(fc) = &self.input else {
                inference_tensor::bail!(
                    "a single-stream MTP head projects concat(embedding, hidden)"
                );
            };
            return fc.forward(&Tensor::cat(&[embeds, hidden], D::Minus1)?);
        }
        let (b, rows, _) = embeds.dims3()?;
        let split = (b, rows, self.streams, self.hidden_size);
        let hidden = hidden.reshape(split)?;
        let per_stream = match &self.input {
            MtpInput::Concat(fc) => {
                let embeds = embeds.unsqueeze(2)?.broadcast_as(split)?;
                fc.forward(&Tensor::cat(&[&embeds, &hidden], D::Minus1)?.contiguous()?)?
            }
            MtpInput::Split {
                embedding,
                hidden: fc_hidden,
            } => fc_hidden
                .forward(&hidden)?
                .broadcast_add(&embedding.forward(embeds)?.unsqueeze(2)?)?,
        };
        per_stream.reshape((b, rows, self.streams * self.hidden_size))
    }

    /// One drafter forward over `[b, rows, hidden]` inputs with `[3, b, rows]` MRoPE positions.
    /// With `AttentionMask::None` and per-row metadata every row is an independent decode query; with the
    /// target's prompt-chunk mask/metadata it runs as one causal prefill over the chunk.
    /// Without hyper-connections the normed hidden state is both outputs; with them `lm_head` reads the mixer's
    /// collapse and the next draft the streams. `qsa` is the head's QSA step and indexer cache on a Flash-Next head.
    pub(super) fn forward(
        &self,
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
        attention: MtpAttentionInputs<'_>,
        qsa: Option<QsaLayerInput<'_>>,
    ) -> Result<MtpDraft> {
        let embeds = self.pre_fc_norm_embedding.forward(input_embeds)?;
        let hidden = self.pre_fc_norm_hidden.forward(target_hidden)?;
        let xs = self.input(&embeds, &hidden)?;
        let rotary_emb = self
            .layer
            .rotary_emb()
            .expect("MTP layer is a full-attention block");
        let cos_sin = match positions.rank() {
            2 => rotary_emb.compute_text_cos_sin(positions, xs.dtype())?,
            3 => rotary_emb.compute_cos_sin(positions, xs.dtype())?,
            rank => inference_tensor::bail!("unexpected Qwen3.5 MTP position rank {rank}"),
        };
        let xs = self.layer.forward_attention_qsa(
            &xs,
            attention.attention_mask,
            &cos_sin,
            Some((attention.kv_cache, attention.metadata)),
            attention.flash_params,
            qsa,
        )?;
        let logits_hidden = self.output.forward(&xs)?;
        let chain_hidden = if self.streams == 1 {
            logits_hidden.clone()
        } else {
            xs
        };
        Ok(MtpDraft {
            logits_hidden,
            chain_hidden,
        })
    }

    pub fn residual_tensors(&self, uvb: &UnVarBuilder) {
        let uvb_mtp = uvb.pp("mtp");
        uvb_mtp
            .pp("pre_fc_norm_embedding")
            .add(&self.pre_fc_norm_embedding);
        uvb_mtp
            .pp("pre_fc_norm_hidden")
            .add(&self.pre_fc_norm_hidden);
        self.output
            .residual_tensors(&uvb_mtp, "norm", "hyper_connection_mixer");
        let uvb_l = uvb_mtp.pp("layers").pp(0);
        match &self.layer.norms {
            super::text::LayerNorms::Rms {
                input,
                post_attention,
            } => {
                uvb_l.pp("input_layernorm").add(input);
                uvb_l.pp("post_attention_layernorm").add(post_attention);
            }
            super::text::LayerNorms::Hyper { attention, mlp } => {
                attention.residual_tensors(&uvb_l.pp("attn_hyper_connection"));
                mlp.residual_tensors(&uvb_l.pp("mlp_hyper_connection"));
            }
        }
        if let super::text::LayerImpl::FullAttention(attn) = &self.layer.layer_impl {
            uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_norm);
            uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_norm);
            if let Some(indexer) = &attn.indexer {
                indexer.residual_tensors(&uvb_l.pp("self_attn").pp("indexer"));
            }
        }
        self.layer.mlp.residual_tensors(&uvb_l.pp("mlp"));
    }
}
