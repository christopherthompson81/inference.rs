//! Qwen3-VL's text stack: the shared decoder with interleaved M-RoPE, fused q/k norm and optional experts.

use inference_quant::ShardedVarBuilder;
use inference_tensor::Result;

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
