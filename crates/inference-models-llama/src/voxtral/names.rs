//! Where a Voxtral Realtime checkpoint keeps its tensors: Mistral's `consolidated.safetensors`, or transformers'
//! `model.safetensors` (the same weights renamed, every Q and K projection's heads in half-split order).

/// One layout's names, each relative to the module that reads it.
#[derive(Clone, Copy, Debug)]
pub struct Names {
    pub encoder: &'static str,
    pub conv1: &'static str,
    pub conv2: &'static str,
    pub encoder_layers: &'static str,
    pub encoder_norm: &'static str,
    pub encoder_attention_norm: &'static str,
    pub encoder_ffn_norm: &'static str,
    pub adapter: &'static str,
    pub adapter_in: &'static str,
    pub adapter_out: &'static str,
    pub embeddings: &'static str,
    pub layers: &'static str,
    pub norm: &'static str,
    pub output: &'static str,
    pub attention_norm: &'static str,
    pub ffn_norm: &'static str,
    pub ada_norm: &'static str,
    pub ada_in: &'static str,
    pub ada_out: &'static str,
    pub attention: &'static str,
    pub q: &'static str,
    pub k: &'static str,
    pub v: &'static str,
    pub o: &'static str,
    pub mlp: &'static str,
    pub gate: &'static str,
    pub down: &'static str,
    pub up: &'static str,
    /// Whether RoPE rotates half-split pairs: transformers' conversion de-interleaves each head of Q and K.
    pub half_split_rope: bool,
}

pub const CONSOLIDATED: Names = Names {
    encoder: "mm_streams_embeddings.embedding_module.whisper_encoder",
    conv1: "conv_layers.0.conv",
    conv2: "conv_layers.1.conv",
    encoder_layers: "transformer.layers",
    encoder_norm: "transformer.norm",
    encoder_attention_norm: "attention_norm",
    encoder_ffn_norm: "ffn_norm",
    adapter: "mm_streams_embeddings.embedding_module.audio_language_projection",
    adapter_in: "0",
    adapter_out: "2",
    embeddings: "mm_streams_embeddings.embedding_module.tok_embeddings",
    layers: "layers",
    norm: "norm",
    output: "output",
    attention_norm: "attention_norm",
    ffn_norm: "ffn_norm",
    ada_norm: "ada_rms_norm_t_cond",
    ada_in: "0",
    ada_out: "2",
    attention: "attention",
    q: "wq",
    k: "wk",
    v: "wv",
    o: "wo",
    mlp: "feed_forward",
    gate: "w1",
    down: "w2",
    up: "w3",
    half_split_rope: false,
};

pub const TRANSFORMERS: Names = Names {
    encoder: "audio_tower",
    conv1: "embedder.conv1",
    conv2: "embedder.conv2",
    encoder_layers: "layers",
    encoder_norm: "norm",
    encoder_attention_norm: "self_attn_layer_norm",
    encoder_ffn_norm: "final_layer_norm",
    adapter: "multi_modal_projector",
    adapter_in: "linear_1",
    adapter_out: "linear_2",
    embeddings: "language_model.model.embed_tokens",
    layers: "language_model.model.layers",
    norm: "language_model.model.norm",
    output: "language_model.lm_head",
    attention_norm: "input_layernorm",
    ffn_norm: "post_attention_layernorm",
    ada_norm: "ada_rms_norm",
    ada_in: "linear1",
    ada_out: "linear2",
    attention: "self_attn",
    q: "q_proj",
    k: "k_proj",
    v: "v_proj",
    o: "o_proj",
    mlp: "mlp",
    gate: "gate_proj",
    down: "down_proj",
    up: "up_proj",
    half_split_rope: true,
};

impl Names {
    /// The layout whose first convolution the checkpoint holds (never quantized, so present under ISQ too).
    pub fn of(vb: &inference_quant::ShardedVarBuilder) -> Self {
        let conv = format!("{}.{}.weight", TRANSFORMERS.encoder, TRANSFORMERS.conv1);
        if vb.contains_tensor(&conv) {
            TRANSFORMERS
        } else {
            CONSOLIDATED
        }
    }
}
