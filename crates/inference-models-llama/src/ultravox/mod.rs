//! Ultravox: Whisper's encoder, a projector over eight stacked encoder positions (RMSNorm, linear, SwiGLU, RMSNorm,
//! linear), and a Llama decoder whose audio placeholders take the projected audio.

pub mod config;
pub mod inputs_processor;

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Linear, Module};
use inference_tensor::{D, Device, Result, Tensor};

use crate::llama::Llama;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    audio::whisper::WhisperEncoder,
    kv_cache::EitherCache,
    layers::{RmsNorm, linear_no_bias},
    media_inputs::audio_prompt::place_audio_features,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};
pub use config::UltravoxConfig;

const MODEL: &str = "Ultravox";

/// One 30 s chunk of a clip: its `(1, n_mels, frames)` log-mel, the encoder positions its real frames reach (the
/// keys a padded chunk's mask leaves), and the tokens it fills.
pub struct AudioChunk {
    pub mel: Tensor,
    pub keys: usize,
    pub tokens: usize,
}

pub struct UltravoxSpecificArgs {
    /// The chunks of the clips whose placeholders this step holds, in prompt order.
    pub audio: Vec<AudioChunk>,
}

pub struct UltravoxModel {
    text: Llama,
    audio: WhisperEncoder,
    ln_pre: RmsNorm,
    linear_1: Linear,
    ln_mid: RmsNorm,
    linear_2: Linear,
    stack: usize,
    audio_token: u32,
}

impl UltravoxModel {
    pub fn new(
        cfg: &UltravoxConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let audio_vb = vb
            .clone()
            .without_lora_registry()
            .set_device(normal_loading_metadata.real_device.clone());
        let audio = WhisperEncoder::new(&cfg.encoder(), audio_vb.pp("audio_tower"))?;
        let projector = audio_vb.pp("multi_modal_projector");
        let (stacked, inner) = (cfg.stacked_width(), cfg.hidden_size);
        // the SwiGLU halves the inner width
        let gated = inner / 2;
        let text_hidden = cfg.text_config.hidden_size;
        let ln_pre = RmsNorm::new(stacked, cfg.norm_eps, projector.pp("ln_pre"))?;
        let linear_1 = linear_no_bias(stacked, inner, projector.pp("linear_1"))?;
        let ln_mid = RmsNorm::new(gated, cfg.norm_eps, projector.pp("ln_mid"))?;
        let linear_2 = linear_no_bias(gated, text_hidden, projector.pp("linear_2"))?;
        let text = Llama::new(
            &cfg.text_config().decoder_spec(),
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            audio,
            ln_pre,
            linear_1,
            ln_mid,
            linear_2,
            stack: cfg.stack_factor,
            audio_token: cfg.audio_token_id,
        })
    }

    // the encoder's positions zero-padded to whole stacks of eight, as the reference's `StackAudioFrames` does
    fn embed_chunk(&self, chunk: &AudioChunk) -> Result<Tensor> {
        let mel = chunk.mel.to_dtype(self.linear_1.weight().dtype())?;
        let xs = self.audio.embed(&mel)?;
        let xs = self.audio.encode_keys(&xs, Some(chunk.keys))?;
        let xs = self.audio.norm(&xs)?.squeeze(0)?;
        let (positions, hidden) = xs.dims2()?;
        let stacks = positions.div_ceil(self.stack);
        let xs = xs.pad_with_zeros(0, 0, stacks * self.stack - positions)?;
        let xs = xs.reshape((stacks, self.stack * hidden))?;
        let xs = self.linear_1.forward(&self.ln_pre.forward(&xs)?)?;
        // SwiGLU with the second half gated: `silu(gate) * x`
        let half = xs.dim(D::Minus1)? / 2;
        let gated = (xs.narrow(D::Minus1, 0, half)? * xs.narrow(D::Minus1, half, half)?.silu()?)?;
        let xs = self.linear_2.forward(&self.ln_mid.forward(&gated)?)?;
        xs.narrow(0, 0, chunk.tokens.min(stacks))
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        args: UltravoxSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut embeds = self.text.get_input_embeddings(input_ids)?;
        if !args.audio.is_empty() {
            let projected = args
                .audio
                .iter()
                .map(|chunk| self.embed_chunk(chunk))
                .collect::<Result<Vec<_>>>()?;
            let projected = Tensor::cat(&projected, 0)?;
            embeds = place_audio_features(&embeds, input_ids, self.audio_token, &projected, MODEL)?;
        }
        self.text.forward_embeds(input_ids, embeds, ctx)
    }
}

impl IsqModel for UltravoxModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(self.text.residual_tensors_m(uvb.pp("language_model")));
        let projector = uvb.pp("multi_modal_projector");
        projector.pp("ln_pre").add(&self.ln_pre);
        projector.pp("linear_1").add(&self.linear_1);
        projector.pp("ln_mid").add(&self.ln_mid);
        projector.pp("linear_2").add(&self.linear_2);
        let mut tensors = uvb.to_safetensors();
        tensors.extend(
            self.audio
                .residual_tensors()
                .into_iter()
                .map(|(name, tensor)| (format!("audio_tower.{name}"), tensor)),
        );
        tensors
    }
}

impl crate::speculative::SpeculativeTargetMixin for UltravoxModel {}
impl crate::model::BlockDiffusionMixin for UltravoxModel {}
impl AnyMoeBaseModelMixin for UltravoxModel {}

impl MultimodalModel for UltravoxModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<UltravoxSpecificArgs>()
            .expect("Cannot downcast into `UltravoxSpecificArgs`");
        self.forward_inner(input_ids, args, ctx)
    }

    fn device(&self) -> &Device {
        self.text.device()
    }

    fn cache(&self) -> &EitherCache {
        self.text.cache()
    }

    fn max_seq_len(&self) -> usize {
        self.text.max_seq_len()
    }

    fn config(&self) -> &ModelConfigMetadata {
        self.text.config()
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(UltravoxSpecificArgs { audio: Vec::new() })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}
