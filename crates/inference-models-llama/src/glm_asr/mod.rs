//! GLM-ASR: a Whisper-shaped encoder with rotary positions, a two-layer projector over four stacked encoder positions,
//! and a Llama decoder whose `<|pad|>` placeholders take the projected audio.

pub mod config;
pub mod inputs_processor;
pub mod transcription;

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Linear, Module};
use inference_tensor::{Device, Result, Tensor};

use crate::mistral::Model as Mistral;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    audio::whisper::WhisperEncoder,
    kv_cache::EitherCache,
    layers::{Activation, linear_b},
    media_inputs::audio_prompt::place_audio_features,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};
pub use config::GlmAsrConfig;

const MODEL: &str = "GLM-ASR";

/// One clip's `(chunks, n_mels, frames)` log-mel, each chunk a 30 s window, with the tokens each chunk keeps.
pub struct AudioClip {
    pub mel: Tensor,
    pub tokens: Vec<usize>,
}

pub struct GlmAsrSpecificArgs {
    /// The clips of the batch whose placeholders this step holds, in prompt order.
    pub audio: Vec<AudioClip>,
}

pub struct GlmAsrModel {
    text: Mistral,
    audio: WhisperEncoder,
    linear_1: Linear,
    linear_2: Linear,
    activation: Activation,
    stacked_width: usize,
    audio_token: u32,
}

impl GlmAsrModel {
    pub fn new(
        cfg: &GlmAsrConfig,
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
        let hidden = cfg.text_config.hidden_size;
        // the projector widens to twice the text width between its layers
        let linear_1 = linear_b(
            cfg.stacked_width(),
            2 * hidden,
            true,
            projector.pp("linear_1"),
        )?;
        let linear_2 = linear_b(2 * hidden, hidden, true, projector.pp("linear_2"))?;
        let text = Mistral::new(
            &cfg.text_config().decoder_spec()?,
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            audio,
            linear_1,
            linear_2,
            activation: cfg.projector_hidden_act,
            stacked_width: cfg.stacked_width(),
            audio_token: cfg.audio_token_id,
        })
    }

    // the encoder reads each whole padded window, as transformers does; each chunk keeps the tokens its audio reaches
    fn embed_audio(&self, clip: &AudioClip) -> Result<Tensor> {
        let encoded = self
            .audio
            .forward(&clip.mel.to_dtype(self.linear_1.weight().dtype())?)?;
        let chunks = encoded.dim(0)?;
        let stacked = encoded.reshape((chunks, (), self.stacked_width))?;
        let xs = self.activation.forward(&self.linear_1.forward(&stacked)?)?;
        let projected = self.linear_2.forward(&xs)?;
        let kept = clip
            .tokens
            .iter()
            .enumerate()
            .filter(|&(_, &tokens)| tokens > 0)
            .map(|(chunk, &tokens)| projected.get(chunk)?.narrow(0, 0, tokens))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&kept, 0)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        args: GlmAsrSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut embeds = self.text.get_input_embeddings(input_ids)?;
        if !args.audio.is_empty() {
            let projected = args
                .audio
                .iter()
                .map(|clip| self.embed_audio(clip))
                .collect::<Result<Vec<_>>>()?;
            let projected = Tensor::cat(&projected, 0)?;
            embeds = place_audio_features(&embeds, input_ids, self.audio_token, &projected, MODEL)?;
        }
        self.text.forward_embeds(input_ids, embeds, ctx)
    }
}

impl IsqModel for GlmAsrModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(
            self.text
                .residual_tensors_m(uvb.pp("language_model").pp("model")),
        );
        let projector = uvb.pp("multi_modal_projector");
        projector.pp("linear_1").add(&self.linear_1);
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

impl crate::speculative::SpeculativeTargetMixin for GlmAsrModel {}
impl crate::model::BlockDiffusionMixin for GlmAsrModel {}
impl AnyMoeBaseModelMixin for GlmAsrModel {}

impl MultimodalModel for GlmAsrModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<GlmAsrSpecificArgs>()
            .expect("Cannot downcast into `GlmAsrSpecificArgs`");
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
        Box::new(GlmAsrSpecificArgs { audio: Vec::new() })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}
