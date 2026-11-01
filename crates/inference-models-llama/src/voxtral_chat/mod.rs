//! Voxtral (Mini 3B, Small 24B): the Whisper encoder, a projector over four stacked encoder positions, and a Mistral
//! decoder whose `[AUDIO]` placeholders take the projected audio.

pub mod config;
pub mod inputs_processor;

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Linear, Module};
use inference_tensor::{Device, Result, Tensor};

use crate::mistral::Model as Mistral;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    audio::whisper::WhisperEncoder,
    kv_cache::EitherCache,
    layers::{Activation, linear_no_bias},
    media_inputs::audio_prompt::place_audio_features,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};
pub use config::VoxtralChatConfig;

const MODEL: &str = "Voxtral";

pub struct VoxtralChatSpecificArgs {
    /// `(chunks, n_mels, frames)` log-mel of each clip whose placeholders this step holds, in prompt order.
    pub audio: Vec<Tensor>,
}

pub struct VoxtralChatModel {
    text: Mistral,
    audio: WhisperEncoder,
    linear_1: Linear,
    linear_2: Linear,
    activation: Activation,
    stacked_width: usize,
    audio_token: u32,
}

impl VoxtralChatModel {
    pub fn new(
        cfg: &VoxtralChatConfig,
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
        let linear_1 = linear_no_bias(cfg.stacked_width(), hidden, projector.pp("linear_1"))?;
        let linear_2 = linear_no_bias(hidden, hidden, projector.pp("linear_2"))?;
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

    /// One clip's `(chunks, n_mels, frames)` log-mel to its `(tokens, hidden)` text-width embeddings.
    fn embed_audio(&self, mel: &Tensor) -> Result<Tensor> {
        let encoded = self
            .audio
            .forward(&mel.to_dtype(self.linear_1.weight().dtype())?)?;
        let stacked = encoded.reshape(((), self.stacked_width))?;
        let xs = self.activation.forward(&self.linear_1.forward(&stacked)?)?;
        self.linear_2.forward(&xs)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        args: VoxtralChatSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut embeds = self.text.get_input_embeddings(input_ids)?;
        if !args.audio.is_empty() {
            let projected = args
                .audio
                .iter()
                .map(|mel| self.embed_audio(mel))
                .collect::<Result<Vec<_>>>()?;
            let projected = Tensor::cat(&projected, 0)?;
            embeds = place_audio_features(&embeds, input_ids, self.audio_token, &projected, MODEL)?;
        }
        self.text.forward_embeds(input_ids, embeds, ctx)
    }
}

impl IsqModel for VoxtralChatModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(self.text.residual_tensors_m(uvb.pp("language_model")));
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

impl crate::speculative::SpeculativeTargetMixin for VoxtralChatModel {}
impl crate::model::BlockDiffusionMixin for VoxtralChatModel {}
impl AnyMoeBaseModelMixin for VoxtralChatModel {}

impl MultimodalModel for VoxtralChatModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<VoxtralChatSpecificArgs>()
            .expect("Cannot downcast into `VoxtralChatSpecificArgs`");
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
        Box::new(VoxtralChatSpecificArgs { audio: Vec::new() })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}
