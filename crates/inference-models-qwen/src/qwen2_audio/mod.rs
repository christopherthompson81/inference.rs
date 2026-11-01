//! Qwen2-Audio: the Whisper encoder, pooled by two positions before its final norm, one linear projector, and a Qwen2
//! decoder whose `<|AUDIO|>` placeholders take the projected audio.

pub mod config;
pub mod inputs_processor;

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Linear, Module};
use inference_tensor::{Device, Result, Tensor};

use crate::qwen2::Model as Qwen2;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    audio::whisper::WhisperEncoder,
    kv_cache::EitherCache,
    layers::linear_b,
    media_inputs::audio_prompt::place_audio_features,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};
pub use config::Qwen2AudioConfig;

const MODEL: &str = "Qwen2-Audio";
// the encoder averages each pair of positions after its layers
const POOL: usize = 2;

/// One clip's `(1, n_mels, frames)` log-mel over the 30 s window, with the encoder positions its real frames give.
pub struct AudioClip {
    pub mel: Tensor,
    pub positions: usize,
}

/// The tokens a clip of `positions` encoder positions fills.
pub fn audio_token_count(positions: usize) -> usize {
    positions.saturating_sub(POOL) / POOL + 1
}

pub struct Qwen2AudioSpecificArgs {
    /// The clips of the batch whose placeholders this step holds, in prompt order.
    pub audio: Vec<AudioClip>,
}

pub struct Qwen2AudioModel {
    text: Qwen2,
    audio: WhisperEncoder,
    projector: Linear,
    audio_token: u32,
}

impl Qwen2AudioModel {
    pub fn new(
        cfg: &Qwen2AudioConfig,
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
        let projector = linear_b(
            cfg.audio_config.d_model,
            cfg.text_config.hidden_size,
            true,
            audio_vb.pp("multi_modal_projector").pp("linear"),
        )?;
        let text = Qwen2::new(
            &cfg.text().decoder_spec()?,
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            audio,
            projector,
            audio_token: cfg.audio_token_index,
        })
    }

    // the layers attend over the clip's real positions only, as transformers' padding mask has them
    fn embed_audio(&self, clip: &AudioClip) -> Result<Tensor> {
        let mel = clip.mel.to_dtype(self.projector.weight().dtype())?;
        let xs = self.audio.embed(&mel)?.narrow(1, 0, clip.positions)?;
        let xs = self.audio.encode(&xs)?;
        let tokens = audio_token_count(clip.positions);
        let (_, _, hidden) = xs.dims3()?;
        let pooled = xs
            .narrow(1, 0, tokens * POOL)?
            .reshape((tokens, POOL, hidden))?
            .mean(1)?;
        self.projector.forward(&self.audio.norm(&pooled)?)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        args: Qwen2AudioSpecificArgs,
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

impl IsqModel for Qwen2AudioModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(self.text.residual_tensors_m(uvb.pp("language_model")));
        uvb.pp("multi_modal_projector")
            .pp("linear")
            .add(&self.projector);
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

impl crate::speculative::SpeculativeTargetMixin for Qwen2AudioModel {}
impl crate::model::BlockDiffusionMixin for Qwen2AudioModel {}
impl AnyMoeBaseModelMixin for Qwen2AudioModel {}

impl MultimodalModel for Qwen2AudioModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<Qwen2AudioSpecificArgs>()
            .expect("Cannot downcast into `Qwen2AudioSpecificArgs`");
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
        Box::new(Qwen2AudioSpecificArgs { audio: Vec::new() })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}
