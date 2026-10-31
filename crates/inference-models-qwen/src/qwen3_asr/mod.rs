//! Qwen3-ASR: the AuT audio encoder in front of a dense Qwen3 decoder, whose `<|audio_pad|>` placeholders take the
//! encoded audio.

pub mod audio;
pub mod config;
pub mod inputs_processor;
pub mod transcription;

use inference_quant::ShardedVarBuilder;
use inference_tensor::{D, DType, Device, Result, Tensor};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    kv_cache::EitherCache,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};
use audio::AudioEncoder;
pub use config::Config;

/// One clip's `(n_mels, frames)` log-mel, padded to whole chunks, with its count of real frames.
pub struct AudioClip {
    pub mel: Tensor,
    pub valid_frames: usize,
}

pub struct Qwen3AsrSpecificArgs {
    /// The clips of the batch whose placeholders this step holds, in prompt order.
    pub audio: Vec<AudioClip>,
}

pub struct Qwen3AsrModel {
    text: crate::qwen3::Model,
    audio: AudioEncoder,
    audio_token: u32,
    layout: config::Layout,
}

impl Qwen3AsrModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let layout = cfg.layout;
        let cfg = &cfg.thinker_config;
        let device = normal_loading_metadata.real_device.clone();
        let audio_vb = vb
            .clone()
            .without_lora_registry()
            .set_device(device.clone());
        let audio = AudioEncoder::new(&cfg.audio_config, audio_vb, layout, &device)?;
        let text = crate::qwen3::Model::new_inner(
            &cfg.text_config.qwen3().decoder_spec(),
            vb.pp(layout.text_model()),
            vb.pp(layout.lm_head()),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            audio,
            audio_token: cfg.audio_token_id,
            layout,
        })
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        args: Qwen3AsrSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut embeds = self.text.get_input_embeddings(input_ids)?;
        if !args.audio.is_empty() {
            let encoded = args
                .audio
                .iter()
                .map(|clip| self.audio.forward(&clip.mel, clip.valid_frames))
                .collect::<Result<Vec<_>>>()?;
            let encoded = Tensor::cat(&encoded, 0)?;
            let positions = input_ids
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?
                .into_iter()
                .enumerate()
                .filter(|&(_, id)| id == self.audio_token)
                .map(|(i, _)| u32::try_from(i).expect("a token position fits u32"))
                .collect::<Vec<_>>();
            if positions.len() != encoded.dim(0)? {
                inference_tensor::bail!(
                    "Qwen3-ASR audio features and audio tokens do not match, tokens: {}, features: {}",
                    positions.len(),
                    encoded.dim(0)?
                );
            }
            let shape = embeds.shape().clone();
            let flat = embeds.reshape(((), embeds.dim(D::Minus1)?))?;
            let positions = Tensor::new(positions.as_slice(), flat.device())?;
            let replace = (encoded.to_dtype(flat.dtype())? - flat.index_select(&positions, 0)?)?;
            embeds = flat.index_add(&positions, &replace, 0)?.reshape(shape)?;
        }
        self.text.forward_embeds(input_ids, embeds, ctx)
    }
}

impl IsqModel for Qwen3AsrModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(
            self.text
                .residual_tensors_m(uvb.pp(self.layout.text_model())),
        );
        let mut tensors = uvb.to_safetensors();
        tensors.extend(self.audio.residual_tensors(self.layout));
        tensors
    }
}

impl crate::speculative::SpeculativeTargetMixin for Qwen3AsrModel {}
impl crate::model::BlockDiffusionMixin for Qwen3AsrModel {}
impl AnyMoeBaseModelMixin for Qwen3AsrModel {}

impl MultimodalModel for Qwen3AsrModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<Qwen3AsrSpecificArgs>()
            .expect("Cannot downcast into `Qwen3AsrSpecificArgs`");
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
        Box::new(Qwen3AsrSpecificArgs { audio: Vec::new() })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}
