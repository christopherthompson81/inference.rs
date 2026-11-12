//! VibeVoice-ASR: an acoustic and a semantic speech-tokenizer encoder over raw 24 kHz audio, a two-branch projector
//! summed into one embedding per 3200 samples, and a Qwen2 decoder that writes the transcript as JSON segments with
//! times and speakers.

pub mod config;
pub mod encoder;
pub mod inputs_processor;
pub mod transcription;

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Linear, Module};
use inference_tensor::{Device, Result, Tensor};

use crate::qwen2::Model as Qwen2;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    kv_cache::EitherCache,
    layers::{RmsNorm, linear_b},
    media_inputs::media::place_token_features,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};
pub use config::VibeVoiceAsrConfig;
use encoder::{EncoderCache, SpeechEncoder};

const MODEL: &str = "VibeVoice-ASR";
/// Seconds of audio the encoders take at once. The causal context they carry makes any whole number of hops give
/// the reference's output; 10 s keeps their activations to tens of MB, where its minute takes about 2 GB.
pub const ENCODE_SECONDS: usize = 10;
const PROJECTOR_EPS: f64 = 1e-6;

pub struct VibeVoiceAsrSpecificArgs {
    /// The clips of the batch whose placeholders this step holds, in prompt order: `(1, 1, samples)` each, normalized
    /// and padded to a multiple of the hop.
    pub audio: Vec<Tensor>,
}

// a latent's linear, RMSNorm and linear up to the text width
struct Branch {
    linear_1: Linear,
    norm: RmsNorm,
    linear_2: Linear,
}

impl Branch {
    fn new(latent: usize, hidden: usize, name: &str, vb: &ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear_b(latent, hidden, true, vb.pp(format!("{name}_linear_1")))?,
            norm: RmsNorm::new(hidden, PROJECTOR_EPS, vb.pp(format!("{name}_norm")))?,
            linear_2: linear_b(hidden, hidden, true, vb.pp(format!("{name}_linear_2")))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear_2
            .forward(&self.norm.forward(&self.linear_1.forward(xs)?)?)
    }
}

/// The audio half: both speech-tokenizer encoders and the projector, one text-width embedding per hop.
pub struct AudioEmbedder {
    acoustic: SpeechEncoder,
    semantic: SpeechEncoder,
    acoustic_branch: Branch,
    semantic_branch: Branch,
    chunk: usize,
}

impl AudioEmbedder {
    /// `vb` is the checkpoint's root.
    pub fn new(cfg: &VibeVoiceAsrConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let hidden = cfg.text().hidden_size;
        let projector = vb.pp("multi_modal_projector");
        let (acoustic, semantic) = (
            &cfg.acoustic_tokenizer_encoder_config,
            &cfg.semantic_tokenizer_encoder_config,
        );
        Ok(Self {
            acoustic: SpeechEncoder::new(acoustic, vb.pp("acoustic_tokenizer_encoder"))?,
            semantic: SpeechEncoder::new(semantic, vb.pp("semantic_tokenizer_encoder"))?,
            acoustic_branch: Branch::new(acoustic.hidden_size, hidden, "acoustic", &projector)?,
            semantic_branch: Branch::new(semantic.hidden_size, hidden, "semantic", &projector)?,
            chunk: cfg.acoustic_tokenizer_chunk_size.min(
                ENCODE_SECONDS * inputs_processor::SAMPLE_RATE as usize / cfg.hop_length()
                    * cfg.hop_length(),
            ),
        })
    }

    /// One embedding per hop of `(1, 1, samples)` audio: both encoders over chunks of `ENCODE_SECONDS`, each
    /// carrying its causal context into the next, as the reference encodes its minutes. The reference adds Gaussian noise to the
    /// acoustic latents (its VAE's sampling); its mean, none, is taken here, so a transcript repeats.
    pub fn embed(&self, audio: &Tensor) -> Result<Tensor> {
        let dtype = self.acoustic_branch.linear_1.weight().dtype();
        let audio = audio.to_dtype(dtype)?;
        let samples = audio.dim(2)?;
        let (mut acoustic_cache, mut semantic_cache) = (EncoderCache::new(), EncoderCache::new());
        let (mut acoustic, mut semantic) = (Vec::new(), Vec::new());
        let mut start = 0;
        while start < samples {
            let chunk = audio.narrow(2, start, self.chunk.min(samples - start))?;
            acoustic.push(self.acoustic.forward(&chunk, &mut acoustic_cache)?);
            semantic.push(self.semantic.forward(&chunk, &mut semantic_cache)?);
            start += self.chunk;
        }
        let acoustic = self.acoustic_branch.forward(&Tensor::cat(&acoustic, 1)?)?;
        let semantic = self.semantic_branch.forward(&Tensor::cat(&semantic, 1)?)?;
        (acoustic + semantic)?.squeeze(0)
    }
}

pub struct VibeVoiceAsrModel {
    text: Qwen2,
    audio: AudioEmbedder,
    audio_token: u32,
}

impl VibeVoiceAsrModel {
    pub fn new(
        cfg: &VibeVoiceAsrConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let audio_vb = vb
            .clone()
            .without_lora_registry()
            .set_device(normal_loading_metadata.real_device.clone());
        Ok(Self {
            audio: AudioEmbedder::new(cfg, &audio_vb)?,
            text: Qwen2::new(
                &cfg.text().decoder_spec()?,
                vb.pp("language_model"),
                is_gptx,
                normal_loading_metadata,
                attention_mechanism,
            )?,
            audio_token: cfg.audio_token_id,
        })
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        args: VibeVoiceAsrSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut embeds = self.text.get_input_embeddings(input_ids)?;
        if !args.audio.is_empty() {
            let projected = args
                .audio
                .iter()
                .map(|clip| self.audio.embed(clip))
                .collect::<Result<Vec<_>>>()?;
            let projected = Tensor::cat(&projected, 0)?;
            embeds = place_token_features(&embeds, input_ids, self.audio_token, &projected, MODEL)?;
        }
        self.text.forward_embeds(input_ids, embeds, ctx)
    }
}

impl IsqModel for VibeVoiceAsrModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(
            self.text
                .residual_tensors_m(uvb.pp("language_model").pp("model")),
        );
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for VibeVoiceAsrModel {}
impl crate::model::BlockDiffusionMixin for VibeVoiceAsrModel {}
impl AnyMoeBaseModelMixin for VibeVoiceAsrModel {}

impl MultimodalModel for VibeVoiceAsrModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<VibeVoiceAsrSpecificArgs>()
            .expect("Cannot downcast into `VibeVoiceAsrSpecificArgs`");
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
        Box::new(VibeVoiceAsrSpecificArgs { audio: Vec::new() })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}
