//! VibeVoice-ASR: an acoustic and a semantic speech-tokenizer encoder over raw 24 kHz audio, a two-branch projector
//! summed into one embedding per 3200 samples, and a Qwen2 decoder that writes the transcript as JSON segments with
//! times and speakers.

pub mod config;
pub mod encoder;
pub mod inputs_processor;
pub mod streaming;
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
pub use config::EncoderConfig;
pub use config::VibeVoiceAsrConfig;
pub use encoder::EncoderNames;
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
    // `names` are the first linear's, the norm's and the second linear's paths under `vb`
    fn new(
        latent: usize,
        hidden: usize,
        names: [String; 3],
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let [linear_1, norm, linear_2] = names;
        Ok(Self {
            linear_1: linear_b(latent, hidden, true, vb.pp(linear_1))?,
            norm: RmsNorm::new(hidden, PROJECTOR_EPS, vb.pp(norm))?,
            linear_2: linear_b(hidden, hidden, true, vb.pp(linear_2))?,
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

/// The audio half's shape and where a checkpoint keeps it.
pub struct AudioSpec<'a> {
    pub acoustic: &'a EncoderConfig,
    pub semantic: &'a EncoderConfig,
    /// The decoder's width.
    pub hidden: usize,
    pub names: EncoderNames,
}

impl AudioEmbedder {
    /// `vb` is the checkpoint's root.
    pub fn new(cfg: &VibeVoiceAsrConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let spec = AudioSpec {
            acoustic: &cfg.acoustic_tokenizer_encoder_config,
            semantic: &cfg.semantic_tokenizer_encoder_config,
            hidden: cfg.text().hidden_size,
            names: cfg.names,
        };
        Self::from_spec(&spec, vb)
    }

    /// `vb` is the checkpoint's root, under either naming.
    pub fn from_spec(spec: &AudioSpec<'_>, vb: &ShardedVarBuilder) -> Result<Self> {
        let (acoustic, semantic, hidden) = (spec.acoustic, spec.semantic, spec.hidden);
        let (encoders, branches): ([ShardedVarBuilder; 2], [[String; 3]; 2]) = match spec.names {
            EncoderNames::Transformers => (
                [
                    vb.pp("acoustic_tokenizer_encoder"),
                    vb.pp("semantic_tokenizer_encoder"),
                ],
                ["acoustic", "semantic"].map(|b| {
                    [
                        format!("multi_modal_projector.{b}_linear_1"),
                        format!("multi_modal_projector.{b}_norm"),
                        format!("multi_modal_projector.{b}_linear_2"),
                    ]
                }),
            ),
            EncoderNames::Upstream => (
                [
                    vb.pp("model.acoustic_tokenizer.encoder"),
                    vb.pp("model.semantic_tokenizer.encoder"),
                ],
                ["acoustic", "semantic"].map(|b| {
                    ["fc1", "norm", "fc2"].map(|part| format!("model.{b}_connector.{part}"))
                }),
            ),
        };
        let [acoustic_vb, semantic_vb] = encoders;
        let [acoustic_names, semantic_names] = branches;
        let hop = acoustic.hop_length();
        Ok(Self {
            acoustic: SpeechEncoder::new(acoustic, spec.names, acoustic_vb)?,
            semantic: SpeechEncoder::new(semantic, spec.names, semantic_vb)?,
            acoustic_branch: Branch::new(acoustic.hidden_size, hidden, acoustic_names, vb)?,
            semantic_branch: Branch::new(semantic.hidden_size, hidden, semantic_names, vb)?,
            chunk: ENCODE_SECONDS * inputs_processor::SAMPLE_RATE as usize / hop * hop,
        })
    }

    /// One embedding per hop of `(1, 1, samples)` audio: both encoders over chunks of `ENCODE_SECONDS`, each
    /// carrying its causal context into the next, as the reference encodes its minutes. The reference adds Gaussian
    /// noise to the acoustic latents (its VAE's sampling); its mean, none, is taken here, so a transcript repeats.
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
    stack: &'static str,
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
        let (stack, head) = cfg.text_prefixes();
        Ok(Self {
            audio: AudioEmbedder::new(cfg, &audio_vb)?,
            text: Qwen2::new_inner(
                &cfg.text().decoder_spec()?,
                vb.pp(stack),
                vb.pp(head),
                is_gptx,
                normal_loading_metadata,
                attention_mechanism,
            )?,
            audio_token: cfg.audio_token_id,
            stack,
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
        uvb.extend(self.text.residual_tensors_m(uvb.pp(self.stack)));
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
