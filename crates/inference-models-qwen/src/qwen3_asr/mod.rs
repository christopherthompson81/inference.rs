//! Qwen3-ASR: the AuT audio encoder in front of a dense Qwen3 decoder, whose `<|audio_pad|>` placeholders take the
//! encoded audio.

pub mod aligner;
pub mod audio;
pub mod config;
pub mod inputs_processor;
pub mod transcription;

use inference_quant::ShardedVarBuilder;
use inference_tensor::{Device, Result, Tensor};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    kv_cache::EitherCache,
    media_inputs::media::place_token_features,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};
use aligner::Aligner;
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
    audio_config: config::AudioConfig,
    // Qwen3-ForcedAligner's time classifier, with the tokenizer it reads transcripts with
    aligner: Option<Aligner>,
}

impl Qwen3AsrModel {
    /// `aligner_tokenizer` is the forced aligner's own tokenizer; a transcriber takes none.
    pub fn new(
        config: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
        aligner_tokenizer: Option<tokenizers::Tokenizer>,
    ) -> Result<Self> {
        let layout = config.layout;
        let cfg = &config.thinker_config;
        let device = normal_loading_metadata.real_device.clone();
        let audio_vb = vb
            .clone()
            .without_lora_registry()
            .set_device(device.clone());
        let audio = AudioEncoder::new(&cfg.audio_config, audio_vb.clone(), layout, &device)?;
        let mut text_config = cfg.text_config.qwen3();
        // the aligner's classifier takes the lm_head's place (the original repo even names it so)
        text_config.tie_word_embeddings |= config.aligner.is_some();
        let aligner = match (&config.aligner, aligner_tokenizer) {
            (Some(head), Some(tokenizer)) => {
                let weight = audio_vb
                    .pp(layout.aligner_head())
                    .get((head.classes, text_config.hidden_size), "weight")?;
                Some(Aligner::new(weight, tokenizer, head).map_err(inference_tensor::Error::msg)?)
            }
            (Some(_), None) => {
                inference_tensor::bail!("Qwen3-ForcedAligner loads with its tokenizer")
            }
            (None, _) => None,
        };
        let text = crate::qwen3::Model::new_inner(
            &text_config.decoder_spec(),
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
            audio_config: cfg.audio_config.clone(),
            aligner,
        })
    }

    // the forced aligner's pass: the clip's placeholders and the transcript's words with their slots, run once outside
    // the KV cache, the head read at the slots
    fn align(
        &self,
        aligner: &Aligner,
        audio: &inference_audio::AudioInput,
        transcript: &str,
        language: Option<&str>,
    ) -> anyhow::Result<Vec<inference_nn::loaders::AlignedWord>> {
        if let Some(language) = language {
            anyhow::ensure!(
                aligner::LANGUAGES
                    .iter()
                    .any(|(code, name)| language.eq_ignore_ascii_case(code)
                        || language.eq_ignore_ascii_case(name)),
                "Qwen3-ForcedAligner does not time language `{language}`"
            );
        }
        let mono = audio.to_mono();
        let pcm = if audio.sample_rate == inputs_processor::SAMPLE_RATE {
            mono
        } else {
            inference_audio::mel::resample(&mono, audio.sample_rate, inputs_processor::SAMPLE_RATE)?
                .into_owned()
        };
        #[allow(clippy::cast_precision_loss)]
        let seconds = pcm.len() as f64 / f64::from(inputs_processor::SAMPLE_RATE);
        anyhow::ensure!(
            seconds <= aligner.max_seconds(),
            "Qwen3-ForcedAligner times clips of up to {} s; this one is {seconds:.1} s",
            aligner.max_seconds()
        );
        let words = aligner::split_words(transcript);
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let device = self.text.device();
        let placeholders = inputs_processor::clip_tokens(&self.audio_config, &pcm);
        let ids = aligner.prompt(&words, self.audio_token, placeholders)?;
        let input_ids = Tensor::from_vec(ids.clone(), (1, ids.len()), device)?;
        let clip = inputs_processor::clip_features(&self.audio_config, &pcm, device)?;
        let encoded = self.audio.forward(&clip.mel, clip.valid_frames)?;
        let embeds = place_token_features(
            &self.text.get_input_embeddings(&input_ids)?,
            &input_ids,
            self.audio_token,
            &encoded,
            "Qwen3-ForcedAligner",
        )?;
        let hidden = self.text.hidden_states(
            &input_ids,
            embeds,
            &crate::attention::FlashParams::empty(true),
        )?;
        aligner.read(&hidden, &ids, words)
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
            embeds =
                place_token_features(&embeds, input_ids, self.audio_token, &encoded, "Qwen3-ASR")?;
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

    fn word_timestamps(
        &self,
        audio: &inference_audio::AudioInput,
        transcript: &str,
        language: Option<&str>,
    ) -> anyhow::Result<Option<Vec<inference_nn::loaders::AlignedWord>>> {
        match &self.aligner {
            Some(aligner) => self.align(aligner, audio, transcript, language).map(Some),
            None => Ok(None),
        }
    }
}
