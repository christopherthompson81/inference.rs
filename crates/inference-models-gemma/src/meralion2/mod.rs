//! MERaLiON-2: the Whisper encoder over 30 s chunks, a layer norm, 15 positions stacked into a gated MLP adapter, and
//! a Gemma 2 decoder whose `<SpeechHere>` placeholders take the adapted audio.
#![allow(clippy::cast_precision_loss)]

pub mod config;
pub mod inputs_processor;
pub mod transcription;

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{LayerNorm, Linear, Module};
use inference_tensor::{Device, Result, Tensor};

use crate::gemma2::Model as Gemma2;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    audio::whisper::WhisperEncoder,
    kv_cache::EitherCache,
    layers::{self, linear_b},
    media_inputs::audio_prompt::place_audio_features,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};
pub use config::MERaLiON2Config;

const MODEL: &str = "MERaLiON-2";
const LN_SPEECH_EPS: f64 = 1e-5;

/// One clip's `(chunks, n_mels, frames)` log-mel, each chunk a padded 30 s window.
pub struct AudioClip {
    pub mel: Tensor,
}

pub struct MERaLiON2SpecificArgs {
    /// The clips of the batch whose placeholders this step holds, in prompt order.
    pub audio: Vec<AudioClip>,
}

pub struct MERaLiON2Model {
    text: Gemma2,
    audio: WhisperEncoder,
    ln_speech: LayerNorm,
    mlp_adapter: Linear,
    gate_proj: Linear,
    pool_proj: Linear,
    out_proj: Linear,
    stacked_width: usize,
    // Gemma scales its input embeddings, the audio placed among them included
    embed_scale: f64,
    speech_token: u32,
}

impl MERaLiON2Model {
    pub fn new(
        cfg: &MERaLiON2Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let audio_vb = vb
            .clone()
            .without_lora_registry()
            .set_device(normal_loading_metadata.real_device.clone());
        let audio = WhisperEncoder::new(&cfg.encoder(), audio_vb.pp("speech_encoder"))?;
        let d = cfg.speech_config.d_model;
        let ln_speech = layers::layer_norm(d, LN_SPEECH_EPS, audio_vb.pp("ln_speech"))?;
        let adapter = audio_vb.pp("speech_audio_adapter");
        let (wide, text) = (cfg.adapter_width(), cfg.text_config());
        let mlp_adapter = linear_b(
            cfg.stacked_width(),
            wide,
            true,
            adapter.pp("mlp_adapter").pp("0"),
        )?;
        let gate_proj = linear_b(wide, wide, true, adapter.pp("gate_proj"))?;
        let pool_proj = linear_b(wide, wide, true, adapter.pp("pool_proj"))?;
        let out_proj = linear_b(wide, text.hidden_size, true, adapter.pp("out_proj"))?;
        let embed_scale = (text.hidden_size as f64).sqrt();
        let text = Gemma2::new(
            &text.decoder_spec()?,
            vb.pp("text_decoder"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            audio,
            ln_speech,
            mlp_adapter,
            gate_proj,
            pool_proj,
            out_proj,
            stacked_width: cfg.stacked_width(),
            embed_scale,
            speech_token: cfg.speech_token_index,
        })
    }

    // the encoder reads every whole padded chunk, unmasked, as transformers' Whisper encoder does
    fn embed_audio(&self, clip: &AudioClip) -> Result<Tensor> {
        let encoded = self
            .audio
            .forward(&clip.mel.to_dtype(self.mlp_adapter.weight().dtype())?)?;
        let xs = self.ln_speech.forward(&encoded)?;
        let (chunks, _, _) = xs.dims3()?;
        let xs = xs.reshape((chunks, (), self.stacked_width))?;
        let xs = self.mlp_adapter.forward(&xs)?.silu()?;
        let gated = (self.gate_proj.forward(&xs)?.silu()? * self.pool_proj.forward(&xs)?)?;
        let projected = self.out_proj.forward(&gated)?;
        let (chunks, tokens, hidden) = projected.dims3()?;
        projected.reshape((chunks * tokens, hidden))? * self.embed_scale
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        args: MERaLiON2SpecificArgs,
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
            embeds =
                place_audio_features(&embeds, input_ids, self.speech_token, &projected, MODEL)?;
        }
        self.text.forward_embeds(input_ids, embeds, ctx)
    }
}

impl IsqModel for MERaLiON2Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(self.text.residual_tensors_m(uvb.pp("text_decoder")));
        uvb.pp("ln_speech").add(&self.ln_speech);
        let adapter = uvb.pp("speech_audio_adapter");
        adapter.pp("mlp_adapter").pp("0").add(&self.mlp_adapter);
        adapter.pp("gate_proj").add(&self.gate_proj);
        adapter.pp("pool_proj").add(&self.pool_proj);
        adapter.pp("out_proj").add(&self.out_proj);
        let mut tensors = uvb.to_safetensors();
        tensors.extend(
            self.audio
                .residual_tensors()
                .into_iter()
                .map(|(name, tensor)| (format!("speech_encoder.{name}"), tensor)),
        );
        tensors
    }
}

impl crate::speculative::SpeculativeTargetMixin for MERaLiON2Model {}
impl crate::model::BlockDiffusionMixin for MERaLiON2Model {}
impl AnyMoeBaseModelMixin for MERaLiON2Model {}

impl MultimodalModel for MERaLiON2Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<MERaLiON2SpecificArgs>()
            .expect("Cannot downcast into `MERaLiON2SpecificArgs`");
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
        Box::new(MERaLiON2SpecificArgs { audio: Vec::new() })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}
