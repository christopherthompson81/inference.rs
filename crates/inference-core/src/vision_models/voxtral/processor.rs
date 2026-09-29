use std::sync::Arc;

use inference_models_llama::voxtral::audio_processing::VoxtralAudioProcessor;
use inference_models_llama::voxtral::config::VoxtralConfig;
use inference_models_llama::voxtral::inputs_processor::VoxtralInputsProcessor;

use crate::pipeline::{InputsProcessor, MessagesAction, Processor};
use crate::vision_models::media_host::MediaInputsProcessor;

const AUDIO_ENCODER_DOWNSAMPLE_FACTOR: usize = 2;

pub struct VoxtralProcessor {
    audio_processor: VoxtralAudioProcessor,
    audio_length_per_tok: usize,
}

impl VoxtralProcessor {
    pub fn new(cfg: &VoxtralConfig) -> Self {
        let enc_args = &cfg.multimodal.whisper_model_args.encoder_args;
        Self {
            audio_processor: VoxtralAudioProcessor::new(&enc_args.audio_encoding_args),
            audio_length_per_tok: AUDIO_ENCODER_DOWNSAMPLE_FACTOR
                * cfg
                    .multimodal
                    .whisper_model_args
                    .downsample_args
                    .downsample_factor,
        }
    }
}

impl Processor for VoxtralProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MediaInputsProcessor(Arc::new(VoxtralInputsProcessor::new(
            VoxtralAudioProcessor::new_from_processor(&self.audio_processor),
            self.audio_length_per_tok,
        ))))
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}
