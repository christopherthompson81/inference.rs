use inference_models_llama::loaders::VoxtralLoader;
use inference_models_llama::voxtral::audio_processing::VoxtralAudioProcessor;
use inference_models_llama::voxtral::config::VoxtralConfig;
use inference_models_llama::voxtral::inputs_processor::VoxtralInputsProcessor;

use crate::pipeline::MessagesAction;
use crate::vision_models::media_host::media_processor;

const AUDIO_ENCODER_DOWNSAMPLE_FACTOR: usize = 2;

processor_factory!(VoxtralLoader => |model_config, _, _, _| {
    let cfg = VoxtralConfig::from_json(model_config).expect("Failed to parse VoxtralConfig");
    let whisper = &cfg.multimodal.whisper_model_args;
    let audio_length_per_tok =
        AUDIO_ENCODER_DOWNSAMPLE_FACTOR * whisper.downsample_args.downsample_factor;
    let audio = VoxtralAudioProcessor::new(&whisper.encoder_args.audio_encoding_args);
    let inputs = VoxtralInputsProcessor::new(audio, audio_length_per_tok);
    media_processor(inputs, &[], MessagesAction::FlattenOnlyText)
});
