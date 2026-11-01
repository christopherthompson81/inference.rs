//! The integration tests, one binary so the stack is monomorphized and linked once.
// Each fixture includes the recording helpers itself, so server-core and ffi can include one fixture alone.
#![allow(clippy::duplicate_mod)]

mod audio_clips;
mod bert_embedding_tiny;
mod decision_label_tiny;
mod decision_tiny;
mod embedding_tiny;
mod ernie4_5_tiny;
mod gemma3_tiny;
mod gguf_embedding_tiny;
mod gguf_iq;
mod gguf_logits_tiny;
mod gguf_lora_tiny;
mod granite4_vision_tiny;
mod granite_speech_tiny;
mod granite_tiny;
mod hidream_o1_tiny;
mod jina_v5_tiny;
mod kokoro_tiny;
mod llama_head_dim_tiny;
mod llama_tiny;
mod llava_tiny;
mod local_attention_tiny;
mod nemotron3_diarization_tiny;
mod olmo_tiny;
mod paddleocr_vl;
mod paddleocr_vl_tiny;
mod parakeet_tiny;
mod qwen3_5_mtp;
mod qwen3_5_text_tiny;
mod qwen3_5_tool_call;
mod qwen3_asr_tiny;
mod qwen3_sliding_window_tiny;
mod qwen4exp_text_tiny;
mod qwen_vl_tiny;
mod rope_encoder_embedding_tiny;
mod silero_tiny;
mod sortformer_tiny;
mod voxtral_chat_tiny;
mod whisper_tiny;
