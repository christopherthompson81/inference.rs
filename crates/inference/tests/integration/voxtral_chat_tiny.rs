//! A tiny random-weight Voxtral: with no template in the checkpoint the loader's Mistral layout applies, and a clip
//! opens the user turn as `[BEGIN_AUDIO]` and one 30 s chunk's run of `[AUDIO]` tokens.

use inference::{
    AudioInput, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole,
};

#[path = "../support/voxtral_chat_tiny.rs"]
mod support;
use support::tiny_checkpoint;

const PROMPT: &str = "Hi";
const SYSTEM: &str = "Brief.";
const MAX_LEN: usize = 4;
const SAMPLE_RATE: u32 = 16_000;
// 3 s pads to one 30 s chunk: 1500 encoder positions over the 4-position stacking
const SAMPLES: usize = 48_000;
const AUDIO_TOKENS: usize = 375;
// `[BEGIN_AUDIO]` before the run
const MARKERS: usize = 1;
// `<s>[INST]Hi[/INST]`: the tiny vocabulary spells "Hi" in two byte tokens
const TEXT_PROMPT_TOKENS: usize = 5;
// `[SYSTEM_PROMPT]` and `[/SYSTEM_PROMPT]` around the system prompt, one byte token per ASCII character
const SYSTEM_TOKENS: usize = 2 + SYSTEM.len();
const TONE_HZ: f32 = 440.0;

fn clip() -> AudioInput {
    AudioInput {
        samples: (0..SAMPLES)
            .map(|i| {
                0.3 * (2.0 * std::f32::consts::PI * TONE_HZ * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect(),
        sample_rate: SAMPLE_RATE,
        channels: 1,
    }
}

#[tokio::test]
async fn the_loader_template_frames_text_system_and_audio_turns() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = MultimodalModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let text = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::User, PROMPT)
                .set_sampler_max_len(MAX_LEN),
        )
        .await?;
    assert_eq!(text.usage.prompt_tokens, TEXT_PROMPT_TOKENS);
    let system = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::System, SYSTEM)
                .add_message(TextMessageRole::User, PROMPT)
                .set_sampler_max_len(MAX_LEN),
        )
        .await?;
    assert_eq!(
        system.usage.prompt_tokens,
        TEXT_PROMPT_TOKENS + SYSTEM_TOKENS
    );
    let audio = model
        .send_chat_request(
            RequestBuilder::from(MultimodalMessages::new().add_audio_message(
                TextMessageRole::User,
                PROMPT,
                vec![clip()],
            ))
            .set_sampler_max_len(MAX_LEN),
        )
        .await?;
    assert_eq!(
        audio.usage.prompt_tokens - text.usage.prompt_tokens,
        AUDIO_TOKENS + MARKERS
    );
    assert!(audio.usage.completion_tokens > 0);
    Ok(())
}
