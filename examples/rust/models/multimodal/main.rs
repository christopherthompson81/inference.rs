//! Multimodal streaming with combined image and audio inputs.
//!
//! Run with: `cargo run --release --example multimodal -p inference-examples`

use std::io::Write;

use anyhow::Result;
use inference::{
    AudioInput, ChatCompletionChunkResponse, ChatStreamEvent, ChunkChoice, Delta, MessageMedia,
    MultimodalMessages, MultimodalModelBuilder, TextMessageRole,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = MultimodalModelBuilder::new("microsoft/Phi-4-multimodal-instruct")
        .with_logging()
        .build()
        .await?;

    let audio_bytes = inference::fetch_url(
        "https://upload.wikimedia.org/wikipedia/commons/4/42/Bird_singing.ogg",
    )
    .await?;
    let audio = AudioInput::from_bytes(&audio_bytes)?;

    let image_bytes =
        inference::fetch_url("https://www.allaboutbirds.org/guide/assets/og/528129121-1200px.jpg")
            .await?;
    let image = image::load_from_memory(&image_bytes)?;

    let media = MessageMedia {
        images: vec![image],
        audios: vec![audio],
        ..MessageMedia::default()
    };
    let messages = MultimodalMessages::new().add_multimodal_message(
        TextMessageRole::User,
        "Describe in detail what is happening.",
        media,
    );

    let mut stream = model.stream_chat_request(messages).await?;

    while let Some(event) = stream.next().await {
        match event {
            ChatStreamEvent::Chunk(ChatCompletionChunkResponse { choices, .. }) => {
                if let Some(ChunkChoice {
                    delta:
                        Delta {
                            content: Some(content),
                            ..
                        },
                    ..
                }) = choices.first()
                {
                    print!("{content}");
                    std::io::stdout().flush()?;
                }
            }
            ChatStreamEvent::Error(error) => return Err(error.into()),
            _ => {}
        }
    }
    Ok(())
}
