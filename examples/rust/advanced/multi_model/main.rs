//! Load and dispatch requests across multiple models simultaneously.
//!
//! Run with: `cargo run --release --example multi_model -p inference-examples`

use anyhow::Result;
use inference::{
    IsqBits, ModelOperationRequest, ModelStatus, MultiModelBuilder, MultimodalModelBuilder,
    RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};

// Model IDs - these are the actual HuggingFace model paths
const GEMMA_MODEL_ID: &str = "google/gemma-4-E4B-it";
const QWEN_MODEL_ID: &str = "Qwen/Qwen3-4B";
// Aliases - these are the short IDs used in API requests
const GEMMA_ALIAS: &str = "gemma-multimodal";
const QWEN_ALIAS: &str = "qwen-text";

fn target(model_id: &str) -> ModelOperationRequest {
    ModelOperationRequest {
        model_id: model_id.to_string(),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Loading multiple models...");

    let model = MultiModelBuilder::new()
        .add_model_with_alias(
            GEMMA_ALIAS,
            MultimodalModelBuilder::new(GEMMA_MODEL_ID).with_auto_isq(IsqBits::Four),
        )
        .add_model_with_alias(
            QWEN_ALIAS,
            TextModelBuilder::new(QWEN_MODEL_ID).with_auto_isq(IsqBits::Four),
        )
        .with_default_model(GEMMA_ALIAS)
        .with_logging()
        .build()
        .await?;

    // List available models
    println!("\n=== Available Models ===");
    let models = model.models()?;
    for card in &models.data {
        println!("  - {}", card.id);
    }

    // Get the default model
    let default_model = model.default_model_id();
    println!("\nDefault model: {:?}", default_model);

    // List models with their status
    println!("\n=== Model Status ===");
    for card in &model.models()?.data {
        println!("  {} -> {:?}", card.id, card.status);
    }

    // Send a request to the default model (Gemma - multimodal model)
    println!("\n=== Request to Default Model ({}) ===", GEMMA_ALIAS);
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "What is 2 + 2? Answer briefly.");

    let response = model.send_chat_request(messages).await?;
    println!(
        "Response: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Send a request to a specific model (Qwen - text model)
    println!("\n=== Request to Specific Model ({}) ===", QWEN_ALIAS);
    let messages = TextMessages::new().add_message(TextMessageRole::User, "Say hello in one word.");

    let response = model
        .send_chat_request(RequestBuilder::from(messages).with_model(QWEN_ALIAS))
        .await?;
    println!(
        "Response: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Change the default model
    println!("\n=== Changing Default Model ===");
    model.set_default_model(target(QWEN_ALIAS))?;
    let new_default = model.default_model_id();
    println!("New default model: {:?}", new_default);

    // Now requests without model_id go to Qwen
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "What is your name? Be brief.");

    let response = model.send_chat_request(messages).await?;
    println!(
        "Response from new default: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Model unloading/reloading demonstration
    println!("\n=== Model Unloading/Reloading ===");

    // Check if Gemma is loaded
    let is_gemma_loaded = model.model_status(target(GEMMA_ALIAS))?.status == ModelStatus::Loaded;
    println!("Is '{}' loaded? {}", GEMMA_ALIAS, is_gemma_loaded);

    // Unload Gemma to free memory
    println!("Unloading '{}' model...", GEMMA_ALIAS);
    model.unload_model(target(GEMMA_ALIAS))?;

    // Check status after unload
    println!("Status after unload:");
    for card in &model.models()?.data {
        println!("  {} -> {:?}", card.id, card.status);
    }

    // Reload Gemma when needed
    println!("Reloading '{}' model...", GEMMA_ALIAS);
    let is_gemma_loaded =
        model.reload_model(target(GEMMA_ALIAS)).await?.status == ModelStatus::Loaded;
    println!(
        "Is '{}' loaded after reload? {}",
        GEMMA_ALIAS, is_gemma_loaded
    );

    // Use the reloaded model
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "Hi! Respond with just 'Hello'.");

    let response = model
        .send_chat_request(RequestBuilder::from(messages).with_model(GEMMA_ALIAS))
        .await?;
    println!(
        "Response from reloaded {}: {}",
        GEMMA_ALIAS,
        response.choices[0].message.content.as_ref().unwrap()
    );

    println!("\n=== Multi-Model Example Complete ===");

    Ok(())
}
