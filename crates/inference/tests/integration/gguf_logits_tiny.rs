//! A quantized GGUF head sums in f32, so a bf16 model still returns f32 logits rather than bf16 values widened.

use inference::{GgufModelBuilder, LogitsOutput, ModelDType, PromptInput, PromptLogitsRequest};
use inference_tensor::quantized::GgmlDType;

#[path = "../support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;
#[path = "../support/llama_tiny_gguf.rs"]
mod tiny_gguf;
use tiny_gguf::write_tiny_gguf_with_head;

const GGUF_FILE: &str = "tiny.gguf";
const PROMPT: [u32; 4] = [40, 41, 42, 43];
// a bf16 value widened to f32 keeps only the high half of its bits
const BF16_LOW_BITS: u32 = 0xffff;

#[tokio::test]
async fn a_bf16_gguf_model_returns_f32_logits() -> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let dir = tempfile::tempdir()?;
    write_tiny_gguf_with_head(
        checkpoint.path(),
        &dir.path().join(GGUF_FILE),
        &[],
        GgmlDType::Q8_0,
    )?;
    let model = GgufModelBuilder::new(dir.path().to_string_lossy(), vec![GGUF_FILE])
        .with_tok_model_id(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_force_cpu()
        .build()
        .await?;
    let scored = model
        .prompt_logits(PromptLogitsRequest {
            model: None,
            prompt: PromptInput::Tokens(PROMPT.to_vec()),
            output: LogitsOutput::Logits,
        })
        .await?;
    let logits = scored.logits.expect("the request asked for logits");
    assert!(
        logits
            .iter()
            .any(|logit| logit.to_bits() & BF16_LOW_BITS != 0),
        "every logit is a bf16 value"
    );
    Ok(())
}
