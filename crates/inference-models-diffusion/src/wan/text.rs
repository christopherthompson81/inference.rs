//! The Wan text condition: umT5-XXL encoder states for a prompt, zero past its tokens up to the pipeline's length.

use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use tokenizers::Tokenizer;

use crate::t5::{Config, T5EncoderModel};

/// Prompt tokens the pipeline encodes (and pads the states to)
pub const MAX_PROMPT_TOKENS: usize = 512;

pub struct WanTextEncoder {
    model: T5EncoderModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl WanTextEncoder {
    /// From the diffusers `text_encoder` weights (in the builder's dtype) and `tokenizer/tokenizer.json`.
    pub fn new(vb: ShardedVarBuilder, cfg: &Config, tokenizer: Tokenizer) -> Result<Self> {
        let device = vb.device().clone();
        Ok(Self {
            model: T5EncoderModel::load(vb, cfg, &device, false)?,
            tokenizer,
            device,
        })
    }

    /// The prompt's token ids as the pipeline makes them: cleaned, end-of-sequence appended, cut at the maximum.
    pub fn tokens(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(clean_prompt(prompt), true)
            .map_err(inference_tensor::Error::msg)?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(MAX_PROMPT_TOKENS);
        Ok(ids)
    }

    /// States `[1, MAX_PROMPT_TOKENS, d_model]` (f32): the prompt's tokens encoded alone (the reference masks its
    /// padding, which leaves the real tokens' states unchanged), then zeros.
    pub fn encode(&mut self, prompt: &str) -> Result<Tensor> {
        let ids = self.tokens(prompt)?;
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let states = self.model.forward(&input)?.to_dtype(DType::F32)?;
        states.pad_with_zeros(1, 0, MAX_PROMPT_TOKENS - ids.len())
    }
}

// The pipeline's prompt_clean without ftfy: HTML entities undone twice, whitespace runs collapsed
fn clean_prompt(text: &str) -> String {
    let unescape = |s: &str| {
        s.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&amp;", "&")
    };
    unescape(&unescape(text))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
