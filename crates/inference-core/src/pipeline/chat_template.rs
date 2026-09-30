//! What core needs beside the templates themselves: the model's `generation_config.json` and its EOS tokens.

use anyhow::Result;
use either::Either;
use inference_protocol::chat_template::ChatTemplate;
use itertools::Itertools;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::{trace, warn};

use crate::ModelGenerationDefaults;

const SUPPORTED_ALTERNATE_EOS: &[&str] = &[
    "<|im_end|>",      // Handle ChatML case
    "<end_of_turn>",   // Handle Gemma2 chat case
    "<|end_of_text|>", // Hermes
    "<|end|>",         // Phi-3, Phi-3.5, Harmony
    "<|eot_id|>",      // Llama 3
];

const HARMONY_ALTERNATE_EOS: &[&str] = &[
    "<|message|>", // Harmony
    "<|start|>",   // Harmony
    "<|channel|>", // Harmony
];

pub fn calculate_eos_tokens(
    chat_template: &ChatTemplate,
    gen_conf: Option<&GenerationConfig>,
    tokenizer: &Tokenizer,
) -> Vec<u32> {
    let mut eos_tok_ids = chat_template.eos_tok().map(|x| vec![x]).unwrap_or_default();
    let mut bos_tok_ids = chat_template.bos_tok().map(|b| vec![b]).unwrap_or_default();

    let templates = chat_template.get_template_contents();

    for alternate in SUPPORTED_ALTERNATE_EOS {
        if tokenizer.get_vocab(true).contains_key(*alternate)
            && templates.iter().any(|t| t.contains(*alternate))
        {
            eos_tok_ids.push(alternate.to_string())
        }
    }
    if chat_template.is_harmony_format() {
        for alternate in HARMONY_ALTERNATE_EOS {
            if tokenizer.get_vocab(true).contains_key(*alternate)
                && templates
                    .iter()
                    .any(|template| template.contains(*alternate))
            {
                eos_tok_ids.push(alternate.to_string());
            }
        }
    }

    if let Some(gen_conf) = gen_conf {
        if let Some(eos_field) = gen_conf.eos_token_id.as_ref() {
            let ids = match eos_field {
                Either::Left(id) => vec![*id],
                Either::Right(ids) => ids.clone(),
            };
            for id in ids {
                let Ok(s) = tokenizer.decode(&[id], false) else {
                    warn!(
                        "Ignoring generation config EOS token id {id}: not in the tokenizer vocabulary"
                    );
                    continue;
                };
                if !eos_tok_ids.contains(&s) {
                    eos_tok_ids.push(s);
                }
            }
        }

        if let Some(bos_field) = gen_conf.bos_token_id.as_ref() {
            let ids = match bos_field {
                Either::Left(id) => vec![*id],
                Either::Right(ids) => ids.clone(),
            };
            for id in ids {
                let s = tokenizer
                    .decode(&[id], false)
                    .unwrap_or_else(|_| panic!("Unable to decode id {id})"));
                if !bos_tok_ids.contains(&s) {
                    bos_tok_ids.push(s);
                }
            }
        }
    }

    eos_tok_ids = eos_tok_ids.into_iter().dedup().collect::<Vec<_>>();
    bos_tok_ids = bos_tok_ids.into_iter().dedup().collect::<Vec<_>>();

    let bos_render = bos_tok_ids
        .iter()
        .map(|val| format!("{val:?}"))
        .collect::<Vec<String>>()
        .join(", ");
    let eos_render = eos_tok_ids
        .iter()
        .map(|val| format!("{val:?}"))
        .collect::<Vec<String>>()
        .join(", ");

    trace!(
        "bos_toks = {bos_render}, eos_toks = {eos_render}, unk_tok = {}",
        chat_template.unk_tok().unwrap_or("`None`".to_string()),
    );

    let mut eos_toks = Vec::new();
    for eos_tok in eos_tok_ids {
        eos_toks.push(
            tokenizer
                .get_vocab(true)
                .get(&eos_tok)
                .copied()
                .unwrap_or_else(|| panic!("Unable to extract `{eos_tok}` EOS token.")),
        )
    }
    eos_toks
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    #[serde(default)]
    #[serde(with = "either::serde_untagged_optional")]
    bos_token_id: Option<Either<u32, Vec<u32>>>,
    #[serde(default)]
    #[serde(with = "either::serde_untagged_optional")]
    eos_token_id: Option<Either<u32, Vec<u32>>>,
    #[serde(default)]
    do_sample: Option<bool>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    min_p: Option<f64>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    max_new_tokens: Option<usize>,
    #[serde(default)]
    max_length: Option<usize>,
    #[serde(default)]
    suppress_tokens: Option<Vec<u32>>,
}

impl GenerationConfig {
    /// HF `GenerationConfig.from_model_config`: without a generation_config.json the model config's own
    /// generation fields apply, with a nested `text_config` filling anything the top level leaves unset.
    pub fn from_model_config(config_json: &str) -> Option<Self> {
        let raw: serde_json::Value = serde_json::from_str(config_json).ok()?;
        let mut conf: GenerationConfig = serde_json::from_value(raw.clone()).ok()?;
        if let Some(nested) = raw.get("text_config").cloned()
            && let Ok(nested) = serde_json::from_value::<GenerationConfig>(nested)
        {
            conf.bos_token_id = conf.bos_token_id.or(nested.bos_token_id);
            conf.eos_token_id = conf.eos_token_id.or(nested.eos_token_id);
            conf.do_sample = conf.do_sample.or(nested.do_sample);
            conf.temperature = conf.temperature.or(nested.temperature);
            conf.top_k = conf.top_k.or(nested.top_k);
            conf.top_p = conf.top_p.or(nested.top_p);
            conf.min_p = conf.min_p.or(nested.min_p);
            conf.repetition_penalty = conf.repetition_penalty.or(nested.repetition_penalty);
        }
        conf.max_new_tokens = None;
        conf.max_length = None;
        Some(conf)
    }

    pub(crate) fn validate_token_ids(&self, vocab_size: usize) -> Result<()> {
        for (field, value) in [
            ("bos_token_id", self.bos_token_id.as_ref()),
            ("eos_token_id", self.eos_token_id.as_ref()),
        ] {
            let Some(value) = value else {
                continue;
            };
            let ids = match value {
                Either::Left(id) => std::slice::from_ref(id),
                Either::Right(ids) => ids.as_slice(),
            };
            for id in ids {
                anyhow::ensure!(
                    usize::try_from(*id).is_ok_and(|id| id < vocab_size),
                    "generation config `{field}` contains token ID {id}, but the tokenizer vocabulary has {vocab_size} entries"
                );
            }
        }
        if let Some(ids) = self.suppress_tokens.as_ref() {
            for id in ids {
                anyhow::ensure!(
                    usize::try_from(*id).is_ok_and(|id| id < vocab_size),
                    "generation config `suppress_tokens` contains token ID {id}, but the tokenizer vocabulary has {vocab_size} entries"
                );
            }
        }
        Ok(())
    }

    pub fn generation_defaults(&self) -> Option<ModelGenerationDefaults> {
        let defaults = ModelGenerationDefaults {
            do_sample: self.do_sample,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            max_new_tokens: self.max_new_tokens,
            max_length: self.max_length,
            suppress_tokens: self.suppress_tokens.clone(),
        };

        if defaults.is_empty() {
            None
        } else {
            Some(defaults)
        }
    }
}

#[cfg(test)]
mod tests {
    use inference_protocol::chat_template::ChatTemplate;
    use tokenizers::Tokenizer;

    use super::{GenerationConfig, calculate_eos_tokens};

    #[test]
    fn generation_config_token_ids_must_fit_the_tokenizer_vocabulary() {
        let valid: GenerationConfig = serde_json::from_value(serde_json::json!({
            "bos_token_id": 0,
            "eos_token_id": [1, 2],
            "suppress_tokens": [3]
        }))
        .unwrap();
        assert!(valid.validate_token_ids(4).is_ok());

        let invalid: GenerationConfig = serde_json::from_value(serde_json::json!({
            "eos_token_id": [1, 4]
        }))
        .unwrap();
        assert!(invalid.validate_token_ids(4).is_err());
    }

    #[test]
    fn muse_channel_tokens_are_not_alternate_eos() {
        use ahash::AHashMap;
        use tokenizers::models::wordlevel::WordLevel;

        let vocab = [
            ("<unk>".to_string(), 0),
            ("<|eot|>".to_string(), 1),
            ("<|start|>".to_string(), 2),
            ("<|message|>".to_string(), 3),
        ]
        .into_iter()
        .collect::<AHashMap<_, _>>();
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("<unk>".to_string())
                .build()
                .unwrap(),
        );
        let template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "eos_token": "<|eot|>",
            "chat_template": "<|start|>assistant to=user<|message|>"
        }))
        .unwrap();

        assert_eq!(calculate_eos_tokens(&template, None, &tokenizer), vec![1]);
    }

    #[test]
    fn generation_config_exposes_sampling_defaults() {
        let config: GenerationConfig = serde_json::from_str(
            r#"{
                "do_sample": true,
                "temperature": 1.0,
                "top_k": 32,
                "top_p": 0.9,
                "min_p": 0.05,
                "repetition_penalty": 1.1,
                "max_new_tokens": 512,
                "suppress_tokens": [258882, 258883]
            }"#,
        )
        .unwrap();

        let defaults = config.generation_defaults().unwrap();
        assert_eq!(defaults.do_sample, Some(true));
        assert_eq!(defaults.temperature, Some(1.0));
        assert_eq!(defaults.top_k, Some(32));
        assert_eq!(defaults.top_p, Some(0.9));
        assert_eq!(defaults.min_p, Some(0.05));
        assert_eq!(defaults.repetition_penalty, Some(1.1));
        assert_eq!(defaults.max_new_tokens, Some(512));
        assert_eq!(defaults.suppress_tokens, Some(vec![258882, 258883]));
    }

    #[test]
    fn generation_config_keeps_omitted_sampling_fields_unset() {
        let config: GenerationConfig = serde_json::from_str(
            r#"{
                "do_sample": true,
                "temperature": 1.0
            }"#,
        )
        .unwrap();

        let defaults = config.generation_defaults().unwrap();
        assert_eq!(defaults.do_sample, Some(true));
        assert_eq!(defaults.temperature, Some(1.0));
        assert_eq!(defaults.top_k, None);
        assert_eq!(defaults.top_p, None);
        assert_eq!(defaults.repetition_penalty, None);
        assert_eq!(defaults.max_new_tokens, None);
        assert_eq!(defaults.max_length, None);
        assert_eq!(defaults.suppress_tokens, None);
    }
}
