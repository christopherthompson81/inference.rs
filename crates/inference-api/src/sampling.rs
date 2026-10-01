//! OpenAI sampling fields converted to the engine's sampling types.

use anyhow::Result;
use inference_core::{DrySamplingParams, StopTokens as InternalStopTokens};

use crate::openai::StopTokens;

/// Helper function to convert from the OpenAI stop tokens to the inference.rs
/// internal stop tokens.
/// A request's `stop` strings and `stop_token_ids` together; `None` when it gives neither.
pub fn convert_stop_tokens(
    stop_seqs: Option<StopTokens>,
    stop_token_ids: Option<Vec<u32>>,
) -> Option<InternalStopTokens> {
    let seqs = match stop_seqs {
        Some(StopTokens::Multi(sequences)) => sequences,
        Some(StopTokens::Single(sequence)) => vec![sequence],
        None => Vec::new(),
    };
    let ids = stop_token_ids.unwrap_or_default();
    (!seqs.is_empty() || !ids.is_empty()).then_some(InternalStopTokens { seqs, ids })
}

/// Helper function to get the dry sampling params.
pub fn get_dry_sampling_params(
    dry_multiplier: Option<f32>,
    dry_sequence_breakers: Option<Vec<String>>,
    dry_base: Option<f32>,
    dry_allowed_length: Option<usize>,
) -> Result<Option<DrySamplingParams>> {
    match dry_multiplier {
        Some(multiplier) => {
            let params = DrySamplingParams::new_with_defaults(
                multiplier,
                dry_sequence_breakers,
                dry_base,
                dry_allowed_length,
            )?;
            Ok(Some(params))
        }
        None => Ok(None),
    }
}
