use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum SpeculativeConfig {
    Off,
    Mtp(MtpConfig),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MtpDraftSamplingMethod {
    #[default]
    Auto,
    Greedy,
    Probabilistic,
}

/// MTP proposer configuration; `model: None` uses the head built into the target checkpoint.
#[derive(Clone, Debug)]
pub struct MtpConfig {
    pub model: Option<String>,
    pub n_predict: Option<usize>,
    pub draft_sampling_method: MtpDraftSamplingMethod,
    /// ISQ type for a draft-only copy of `lm_head`, so drafting skips the promoted (wider)
    /// sensitive-tensor type; the target still verifies with the promoted head.
    pub draft_lm_head_isq: Option<inference_quant::IsqType>,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MtpRuntimeConfig {
    prefix_cache_capacity: usize,
}

impl MtpRuntimeConfig {
    pub fn new(prefix_cache_capacity: usize) -> Self {
        Self {
            prefix_cache_capacity,
        }
    }

    pub fn prefix_cache_capacity(self) -> usize {
        self.prefix_cache_capacity
    }
}

impl MtpConfig {
    pub fn new(model: impl Into<String>, n_predict: Option<usize>) -> Self {
        Self {
            model: Some(model.into()),
            n_predict,
            draft_sampling_method: MtpDraftSamplingMethod::default(),
            draft_lm_head_isq: None,
        }
    }

    pub fn builtin(n_predict: Option<usize>) -> Self {
        Self {
            model: None,
            n_predict,
            draft_sampling_method: MtpDraftSamplingMethod::default(),
            draft_lm_head_isq: None,
        }
    }

    pub fn with_draft_sampling_method(mut self, method: MtpDraftSamplingMethod) -> Self {
        self.draft_sampling_method = method;
        self
    }

    pub fn with_draft_lm_head_isq(mut self, isq: Option<inference_quant::IsqType>) -> Self {
        self.draft_lm_head_isq = isq;
        self
    }

    pub fn is_builtin(&self) -> bool {
        self.model.is_none()
    }

    /// The assistant checkpoint directory; core resolves hub ids to a local snapshot before attaching.
    pub fn resolve_path(&self) -> candle_core::Result<PathBuf> {
        let Some(model) = &self.model else {
            candle_core::bail!("this MTP proposer requires a separate assistant model (`--mtp-model`), not the built-in head");
        };
        Ok(PathBuf::from(model))
    }
}
