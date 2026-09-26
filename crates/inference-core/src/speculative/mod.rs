pub mod cache;
pub mod config;
pub mod dflash;
pub mod driver;
pub(crate) mod staging;
pub mod verifier;

#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use crate::cuda::speculative_rejection::CudaSparseRejectionWorkspace;
pub use config::{
    reserve_external_mtp_memory, reserve_external_mtp_memory_with_runtime,
    resolve_speculative_model,
};
pub use dflash::DFlashDraftModel;
pub use inference_nn::speculative::*;
pub use inference_nn::speculative::{logging, paged_rows, policy, proposer, target};

#[cfg(test)]
mod tests {
    use super::{MtpConfig, MtpDraftSamplingMethod};

    #[test]
    fn mtp_config_supports_public_struct_literals() {
        let config = MtpConfig {
            model: Some("assistant".to_string()),
            n_predict: Some(3),
            draft_sampling_method: MtpDraftSamplingMethod::Probabilistic,
            draft_lm_head_isq: None,
        };

        assert_eq!(config.n_predict, Some(3));
    }
}
