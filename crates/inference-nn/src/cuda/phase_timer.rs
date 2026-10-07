use std::sync::{Arc, OnceLock};

use inference_tensor::cuda_backend::cudarc::driver::{CudaEvent, CudaStream, sys};

const CUDA_PHASE_TIMINGS_ENV: &str = "INFERENCE_RS_CUDA_PHASE_TIMINGS";
static CUDA_PHASE_TIMINGS_ENABLED: OnceLock<bool> = OnceLock::new();

pub struct CudaPhaseTimer {
    start: CudaEvent,
    stream: Arc<CudaStream>,
}

impl CudaPhaseTimer {
    pub fn start(stream: &Arc<CudaStream>) -> inference_tensor::Result<Option<Self>> {
        let enabled = *CUDA_PHASE_TIMINGS_ENABLED.get_or_init(|| {
            std::env::var(CUDA_PHASE_TIMINGS_ENV)
                .ok()
                .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        });
        if !enabled {
            return Ok(None);
        }
        let capture_status = stream
            .capture_status()
            .map_err(inference_tensor::Error::wrap)?;
        if capture_status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Ok(None);
        }
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(inference_tensor::Error::wrap)?;
        Ok(Some(Self {
            start,
            stream: stream.clone(),
        }))
    }

    pub fn finish(
        self,
        component: &'static str,
        batch: usize,
        rows: usize,
    ) -> inference_tensor::Result<()> {
        let end = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(inference_tensor::Error::wrap)?;
        end.synchronize().map_err(inference_tensor::Error::wrap)?;
        let latency_ms = self
            .start
            .elapsed_ms(&end)
            .map_err(inference_tensor::Error::wrap)?;
        tracing::info!(component, batch, rows, latency_ms, "CUDA phase timing");
        Ok(())
    }
}
