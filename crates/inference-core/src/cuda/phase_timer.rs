use std::sync::{Arc, OnceLock};

use candle_core::cuda_backend::cudarc::driver::{sys, CudaEvent, CudaStream};

const CUDA_PHASE_TIMINGS_ENV: &str = "INFERENCE_RS_CUDA_PHASE_TIMINGS";
static CUDA_PHASE_TIMINGS_ENABLED: OnceLock<bool> = OnceLock::new();

pub(crate) struct CudaPhaseTimer {
    start: CudaEvent,
    stream: Arc<CudaStream>,
}

impl CudaPhaseTimer {
    pub(crate) fn start(stream: &Arc<CudaStream>) -> candle_core::Result<Option<Self>> {
        let enabled = *CUDA_PHASE_TIMINGS_ENABLED.get_or_init(|| {
            std::env::var(CUDA_PHASE_TIMINGS_ENV)
                .ok()
                .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        });
        if !enabled {
            return Ok(None);
        }
        let capture_status = stream.capture_status().map_err(candle_core::Error::wrap)?;
        if capture_status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            return Ok(None);
        }
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        Ok(Some(Self {
            start,
            stream: stream.clone(),
        }))
    }

    pub(crate) fn finish(
        self,
        component: &'static str,
        batch: usize,
        rows: usize,
    ) -> candle_core::Result<()> {
        let end = self
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::wrap)?;
        end.synchronize().map_err(candle_core::Error::wrap)?;
        let latency_ms = self
            .start
            .elapsed_ms(&end)
            .map_err(candle_core::Error::wrap)?;
        tracing::info!(component, batch, rows, latency_ms, "CUDA phase timing");
        Ok(())
    }
}
