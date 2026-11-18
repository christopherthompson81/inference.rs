//! The STL the image jobs write beside their models to 3D print: the object as one closed solid standing on z = 0,
//! its longest side a chosen size in millimetres.

use std::ops::RangeInclusive;
use std::path::Path;
use std::time::Instant;

use inference_recon::printable::Solid;
use serde::Serialize;

pub use inference_recon::printable::DEFAULT_PRINT_SIZE_MM;

pub const PRINT_FILE: &str = "print.stl";
/// The sizes a print accepts, in millimetres
pub const PRINT_SIZE_RANGE_MM: RangeInclusive<f64> = 5.0..=2000.0;
const MM3_PER_CM3: f64 = 1000.0;

#[derive(Clone, Debug, Serialize)]
pub struct PrintReport {
    /// The longest side, in millimetres
    pub size_mm: f64,
    pub faces: usize,
    pub volume_cm3: f64,
    /// Making the solid and its surface, on the CPU
    pub seconds: f64,
}

/// Writes `solid` as `print.stl` into `dir`; `None` (and no file) when nothing came out solid. `started` is when
/// making the solid began.
pub fn write_print(
    solid: &Solid,
    dir: &Path,
    size_mm: f64,
    started: Instant,
) -> anyhow::Result<Option<PrintReport>> {
    let Some(print) = solid.printable(size_mm) else {
        tracing::warn!("the object came out with no solid to print");
        return Ok(None);
    };
    std::fs::write(dir.join(PRINT_FILE), &print.stl)?;
    Ok(Some(PrintReport {
        size_mm,
        faces: print.faces,
        volume_cm3: print.volume_mm3 / MM3_PER_CM3,
        seconds: started.elapsed().as_secs_f64(),
    }))
}
