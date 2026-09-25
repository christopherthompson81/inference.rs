use std::ffi::{c_char, CString};
use std::sync::Arc;

use image::RgbImage;
use inference_layout::pp_doclayout_v3::{
    LayoutDetection, PPDocLayoutV3Detector, DEFAULT_THRESHOLD,
};

use crate::inference_status::*;
use crate::FfiResult;
use crate::{
    arg_str, backend_from, guard, guard_value, inference_backend_config, inference_status,
    write_opt, Failure,
};

// the header promises a model handle may be shared across threads
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PPDocLayoutV3Detector>();
};

/// Largest accepted image, in pixels (a 600-dpi A0 page is ~70M); bigger inputs are rejected instead of allocated.
pub const MAX_IMAGE_PIXELS: usize = 1 << 28;

/// Mirrors the `inference.h` constants of the same names.
pub const INFERENCE_LAYOUT_DEFAULT_THRESHOLD: f32 = -1.0;
pub const INFERENCE_PIXEL_RGB8: i32 = 0;
pub const INFERENCE_PIXEL_BGR8: i32 = 1;
pub const INFERENCE_PIXEL_RGBA8: i32 = 2;
pub const INFERENCE_PIXEL_BGRA8: i32 = 3;
pub const INFERENCE_PIXEL_GRAY8: i32 = 4;

/// Opaque `inference_layout_model`.
#[allow(non_camel_case_types)]
pub struct inference_layout_model {
    detector: PPDocLayoutV3Detector,
    /// NUL-terminated class names by id, shared with every result so labels outlive the model.
    labels: Arc<[CString]>,
}

/// Opaque `inference_layout_result`; detections in reading order.
#[allow(non_camel_case_types)]
pub struct inference_layout_result {
    detections: Vec<LayoutDetection>,
    labels: Arc<[CString]>,
}

/// Mirrors `inference_image`.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct inference_image {
    pub pixels: *const u8,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: i32,
}

/// `(bytes per pixel, R/G/B byte offsets)` for an `inference_pixel_format` value.
fn pixel_layout(format: i32) -> FfiResult<(usize, [usize; 3])> {
    Ok(match format {
        INFERENCE_PIXEL_RGB8 => (3, [0, 1, 2]),
        INFERENCE_PIXEL_BGR8 => (3, [2, 1, 0]),
        INFERENCE_PIXEL_RGBA8 => (4, [0, 1, 2]),
        INFERENCE_PIXEL_BGRA8 => (4, [2, 1, 0]),
        INFERENCE_PIXEL_GRAY8 => (1, [0, 0, 0]),
        other => return Err(Failure::invalid(format!("unknown pixel format {other}"))),
    })
}

/// Copies a caller image into an `RgbImage`. Safety: `img.pixels` covers `stride * (height - 1) + width * bpp` bytes.
unsafe fn to_rgb(img: &inference_image) -> FfiResult<RgbImage> {
    if img.pixels.is_null() {
        return Err(Failure::invalid("image.pixels is NULL"));
    }
    if img.width == 0 || img.height == 0 {
        return Err(Failure::invalid(format!(
            "image is {}x{}",
            img.width, img.height
        )));
    }
    let (bpp, [r, g, b]) = pixel_layout(img.format)?;
    let (w, h) = (img.width as usize, img.height as usize);
    let row_bytes = w
        .checked_mul(bpp)
        .ok_or_else(|| Failure::invalid("image row size overflows"))?;
    let stride = if img.stride == 0 {
        row_bytes
    } else {
        img.stride as usize
    };
    if stride < row_bytes {
        return Err(Failure::invalid(format!(
            "stride {stride} is smaller than a {row_bytes}-byte row"
        )));
    }
    let pixels = w
        .checked_mul(h)
        .filter(|&n| n <= MAX_IMAGE_PIXELS)
        .ok_or_else(|| {
            Failure::invalid(format!("image {w}x{h} exceeds {MAX_IMAGE_PIXELS} pixels"))
        })?;
    let mut out = Vec::new();
    out.try_reserve_exact(pixels * 3).map_err(|_| {
        Failure::new(
            INFERENCE_ERR_RUNTIME,
            format!("out of memory for a {w}x{h} image"),
        )
    })?;
    out.resize(pixels * 3, 0u8);
    for (y, dst) in out.chunks_exact_mut(w * 3).enumerate() {
        let src = std::slice::from_raw_parts(img.pixels.add(y * stride), row_bytes);
        for (px, d) in src.chunks_exact(bpp).zip(dst.as_chunks_mut::<3>().0) {
            d.copy_from_slice(&[px[r], px[g], px[b]]);
        }
    }
    RgbImage::from_raw(img.width, img.height, out)
        .ok_or_else(|| Failure::invalid("image buffer size mismatch"))
}

fn threshold_arg(threshold: f32) -> FfiResult<f32> {
    if threshold == INFERENCE_LAYOUT_DEFAULT_THRESHOLD {
        return Ok(DEFAULT_THRESHOLD);
    }
    if !(0. ..=1.).contains(&threshold) {
        return Err(Failure::invalid(format!(
            "threshold {threshold} is outside [0, 1] (pass INFERENCE_LAYOUT_DEFAULT_THRESHOLD for the default)"
        )));
    }
    Ok(threshold)
}

/// Safety: `model_dir` is a C string, `backend` NULL or valid, `out_model` valid for a write.
#[no_mangle]
pub unsafe extern "C" fn inference_layout_model_load(
    model_dir: *const c_char,
    backend: *const inference_backend_config,
    out_model: *mut *mut inference_layout_model,
) -> inference_status {
    guard(|| {
        if out_model.is_null() {
            return Err(Failure::invalid("out_model is NULL"));
        }
        out_model.write(std::ptr::null_mut());
        let dir = arg_str(model_dir, "model_dir")?;
        let backend = backend_from(backend)?;
        let load_failed =
            |e: candle_core::Error| Failure::new(INFERENCE_ERR_LOAD_FAILED, format!("{dir}: {e}"));
        let mut detector =
            PPDocLayoutV3Detector::load(dir, &backend.device).map_err(load_failed)?;
        if let Some(threads) = backend.cpu_threads {
            detector = detector.with_cpu_threads(threads).map_err(load_failed)?;
        }
        let labels = detector
            .labels()
            .iter()
            .map(|l| CString::new(l.replace('\0', "")).unwrap_or_default())
            .collect();
        out_model.write(Box::into_raw(Box::new(inference_layout_model {
            detector,
            labels,
        })));
        Ok(())
    })
}

/// Safety: `model` is NULL or a handle from `inference_layout_model_load` that is not used again.
#[no_mangle]
pub unsafe extern "C" fn inference_layout_model_free(model: *mut inference_layout_model) {
    if !model.is_null() {
        guard_value((), || drop(Box::from_raw(model)));
    }
}

/// Safety: `model` is NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn inference_layout_model_label_count(
    model: *const inference_layout_model,
) -> usize {
    guard_value(0, || model.as_ref().map_or(0, |m| m.labels.len()))
}

/// Safety: `out_label` is NULL or valid for a write.
#[no_mangle]
pub unsafe extern "C" fn inference_layout_model_label(
    model: *const inference_layout_model,
    index: usize,
    out_label: *mut *const c_char,
) -> inference_status {
    guard(|| {
        let model = model
            .as_ref()
            .ok_or_else(|| Failure::invalid("model is NULL"))?;
        let label = model.labels.get(index).ok_or_else(|| {
            Failure::new(
                INFERENCE_ERR_OUT_OF_RANGE,
                format!("label {index} of {}", model.labels.len()),
            )
        })?;
        write_opt(out_label, label.as_ptr());
        Ok(())
    })
}

/// Safety: `model` is a live handle, `images` points to `count` valid images, `out` to `count` writable slots.
unsafe fn detect_into(
    model: *const inference_layout_model,
    images: *const inference_image,
    count: usize,
    threshold: f32,
    out: *mut *mut inference_layout_result,
) -> FfiResult {
    let model = model
        .as_ref()
        .ok_or_else(|| Failure::invalid("model is NULL"))?;
    if images.is_null() && count > 0 {
        return Err(Failure::invalid("images is NULL"));
    }
    let threshold = threshold_arg(threshold)?;
    let rgb = (0..count)
        .map(|i| {
            to_rgb(&*images.add(i))
                .map_err(|f| Failure::new(f.status, format!("image {i}: {}", f.message)))
        })
        .collect::<FfiResult<Vec<_>>>()?;
    let results = model
        .detector
        .detect_batch(&rgb, threshold)
        .map_err(|e| Failure::new(INFERENCE_ERR_RUNTIME, e.to_string()))?;
    for (i, detections) in results.into_iter().enumerate() {
        out.add(i)
            .write(Box::into_raw(Box::new(inference_layout_result {
                detections,
                labels: model.labels.clone(),
            })));
    }
    Ok(())
}

/// Safety: `image` points to a valid `inference_image`; `out_result` is valid for a write.
#[no_mangle]
pub unsafe extern "C" fn inference_layout_detect(
    model: *const inference_layout_model,
    image: *const inference_image,
    threshold: f32,
    out_result: *mut *mut inference_layout_result,
) -> inference_status {
    guard(|| {
        if out_result.is_null() {
            return Err(Failure::invalid("out_result is NULL"));
        }
        out_result.write(std::ptr::null_mut());
        if image.is_null() {
            return Err(Failure::invalid("image is NULL"));
        }
        detect_into(model, image, 1, threshold, out_result)
    })
}

/// Safety: `images` points to `count` valid images and `out_results` to `count` writable slots.
#[no_mangle]
pub unsafe extern "C" fn inference_layout_detect_batch(
    model: *const inference_layout_model,
    images: *const inference_image,
    count: usize,
    threshold: f32,
    out_results: *mut *mut inference_layout_result,
) -> inference_status {
    guard(|| {
        if out_results.is_null() && count > 0 {
            return Err(Failure::invalid("out_results is NULL"));
        }
        for i in 0..count {
            out_results.add(i).write(std::ptr::null_mut());
        }
        // all-or-nothing: detect_into only writes the slots after every fallible step has succeeded
        detect_into(model, images, count, threshold, out_results)
    })
}

/// Safety: `result` is NULL or a handle from `inference_layout_detect*` that is not used again.
#[no_mangle]
pub unsafe extern "C" fn inference_layout_result_free(result: *mut inference_layout_result) {
    if !result.is_null() {
        guard_value((), || drop(Box::from_raw(result)));
    }
}

/// Safety: `result` is NULL or a live handle.
#[no_mangle]
pub unsafe extern "C" fn inference_layout_result_count(
    result: *const inference_layout_result,
) -> usize {
    guard_value(0, || result.as_ref().map_or(0, |r| r.detections.len()))
}

/// Safety: `result` is a live handle; each out-pointer is NULL or valid (`out_bbox` for 4 floats).
#[no_mangle]
pub unsafe extern "C" fn inference_layout_result_detection(
    result: *const inference_layout_result,
    index: usize,
    out_class_id: *mut i32,
    out_label: *mut *const c_char,
    out_score: *mut f32,
    out_bbox: *mut f32,
) -> inference_status {
    guard(|| {
        let result = result
            .as_ref()
            .ok_or_else(|| Failure::invalid("result is NULL"))?;
        let d = result.detections.get(index).ok_or_else(|| {
            Failure::new(
                INFERENCE_ERR_OUT_OF_RANGE,
                format!("detection {index} of {}", result.detections.len()),
            )
        })?;
        write_opt(out_class_id, d.class_id as i32);
        write_opt(out_label, result.labels[d.class_id].as_ptr());
        write_opt(out_score, d.score);
        if !out_bbox.is_null() {
            std::ptr::copy_nonoverlapping(d.bbox.as_ptr(), out_bbox, 4);
        }
        Ok(())
    })
}
