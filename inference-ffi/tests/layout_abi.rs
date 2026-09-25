//! Exercises the C ABI through the exported functions. Model-backed tests run when `INFERENCE_TEST_LAYOUT_MODEL` (an
//! HF PP-DocLayoutV3 directory) and `INFERENCE_TEST_LAYOUT_IMAGE` (any page image) are set, and skip otherwise.

use std::ffi::{c_char, CStr, CString};
use std::ptr::{null, null_mut};

use inference_ffi::inference_status::{self, *};
use inference_ffi::layout::*;
use inference_ffi::*;

const MODEL_ENV: &str = "INFERENCE_TEST_LAYOUT_MODEL";
const IMAGE_ENV: &str = "INFERENCE_TEST_LAYOUT_IMAGE";

fn last_error() -> String {
    unsafe { CStr::from_ptr(inference_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn backend(name: &str) -> (CString, inference_backend_config) {
    let name = cstr(name);
    let cfg = inference_backend_config {
        backend: name.as_ptr(),
        device: 0,
        threads: 0,
    };
    (name, cfg)
}

#[test]
fn versions_and_status_strings() {
    let v = inference_abi_version();
    assert_eq!(v >> 16, ABI_VERSION_MAJOR);
    assert_eq!((v >> 8) & 0xff, ABI_VERSION_MINOR);
    let build = unsafe { CStr::from_ptr(inference_build_version()) }
        .to_str()
        .unwrap();
    assert!(build.starts_with("inference.rs "), "{build}");
    let name = |s: i32| {
        unsafe { CStr::from_ptr(inference_status_string(s)) }
            .to_str()
            .unwrap()
            .to_string()
    };
    assert_eq!(
        name(INFERENCE_ERR_LOAD_FAILED as i32),
        "INFERENCE_ERR_LOAD_FAILED"
    );
    assert_eq!(name(12345), "INFERENCE_UNKNOWN_STATUS");
    assert_eq!(last_error(), "");
}

#[test]
fn load_errors_set_status_and_detail() {
    let mut model: *mut inference_layout_model = null_mut();
    unsafe {
        assert_eq!(
            inference_layout_model_load(null(), null(), &mut model),
            INFERENCE_ERR_INVALID_ARGUMENT
        );
        assert!(last_error().contains("model_dir"));
        assert!(model.is_null());

        let missing = cstr("/nonexistent/layout-model");
        assert_eq!(
            inference_layout_model_load(missing.as_ptr(), null(), &mut model),
            INFERENCE_ERR_LOAD_FAILED
        );
        assert!(
            last_error().contains("/nonexistent/layout-model"),
            "{}",
            last_error()
        );

        let (_n, cfg) = backend("tpu");
        assert_eq!(
            inference_layout_model_load(missing.as_ptr(), &cfg, &mut model),
            INFERENCE_ERR_INVALID_ARGUMENT
        );
        assert!(last_error().contains("tpu"));

        if !cfg!(feature = "cuda") {
            let (_n, cfg) = backend("cuda");
            assert_eq!(
                inference_layout_model_load(missing.as_ptr(), &cfg, &mut model),
                INFERENCE_ERR_NOT_AVAILABLE
            );
        }
        assert_eq!(
            inference_layout_model_load(missing.as_ptr(), null(), null_mut()),
            INFERENCE_ERR_INVALID_ARGUMENT
        );
    }
}

#[test]
fn null_handles_are_safe() {
    unsafe {
        inference_layout_model_free(null_mut());
        inference_layout_result_free(null_mut());
        assert_eq!(inference_layout_result_count(null()), 0);
        let mut out = null_mut();
        let img = inference_image {
            pixels: [0u8; 3].as_ptr(),
            width: 1,
            height: 1,
            stride: 0,
            format: 0,
        };
        assert_eq!(
            inference_layout_detect(null(), &img, -1., &mut out),
            INFERENCE_ERR_INVALID_ARGUMENT
        );
        assert!(out.is_null());
        let mut label: *const c_char = null();
        assert_eq!(
            inference_layout_model_label(null(), 0, &mut label),
            INFERENCE_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            inference_layout_result_detection(
                null(),
                0,
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut()
            ),
            INFERENCE_ERR_INVALID_ARGUMENT
        );
    }
}

/// Decoded test page as tightly packed RGB8, or `None` when the model-backed tests are not configured.
fn fixture() -> Option<(CString, Vec<u8>, u32, u32)> {
    let (Ok(model), Ok(image)) = (std::env::var(MODEL_ENV), std::env::var(IMAGE_ENV)) else {
        eprintln!("skipping: set {MODEL_ENV} and {IMAGE_ENV}");
        return None;
    };
    let rgb = image::ImageReader::open(&image)
        .unwrap()
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();
    let (w, h) = rgb.dimensions();
    Some((cstr(&model), rgb.into_raw(), w, h))
}

struct Detection {
    class_id: i32,
    label: String,
    score: f32,
    bbox: [f32; 4],
}

unsafe fn collect(result: *const inference_layout_result) -> Vec<Detection> {
    (0..inference_layout_result_count(result))
        .map(|i| {
            let mut d = Detection {
                class_id: -1,
                label: String::new(),
                score: 0.,
                bbox: [0.; 4],
            };
            let mut label: *const c_char = null();
            let st = inference_layout_result_detection(
                result,
                i,
                &mut d.class_id,
                &mut label,
                &mut d.score,
                d.bbox.as_mut_ptr(),
            );
            assert_eq!(st, INFERENCE_OK);
            d.label = CStr::from_ptr(label).to_str().unwrap().to_string();
            d
        })
        .collect()
}

unsafe fn detect(model: *const inference_layout_model, img: &inference_image) -> Vec<Detection> {
    let mut result = null_mut();
    let st: inference_status =
        inference_layout_detect(model, img, INFERENCE_LAYOUT_DEFAULT_THRESHOLD, &mut result);
    assert_eq!(st, INFERENCE_OK, "{}", last_error());
    let dets = collect(result);
    inference_layout_result_free(result);
    dets
}

fn same(a: &[Detection], b: &[Detection]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert_eq!((x.class_id, &x.label), (y.class_id, &y.label));
        assert!(
            (x.score - y.score).abs() < 1e-4,
            "{} vs {}",
            x.score,
            y.score
        );
        for (p, q) in x.bbox.iter().zip(&y.bbox) {
            assert!((p - q).abs() < 0.05, "{:?} vs {:?}", x.bbox, y.bbox);
        }
    }
}

#[test]
fn detections_through_the_abi() {
    let Some((dir, rgb, w, h)) = fixture() else {
        return;
    };
    unsafe {
        let mut model = null_mut();
        let (_n, mut cfg) = backend("cpu");
        cfg.threads = 4;
        assert_eq!(
            inference_layout_model_load(dir.as_ptr(), &cfg, &mut model),
            INFERENCE_OK,
            "{}",
            last_error()
        );

        assert_eq!(inference_layout_model_label_count(model), 25);
        let mut label: *const c_char = null();
        assert_eq!(
            inference_layout_model_label(model, 21, &mut label),
            INFERENCE_OK
        );
        assert_eq!(CStr::from_ptr(label).to_str().unwrap(), "table");
        assert_eq!(
            inference_layout_model_label(model, 25, &mut label),
            INFERENCE_ERR_OUT_OF_RANGE
        );

        let rgb_img = inference_image {
            pixels: rgb.as_ptr(),
            width: w,
            height: h,
            stride: 0,
            format: INFERENCE_PIXEL_RGB8,
        };
        let base = detect(model, &rgb_img);
        assert!(!base.is_empty());
        for d in &base {
            assert!(d.score >= 0.5 && d.bbox[0] < d.bbox[2] && d.bbox[1] < d.bbox[3]);
        }

        // BGRA with row padding must give the same detections as tight RGB
        let stride = w * 4 + 12;
        let mut bgra = vec![0u8; (stride * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let s = (y * w as usize + x) * 3;
                let d = y * stride as usize + x * 4;
                bgra[d..d + 4].copy_from_slice(&[rgb[s + 2], rgb[s + 1], rgb[s], 255]);
            }
        }
        let bgra_img = inference_image {
            pixels: bgra.as_ptr(),
            width: w,
            height: h,
            stride,
            format: INFERENCE_PIXEL_BGRA8,
        };
        same(&base, &detect(model, &bgra_img));

        // batch of two == two singles; a bad image fails the whole batch and leaves every slot NULL
        let imgs = [rgb_img, bgra_img];
        let mut out = [null_mut(); 2];
        assert_eq!(
            inference_layout_detect_batch(model, imgs.as_ptr(), 2, -1., out.as_mut_ptr()),
            INFERENCE_OK
        );
        for r in out {
            same(&base, &collect(r));
            inference_layout_result_free(r);
        }
        let bad = [
            rgb_img,
            inference_image {
                stride: 5,
                ..bgra_img
            },
        ];
        let mut out = [null_mut(); 2];
        assert_eq!(
            inference_layout_detect_batch(model, bad.as_ptr(), 2, -1., out.as_mut_ptr()),
            INFERENCE_ERR_INVALID_ARGUMENT
        );
        assert!(last_error().contains("image 1"), "{}", last_error());
        assert!(out.iter().all(|r| r.is_null()));

        let mut r = null_mut();
        assert_eq!(
            inference_layout_detect(model, &rgb_img, 1.5, &mut r),
            INFERENCE_ERR_INVALID_ARGUMENT
        );
        let bad_fmt = inference_image {
            format: 9,
            ..rgb_img
        };
        assert_eq!(
            inference_layout_detect(model, &bad_fmt, -1., &mut r),
            INFERENCE_ERR_INVALID_ARGUMENT
        );

        let mut result = null_mut();
        assert_eq!(
            inference_layout_detect(model, &rgb_img, -1., &mut result),
            INFERENCE_OK
        );
        let n = inference_layout_result_count(result);
        assert_eq!(
            inference_layout_result_detection(
                result,
                n,
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut()
            ),
            INFERENCE_ERR_OUT_OF_RANGE
        );
        // results do not reference the model: free the model first
        inference_layout_model_free(model);
        assert_eq!(collect(result).len(), n);
        inference_layout_result_free(result);
    }
}
