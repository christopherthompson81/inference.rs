//! C ABI tests; model-backed ones need `INFERENCE_TEST_LAYOUT_MODEL` (HF dir) and `INFERENCE_TEST_LAYOUT_IMAGE`.

use std::ffi::{CStr, CString, c_char};
use std::ptr::{null, null_mut};

use inference_ffi::inference_status::{self, *};
use inference_ffi::layout::*;
use inference_ffi::*;
use inference_tensor::quantized::GgmlDType;

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
    // an unstable 0.0.x ABI is only compatible with the exact version
    assert_eq!(
        v,
        (ABI_VERSION_MAJOR << 16) | (ABI_VERSION_MINOR << 8) | ABI_VERSION_PATCH
    );
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
        assert!(last_error().contains("path"));
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

        let (_n, mut cfg) = backend("cpu");
        cfg.threads = MAX_CPU_THREADS + 1;
        assert_eq!(
            inference_layout_model_load(missing.as_ptr(), &cfg, &mut model),
            INFERENCE_ERR_INVALID_ARGUMENT
        );
        assert!(last_error().contains("threads"), "{}", last_error());

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
        assert_eq!(inference_layout_model_label_count(null()), 0);
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
        assert_eq!(
            inference_layout_result_polygon(null(), 0, null_mut(), null_mut()),
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
    polygon: Vec<[f32; 2]>,
}

unsafe fn collect(result: *const inference_layout_result) -> Vec<Detection> {
    unsafe {
        (0..inference_layout_result_count(result))
            .map(|i| {
                let mut d = Detection {
                    class_id: -1,
                    label: String::new(),
                    score: 0.,
                    bbox: [0.; 4],
                    polygon: Vec::new(),
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
                let (mut points, mut count): (*const f32, usize) = (null(), 0);
                assert_eq!(
                    inference_layout_result_polygon(result, i, &mut points, &mut count),
                    INFERENCE_OK
                );
                d.polygon = std::slice::from_raw_parts(points.cast::<[f32; 2]>(), count).to_vec();
                d
            })
            .collect()
    }
}

unsafe fn detect(model: *const inference_layout_model, img: &inference_image) -> Vec<Detection> {
    unsafe {
        let mut result = null_mut();
        let st: inference_status =
            inference_layout_detect(model, img, INFERENCE_LAYOUT_DEFAULT_THRESHOLD, &mut result);
        assert_eq!(st, INFERENCE_OK, "{}", last_error());
        let dets = collect(result);
        inference_layout_result_free(result);
        dets
    }
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
            // the box's corners at the least, every vertex a finite pixel coordinate
            assert!(d.polygon.len() >= 4, "{:?}", d.polygon);
            assert!(
                d.polygon.iter().flatten().all(|v| v.is_finite()),
                "{:?}",
                d.polygon
            );
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
        for bad in [1.5, -0.3, f32::NAN] {
            assert_eq!(
                inference_layout_detect(model, &rgb_img, bad, &mut r),
                INFERENCE_ERR_INVALID_ARGUMENT,
                "{bad}"
            );
        }
        // rejected on its dimensions before anything is allocated or read
        let huge = inference_image {
            width: 100_000,
            height: 100_000,
            ..rgb_img
        };
        assert_eq!(
            inference_layout_detect(model, &huge, -1., &mut r),
            INFERENCE_ERR_INVALID_ARGUMENT
        );
        assert!(last_error().contains("exceeds"), "{}", last_error());
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

unsafe fn load_and_detect(path: &CStr, rgb: &[u8], w: u32, h: u32) -> Vec<Detection> {
    unsafe {
        let mut model = null_mut();
        let (_n, cfg) = backend("cpu");
        assert_eq!(
            inference_layout_model_load(path.as_ptr(), &cfg, &mut model),
            INFERENCE_OK,
            "{}",
            last_error()
        );
        let img = inference_image {
            pixels: rgb.as_ptr(),
            width: w,
            height: h,
            stride: 0,
            format: INFERENCE_PIXEL_RGB8,
        };
        let dets = detect(model, &img);
        inference_layout_model_free(model);
        dets
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.);
    let area = |r: &[f32; 4]| (r[2] - r[0]) * (r[3] - r[1]);
    inter / (area(a) + area(b) - inter)
}

#[test]
fn detections_from_gguf() {
    // F16 keeps outlines and order; Q8_0 keeps every region within a pixel or so but may swap close reading orders
    const F16_BOX_PX: f32 = 0.1;
    const F16_SCORE: f32 = 5e-3;
    const Q8_MIN_IOU: f32 = 0.99;
    const Q8_SCORE: f32 = 0.05;
    let Some((dir, rgb, w, h)) = fixture() else {
        return;
    };
    let base = unsafe { load_and_detect(&dir, &rgb, w, h) };
    let tmp = tempfile::tempdir().unwrap();
    let convert = |dtype| {
        let path = tmp.path().join(format!("{dtype:?}.gguf"));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        inference_layout::pp_doclayout_v3::gguf::write_gguf(
            dir.to_str().unwrap().as_ref(),
            &mut file,
            dtype,
        )
        .unwrap();
        drop(file);
        unsafe { load_and_detect(&cstr(path.to_str().unwrap()), &rgb, w, h) }
    };

    let f32 = convert(GgmlDType::F32);
    assert_eq!(f32.len(), base.len());
    for (x, y) in base.iter().zip(&f32) {
        assert_eq!(
            (x.class_id, x.score, x.bbox, &x.polygon),
            (y.class_id, y.score, y.bbox, &y.polygon)
        );
    }

    let f16 = convert(GgmlDType::F16);
    assert_eq!(f16.len(), base.len());
    for (x, y) in base.iter().zip(&f16) {
        assert_eq!((x.class_id, &x.polygon), (y.class_id, &y.polygon));
        assert!(
            (x.score - y.score).abs() < F16_SCORE,
            "{} vs {}",
            x.score,
            y.score
        );
        for (p, q) in x.bbox.iter().zip(&y.bbox) {
            assert!((p - q).abs() < F16_BOX_PX, "{:?} vs {:?}", x.bbox, y.bbox);
        }
    }

    let q8 = convert(GgmlDType::Q8_0);
    assert_eq!(q8.len(), base.len());
    for x in &base {
        let y = q8
            .iter()
            .filter(|y| y.class_id == x.class_id)
            .max_by(|a, b| iou(&x.bbox, &a.bbox).total_cmp(&iou(&x.bbox, &b.bbox)))
            .unwrap();
        assert!(
            iou(&x.bbox, &y.bbox) > Q8_MIN_IOU,
            "{:?} vs {:?}",
            x.bbox,
            y.bbox
        );
        assert!(
            (x.score - y.score).abs() < Q8_SCORE,
            "{} vs {}",
            x.score,
            y.score
        );
    }
}
