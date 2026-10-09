# PP-DocLayoutV3 GGUF investigation

Goal (#389, second half): a GGUF of PP-DocLayoutV3 that `PPDocLayoutV3Detector::load` and the C ABI accept, in F16 and in whichever quant types keep the detections stable.

## Run 1 - 2026-10-09 08:58

Question: what shape does the checkpoint have for GGUF?

Command: read the safetensors header (Python, `-I`).

Finding:
- 858 tensors, all F32, at most rank 4.
- The longest name is 88 bytes, and 425 names are over 63, which llama.cpp's reader rejects (`GGML_MAX_NAME` is 64 with the NUL).
- Ten whole-segment abbreviations bring the longest to 57 bytes: backbone, encoder, stages, blocks, layers, normalization, running_mean, running_var, convolution, bottlenecks. None of the short forms is a segment of the checkpoint, so the mapping inverts exactly.
- Parameters: 33.3M in total; convolution kernels hold 25.5M (77%), and rows a 32-block divides hold 7.6M.
- llama.cpp has no PP-DocLayout support, so no upstream tensor naming exists to follow.

Next: write `gguf::write_gguf`/`read_gguf`, keeping the configs as string metadata and the vectors in F32.

## Run 2 - 2026-10-09 08:58

Question: do the detections hold up per storage type when kernels are kept 4-D, so Q8_0 only reaches the linear layers?

Command: `pp_doclayout_v3_gguf --dtype {f32,f16,bf16,q8_0}`, then `pp_doclayout_v3_detect` on 10 pages (6 sample document images: scans, tables, a mixed layout; 4 synthetic fixture pages), compared to the safetensors load. Detections are paired by label and IoU >= 0.5.

Finding:
- Sizes: F32 133 MB, F16/BF16 67 MB, Q8_0 60 MB, so Q8_0 saves only 7 MB because the kernels fall back to F16.
- gguf-py reads the Q8_0 file: 644 F32, 131 F16 and 83 Q8_0 tensors, longest name 57.
- The first comparison paired detections by position and reported a 646 px box change for BF16. It was really two low-score regions (0.55 and 0.62) swapping reading order. Pairing by IoU fixed the metric.

Next: test kernels stored flat, (out, in*kh*kw), so the block types reach them.

## Run 3 - 2026-10-09 08:58

Question: with kernels flattened at write time and reshaped at load (a `SimpleBackend` that accepts exactly (out, prod(rest)) for a requested shape of rank 3 or more), which types keep the detections stable?

Command: as in Run 2, adding q5_0 and q4_0; CPU and CUDA.

Finding (identical on CPU and CUDA), against safetensors over 77 detections:

| type | size | detections | unmatched | max box px | max score | order changed | polygons equal |
|---|---|---|---|---|---|---|---|
| f32 | 133 MB | 77 | none | 0 | 0 | 0 | 77 |
| f16 | 67 MB | 77 | none | 0.016 | 0.0012 | 0 | 77 |
| bf16 | 67 MB | 76 | one ref at 0.501 | 1.29 | 0.0094 | 2 | 64 |
| q8_0 | 36 MB | 77 | none | 0.92 | 0.0195 | 2 | 55 |
| q5_0 | 24 MB | 79 | 3 new, 1 ref (0.52-0.57) | 8.7 | 0.090 | 10 | 16 |
| q4_0 | 20 MB | 78 | 6 new, 7 ref (incl. 0.78/0.80) | 8.9 | 0.30 | 22 | 2 |

The two order changes for q8_0 and bf16 are the same pair of low-score regions swapping.

Conclusion:
- F16 is effectively lossless.
- Q8_0 keeps every region, within a pixel, at about half the F16 size.
- BF16 is the same size as F16 and strictly worse.
- Q5_0 and Q4_0 add and drop regions.
- The converter offers f32, f16 and q8_0.

Next: ABI and bindings docs, a deep test (F32 exact, F16 and Q8_0 within tolerance), full CI.

## Run 4 - 2026-10-09 09:00

Question: does the C ABI load the GGUFs as well as the directory, and what does the load cost?

Command:
- `cargo nextest run --profile deep -E 'package(inference-ffi) & test(detections)'`. The new `detections_from_gguf` converts F32, F16 and Q8_0 into a tempdir and compares each through the ABI against the directory load, on the layout test page.
- Load times through the Python bindings: best of 3.

Finding:
- Both deep tests pass. F32 gives identical classes, scores, boxes and polygons. F16 stays within 0.1 px and 5e-3 in score. Every Q8_0 region matches at IoU > 0.99 with its score within 0.05.
- Load (ms):

| source | CPU | CUDA |
|---|---|---|
| safetensors dir | 122 | 41 |
| GGUF F32 | 139 | 55 |
| GGUF F16 | 142 | 53 |
| GGUF Q8_0 | 145 | 50 |

The 10-20 ms extra is the read and dequantize on CPU, compared with an mmap of F32. That is negligible next to a detection.

Next: full CI.

## Run 5 - 2026-10-09 09:04

Question: do the review's hardening fixes hold?

Fixes:
- Write refuses any name that doesn't map back exactly.
- A clear error for a missing path.
- The F16 deep check also compares polygons.
- CPU unit tests write and read a GGUF in memory, so normal CI covers the converter: a Q8_0 round trip of a kernel and a vector, plus refusal of names that wouldn't read back, names that are too long, a foreign architecture and a non-GGUF file.

Finding: the first round-trip run failed with `Q8_0 kernel error 0.0386` against a 0.01 bound. That was the test's own fault: its kernel values reached 18.4, so the Q8_0 step was about 0.07. With values scaled to [0, 1) all 5 gguf unit tests pass.

Not changed: the reshape rule accepts any stored (out, prod(rest)) for a requested shape. The model always requests the true shape, and the F32 deep check compares outputs exactly, so recording the original dims in metadata would add nothing today.

Note: Runs 1-3 share one timestamp because they were written into the log together after the runs, not as each finished.

## Run 6 - 2026-10-09 09:07

Commands:
- `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep`
- the deep `detections` tests
- the CUDA layout tests

Finding:
- The first CI attempt failed clippy over an unused `mut` in `detections_from_gguf`; that is fixed.
- Rerun green: CPU 2520 passed, CUDA 2911 passed, Python 43 OK, scripts 21 OK.
- Both deep tests pass, now with F16 polygons compared too.
- The CUDA layout tests pass (23).
