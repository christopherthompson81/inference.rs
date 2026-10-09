# PP-DocLayoutV3 region outlines

The model predicts a mask per query, but only boxes were returned. transformers' `PPDocLayoutV3ImageProcessor`
(5.19) turns each kept query's mask into `polygon_points`: the thresholded mask cropped to the box (in mask
coordinates, a quarter of the model input), resized with `cv2.INTER_NEAREST` to the box's whole-pixel size,
`cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`, the largest contour by area, `cv2.approxPolyDP` with epsilon
0.4% of its perimeter, then `extract_custom_vertices` (only corners turning one way survive; one within a degree of 45
moves into its wedge along the bisector), falling back to the box's corners below four vertices. Issue #389.

## Run 1 - 2026-10-09 08:25

Question: can a port without OpenCV give transformers' vertices exactly?

Command: `make_outline_goldens.py` (committed beside `outline.rs`) runs the processor's own
`_extract_polygon_points_by_masks` on eleven synthetic masks (a rectangle, a skewed quad, an ellipse, an L, a frame
with a hole, two separate parts, a 2-pixel strip, one pixel, nothing, a triangle, a notched shape) and boxes offset so
some crops are partial; `outlines_match_transformers` holds `outline.rs` to them.

Raw, in order:
- First port (border following after Suzuki and Abe, `approxPolyDP` from memory): ten of eleven exact; the notched
  shape had one extra vertex on its diagonal edge.
- `approxPolyDP` as in OpenCV 4.x's source now measures distance to the segment, not its line (a point past an end
  counts by its distance to that end). Ported; the notched case still differed.
- The full contours differed from point 59: the nearest-neighbour resize picked another source column at one exact
  boundary. OpenCV's `resizeNN` computes the index as `floor(x * (1 / (dst / src)))`; that reciprocal rounds
  differently from `src / dst`. With the same arithmetic all eleven match vertex for vertex.

## Run 2 - 2026-10-09 08:25

Question: on a real page, do the outlines match transformers'?

Command: transformers' `post_process_object_detection` (threshold 0.5) and our detector (CPU and CUDA) on the layout
test page (a two-column printed report page), 13 detections each.

Raw: all 13 polygons identical, vertex for vertex, on both devices (4 to 9 vertices each).

Cost (`detect` on that page, mean of 10 after 2 warm-ups): CUDA 38.9 -> 41.0 ms, of which the mask head is 0.6 ms
(forward 29.6 -> 30.2 ms) and the rest the outlines traced at the page's full pixel size; CPU (dev profile)
1134 -> 1191 ms.

Change: `LayoutDetection` gains `polygon` (the decoder query it came from rides along unserialized); the detector
asks the model for masks, moves only the kept queries' masks to the host and traces outlines in parallel. The C ABI
adds `inference_layout_result_polygon` (a borrowed array of x, y pairs; ABI 0.0.22), the Python and C#
`LayoutDetection` gain `polygon`/`Polygon`, and `detections_through_the_abi` checks every outline has four or more
vertices on its box.

## Run 3 - 2026-10-09 08:39

Question: after the review fixes (last-wins tie, `(1/w)*pw` scale, f32 per-edge arcLength), do the real-page polygons still match transformers?

Command: the Python bindings (`inference_rs.LayoutModel`) on the same real scanned page, CPU and CUDA, compared to the transformers dump.

Finding: the first try failed with `ctypes.ArgumentError: argument 3 ... expected LP_c_void_p`: the generated signature table types the out pointer as `void**`, so `_layout.py` passed the wrong pointer type. No Python test exercises a model-backed detection, so the bindings suite would not have caught it. After reading through a `c_void_p` and casting: CPU 13/13 and CUDA 13/13 polygons identical.

Next: full CI.

## Run 4 - 2026-10-09 08:39

Question: can a committed test catch the binding bug from Run 3?

Command: `python3 -m unittest bindings/python/tests/test_layout.py` with `INFERENCE_TEST_LAYOUT_MODEL`/`INFERENCE_TEST_LAYOUT_IMAGE` exported (it skips without them or without Pillow).

Finding: with the fix it passes in 1.4 s on CPU. With `_layout.py` stashed it errors, but stashing also removed the polygon field, so this run doesn't isolate the pointer bug itself. local_ci.sh doesn't export the layout variables, so CI skips the test, just as it skips the model-backed Rust ABI test.

Next: full CI (`--lint --tests --cuda --slim --bindings --docs --sweep`).

## Run 5 - 2026-10-09 08:43

Command: `scripts/local_ci.sh --lint --tests --cuda --slim --bindings --docs --sweep`

Finding: the first attempt failed clippy (`chunks_exact_to_as_chunks` in the ABI test's polygon read). Reading through `points.cast::<[f32; 2]>()` fixed it. Rerun green: CPU 2515 passed, CUDA 2906 passed, Python 43 OK (1 skipped: the layout test, since its env vars aren't exported), scripts 21 OK.

Next: commit; push only on approval.
