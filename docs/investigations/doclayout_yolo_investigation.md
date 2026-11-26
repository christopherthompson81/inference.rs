# DocLayout-YOLO investigation

DocLayout-YOLO (YOLOv10-m with G2L_CRM backbone stages) as a second layout detector for the document reading
workflow, beside PP-DocLayoutV3. The weights (`juliozhao/DocLayout-YOLO-DocStructBench`) are Apache-2.0; the
reference code is AGPL-3.0, so this port is written from the papers (DocLayout-YOLO, arXiv 2410.12628; YOLOv10,
arXiv 2405.14458) and the checkpoint's own structure, with behavior established by black-box comparison against the
reference package's outputs. The reference source was not read.

Reference environment: `/mnt/data/doclayout-venv` (system site packages, `pip install doclayout-yolo` 0.0.4, torch
2.11). Probe scripts and dumps: `/mnt/data/models/doclayout-ref`. Checkpoint:
`/mnt/data/models/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt`.

## Run 1 - 2026-11-25 19:35

Question: what is in the checkpoint, and what does each module compute?

`dump_structure.py` prints the pickled `YOLOv10DetectionModel`'s module tree, state dict shapes and `yaml`:
`yolov10m-doclayout.yaml`, scale `m` (`[0.67, 0.75, 768]`), 10 classes (title, plain text, abandon, figure,
figure_caption, table, table_caption, table_footnote, isolate_formula, formula_caption). The checkpoint dict's
`model` entry is a whole pickled module in f16 (not a state dict). BatchNorm eps is 1e-3. `attrs.py` gives the
non-tensor attributes: G2L_CRM dilated blocks at layer 4 use dilations `[1, 2, 3]` with a 5x5 kernel, at layer 6
`[1, 3, 5]` with 3x3, fuse `glu`; layer 8 is a G2L_CRM made of CIB blocks; PSA has 4 heads, head_dim 72, key_dim
36; the head is `v10Detect` with reg_max 16, strides 8/16/32.

Black-box fits (`probe_*.py`), each comparing a hypothesis built from the module's own submodules to its output on
random inputs:

- `DilatedBottleneck`: `x + cv2(dilated_block(cv1(x)))`, error 0.0 (the other orders 0.47-0.92).
- `DilatedBlock`: the first guesses (gating on the concatenated branches before `cv2`, dcv with its activation)
  were all wrong (relative error ~1). Forward hooks on its submodules showed `dcv` is never called as a module:
  its weight runs functionally once per dilation, through `dcv.bn` then SiLU, then the module's `cv2` per branch;
  the three branches concatenate along channels (dilation-major), `conv_gating` (depthwise 1x1 + BN + SiLU) runs
  on the concatenation, the gate is its sigmoid, and the grouped `conv1x1` (groups = c, 3 inputs per group) fuses
  `cat * sigmoid(gate)`, plus the residual. Error 0.0. (A first hook pass matched SiLU outputs falsely: one SiLU
  instance is shared by every Conv.)
- C2f / G2L_CRM / C2fCIB (split in halves, chain of blocks each kept, concat, cv2), SCDown, SPPF (three chained
  5x5 max pools), CIB, PSA attention: all error 0.0.
- Head: `one2one_cv2/cv3` per level, DFL softmax expectation over 16 bins, anchors at cell centres; the head emits
  `[cx, cy, w, h]` times stride and sigmoid scores (error 1.2e-4 absolute on pixel coordinates, xyxy guess 1018).
- Selection: top 300 anchors by best class score, then top 300 (anchor, class) pairs, then the threshold; no NMS.
  Boxes map back with the gain and padding recomputed from the letterboxed shape. Exact on all 13 boxes.
- Letterbox: long side to 1024 (`round` half-even), cv2 INTER_LINEAR, pad each side to a multiple of 32 centred
  (`round(d - 0.1)` / `round(d + 0.1)`), value 114, BGR to RGB, /255. Error 0.0 on the model input.
- cv2 INTER_LINEAR on u8 (`cv2_linear.py`): the x86 SIMD form (11-bit weights, horizontal sums `>> 4`, `mulhi`,
  `(v + 2) >> 2`) matches cv2 exactly when shrinking (0 of 2.2M values off); the scalar form is off by 1 on 64K.
  Enlarging differs on 17 values (a scalar row tail in cv2). Taken as is.

Implies: the whole forward is specified; write the Rust port.

## Run 2 - 2026-11-25 20:30

Question: does the Rust port match the reference end to end?

`dump_reference.py` runs the reference predictor on `layout_demo.png` (the test page decoded once and written as PNG
so both sides read the same pixels) and saves the model input, the three head features and the boxes.
`cargo run -p inference-layout --example doclayout_yolo_parity -- --cpu ...`:

```
input    max_abs=0.000e0 ref_max=1.000e0
feat0    max_abs=4.148e-5 ref_max=1.100e1
feat1    max_abs=2.670e-5 ref_max=1.409e1
feat2    max_abs=2.527e-5 ref_max=9.699e0
boxes: ours 13, reference 13
worst box offset 0.00 px, scores within 4.8e-7
```

(The reference predictor fuses BatchNorm into its convs in f32; so does the port, at load.) Implies: CPU parity
holds; wire it into the layout model loading, check CUDA through the server.

## Run 3 - 2026-11-25 20:05

Question: does it hold on CUDA and through the server and web UI, and is the reading order usable?

- `scripts/doclayout_yolo_parity.sh` on a second page (`pp_structure_v3_demo.png`, CPU): 35 boxes, all 0.00 px off,
  scores within 2.2e-6.
- Tiny random-weight checkpoint (`doclayout_yolo::model::tests`, one block per stage, PSA at 128 channels): CUDA
  matches CPU (boxes relative error and scores under 1e-4).
- Dev server (`serve -p 18082`, CUDA build): the inventory lists `DocLayout-YOLO-DocStructBench` as `layout`
  (`doclayout_yolo`); load and `POST /v1/layout/detect` on `layout_demo.jpg`: 13 regions, 0.18 s first call, then
  0.10 s (base64 and JPEG decode included). Boxes as the reference's.
- Reading order, first cut (bands, merging neighbouring bands whose union has a column gutter): the body read down
  each column, but the footer came out as page number, footnote, running footer: the margin page number (x 107-136)
  made a "gutter" with the footer band. Fix: merge only bands that are multi-column themselves; a single-column band
  separates. Unit test pins both shapes.
- Web UI (Playwright Chromium, Documents tab, PaddleOCR-VL-1.5 as the OCR model): Detect layout then Read: 13
  regions read in 1.9 s, two columns in order, the title between paragraphs, `abandon` regions last (left out of
  the exports with the other page furniture).

Implies: done; the remaining differences from PP-DocLayoutV3 are by design (no masks, so polygons are box corners;
order from geometry, not predicted).
