# #251: reproducible reference builds and test GGUFs for the GGUF parity checks

## Run 1 - 2026-10-08 10:05

Question: how were the references and the `qwen35-0.8b-{iq,kt,iqk}` directories behind GGUF investigation Run 15 made?

Nothing in the repo recorded it. Recovered from the session that made them and from the GGUF investigation doc:
- Mainline: the local checkout at `4617ccc1a`, which is the owner's fork branch; its only commit past upstream is a
  Vulkan shader fix, on top of upstream `4a8993735` (2026-09-13). The CUDA build is the same, so the pin is upstream.
- ik_llama.cpp: a shallow clone at `5f89bfc81268` (2026-10-02), built in a session `/tmp` scratchpad (gone since).
- Build: `cmake -B build_cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`,
  targets `llama-quantize llama-perplexity llama-imatrix llama-simple`.
- F16: mainline `convert_hf_to_gguf.py <Qwen3.5-0.8B> --no-mtp --outtype f16` (with the MTP layer, quantizing fails
  with "Missing importance matrix for tensor blk.24").
- Calibration text: `cat README.md docs/build.md` in the mainline checkout.
- Imatrix: mainline `llama-imatrix -c 512 --chunks 32 -ngl 99`; ik `llama-imatrix -c 512 -ngl 99` (ik cannot read the
  GGUF-format mainline imatrix: "failed reading number of values for entry 1").
- iq: mainline `llama-quantize --imatrix` per type. kt: ik `llama-quantize --pure --token-embedding-type q8_0`
  (ik's default KT mixes add IQ*_K tensors). iqk: ik `llama-quantize` with its default rules, plus a default IQ2_KT mix.

Written as `scripts/build_gguf_references.sh` (pinned commits, CUDA arch `native`) and
`scripts/make_gguf_test_dirs.sh`. Next: build both references with the script, regenerate the directories into a
scratch location, and check the reference perplexities against Run 15's.

## Run 2 - 2026-10-08 10:49

Question: do the scripts rebuild the references and directories, and do the parity checks pass on what they produce?

Commands: `scripts/build_gguf_references.sh` (19.5 min, both references with CUDA, plus a venv), then
`scripts/make_gguf_test_dirs.sh /mnt/data/models/Qwen3.5-0.8B <scratch>/gguf-regen` (15 min, 23 files), then
`scripts/gguf_perplexity_parity.sh` per directory.

Raw findings:
- First regeneration failed in `convert_hf_to_gguf.py`: the user-site torch cannot load
  (`libtorch_global_deps.so: cannot open shared object file`). The build script now creates `<dir>/venv` from mainline's
  pinned `requirements-convert_hf_to_gguf.txt` (CPU torch 2.11, transformers 4.57.6) and the conversion runs there.
- Every regenerated file is 128 bytes smaller than its original: `quantize.imatrix.file` and `.dataset` record
  paths, and the old ones were long `/tmp` scratchpad paths.
- The mainline files also differ in about half their tensors (the imatrix-guided ones). The imatrix is deterministic
  (recomputed, byte identical) and the calibration text matches the old checkout, but the original mainline binaries
  were built 2026-08-28 at `d7bd3bfca` and never rebuilt; the checkout moved to `4617ccc1a` later. So Run 15 of the GGUF
  investigation mislabeled its mainline build (corrected there), and the old iq files mixed a `4617ccc1a` conversion
  with `d7bd3bfca` quantization. The pin stays at upstream `4a8993735`, a clean commit both scripts use throughout.
- kt and iqk: ik's reference perplexities equal Run 15's to every printed digit, so those files are equivalent. Both
  pass (worst IQ2_K 1.62%, IQ3_KT 1.44%). Ours moved since Run 15 on some files (IQ1_KT 161.46 -> 162.60, IQ4_KT
  9.628 -> 9.677, IQ6_K 8.947 -> 8.896), within tolerance.

```
iq (regenerated), reference 4a8993735 CUDA      reference       ours     drift
qwen35-0.8b-IQ1_M.gguf                           284.6210   296.4079    0.0414
qwen35-0.8b-IQ1_S.gguf                           838.9183   871.2956    0.0386
qwen35-0.8b-IQ2_S.gguf                            24.4528    24.6216    0.0069
qwen35-0.8b-IQ2_XS.gguf                           31.0333    30.9941    0.0013
qwen35-0.8b-IQ2_XXS.gguf                          62.6469    62.3462    0.0048
qwen35-0.8b-IQ3_S.gguf                            10.3661    10.3152    0.0049
qwen35-0.8b-IQ3_XXS.gguf                          13.2224    13.2079    0.0011
iq (old files), same reference
qwen35-0.8b-IQ1_M.gguf                           288.4148   292.0190    0.0125
qwen35-0.8b-IQ1_S.gguf                           818.2083   826.6493    0.0103
(IQ2/IQ3 all under 1%)
```

The IQ1 drift on the regenerated files, split by backend (regenerated files):

| | mainline CUDA | mainline CPU (`CUDA_VISIBLE_DEVICES=`) | ours CUDA | ours CPU |
|---|---|---|---|---|
| IQ1_M | 284.62 | 293.60 | 296.41 | 291.37 |
| IQ1_S | 838.92 | 858.02 | 871.30 | 825.86 |
| IQ2_XXS (control) | 62.65 | 62.24 | 62.35 | 61.29 |

Mainline's own CPU and CUDA paths differ by 3.2% (IQ1_M) and 2.3% (IQ1_S); ours CUDA is 1.0% and 1.5% from mainline's
CPU. At perplexity 300-800 the IQ1 types spread a few percent across backends, so this is no sign of a kernel bug; the
old files sat inside 2% by chance. (`-ngl 0` alone is not a CPU run: mainline still offloads the matmuls and returns the
CUDA numbers.) The docs pass 5% for the iq directory.

A review then found rerun hazards (partial F16 or imatrix reused after an interrupt, stale ones after a pin change,
silent configure failures, a broken venv blocking reruns); fixed with tmp-then-move writes and a pins stamp in `work`.

## Run 3 - 2026-10-08 11:07

Question: are the fixed scripts deterministic end to end?

Command: `sha256sum` of the 23 regenerated files, then `scripts/build_gguf_references.sh` (no-op rebuild) and
`scripts/make_gguf_test_dirs.sh` into the same directory; the new pins stamp was missing, so `work` was cleared and
the F16, both imatrices and every quantization were redone (15.4 min).

Raw: `sha256sum -c` passes for all 23 files. The pipeline reproduces itself byte for byte on this machine, so a
parity drift on these directories now points at code, not inputs.
