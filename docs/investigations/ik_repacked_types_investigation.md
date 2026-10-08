# #249: ik_llama.cpp repacked (_R4 / _R8) GGUF types

## Run 1 - 2026-10-08 16:42

Question: can ik's CPU-repacked GGUF types load by de-interleaving them to the plain layout, and which ones?

Scope (owner's choice): the repacked forms of types we already load, plus IQ1_BN / IQ2_BN later. ik defines 31
repacked types; IQ1_S_R4 and IQ1_M_R4 are separate quantizations in ik (no `repack_*` for them), and Q6_0_R4,
Q8_KV_R8, Q8_K_R8, Q1_0_G128_R8, PQ2_0_R8 and PTQ1_0_R8 have base types we do not load, so 22 are in scope:
Q4_0_R8, Q5_0_R4, Q8_0_R8, Q2_K_R4..Q6_K_R4, IQ4_NL_R4, IQ4_XS_R8, IQ2_XXS_R4, IQ2_XS_R4, IQ2_S_R4, IQ3_XXS_R4,
IQ3_S_R4, IQ2_K_R4, IQ3_K_R4, IQ4_K_R4, IQ5_K_R4, IQ4_KS_R4, IQ5_KS_R4, MXFP4_R8.

Test data: ik (`5f89bfc`) quantizations of Qwen3.5-0.8B from the F16 `scripts/make_gguf_test_dirs.sh` makes,
`llama-quantize --pure [--imatrix ik.imatrix] <f16> plain-T.gguf T`, then `llama-quantize --repack plain-T.gguf
repack-T.gguf T`. Raw findings while making them:
- `--repack` alone on an F16 input does nothing ("nothing to do for only_repack option"); it repacks a quantized file.
- IQ2_XXS / IQ2_XS / IQ2_S with `--pure` abort (`GGML_ASSERT(imatrix != NULL)`): the token embedding has no imatrix
  entry; `--token-embedding-type q8_0 --output-tensor-type q8_0` avoids it. The repack step also refuses them without
  `--imatrix`, though it only repacks.
- The `IQ2_S` file type writes IQ2_XS tensors (as in llama.cpp); `IQ2_M` writes IQ2_S.
- Every repacked file has the plain file's byte size; 186 tensors per file are repacked.

Change: `gguf/repack.rs` holds one inverse per type (decode the interleaved quants and scales, re-encode the base
block; ik scrambles IQ2_XXS / IQ2_XS / IQ3_XXS sign indices through a 128-entry permutation, inverted at compile time;
IQ4_KS / IQ5_KS groups lead with their rows' f32 scales). The archive sizes repacked ids as their base, reports the base
dtype and unpacks a tensor lazily on first read, so every loader sees plain blocks. Fixtures: the first 8 rows of
`blk.0.attn_gate.weight` per type, plain and repacked (`tests/fixtures/gguf_ik_repack`).

Raw: all 22 goldens unpack to ik's plain bytes exactly. Perplexity (`examples/rust/advanced/perplexity --file
README.md --llama-cpp-ctx 512`, CUDA), plain vs repacked, identical for every loadable type: Q4_K 10.5260, Q8_0
8.9383, IQ4_XS 10.1893, IQ3_S 11.4175, IQ4_KS 9.7646, IQ2_K 46.2487, IQ4_NL 10.0371, IQ2_XXS 83.7725, IQ2_XS 34.4499,
IQ2_S 27.4870, IQ3_XXS 15.3067, IQ3_K 12.7081, IQ4_K 9.5045, IQ5_K 9.1313, IQ5_KS 9.0885. MXFP4 does not load on this
model plain or repacked (MXFP4 loads only through the GPT-OSS binding), so MXFP4_R8 has only its golden.

Next: IQ1_BN / IQ2_BN (new dequantizers and kernels) as their own change.

Review of the change, and what changed: the first version cached each unpacked tensor in a `OnceLock` held by the
archive, which outlives loading (the prepared weight source keeps it for online ISQ), so a repacked model kept a
second copy of its weights in host memory for the server's life. Unpacking now returns an owned buffer per read
(`GgufTensorData` holds a `Cow`), freed when the loader drops it. Peak RSS loading the 0.8B (`/usr/bin/time -v`, the
perplexity run): plain-Q8_0 3,614,692 kB; repack-Q8_0 4,131,316 kB with the cache, 3,616,252 kB without. Out-of-scope
repacked ids (IQ1_S_R4, Q6_0_R4, IQ2_BN_R4, ...) are named and fail with "ik_llama.cpp CPU-repacked type ... is not
supported; use the non-repacked GGUF" when read or sized.
