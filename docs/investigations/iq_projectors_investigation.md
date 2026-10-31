# IQ projectors

Question: do IQ-quantized multimodal projectors (mmproj GGUFs with IQ tensors) load and work, and is the
"unsupported IQ artifacts" rule in automatic projector selection (`resolve_gguf_projector`,
crates/inference-selection/src/quant/gguf_discovery.rs) still right?

## Run 1 - 2026-10-31 16:44

**Where the rule lives.** `resolve_gguf_projector` drops every projector group whose filename label starts with
`IQ` (`is_iq_projector`) and errors when only IQ projectors remain. It decides by filename alone, never by the
tensors. The rule dates from #93 (an early refactor that moved `quant` resolution into core), before the IQ GGUF
kernels existed.

**Do IQ projectors exist?** A Hub search ("mmproj IQ4", "mmproj IQ", "GGUF mmproj", 50 repos each, filenames
containing both `mmproj` and `iq`) found none. llama.cpp's `llama-quantize` crashes on a projector GGUF (it only
knows text architectures), and gguf-py's `quants.py` implements quantization only for BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0,
TQ1_0/TQ2_0, Q1_0/Q2_0 and MXFP4. The common tools can't produce one.

**Test file.** `scratchpad/iqproj/requant_projector.py` calls ggml's own `ggml_quantize_chunk` through ctypes
(llama.cpp build_cuda `libggml-base.so`) and requantizes every `.weight` whose row length is a multiple of the block.
IQ4_XS (256-wide blocks) would leave nearly everything alone, since Qwen3.8-27B's vision tower is 1152 wide, so the
test uses IQ4_NL (32-wide):

```
python3 requant_projector.py mmproj-Qwen3.8-27B-Q8_0.gguf mmproj-Qwen3.8-27B-IQ4_NL.gguf IQ4_NL
-> 84 tensors IQ4_NL, 250 kept (223 F32, 27 F16); 459 MB (Q8_0 was 629 MB); clip.projector_type qwen3vl_merger
```

Next: load it through our engine (explicit projector) beside recipe-A, describe the cafe test image, and compare
with the Q8_0 projector and with llama.cpp's mtmd on the same file.

## Run 2 - 2026-10-31 16:58

**Explicit projector, our engine.** recipe-A with `--mmproj` naming the Q8_0 and then the IQ4_NL projector, cafe image,
"Describe this image in one sentence.":

- Q8_0: "A cozy, warmly lit cafe with string lights and a green awning sits along a rainy city street, where three
  people chat at an outdoor table while pedestrians with umbrellas and cars pass by on the wet pavement." Prompt
  3665 tokens at 1863 T/s.
- IQ4_NL: "...a warm, inviting cafe with glowing string lights and large windows on a rainy evening, where three
  people sit at an outdoor table chatting over coffee while pedestrians with umbrellas and cars pass along the wet,
  reflective street." Prompt 3665 tokens at 1859 T/s.

Both read the picture correctly (sampling at temperature 1, so the wording differs).

**Automatic selection.** The same model with only the IQ4_NL projector beside it fails:
`Automatic GGUF projector selection only found unsupported IQ artifacts: mmproj-Qwen3.8-27B-IQ4_NL.gguf`.

**llama.cpp (970b51ce7) on the same file.** `llama-mtmd-cli` with Qwen3.8-27B-IQ4_XS.gguf aborted with
`ggml.c:5191: GGML_ASSERT(a->type == GGML_TYPE_F32)`; the Q8_0 projector loaded. Cause: the first test file
requantized every 2D `.weight`, including one tensor the source kept in F32 (84 converted, the source has 83 Q8_0).
Dead end in the test file, not in either engine. The script now requantizes only the source's Q8_0 tensors (83).

With the corrected file llama.cpp answers "A cozy, warmly-lit cafe with patrons enjoying drinks sits on a
rain-slicked, cobblestone street at dusk...", and ours, run again, reasons through the same scene (three people at
an outdoor table, rainy street, umbrellas, painterly style) in its thinking.

**Conclusion.** IQ projectors load and work in both engines; only our automatic selection refuses them, by filename.
Next: drop the IQ refusal from `resolve_gguf_projector` so an IQ projector is selected like any other quant.

## Run 3 - 2026-10-31 16:51

**Change.** `resolve_gguf_projector` no longer refuses IQ projectors: every projector group is a candidate, and
`projector_rank` puts an IQ projector one step after a Q/K projector of the same width (so `IQ4_XS` beside
`Q4_K_M` picks `Q4_K_M` rather than reporting ambiguity). The three tests that pinned the refusal now pin
selection (IQ only, IQ4_XS vs Q4_K_M, vision Q4_K_M + audio IQ4_XS); the generic-plus-role-specific case is the
existing mixing test. `inference-selection`: 74 passed, 1 skipped.

**Rerun of the failing case.** The same folder (recipe-A + only the IQ4_NL projector), no `--mmproj`:

```
GGUF: selected IQ4_NL projector `mmproj-Qwen3.8-27B-IQ4_NL.gguf`
A cozy, warmly lit cafe on a rainy evening, with three people chatting at an outdoor table while pedestrians with
an umbrella walk along the wet cobblestone street.
```

Not covered: `--quant` with a bit width still skips IQ text-model files (a separate rule, documented in
run-gguf.mdx); IQ types other than IQ4_NL were not run as projectors (IQ4_XS can't cover a 1152-wide tower).
