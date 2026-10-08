# #248: mmq requant (_q8) loaders for IQ2_K..IQ5_K prefill

## Run 1 - 2026-10-08 13:55

Question: do ik_llama.cpp's `_q8` mmq loaders close the prefill gap between IQ2_K..IQ5_K and the `_KS` types?

ik's `mmq_type_traits_id` for IQ2_K/IQ3_K/IQ4_K/IQ5_K switch to `load_tiles_iq*_k_q8` plus `vec_dot_q8_0_q8_1_mma`
(D4 layout) once `mmq_x` reaches 48/40/32/48 (`mmq_requant`, tensor-core MMA only); IQ6_K has no `_q8` loader. Our
port (#245) had only the block-16 path. Ported from the reference checkout (`5f89bfc`, built by
`scripts/build_gguf_references.sh`): the four loaders, their traits, `mmq_requant`, `requant_int_q8`,
`get_int_from_table_16_q8` (ik's `iq3k_table`/`iq4k_table` lookups serve only its `_R4` loaders, not ported). The only change is the
MMA guard (ik's `INT8_MMA_AVAILABLE` is our `TURING_MMA_AVAILABLE`). The Q8_0 MMA tile row (76 ints) fits the Q3_K
row these types allocate (84).

Command: `target/debug/inference bench --format gguf -m qwen35-0.8b-iqk -f <file> --prompt-len {512,2048,4096}
--gen-len 1` (RTX 3090, CUDA build), TTFT tok/s before and after; the files are the ones
`scripts/make_gguf_test_dirs.sh` makes.

Raw (before -> after):

| File | 512 | 2048 | 4096 |
|---|---|---|---|
| IQ2_K | 18677 -> 19918 | 20832 -> 22140 | 20628 -> 21991 |
| IQ3_K | 17980 -> 20863 | 20203 -> 21628 | 20117 -> 21423 |
| IQ4_K | 18225 -> 20895 | 20246 -> 21798 | 20304 -> 21628 |
| IQ5_K | 17361 -> 19522 | 19408 -> 19826 | 19325 -> 19777 |
| IQ6_K (unchanged) | 17689 -> 18624 | 19502 -> 19533 | 19534 -> 19480 |
| IQ2_KS (unchanged) | 21057 -> 21179 | 22142 -> 22607 | 22129 -> 22278 |
| IQ4_KS (unchanged) | 19697 -> 20989 | 21907 -> 21781 | 22057 -> 21809 |

After: IQ3_KS 21159 / 21241, IQ5_KS 20401 / 20142 (2048 / 4096).

The unchanged types move up to 6.6% at 512 tokens, so 512 is noise; at 2048/4096 they hold within 2%. There IQ2_K,
IQ3_K and IQ4_K gain 6-8% and reach their `_KS` rates (IQ3_K now above IQ3_KS); IQ5_K gains 2% and sits within 2% of
IQ5_KS.

Parity, `scripts/gguf_perplexity_parity.sh <ik llama-perplexity> qwen35-0.8b-iqk`: all 12 within 0.9% (IQ2_K 30.1513
vs 30.3753, 0.74%, was 1.62%; IQ3_K 0.76%; IQ4_K 0.24%; IQ5_K 0.21%). `gguf::raw` CUDA kernel tests pass.
