# Loop guidance investigation

LoopCD (arXiv:2610.02185, "Decoding Looped Transformers Better for (Almost) Free"): a looped model's first pass is a
weaker prediction of the same token, so steering away from it sharpens the final one. LoopCD-Hidden, implemented
here for Inkling 2: the coda's input becomes `h + w (h - h1)`, with `h1` the first pass's core output and `h` the last
pass's. Inkling 2 Mini is not one of the paper's models, and its loop differs from theirs (each pass restarts from the
prelude's output and reads the previous pass through learned depth mixing), so the strength needs measuring here.

## Run 1 - 2026-11-25 01:40

Question: does loop guidance change Inkling 2 Mini's answers on a reasoning set, and at what strength?

Setup: this branch's `inference serve -m inkling2-mini-official --pa-cache-type q8 --max-seqs 4` (recipe C, no
drafter), GSM8K test (openai/gsm8k, MIT), the first N problems, greedy (temperature 0), the prompt asking for the steps
then `#### <number>`; the strength switched live through `/control_vectors/set` between runs (`gsm8k_eval.py`,
scratch). Thinking off is `reasoning_effort: none`; thinking on is the template's default effort.

```
GSM8K, thinking off, 200 problems    accuracy        gained / lost vs none   mean tokens
none                                 0.920 (184)     -                       106
0.1                                  0.915 (183)     2 / 3                   107
0.2                                  0.915 (183)     2 / 3                   110
0.3                                  0.925 (185)     3 / 2                   104

GSM8K, thinking on, 100 problems
none                                 0.950 (95)      -                       607
0.2                                  0.970 (97)      2 / 0                   630
0.3                                  0.960 (96)      1 / 0                   561
```

A 20-problem smoke run first showed 0.2 at 17/20 against 19/20, with the guided replies writing `#### <answer>` first
and no working; on 200 problems that pattern is as common unguided (30 replies) as guided (32-33), and the accuracy
differences are 1-2 problems either way: noise. With thinking, guidance flipped only wrong answers to right (2 at 0.2,
1 at 0.3), which points the paper's way but is within noise at 100 problems. GSM8K sits at 92-95% here, too close to
its ceiling to show a difference the size the paper reports; a harder set (MATH-500, AIME) with thinking is what would.

Generation is intact at these strengths: no degenerate or truncated replies, lengths within 10% of unguided.

## Run 2 - 2026-11-25 05:30

The authors' Inkling 2 Mini recipe (thread by the paper's first author, 2026-11-24): LoopCD-Logits, the first pass's
coda keeping its own KV cache, adaptive strength `w = w_max (1 - (p1 - p2))` with `w_max = 0.5`; for agentic coding
only outside thinking blocks, control tokens untouched. Built here as a load-time mode (`loop_guidance_logits` in the
config, through `--hf-overrides`, which GGUF loads now take): the coda's six layers run again on the first pass's output
in cache slots of their own (five of them sliding, so one full-context layer's KV), every token, and the strength is
`w_max`; the control tokens (ids from 199998, the tokenizer's specials) keep the last pass's logits. The thinking-block
gate is not built. Also `loop_config.run_passes` (fewer passes than trained, the drafter's taps mapped onto them).

Benchmark: MATH-500 levels 4-5 with integer answers (156 problems, HuggingFaceH4/MATH-500, MIT), thinking on, greedy,
the last `\boxed{}` graded, 8192-token cap, 4 at a time (`math_eval.py`, scratch).

```
3 passes                                unguided         LoopCD-Logits w_max 0.5   LoopCD fixed / broke
no thinking budget                      146 (9 cut off)  148 (8 cut off)           5 / 3
thinking_budget 6144                    150 (3 cut off)  153 (1 cut off)           4 / 1
```

The first run gave no thinking budget, so 8-9 answers thought past the cap with no answer; the engine's thinking budget
(the end-of-thinking token pushed up past 6144 reasoning tokens) turned most of them into answers, most right (5 fixed
each arm, 1 broken unguided). With it, LoopCD is +3 (96.2% to 98.1%), mean length 1433 against 1481 tokens. Direction
and size agree with the authors' table; one benchmark, small counts.

Fewer passes is not usable on this model: at `run_passes: 2` the reply degenerates at once (`<|message_model|>
multiplication 17 23 17 23 ...` for "What is 17 times 23?"), unguided and with LoopCD alike; the 2-pass MATH arm ran
nearly every problem to the cap and was stopped. Inkling 2 Mini's config says `is_varloop: false`: its coda only saw
the third pass's output in training, unlike the paper's half-depth models (Huginn, Ouro, Parcae).

Decode speed (`inference bench --depth 8,8192`, recipe C, q8 KV, one sequence):

```
                                          d8       d8192
plain                                     106.2    95.3
loop_guidance_logits loaded, no strength  94.9     85.2    (-11%: the extra coda every token)
loop_guidance_logits, w_max 0.5           72.4     66.6    (-32%)
```

The active contrast costs 3.3 ms a token where a second lm_head read and a few vocabulary-wide ops should cost about
0.5 ms: something in it leaves the CUDA graph or syncs with the host. Next: profile it.
