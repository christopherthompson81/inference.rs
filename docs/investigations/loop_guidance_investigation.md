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
