# Seed-OSS on real checkpoints

Seed-OSS (#576) is a plain pre-norm Llama decoder (q/k/v bias, no o_proj or MLP bias, 80 x 128 query heads over a
5120 hidden, 8 KV heads, RoPE theta 1e7) put on the shared decoder. Its decoding is checked against llama.cpp
(970b51ce7, `build_cuda`) on the same GGUF, greedy, on the RTX 3090. Target: unsloth/Seed-OSS-36B-Instruct-GGUF
Q4_K_M (Apache 2.0; 20.3 GiB).

The GGUF's layout, from llama.cpp's `seed_oss` loader: the standard names, the post-attention norm as
`post_attention_norm`, NeoX RoPE (q/k stored as converted, so half-split), and the `seed-coder` pre-tokenizer, whose
split is the tokenizer.json's (which also normalizes NFC; llama.cpp does not).

Probes: the chat request "Explain in two sentences why the sky is blue." (greedy, top-2 logprobs), with thinking on
(200 tokens) and off (80 tokens; ours `enable_thinking: false`, llama.cpp `chat_template_kwargs.thinking_budget: 0`),
and a one-tool request ("What's the weather in Lyon right now, in celsius?") with its result sent back. Both servers
use the GGUF's own tokenizer and chat template (Unsloth's fixes to the Seed template).

## Run 1 - 2026-11-16 17:05

**Question:** does the 36B decode as llama.cpp does, and do its reasoning and tool calls come back split?

```
llama-server -m Seed-OSS-36B-Instruct-Q4_K_M.gguf -ngl 99 -c 4096 --jinja
inference serve -f Seed-OSS-36B-Instruct-Q4_K_M.gguf --pa-context-len 4096

thinking on   prompt 19 tokens on both; 175 of 200 tokens equal, max |d logprob| 0.138
              step 175 a near tie on both sides: llama.cpp ' light' -0.619 / ')' -0.780, ours ')' -0.515 / ' light' -0.919
thinking off  prompt 72 tokens on both; 80 of 80 equal, max |d logprob| 0.125
```

Thinking on, the reply opens inside `<seed:think>` and comes back as `reasoning_content` as llama.cpp's does. Thinking
off, the tokens match, but the split did not: this GGUF's template ends a zero-budget generation prompt with
`<seed:think><seed:cot_budget_reflect>` (the official template closes the block itself), so the reply starts inside
the reasoning block and our parser read it all as content. It now starts inside the block when the prompt's last
`<seed:think>` follows its last `</seed:think>`. After (`--pa-context-len 2048`): the budget note is the reasoning, the
answer the content, tokens still 80 of 80 and 175 of 200 equal.

Tool call, thinking on and off: `get_weather({"city":"Lyon","unit":"celsius"})`, finish `tool_calls`, no content.
With the result sent back (`{"temp": 17, "sky": "light rain"}`), the reply gives Lyon at 17 degrees with light rain
(283 prompt tokens).

Placement: at 4096 tokens layers 62-63 went to the CPU (decode 14 tok/s), at 2048 layer 63 (18.5 tok/s), where
llama.cpp holds every layer on the card. Weights 21.76 GB, activations 0.16 GB and KV 0.54 GB (2048) exceed our
22.2 GB cap (22.7 GB free less the 512 MB GPU reserve); llama.cpp keeps `token_embd` (447 MB, Q4_K) in host memory
and reserves nothing. That holds for every model, not Seed-OSS; by the same sums the Q4_K_S file (19.3 GiB) would fit whole on
this card (not run).
