# Plain decode below the model card's figures

The model card's sweep (2026-10-22, master at #545) read plain decode (no drafter) 10-14% below the card's earlier
figures: 39 / 37 / 29 tok/s at short / 26K / 110K against 45 / 41 / 32 (2026-10-18, after #495). Question: a regression
in the non-speculative path, or the measurement?

## Run 1 - 2026-10-22 18:20

`plain/run.sh` (scratchpad): one plain server (recipe A, q3 cache, 262K window, master's LAN copy), the card's earlier
sampling (0.7 / 0.8 / 20) and the model's defaults (1.0 / 0.95 / 20) alternating, short and 26K prompts, twice:

```
round   sampling   short      26K
1       old        48.0       45.0 tok/s
1       default    47.7       38.3
2       old        36.2       32.6
2       default    39.1       35.0
```

Finding: round 1 matches or beats the earlier card; round 2 falls 20-25% on identical requests. The round-2 prompts were
prefix-cache hits, so either the restore path decodes slower, or the card's state changed between rounds.

## Run 2 - 2026-10-22 18:35

`plain/clocks.sh`: eight 512-token plain decodes on a fresh server, two prompts alternating, with
`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active -lms 500` during each:

```
request   tok/s   SM clock   GPU temp   power   throttle reasons (samples)
1         47.7    1658 MHz   65 C       342 W   0x4 (power cap) 20
2         47.6    1635       68         356     0x4 22
3         44.1    1346       67         321     0x20 (SW thermal slowdown) 14, 0x4 9
4         34.0     826       64         255     0x20 31
5         38.2    1023       64         279     0x20 23, 0x4 4
6         38.5    1016       64         278     0x20 24, 0x4 3
7         36.4     879       63         261     0x20 29
8         37.4     913       63         265     0x20 28
```

Finding: not a regression. From the third request the card holds SW thermal slowdown and the SM clock falls to
830-1,020 MHz, with decode following it, while the core reads 63-68 C, far below its limit. Repeated prompts and fresh
ones fall alike, so the prefix cache is not involved. On a 3090 the trigger at these core temperatures is most likely the
GDDR6X memory junction, which nvidia-smi does not report on GeForce.

Implication: the card's no-speculation column was measured throttled, and sustained runs on this machine (the decode
columns too) can read low. Benchmarks here should log `clocks_throttle_reasons.active` and discard throttled samples, or
alternate builds within one session as the kernel A/B runs did. The card's memory cooling is worth checking.
