# `inference-drafters`

External speculative drafters: separate models that draft tokens for a target model from the target's tapped hidden
states (DFlash, DFlash 2 and DSpark). They depend on no target model; a target supplies its taps, embedding table and
lm_head through `inference-nn`'s speculative traits. Built on `inference-nn`; target model crates depend on it.
