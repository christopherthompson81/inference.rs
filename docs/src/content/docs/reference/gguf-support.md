---
title: GGUF compatibility
description: Model families, multimodal projectors, and storage formats supported for GGUF files.
---

This page is the compatibility reference for GGUF models. For commands and workflows, see
[Run GGUF models](/guides/models/run-gguf/).

## Text model families

The `GGUF architecture` column is the value stored in the file's `general.architecture` metadata.
Fine-tunes and model sizes within a family use the same entry.

| Model family | GGUF architecture |
|---|---|
| Llama, Mistral, Mixtral | `llama` |
| Mistral 3 text weights | `mistral3` |
| Gemma | `gemma` |
| Gemma 2 | `gemma2` |
| Gemma 3 text weights | `gemma3` |
| Phi-2 | `phi2` |
| Phi-3 and Phi-3.5 | `phi3` |
| Phi-3.5 MoE | `phimoe` |
| Qwen (Qwen 1) | `qwen` |
| Qwen2 and Qwen2.5 | `qwen2` |
| Qwen1.5-MoE and Qwen2-MoE | `qwen2moe` |
| Qwen3 | `qwen3` |
| Qwen3 MoE | `qwen3moe` |
| Qwen3-Next and Qwen3-Coder-Next | `qwen3next` |
| Qwen3.5 and Qwen3.6 dense | `qwen35` |
| Qwen3.5 and Qwen3.6 MoE | `qwen35moe` |
| Qwen3.8-Flash-Next | `qwen4exp` |
| StarCoder2 | `starcoder2` |
| DeepSeek-V2, DeepSeek-V3, DeepSeek-R1 (non-distill), GLM-4 MoE Lite | `deepseek2` |
| GLM-4 dense | `glm4` |
| GLM-4 MoE | `glm4moe` |
| SmolLM3 | `smollm3` |
| Granite dense | `granite` |
| Granite MoE | `granitemoe` |
| Granite hybrid | `granitehybrid` |
| Granite SWA (swash) dense and MoE | `granite_swa` |
| Nemotron-H dense and MoE | `nemotron_h`, `nemotron_h_moe` |
| Falcon-H1 | `falcon-h1` |
| OLMo 1 | `olmo` |
| OLMo 2 and Olmo 3 | `olmo2` |
| OLMoE | `olmoe` |
| ERNIE 4.5 dense | `ernie4_5` |
| ERNIE 4.5 MoE | `ernie4_5-moe` |
| GPT-OSS | `gpt-oss` |
| Hunyuan dense | `hunyuan-dense` |
| Hunyuan MoE | `hunyuan-moe` |
| LFM2 and LFM2.5 dense | `lfm2` |
| LFM2 and LFM2.5 MoE | `lfm2moe` |

Some GGUF architectures cover more than one model family. Repository files and GGUF metadata are
used together to select the family. For a standalone file that cannot be identified unambiguously,
pass the original model with `--tok-model-id`.

GLM GGUFs configured for multimodal input are not supported as text-only models.

## Multimodal model families

Multimodal GGUF requires a compatible companion projector. Depending on the family, the loader may
also need the original configuration or processor assets. Repository loading selects supporting
files when the repository and GGUF metadata identify them unambiguously. The direct local
`-f /path/model.gguf` shorthand also selects an unambiguous projector stored beside the model.

| Model family | GGUF architecture |
|---|---|
| Gemma 3 | `gemma3` |
| Gemma 3n | `gemma3n` |
| Gemma 4 dense and MoE | `gemma4` |
| Granite 4 Vision | `granite` (projector `granite4_vision`) |
| Granite Speech | `granite` (projector `granite_speech`) |
| Idefics3 and SmolVLM | `llama` |
| Mistral 3 and Pixtral | `mistral3` |
| Llama 4 | `llama4` |
| LFM2-VL and LFM2.5-VL | `lfm2` |
| Muse Glimmer | `muse-glimmer` |
| Nemotron 3.5 Bolt (Nemotron-H Omni) | `nemotron_h_moe` (projector `nemotron_omni`) |
| Nemotron 3.5 Super VL (Nemotron-H Omni) | `nemotron_h_moe` (projector `nemotron_omni`) |
| Nemotron 3 Nano Omni (Nemotron-H Omni) | `nemotron_h_moe` (the checkpoint's safetensors shard as projector) |
| Nemotron Nano 12B v2 VL (Nemotron-H Omni) | `nemotron_h` (projector `nemotron_v2_vl`, or the checkpoint's shard) |
| PaddleOCR-VL (1.5, 1.6) | `paddleocr` |
| Qwen2-VL and Qwen2.5-VL | `qwen2vl` |
| Qwen3-VL | `qwen3vl` |
| Qwen3-ASR | `qwen3vl` (projector `qwen3a`) |
| Qwen3-VL MoE | `qwen3vlmoe` |
| Qwen3.5 and Qwen3.6 multimodal dense | `qwen35` |
| Qwen3.5 and Qwen3.6 multimodal MoE | `qwen35moe` |
| Audio Flamingo 3 and Music Flamingo | `qwen2` (projector `musicflamingo`) |
| GLM-ASR | `llama` (projector `glma`) |
| MERaLiON-2 | `gemma2` (projector `meralion`) |
| Qwen2-Audio | `qwen2` (projector `qwen2a`) |
| Ultravox | `llama` (projector `ultravox`) |
| Voxtral Mini and Small | `llama` (projector `voxtral`) |

Input modalities depend on the model. A listed family does not imply that every checkpoint accepts
image, audio, and video. Use the model card and [multimodal input guide](/guides/models/multimodal-input/)
for its supported request types.

Multimodal Qwen3.5 and Qwen3.6 accept image and video input, but not audio. Gemma 4 accepts
image/video and audio when the matching components are present in the model configuration and
projector files.

The audio GGUFs (Voxtral, Ultravox, Qwen2-Audio, Audio Flamingo 3, GLM-ASR, MERaLiON-2) load with no base repository: the config comes
from the text and projector GGUFs' metadata. For Voxtral and GLM-ASR the loader's chat template replaces the one the
published GGUFs embed (another Mistral model's; GLM-ASR's original, which writes no audio markers). llama.cpp converts
GLM-ASR's original checkpoint, so its GGUFs load onto transformers' layout, as the safetensors loader reads it.
MERaLiON-2's official GGUFs come from the repo's patched llama.cpp (`meralion` projector). A GGUF's end-of-turn and
end-of-message tokens end a generation as its EOS does.

NVIDIA's Nemotron 3.5 Bolt GGUF and its `mmproj` load with no base repository: the config comes from the two files'
metadata, and the projector holds the checkpoint's C-RADIO tower and projector unchanged. Images and video are
supported (a video's tokens pruned with EVS at the released checkpoints' rate, 0.7). Nemotron 3.5 Super VL's GGUF and
mmproj (`ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF`) load the same way, its tower's final LayerNorm from
the projector's `v.post_ln`. On a 24 GB GPU the text GGUF and the tower
together do not fit: map a few blocks to the CPU (`-n "0:43"`), as Bolt's IQ2_XXS experts cannot stay in host memory
on their own.

Nemotron 3 Nano Omni takes images and audio, but no published projector carries its sound encoder, and the published
`nemotron_v2_vl` projectors keep its position table cut to one 512-pixel image. Its towers load from NVIDIA's own
checkpoint instead: the first safetensors shard of `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (4 GB) holds
the whole C-RADIO tower, the Parakeet sound encoder and both projectors. Pass it as the projector beside a text GGUF,
with that repository (or a directory with its `config.json`, tokenizer and processor files) as `--tok-model-id`:

```bash
inference serve --format gguf -m lmstudio-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  -f Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Q4_K_M.gguf \
  --mmproj /path/to/model-00001-of-00017.safetensors \
  --tok-model-id nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
```

Images run at the checkpoint's dynamic resolution, as with Bolt; each clip takes one token per encoder frame (one per
80 ms). Video is not yet supported.

`inference quantize mmproj` writes the same towers as one llama.cpp-style projector, so the pair then loads with no
base repository and runs in llama.cpp too: C-RADIO with its full position table under Bolt's `nemotron_omni` projector
type, the sound encoder as llama.cpp's `parakeet`, and the checkpoint's chat template (the published text GGUF's has no
media parts). It reads a checkpoint directory with only the shards that hold the towers present:

```bash
inference quantize mmproj -m /path/to/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 -o mmproj-nano-omni.gguf
inference serve --format gguf -m lmstudio-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  -f Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Q4_K_M.gguf --mmproj mmproj-nano-omni.gguf
```

Nemotron Nano 12B v2 VL loads the same way from a dense `nemotron_h` text GGUF: its first safetensors shard
(`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`, 3.8 GB) holds the C-RADIO tower and `mlp1`. Images are cut into InternVL's
tiles (up to 12 of 512 pixels on the closest aspect ratio, and a thumbnail), as the checkpoint's processor does. A
`nemotron_v2_vl` projector loads too, the published ones included, and its images are tiled the same way (llama.cpp
runs that projector as one fixed 512-pixel image). `inference quantize mmproj` writes Nano VL's as `nemotron_v2_vl` with
the checkpoint's CLIP normalization (the published files carry ImageNet's), its position table resized for a 512-pixel
tile as RADIO resizes it, and its tile count.

Matrices keep the checkpoint's float type unless `--matrix-type` names another (`f16`, `q8_0`, ...); norms, biases,
convolutions and embeddings stay f32, as llama.cpp's converter keeps them.

Muse Glimmer GGUFs require a companion `muse-glimmer` projector. Current published GGUF
repositories also require `--tok-model-id meta-models/Muse-Glimmer-30B` because they do not carry
enough base-model configuration metadata for standalone loading. Image input is supported. Video
input is rejected because the llama.cpp conversion irreversibly sums each pair of temporal patch
weights in the projector.

The `gemma3n`, `gemma4`, `llama4`, `qwen2vl`, `qwen3vl`, and `qwen3vlmoe` architectures always need a
projector. Architectures that appear in both tables above, including `gemma3`, `llama`, `mistral3`,
`lfm2`, `qwen35`, and `qwen35moe`, load as text models when no projector is supplied.

## Storage formats

inference.rs accepts GGUF files using the following storage types. A file can mix types, as common
`_K_M` and `_K_S` artifacts do.

| Category | Supported storage types |
|---|---|
| Floating point | `F32`, `F16`, `BF16` |
| Legacy block quants | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` |
| K-quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` |
| IQ quants | `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS` |
| ik_llama.cpp trellis quants | `IQ1_KT`, `IQ2_KT`, `IQ3_KT`, `IQ4_KT` |
| ik_llama.cpp IQK quants | `IQ2_K`, `IQ3_K`, `IQ4_K`, `IQ5_K`, `IQ6_K`, `IQ2_KS`, `IQ3_KS`, `IQ4_KS`, `IQ5_KS`, `IQ4_KSS`, `IQ2_KL` |
| FP4 | `NVFP4` (64-value blocks of E2M1 with a UE4M3 scale per 16, and the tensor-wide F32 `.scale` llama.cpp pairs with it) |
| GPT-OSS | The GPT-OSS MXFP4 representation |

The IQ, trellis and IQK types run on CUDA and Metal; on the CPU they are dequantized for each matmul. On CUDA, MoE
expert stacks of these types run through the expert-indexed matvec kernels. Experts kept in host memory by
[expert offload](/guides/perf/moe-expert-offload/) run on the CPU kernels for IQ3_S, IQ4_XS, IQ4_KSS and IQ4_NL (and
Q2_K-Q6_K, Q8_0). `NVFP4` runs on CUDA (any GPU with dp4a, through the same matvec and tiled kernels as the
legacy block quants) and is dequantized on the CPU; Metal has no kernel for it. Tensor parallelism cannot split the rows of
the trellis and `_KS` / `_KSS` / `_KL` types, since each of their rows keeps one scale. llama-quantize's IQ mixes (`IQ2_M`,
`IQ3_XS`, `IQ3_M`) and ik_llama.cpp's default mixes use only the types above. ik_llama.cpp's bitnet types
and its CPU-repacked `_R4` / `_R8` layouts are not supported. Other storage types not listed above
are also unsupported. Big-endian GGUF files are not supported.

## Feature compatibility

| Capability | GGUF support |
|---|---|
| Local exact-file loading | Yes, with `-f` |
| Hugging Face exact-file loading | Yes, with `-m` and `-f` |
| Automatic artifact selection | Yes, with `-m` and `--quant` |
| Tokenizer and chat-template discovery | From embedded GGUF metadata or supplied model assets |
| Multimodal projector discovery | From an unambiguous GGUF repository, or an adjacent projector with the direct local `-f` shorthand |
| Serving, tool calling, and agents | Same runtime paths as other loads; checkpoint and chat-template support still apply |
| Dynamic LoRA | Language-model adapters; the architectures below are rejected |
| Multimodal LoRA | Language-model adapters only; projector, vision, and audio adapters are not supported |
| ISQ requantization | Yes, for compatible weights selected with `-f` |
| Offline loading | Yes, when every required file is local or cached |

Text GGUF models whose converter reordered Q/K rows into adjacent rotary pairs (`llama`,
`mistral3`, `smollm3`, `granite`, `granitemoe`, `granitehybrid`) accept dynamic LoRA: adapter rows
are mapped onto that order when the adapter loads. Dynamic LoRA is rejected for `deepseek2` and
`glm4`, which rotate only part of each head, and for multimodal models routed through
adjacent-RoPE architectures, including Idefics3, Mistral 3/Pixtral, Llama 4, and Muse Glimmer.
Base-model loading is unaffected.

Beyond the text and multimodal families above, GGUF also loads:

- Encoder embedders and rerankers as llama.cpp's converter writes them (`bert`, `nomic-bert`, `jina-bert-v3`, `eurobert` and others); see [embeddings](/guides/models/use-embeddings/#gguf-embedders).
- Speech: Parakeet ASR GGUFs (NVIDIA's own, or written by `inference quantize asr`), Qwen3-ASR from a `qwen3vl` text GGUF with its `qwen3a` projector, Silero VAD, and audio.cpp's Kokoro GGUF; see [speech models](/guides/models/use-speech-models/). Whisper does not load from GGUF.
- Image generation: FLUX transformers (with a T5 encoder GGUF) and HiDream-O1 in ComfyUI-GGUF's layout; see [image generation](/guides/models/use-image-generation/).
- World generation: WorldStereo-2 and the umT5 text encoder; see [3D worlds](/guides/models/generate-3d-worlds/).
- Decision models, from llama.cpp's GGUFs of them; see [decision models](/guides/models/use-decision-models/).
- PP-DocLayoutV3 layout detection.
