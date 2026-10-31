---
title: The multimodal pipeline
description: How image, video, and audio inputs reach the model.
---

Multimodal requests carry non-text content parts (`image_url`, `audio`, `video`). Each part goes through a per-modality preprocessing and encoder path before the tokens reach the transformer.

## Request shape

Content parts are ordered in the request body. The engine preserves that order when it interleaves media tokens with text tokens, so text surrounding an image appears on either side of the image tokens in the final token sequence. The transformer sees one uniform token stream.

## Image path

1. **Decode.** The source is resolved to bytes, then to a pixel buffer. The engine itself reads only `data:image/...;base64,` URLs and attachments passed beside the request (`media://<index>`); any other source goes to the client's media fetcher. The server's fetcher reads public `http(s)://` addresses and the web UI's `inference-upload:` references and rejects `file://`; a local client's (the CLI's) also reads file paths, `file://` URLs and any address.
2. **Preprocess.** Model-specific: resize to the vision encoder's expected resolution, normalize per-channel, tensorize.
3. **Encode.** The vision encoder produces a sequence of patch embeddings.
4. **Project.** A learned projection maps patch embeddings to the transformer's hidden dimension.
5. **Place.** The projected tokens are inserted at the position corresponding to the content part in the user's request.

Multiple images in one request are encoded as a batch.

## Video path

Video is decoded to frames before model preprocessing. Each selected frame then flows through the image path.

Supported containers and FFmpeg installation are covered in [Set up video input](/guides/models/video-setup/).

## Audio path

1. **Decode.** The audio is decoded to PCM via symphonia (WAV via hound) at the file's native sample rate, then resampled to the model's expected rate.
2. **Feature extraction.** The PCM is converted to a mel-spectrogram (or a similar feature representation) expected by the encoder.
3. **Encode.** A model-specific audio encoder produces a sequence of vectors.
4. **Project and place.** As with images.

## Encoder cache

Encoder outputs are cached by content hash and modality. When the same image, video, or audio clip appears in a later request, or in a later turn of the same session, the encoder pass is skipped and the cached tokens are reused.

The modality is part of the key: identical bytes processed as an image versus as a video frame can yield different token counts, and the cache keeps them separate.

The cache is LRU (least-recently-used). It retains 32 entries per model by default. Serving commands can instead bound retained tensor memory with `--encoder-cache-memory-mb`; matching headroom is reserved before PagedAttention sizes the KV cache, and cached views are copied into compact storage. Hit, miss, eviction, and resident-memory metrics are exposed for observability.

## See also

- Guide: [multimodal input](/guides/models/multimodal-input/).
- Setup: [set up video input](/guides/models/video-setup/).
- Reference: [supported models](/reference/supported-models/).
