# BiRefNet investigation

Native BiRefNet (ZhengPeng7/BiRefNet, MIT) for background removal ahead of image-to-3D (JoVox takes an object on a
transparent background; its reference removes backgrounds with RMBG-2.0, a BiRefNet retrained under a non-commercial
license, which is not ported). The model card's own code is the reference. Weights and dumps are never committed.

## Run 1 - 2026-11-10 09:20

Question: does the reference run here, and what does it compute at inference?

Reference: the model repo at e2bf8e4 (`birefnet.py`, `BiRefNet_config.py`), imported as a package from scratch with
timm 1.0.25 and einops installed to a scratch `--target` dir and kornia stubbed (only its `laplacian`, a training
target, is imported). Weights: `model.safetensors`, 444 MB, F16 (687 tensors) plus 67 BatchNorm counters; loaded as
f32. Test image: scikit-image's `chelsea.png` (451 x 300 RGB, public domain). `ref/ref.py` (scratch) runs the model
card's preprocessing (`Resize((1024, 1024))` on the PIL image, ImageNet normalization), CUDA f32 with TF32 off, and
records the backbone and encoder outputs, the squeeze block, the decoder blocks and the deformable ASPPs, the logits,
and the mask as the card makes it (sigmoid, `ToPILImage`, PIL resize back).

The inference path: Swin-L (window 12, depths 2/2/18/2) on the image and on it bilinearly halved (`align_corners`),
each of the 4 scales concatenated with its half-size twin upsampled; the 3 shallower scales resized onto the deepest
(5760 channels at 32^2) and squeezed by one decoder block to 3072; then 4 decoder blocks (conv, BN, ReLU, a deformable
ASPP of a 1x1, deformable 1x1/3x3/7x7 branches and a global pool, conv, BN), each output gated by a sigmoid map from a
16-wide side branch, lateral 1x1 skips, and the normalized image stacked into channels as patches matching each scale
fed in beside every block. Multi-scale supervision outputs are training-only. Reference forward: 0.39-0.48 s f32.

## Run 2 - 2026-11-10 09:45

Question: does the port (`inference_recon::birefnet`, f32) match, and where does any gap come from?

`BIREFNET_DIR=... BIREFNET_REF=.../chelsea.safetensors BIREFNET_IMAGE=.../chelsea.png cargo nextest run -p
inference-recon --features cuda --lib --run-ignored all -E 'test(/birefnet/)'`. First pass:
```
bb.0-3:  rms 5e-7 .. 2e-6        enc.0-3: rms 4e-7 .. 2e-6
squeeze_module: rms 9.0e-2       logits: rms 3.2e-2
```
The Swin encoder matched; the squeeze block did not. Dumping its insides: the offset convolution matched (rms 8e-7),
the deformable 1x1 branches were off by 5e-3 and the 3x3/7x7 by 1.1e-1. Cause: the sampler clamped positions in
[H-1, H) to the last row, as RoIAlign's bilinear does; `deform_conv2d`'s bilinear does not clamp, it takes the
out-of-range corners as zero. Without the clamp:
```
squeeze_module: max 1.0e-5 rms 2.1e-6
logits:         max 1.1e-4 rms 5.4e-7
mask:           0 of 135300 pixels differ
forward 0.66 s
```
The mask matches bit for bit, through the PIL bilinear resize in and the bicubic resize out (`inference_vision::
pil_resize`, generalized from the bicubic RGB path to any channel count and PIL's bilinear and Lanczos filters).
The deformable convolution is tensor ops (4 bilinear gathers and a matmul per tap, 60 taps per block); the forward is
0.66 s against the reference's 0.4 s, not yet profiled.
