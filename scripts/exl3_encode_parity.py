#!/usr/bin/env python3
"""Our EXL3 encoder against exllamav3's: one linear of a safetensors model and its input Hessian (a `hin` file, as
the `exl3_hessians` example writes them) encoded by exllamav3's `quantize_exl3` and by the `exl3_encode` example,
their output errors tr(E H E^T) / tr(W H W^T) compared, and ours read back through exllamav3's `LinearEXL3` (the
format check: its reconstruction of our tensors gives our error).

Run with a Python that has exllamav3 and its extension (e.g. a venv with `pip install exllamav3`), from outside
exllamav3's source tree:
  <python> scripts/exl3_encode_parity.py <model.safetensors> <hessian dir> <bits> <3inst|mcg|mul1> <key> ...
"""

import os
import subprocess
import sys
import tempfile

import torch
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
from safetensors import safe_open
from safetensors.torch import load_file

# ours may be this much worse than exllamav3's before the comparison fails (the sign flips differ, so the two
# errors are draws from one distribution, not one number)
TOLERANCE = 1.10
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "target", "debug", "examples", "exl3_encode")


def output_error(w, q, h):
    # w, q are (in, out), h (in, in)
    e = w - q
    return ((h @ e) * e).sum().item() / ((h @ w) * w).sum().item()


def main():
    model, hessians, bits, codebook = sys.argv[1:5]
    keys = sys.argv[5:]
    worst = 0.0
    with safe_open(model, "pt") as f:
        for key in keys:
            # exllamav3 keeps weights as (in, out)
            w = f.get_tensor(f"{key}.weight").float().T.contiguous().cuda()
            h = (
                load_file(os.path.join(hessians, f"{key}.safetensors"))["hin"]
                .float()
                .cuda()
            )
            quant_args = {
                "K": int(float(bits)) if float(bits).is_integer() else float(bits),
                "seed": 1,
                "devices": [0],
                "apply_out_scales": None,
                "sigma_reg": 0.025,
                "mcg": codebook == "mcg",
                "mul1": codebook == "mul1",
            }
            h_data = {"H": h.clone(), "count": 1, "finalized": False, "first_key": key}
            q_ref, _, _ = quantize_exl3(
                w.clone(), h_data, quant_args, return_weight_q=True
            )
            ref = output_error(w, q_ref.float(), h)

            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "enc.safetensors")
                hfile = os.path.join(hessians, f"{key}.safetensors")
                subprocess.run(
                    [EXAMPLE, model, key, hfile, bits, codebook, "1", out],
                    check=True,
                    capture_output=True,
                )
                ours_t = load_file(out, device="cuda")
            linear = LinearEXL3(
                None,
                w.shape[0],
                w.shape[1],
                scale=None,
                su=None,
                sv=None,
                suh=ours_t[f"{key}.suh"],
                svh=ours_t[f"{key}.svh"],
                trellis=ours_t[f"{key}.trellis"],
                mcg=ours_t.get(f"{key}.mcg"),
                mul1=ours_t.get(f"{key}.mul1"),
                bias=None,
                out_dtype=torch.half,
            )
            ours = output_error(w, linear.get_weight_tensor().float(), h)
            ratio = ours / ref
            worst = max(worst, ratio)
            print(f"{key}: exllamav3 {ref:.6f}, ours {ours:.6f} (x{ratio:.3f})")
    print(f"worst ratio {worst:.3f}")
    sys.exit(0 if worst < TOLERANCE else 1)


if __name__ == "__main__":
    main()
