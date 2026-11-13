#!/usr/bin/env python3
"""Our EXL3 reconstruction against exllamav3's: each EXL3 linear of a safetensors file through exllamav3's
`LinearEXL3.get_weight_tensor` (CUDA) and through the `exl3_reconstruct` example, compared value by value.

Run with a Python that has exllamav3 and its extension (e.g. a venv with `pip install exllamav3`):
  <python> scripts/exl3_parity.py <model.safetensors> [max tensors]
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from exllamav3.modules.quant.exl3 import LinearEXL3
from safetensors import safe_open

# exllamav3 rotates and scales in f16; ours runs in f32, so they agree to f16 rounding of the largest values
TOLERANCE = 2e-3


def main():
    path = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        ["cargo", "build", "-q", "-p", "inference-quant", "--example", "exl3_reconstruct"],
        cwd=root,
        check=True,
    )
    example = os.path.join(root, "target", "debug", "examples", "exl3_reconstruct")
    failed = False
    with safe_open(path, "pt", device="cuda") as f, tempfile.TemporaryDirectory() as tmp:
        names = set(f.keys())
        keys = sorted(k[: -len(".trellis")] for k in names if k.endswith(".trellis"))[:limit]
        for key in keys:
            get = lambda n: f.get_tensor(f"{key}.{n}") if f"{key}.{n}" in names else None
            trellis = get("trellis")
            k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
            lin = LinearEXL3(None, k, n, suh=get("suh"), svh=get("svh"), trellis=trellis, mcg=get("mcg"), mul1=get("mul1"))
            ref = lin.get_weight_tensor().float().T.contiguous().cpu().numpy()
            out = os.path.join(tmp, "w.f32")
            subprocess.run([example, path, key, out], check=True)
            ours = np.fromfile(out, dtype=np.float32).reshape(ref.shape)
            err = float(np.abs(ours - ref).max() / np.abs(ref).max())
            cb = "mcg" if get("mcg") is not None else "mul1" if get("mul1") is not None else "3inst"
            print(f"{key:60s} {cb:5s} words {trellis.shape[-1]:3d}  max rel err {err:.2e}")
            failed |= err > TOLERANCE
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
