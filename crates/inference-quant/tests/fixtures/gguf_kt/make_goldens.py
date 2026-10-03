#!/usr/bin/env python3
"""Writes goldens.json: random trellis (IQ*_KT) rows and their values from ik_llama.cpp's reference dequantizers.

ik_llama.cpp has no Python dequantizers, so this calls `dequantize_row_iq*_kt` in its shared ggml library:
`python3 make_goldens.py <ik_llama.cpp build>/ggml/src/libggml.so`.
"""
import ctypes
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).parent
SEED = 20261003
ROW_META = 4
QK_K = 256
ROWS = 2
# ik's CUDA and CPU-GEMM kernels scale IQ2_KT by 1.05; its reference `dequantize_row_iq2_kt` leaves the factor out,
# so the reference reads a row scale with the factor folded in (as `dequantize_row_iq3_kt` folds in its 1.01)
IQ2_KT_SCALE = np.float32(1.05)

rng = np.random.default_rng(SEED)
ggml = ctypes.CDLL(sys.argv[1])


def row_size(name: str, type_size: int, cols: int) -> int:
    tails = (cols % QK_K) // 32
    tail = {"IQ3_KT": (tails + 1) // 2 + 12 * tails, "IQ4_KT": 16 * tails}.get(name, 0)
    size = ROW_META + type_size * (cols // QK_K) + tail
    return (size + 3) // 4 * 4


goldens = []
# IQ3_KT and IQ4_KT rows also get tails after their whole blocks: one at 544 columns, seven at 480
for name, type_size, cols in [
    ("IQ1_KT", 56, 512),
    ("IQ2_KT", 68, 512),
    ("IQ3_KT", 100, 544),
    ("IQ3_KT", 100, 480),
    ("IQ4_KT", 128, 544),
    ("IQ4_KT", 128, 480),
]:
    dequantize = getattr(ggml, f"dequantize_row_{name.lower()}")
    dequantize.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int64]
    data, values = bytearray(), []
    for _ in range(ROWS):
        raw = bytearray(rng.integers(0, 256, row_size(name, type_size, cols), dtype=np.uint8).tobytes())
        raw[0:ROW_META] = np.float32(rng.uniform(-0.002, 0.002)).tobytes()
        reference = bytearray(raw)
        if name == "IQ2_KT":
            reference[0:ROW_META] = (np.frombuffer(raw[0:ROW_META], dtype=np.float32) * IQ2_KT_SCALE).tobytes()
        out = np.zeros(cols, dtype=np.float32)
        dequantize(bytes(reference), out.ctypes.data, cols)
        data += raw
        values += [float(v) for v in out]
    goldens.append({"ty": name, "cols": cols, "bytes": list(data), "values": values})

(ROOT / "goldens.json").write_text(json.dumps(goldens) + "\n")
