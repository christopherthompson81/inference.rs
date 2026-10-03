#!/usr/bin/env python3
"""Writes goldens.json: random rows of ik_llama.cpp's types and their values from its reference dequantizers.

ik_llama.cpp has no Python dequantizers, so this calls `dequantize_row_*` in its shared ggml library:
`python3 make_goldens.py <ik_llama.cpp build>/ggml/src/libggml.so`.
"""
import ctypes
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).parent
SEED = 20261003
QK_K = 256
ROWS = 2
# ik's CUDA and CPU-GEMM kernels scale IQ2_KT by 1.05; its reference `dequantize_row_iq2_kt` leaves the factor out,
# so the reference reads a row scale with the factor folded in (as `dequantize_row_iq3_kt` folds in its 1.01)
IQ2_KT_SCALE = np.float32(1.05)
# ik's kernels read IQ6_K through a table that rounds the cubic its reference evaluates, so IQ6_K matches to
# within this fraction of a row's largest value instead of to the bit
IQ6_K_TOLERANCE = 0.01

rng = np.random.default_rng(SEED)
ggml = ctypes.CDLL(sys.argv[1])


class InitParams(ctypes.Structure):
    _fields_ = [("mem_size", ctypes.c_size_t), ("mem_buffer", ctypes.c_void_p), ("no_alloc", ctypes.c_bool)]


# ggml's f16 conversions read a table that ggml_init fills
ggml.ggml_init.argtypes = [InitParams]
ggml.ggml_init.restype = ctypes.c_void_p
ggml.ggml_init(InitParams(1 << 20, None, True))


def row_size(name: str, type_size: int, meta: int, cols: int) -> int:
    tails = (cols % QK_K) // 32
    tail = {"IQ3_KT": (tails + 1) // 2 + 12 * tails, "IQ4_KT": 16 * tails}.get(name, 0)
    size = meta + type_size * (cols // QK_K) + tail
    return (size + 3) // 4 * 4 if name.endswith("_KT") else size


def row_scale(meta: int) -> bytes:
    scale = rng.uniform(-0.002, 0.002)
    return {0: b"", 2: np.float16(scale).tobytes(), 4: np.float32(scale).tobytes()}[meta]


def block_scale(name: str) -> bytes:
    # IQ*_K blocks without a row scale lead with an f16 one
    return np.float16(rng.uniform(-0.002, 0.002)).tobytes() if name.endswith("_K") else b""


goldens = []
# (name, block bytes, row-scale bytes, columns); IQ3_KT / IQ4_KT rows also get tails: one at 544, seven at 480
for name, type_size, meta, cols in [
    ("IQ1_KT", 56, 4, 512),
    ("IQ2_KT", 68, 4, 512),
    ("IQ3_KT", 100, 4, 544),
    ("IQ3_KT", 100, 4, 480),
    ("IQ4_KT", 128, 4, 544),
    ("IQ4_KT", 128, 4, 480),
    ("IQ2_K", 76, 0, 512),
    ("IQ3_K", 110, 0, 512),
    ("IQ4_K", 144, 0, 512),
    ("IQ5_K", 176, 0, 512),
    ("IQ6_K", 212, 0, 512),
    ("IQ2_KS", 70, 2, 512),
    ("IQ3_KS", 102, 2, 512),
    ("IQ2_KL", 86, 2, 512),
    ("IQ4_KS", 136, 4, 512),
    ("IQ4_KSS", 128, 4, 512),
    ("IQ5_KS", 168, 4, 512),
]:
    dequantize = getattr(ggml, f"dequantize_row_{name.lower()}")
    dequantize.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int64]
    data, values = bytearray(), []
    for _ in range(ROWS):
        raw = bytearray(rng.integers(0, 256, row_size(name, type_size, meta, cols), dtype=np.uint8).tobytes())
        raw[0:meta] = row_scale(meta)
        for b in range(cols // QK_K):
            scale = block_scale(name)
            at = meta + b * type_size
            raw[at:at + len(scale)] = scale
        reference = bytearray(raw)
        if name == "IQ2_KT":
            reference[0:meta] = (np.frombuffer(raw[0:meta], dtype=np.float32) * IQ2_KT_SCALE).tobytes()
        out = np.zeros(cols, dtype=np.float32)
        dequantize(bytes(reference), out.ctypes.data, cols)
        data += raw
        values += [float(v) for v in out]
    golden = {"ty": name, "cols": cols, "bytes": list(data), "values": values}
    if name == "IQ6_K":
        golden["tolerance"] = IQ6_K_TOLERANCE
    goldens.append(golden)

(ROOT / "goldens.json").write_text(json.dumps(goldens) + "\n")
