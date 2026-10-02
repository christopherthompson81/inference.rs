#!/usr/bin/env python3
"""Writes goldens.json: random IQ4_NL / IQ4_XS blocks and their values from gguf-py's reference dequantizers.

Run with llama.cpp's gguf-py importable, e.g. `PYTHONPATH=llama.cpp/gguf-py python3 make_goldens.py`.
"""
import json
import pathlib

import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize

ROOT = pathlib.Path(__file__).parent
SEED = 20261002
BLOCKS = 4

rng = np.random.default_rng(SEED)


def block(type_size: int) -> bytes:
    raw = bytearray(rng.integers(0, 256, type_size, dtype=np.uint8).tobytes())
    # a finite, sign-varying f16 scale; the remaining bytes are any scale bits and indices
    raw[0:2] = np.float16(rng.uniform(-0.05, 0.05)).tobytes()
    return bytes(raw)


goldens = []
for name, qtype, block_size, type_size in [
    ("IQ4_NL", GGMLQuantizationType.IQ4_NL, 32, 18),
    ("IQ4_XS", GGMLQuantizationType.IQ4_XS, 256, 136),
]:
    data = b"".join(block(type_size) for _ in range(BLOCKS))
    values = dequantize(np.frombuffer(data, dtype=np.uint8), qtype).astype(np.float32)
    assert values.size == BLOCKS * block_size
    goldens.append({"ty": name, "bytes": list(data), "values": [float(v) for v in values]})

(ROOT / "goldens.json").write_text(json.dumps(goldens) + "\n")
