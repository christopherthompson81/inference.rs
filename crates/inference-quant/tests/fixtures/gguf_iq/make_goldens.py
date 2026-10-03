#!/usr/bin/env python3
"""Writes goldens.json: random IQ blocks and their values from gguf-py's reference dequantizers.

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


# IQ1_M has no leading scale: its f16 is the top nibble of each of the four u16 scale words at this offset
IQ1_M_SCALES = 48


def block(name: str, type_size: int) -> bytes:
    raw = bytearray(rng.integers(0, 256, type_size, dtype=np.uint8).tobytes())
    # a finite, sign-varying f16 scale; the remaining bytes are any scale bits and indices
    scale = np.float16(rng.uniform(-0.05, 0.05)).view(np.uint16)
    if name == "IQ1_M":
        for k in range(4):
            at = IQ1_M_SCALES + 2 * k
            word = int.from_bytes(raw[at:at + 2], "little") & 0x0FFF | ((int(scale) >> (4 * k)) & 0xF) << 12
            raw[at:at + 2] = word.to_bytes(2, "little")
    else:
        raw[0:2] = int(scale).to_bytes(2, "little")
    return bytes(raw)


goldens = []
for name, block_size, type_size in [
    ("IQ4_NL", 32, 18),
    ("IQ4_XS", 256, 136),
    ("IQ2_XXS", 256, 66),
    ("IQ2_XS", 256, 74),
    ("IQ2_S", 256, 82),
    ("IQ3_XXS", 256, 98),
    ("IQ3_S", 256, 110),
    ("IQ1_S", 256, 50),
    ("IQ1_M", 256, 56),
]:
    qtype = GGMLQuantizationType[name]
    data = b"".join(block(name, type_size) for _ in range(BLOCKS))
    values = dequantize(np.frombuffer(data, dtype=np.uint8), qtype).astype(np.float32)
    assert values.size == BLOCKS * block_size
    goldens.append({"ty": name, "bytes": list(data), "values": [float(v) for v in values]})

(ROOT / "goldens.json").write_text(json.dumps(goldens) + "\n")
