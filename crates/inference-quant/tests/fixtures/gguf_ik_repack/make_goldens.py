#!/usr/bin/env python3
"""Writes goldens.json: the first rows of one tensor in plain and ik_llama.cpp CPU-repacked layouts.

The inputs are Qwen3.5-0.8B (Apache-2.0) quantized by ik_llama.cpp from `scripts/make_gguf_test_dirs.sh`'s F16 GGUF:
`llama-quantize --pure <f16> plain-T.gguf T`, then `llama-quantize --repack plain-T.gguf repack-T.gguf T`, so both
hold the same quantization and only the row-interleaved layout differs. Row-scaled types (IQ4_KS, IQ5_KS) keep their
byte size: a repacked group leads with its rows' f32 scales, then the interleaved blocks.
`python3 make_goldens.py <directory with plain-T.gguf and repack-T.gguf>`.
"""
import base64
import json
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).parent
TENSOR = "blk.0.attn_gate.weight"
ROWS = 8
DEFAULT_ALIGNMENT = 32
QK = 32
QK_K = 256
# (type, base ggml id, repacked ggml id, rows per repacked group, elements per block, bytes per block, row-scale bytes)
TYPES = [
    ("Q4_0", 2, 202, 8, QK, 18, 0),
    ("Q5_0", 6, 206, 4, QK, 22, 0),
    ("Q8_0", 8, 208, 8, QK, 34, 0),
    ("Q2_K", 10, 210, 4, QK_K, 84, 0),
    ("Q3_K", 11, 211, 4, QK_K, 110, 0),
    ("Q4_K", 12, 212, 4, QK_K, 144, 0),
    ("Q5_K", 13, 213, 4, QK_K, 176, 0),
    ("Q6_K", 14, 214, 4, QK_K, 210, 0),
    ("IQ2_XXS", 16, 216, 4, QK_K, 66, 0),
    ("IQ2_XS", 17, 217, 4, QK_K, 74, 0),
    ("IQ3_XXS", 18, 218, 4, QK_K, 98, 0),
    ("IQ4_NL", 20, 220, 4, QK, 18, 0),
    ("IQ3_S", 21, 221, 4, QK_K, 110, 0),
    ("IQ2_S", 22, 222, 4, QK_K, 82, 0),
    ("IQ4_XS", 23, 223, 8, QK_K, 136, 0),
    ("IQ2_K", 137, 337, 4, QK_K, 76, 0),
    ("IQ3_K", 138, 338, 4, QK_K, 110, 0),
    ("IQ4_K", 139, 339, 4, QK_K, 144, 0),
    ("IQ5_K", 140, 340, 4, QK_K, 176, 0),
    ("IQ4_KS", 144, 344, 4, QK_K, 136, 4),
    ("IQ5_KS", 152, 352, 4, QK_K, 168, 4),
    ("MXFP4", 39, 353, 8, QK, 17, 0),
]
SCALAR_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9


def tensor_rows(path: pathlib.Path, name: str, rows: int, row_bytes):
    """The tensor's type, columns and first `rows` rows, with `row_bytes(cols)` bytes per row."""
    with open(path, "rb") as f:
        def u32():
            return struct.unpack("<I", f.read(4))[0]

        def u64():
            return struct.unpack("<Q", f.read(8))[0]

        def string():
            return f.read(u64()).decode()

        def value(ty):
            if ty == STRING:
                return string()
            if ty == ARRAY:
                inner, count = u32(), u64()
                return [value(inner) for _ in range(count)]
            fmt = SCALAR_FORMATS[ty]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        assert f.read(4) == b"GGUF"
        u32()
        tensor_count, kv_count = u64(), u64()
        kv = {}
        for _ in range(kv_count):
            key = string()
            kv[key] = value(u32())
        infos = []
        for _ in range(tensor_count):
            tensor = string()
            dims = [u64() for _ in range(u32())]
            infos.append((tensor, dims, u32(), u64()))
        align = kv.get("general.alignment", DEFAULT_ALIGNMENT)
        start = (f.tell() + align - 1) // align * align
        for tensor, dims, ty, offset in infos:
            if tensor != name:
                continue
            cols = dims[0]
            size = row_bytes(cols)
            f.seek(start + offset)
            data = f.read(size * rows)
            assert len(data) == size * rows, (path, name)
            return ty, cols, data
    raise KeyError(name)


goldens = []
src = pathlib.Path(sys.argv[1])
for name, base, repacked, rows_per_group, block_elems, block_bytes, row_meta in TYPES:
    def row_bytes(cols):
        assert cols % block_elems == 0, (name, cols)
        return row_meta + cols // block_elems * block_bytes

    plain_ty, cols, plain = tensor_rows(src / f"plain-{name}.gguf", TENSOR, ROWS, row_bytes)
    repack_ty, repack_cols, packed = tensor_rows(src / f"repack-{name}.gguf", TENSOR, ROWS, row_bytes)
    assert (plain_ty, repack_ty) == (base, repacked), (name, plain_ty, repack_ty)
    assert cols == repack_cols
    goldens.append({
        "ty": f"{name}_R{rows_per_group}",
        "id": repacked,
        "base": base,
        "rows_per_group": rows_per_group,
        "cols": cols,
        "plain": base64.b64encode(plain).decode(),
        "repacked": base64.b64encode(packed).decode(),
    })

(ROOT / "goldens.json").write_text(json.dumps(goldens, indent=1) + "\n")
