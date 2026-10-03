#!/usr/bin/env python3
"""Writes the IQ lookup tables from llama.cpp's ggml-common.h as a CUDA header and a Rust module.

Usage: make_iq_tables.py <path to ggml/src/ggml-common.h> <llama.cpp commit>
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CUDA_OUT = ROOT / "kernels/cuda/gguf_iq_tables.cuh"
RUST_OUT = ROOT / "src/gguf/iq_tables.rs"
CUDA_TABLES = ["kmask_iq2xs", "ksigns_iq2xs", "ksigns64", "iq2xxs_grid", "iq2xs_grid", "iq2s_grid", "iq3xxs_grid",
               "iq3s_grid", "iq1s_grid_gpu"]
RUST_TABLES = ["kmask_iq2xs", "ksigns_iq2xs", "iq2xxs_grid", "iq2xs_grid", "iq2s_grid", "iq3xxs_grid", "iq3s_grid",
               "iq1s_grid"]
RUST_TYPES = {"uint8_t": "u8", "uint32_t": "u32", "uint64_t": "u64"}
SIZES = {"NGRID_IQ1S": 2048}

header, commit = sys.argv[1], sys.argv[2]
lines = pathlib.Path(header).read_text().split("\n")
tables = {}
i = 0
while i < len(lines):
    m = re.match(r"GGML_TABLE_BEGIN\((\w+), (\w+), (\w+)\)", lines[i])
    if m:
        end = next(j for j in range(i, len(lines)) if lines[j].startswith("GGML_TABLE_END()"))
        size = SIZES.get(m.group(3)) or int(m.group(3))
        tables[m.group(2)] = (m.group(1), size, "\n".join(lines[i + 1:end]))
        i = end
    i += 1

provenance = f"llama.cpp ggml-common.h at {commit} (MIT, see third_party/README.md); written by scripts/make_iq_tables.py"
cuda = [f"// IQ lookup tables from {provenance}.", "#pragma once", "", "#include <cstdint>", ""]
for name in CUDA_TABLES:
    ty, size, body = tables[name]
    cuda += [f"static const __device__ {ty} {name}[{size}] = {{", body, "};", ""]
CUDA_OUT.write_text("\n".join(cuda))

rust = [f"//! IQ lookup tables from {provenance}.", "", "#![allow(clippy::unreadable_literal)]", ""]
for name in RUST_TABLES:
    ty, size, body = tables[name]
    values = re.findall(r"0x[0-9a-fA-F]+|\d+", re.sub(r"//.*", "", body))
    assert len(values) == size, (name, len(values), size)
    rust.append(f"pub(crate) const {name.upper()}: [{RUST_TYPES[ty]}; {size}] = [")
    rust += ["    " + ", ".join(values[k:k + 8]) + "," for k in range(0, size, 8)]
    rust += ["];", ""]
RUST_OUT.write_text("\n".join(rust))
