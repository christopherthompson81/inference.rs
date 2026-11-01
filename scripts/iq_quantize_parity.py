#!/usr/bin/env python3
"""Our quantizers against ggml's: the same rows (and imatrix) through `ggml_quantize_chunk` in a llama.cpp
(libggml-base) or ik_llama.cpp (libggml, for the IQ*_K types) build and through the `quantize_rows` example, compared
byte for byte.

Usage: scripts/iq_quantize_parity.py <build lib dir> [type ...]   (default: iq4_nl iq4_xs; iq4_ks needs ik_llama.cpp)
  e.g. q4_k q6_k for the k-quants
"""
import ctypes, os, subprocess, sys, tempfile
import numpy as np

TYPES = {"q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7, "q8_0": 8, "q2_k": 10, "q3_k": 11, "q4_k": 12, "q5_k": 13, "q6_k": 14, "iq4_nl": 20, "iq4_xs": 23, "iq4_ks": 144}
DEFAULT_TYPES = ["iq4_nl", "iq4_xs"]
ROWS, COLS = 64, 4096
SEED = 0


def ggml(lib_dir):
    names = [n for n in ("libggml-base.so", "libggml.so") if os.path.exists(os.path.join(lib_dir, n))]
    if not names:
        sys.exit(f"no libggml-base.so or libggml.so in {lib_dir}")
    lib = ctypes.CDLL(os.path.join(lib_dir, names[0]))
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    return lib


def main():
    lib = ggml(sys.argv[1])
    types = sys.argv[2:] or DEFAULT_TYPES
    rng = np.random.default_rng(SEED)
    # weights with outliers and a skewed imatrix, as real layers have
    x = (rng.standard_normal((ROWS, COLS)) * rng.choice([0.02, 0.2], size=(ROWS, COLS), p=[0.95, 0.05])).astype(np.float32)
    imatrix = (rng.gamma(0.5, 2.0, COLS)).astype(np.float32)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(["cargo", "build", "-q", "-p", "inference-quant", "--example", "quantize_rows"], cwd=root, check=True)
    example = os.path.join(root, "target", "debug", "examples", "quantize_rows")
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        values, weights = os.path.join(tmp, "x.f32"), os.path.join(tmp, "w.f32")
        x.tofile(values)
        imatrix.tofile(weights)
        for name in types:
            ty = TYPES[name]
            for label, w in (("plain", None), ("imatrix", imatrix)):
                out = np.zeros(lib.ggml_row_size(ty, COLS) * ROWS, dtype=np.uint8)
                lib.ggml_quantize_chunk(ty, x.ctypes.data, out.ctypes.data, 0, ROWS, COLS,
                                        None if w is None else w.ctypes.data, None)
                ours_path = os.path.join(tmp, f"{name}.bin")
                subprocess.run([example, str(ty), str(COLS), values, "-" if w is None else weights, ours_path], check=True)
                ours = np.fromfile(ours_path, dtype=np.uint8)
                differ = int((ours != out).sum()) if ours.size == out.size else -1
                print(f"{name:7s} {label:8s} bytes {out.size}  differing {differ}")
                failed |= differ != 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
