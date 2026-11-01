#!/usr/bin/env python3
"""Our IQ quantizers against ggml's: the same rows (and imatrix) through `ggml_quantize_chunk` in llama.cpp's
libggml-base and through the `quantize_rows` example, compared byte for byte.

Usage: scripts/iq_quantize_parity.py <llama.cpp build lib dir>
"""
import ctypes, os, subprocess, sys, tempfile
import numpy as np

TYPES = {"iq4_nl": 20, "iq4_xs": 23}
ROWS, COLS = 64, 4096
SEED = 0


def ggml(lib_dir):
    lib = ctypes.CDLL(os.path.join(lib_dir, "libggml-base.so"))
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    return lib


def main():
    lib = ggml(sys.argv[1])
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
        for name, ty in TYPES.items():
            for label, w in (("plain", None), ("imatrix", imatrix)):
                out = np.zeros(lib.ggml_row_size(ty, COLS) * ROWS, dtype=np.uint8)
                lib.ggml_quantize_chunk(ty, x.ctypes.data, out.ctypes.data, 0, ROWS, COLS,
                                        None if w is None else w.ctypes.data)
                ours_path = os.path.join(tmp, f"{name}.bin")
                subprocess.run([example, str(ty), str(COLS), values, "-" if w is None else weights, ours_path], check=True)
                ours = np.fromfile(ours_path, dtype=np.uint8)
                differ = int((ours != out).sum()) if ours.size == out.size else -1
                print(f"{name:7s} {label:8s} bytes {out.size}  differing {differ}")
                failed |= differ != 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
