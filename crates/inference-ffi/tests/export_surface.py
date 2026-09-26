"""Checks that a built libinference_ffi exports exactly the functions declared in include/inference.h.

Usage: python3 export_surface.py <path/to/libinference_ffi.so> [path/to/inference.h]
Exit codes: 0 match, 1 mismatch.
"""
import pathlib
import re
import subprocess
import sys

lib = sys.argv[1]
header = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(__file__).parent.parent / "include/inference.h")

declared = set(re.findall(r"INFERENCE_API\b[^;(]*?\b(inference_\w+)\s*\(", header.read_text(), re.S))
nm = subprocess.run(["nm", "-D", "--defined-only", lib], capture_output=True, text=True, check=True).stdout
exported = {line.split()[-1] for line in nm.splitlines() if line.strip()}

missing = sorted(declared - exported)
extra = sorted(exported - declared)
for name in missing:
    print(f"declared but not exported: {name}")
for name in extra:
    print(f"exported but not declared: {name}")
print(f"{len(declared)} declared, {len(exported)} exported")
sys.exit(1 if missing or extra else 0)
