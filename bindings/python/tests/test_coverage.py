"""Every entry point inference.h declares is declared here with its parameter count, and nothing else is."""

import ctypes
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_rs import _native

HEADER_VARIABLE = "INFERENCE_HEADER"
HEADER_PATH = "crates/inference-ffi/include/inference.h"


def find_header():
    configured = os.environ.get(HEADER_VARIABLE)
    if configured:
        return Path(configured)
    for parent in Path(__file__).resolve().parents:
        if (parent / HEADER_PATH).is_file():
            return parent / HEADER_PATH
    return None


# The ctypes each C type may bind as; inputs with a length may be c_char_p, borrowed returns must be c_void_p.
STRUCTS = {
    "inference_backend_config": _native.BackendConfig,
    "inference_image": _native.Image,
    "inference_media": _native.Media,
    "inference_skill_file": _native.SkillFile,
    "inference_host_callbacks": _native.HostCallbacks,
}
SCALARS = {
    "uint32_t": ctypes.c_uint32,
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
    "size_t": ctypes.c_size_t,
    "float": ctypes.c_float,
    "inference_status": ctypes.c_int32,
}


def expected(c_type: str, returned: bool):
    """The ctypes a C parameter or return type may bind as."""
    c_type = re.sub(r"\bconst\b", "", c_type)
    base = c_type.replace("*", " ").split()[0]
    depth = c_type.count("*")
    if depth == 0:
        return {SCALARS[base]} if base in SCALARS else {None}
    if depth == 2:
        return {_native.out}
    if returned:
        return {ctypes.c_void_p}
    if base in STRUCTS:
        return {ctypes.POINTER(STRUCTS[base])}
    if base in SCALARS:
        return {ctypes.POINTER(SCALARS[base]), ctypes.c_void_p}
    if base in ("char", "uint8_t"):
        return {ctypes.c_char_p, ctypes.c_void_p}
    return {ctypes.c_void_p}


def declared_types(header: str):
    """Each entry point's return type and parameter types, without parameter names."""
    code = re.sub(r"/\*.*?\*/", "", header, flags=re.DOTALL)
    matches = re.finditer(
        r"INFERENCE_API\s+([A-Za-z_][A-Za-z0-9_ *]*?)\b(inference_[a-z0-9_]+)\s*\(([^)]*)\)",
        code,
    )
    result = {}
    for match in matches:
        raw = match.group(3).strip()
        params = [] if raw in ("", "void") else [p.strip() for p in raw.split(",")]
        types = [re.sub(r"\b[a-z_][a-z0-9_]*$", "", param).strip() for param in params]
        result[match.group(2)] = (match.group(1).strip(), types)
    return result


def declared(header: str):
    code = re.sub(r"/\*.*?\*/", "", header, flags=re.DOTALL)
    matches = re.finditer(
        r"INFERENCE_API\s+[A-Za-z_][A-Za-z0-9_ *]*?\b(inference_[a-z0-9_]+)\s*\(([^)]*)\)",
        code,
    )
    return {m.group(1): 0 if m.group(2).strip() in ("", "void") else len(m.group(2).split(",")) for m in matches}


class Coverage(unittest.TestCase):
    def setUp(self):
        path = find_header()
        if path is None:
            self.skipTest(f"inference.h not found; set {HEADER_VARIABLE}")
        self.header = path.read_text()

    def test_every_entry_point_is_declared_with_its_parameters(self):
        header = declared(self.header)
        bound = {name: len(argtypes) for name, (_, argtypes) in _native.SIGNATURES.items()}
        self.assertEqual(sorted(header.keys() - bound.keys()), [], "not declared in _native")
        self.assertEqual(sorted(bound.keys() - header.keys()), [], "not in the header")
        self.assertEqual(
            {n: c for n, c in bound.items() if header[n] != c},
            {},
            "parameter counts differ",
        )

    def test_every_parameter_binds_as_its_c_type(self):
        mismatches = []
        for name, (returns, params) in declared_types(self.header).items():
            restype, argtypes = _native.SIGNATURES[name]
            if restype not in expected(returns, returned=True):
                mismatches.append(f"{name} returns {returns}, bound as {restype}")
            for index, (c_type, bound) in enumerate(zip(params, argtypes)):
                if bound not in expected(c_type, returned=False):
                    mismatches.append(f"{name} parameter {index}: {c_type} bound as {bound}")
        self.assertEqual(mismatches, [])

    def test_the_package_version_is_the_workspaces(self):
        root = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").is_file())
        cargo = (root / "Cargo.toml").read_text()
        workspace = re.search(r'\[workspace\.package\][^\[]*?version = "([^"]+)"', cargo, re.DOTALL)
        pyproject = (root / "bindings/python/pyproject.toml").read_text()
        project = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
        self.assertEqual(project.group(1), workspace.group(1))

    def test_the_package_expects_the_headers_abi_version(self):
        part = {
            name: int(re.search(rf"#define INFERENCE_ABI_VERSION_{name} (\d+)", self.header).group(1))
            for name in ("MAJOR", "MINOR", "PATCH")
        }
        self.assertEqual(
            _native.ABI_VERSION,
            (part["MAJOR"] << 16) | (part["MINOR"] << 8) | part["PATCH"],
        )

    def test_the_built_library_exports_every_entry_point(self):
        path = next((p for p in _native.search_paths() if p.is_file()), None)
        if path is None:
            self.skipTest("libinference_ffi is not built")
        library = ctypes.CDLL(str(path))
        self.assertEqual([name for name in _native.SIGNATURES if not hasattr(library, name)], [])


if __name__ == "__main__":
    unittest.main()
