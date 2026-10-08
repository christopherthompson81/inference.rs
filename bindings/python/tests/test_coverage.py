"""The ctypes declarations are the generator's rendering of inference.h, and the built library exports them all."""

import ctypes
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_rs import _native

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import generate_native

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


class Coverage(unittest.TestCase):
    def setUp(self):
        path = find_header()
        if path is None:
            self.skipTest(f"inference.h not found; set {HEADER_VARIABLE}")
        self.header = path.read_text()

    def test_the_declarations_are_generated_from_the_header(self):
        self.assertEqual(generate_native.stale(), [], "run bindings/scripts/generate_native.py")

    def test_the_package_version_is_the_workspaces(self):
        root = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").is_file())
        cargo = (root / "Cargo.toml").read_text()
        workspace = re.search(r'\[workspace\.package\][^\[]*?version = "([^"]+)"', cargo, re.DOTALL)
        pyproject = (root / "bindings/python/pyproject.toml").read_text()
        project = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
        self.assertEqual(project.group(1), workspace.group(1))

    def test_the_built_library_exports_every_entry_point(self):
        path = next((p for p in _native.search_paths() if p.is_file()), None)
        if path is None:
            self.skipTest("libinference_ffi is not built")
        library = ctypes.CDLL(str(path))
        self.assertEqual([name for name in _native.SIGNATURES if not hasattr(library, name)], [])


if __name__ == "__main__":
    unittest.main()
