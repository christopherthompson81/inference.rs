"""The package's version is the workspace's, and the built library exports every declared entry point."""

import ctypes
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_rs import _native


class Coverage(unittest.TestCase):
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
