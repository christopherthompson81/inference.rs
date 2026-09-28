"""A wheel from scripts/release/build_wheels.py installs into a fresh environment and serves from its bundled library."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import venv
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_rs import _native

REPO = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = REPO / "scripts" / "release" / "build_wheels.py"
MODEL_VARIABLE = "INFERENCE_TEST_TINY_CHECKPOINT"
PROBE = """
import json, sys
import inference_rs as ir
from inference_rs import _native, types as t
loaded = next(str(p) for p in _native.search_paths() if p.is_file())
result = {"loaded": loaded, "abi": _native.lib.inference_abi_version()}
if len(sys.argv) > 1:
    spec = t.EngineSpec(
        model=t.ModelSelectedMultimodalPlain(model_id=sys.argv[1], dtype=t.ModelDType.F32),
        runtime=t.RuntimeSpec(device="cpu"),
    )
    with ir.Engine(spec) as engine:
        request = t.ChatCompletionRequest(model="default", messages=[t.Message(role="user", content="hi")], max_tokens=2)
        result["object"] = engine.chat(request).object
print(json.dumps(result))
"""


class Wheel(unittest.TestCase):
    def test_a_built_wheel_installs_and_loads_its_own_library(self):
        library = next((path for path in _native.search_paths() if path.is_file()), None)
        if library is None:
            self.skipTest("no built libinference_ffi to package")
        if importlib.util.find_spec("setuptools") is None:
            self.skipTest("building a wheel needs setuptools in this interpreter")
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            built = subprocess.run(
                [sys.executable, str(BUILD_SCRIPT), "--library", str(library), "--out", str(scratch / "wheels")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            (wheel,) = (scratch / "wheels").glob("*.whl")
            self.assertRegex(wheel.name, r"^inference_rs-[^-]+-py3-none-(manylinux_2_\d+|macosx_\d+_\d+|win)_")
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
            self.assertIn(f"inference_rs/{_native.BUNDLED_DIR}/{library.name}", names)
            self.assertIn("inference_rs/types.py", names)

            # Resolved, as the loader resolves the path it reports (macOS's /var is /private/var).
            environment = (scratch / "venv").resolve()
            venv.create(environment, with_pip=True)
            python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            install = [str(python), "-m", "pip", "install", "-q", "--no-deps", "--no-index", str(wheel)]
            installed = subprocess.run(install, capture_output=True, text=True, check=False)
            self.assertEqual(installed.returncode, 0, installed.stderr)

            # Outside the checkout and without INFERENCE_NATIVE_DIR, only the bundled library can load.
            env = {key: value for key, value in os.environ.items() if key != _native.NATIVE_DIR_VARIABLE}
            model = os.environ.get(MODEL_VARIABLE)
            probe = [str(python), "-c", PROBE, *([model] if model else [])]
            ran = subprocess.run(probe, capture_output=True, text=True, cwd=scratch, env=env, check=False)
            self.assertEqual(ran.returncode, 0, ran.stderr)
            result = json.loads(ran.stdout)
            self.assertTrue(Path(result["loaded"]).is_relative_to(environment), result)
            self.assertEqual(result["abi"], _native.ABI_VERSION)
            if model:
                self.assertEqual(result["object"], "chat.completion")


if __name__ == "__main__":
    unittest.main()
