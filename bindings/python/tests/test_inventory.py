"""Model inventory through the bindings, with no engine: inspect, scan, cache deletion and a download stream."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_rs as ir
from inference_rs import _native

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_STRING = 8
POLL_SECONDS = 60


def gguf_string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def write_gguf(path: Path, architecture: str) -> None:
    key = gguf_string("general.architecture") + struct.pack("<I", GGUF_STRING) + gguf_string(architecture)
    path.write_bytes(GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, 0, 1) + key)


@unittest.skipUnless(any(p.is_file() for p in _native.search_paths()), "libinference_ffi is not built")
class Inventory(unittest.TestCase):
    def test_inspect_and_scan_classify_and_errors_are_invalid_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "Tiny-Q8_0.gguf"
            write_gguf(model, "llama")
            write_gguf(Path(folder) / "foreign-Q8_0.gguf", "audiocpp")

            inspected = json.loads(ir.inspect_model(json.dumps({"path": str(model)})))
            self.assertEqual(inspected["kind"], "text")
            self.assertEqual(inspected["quantization"], "Q8_0")

            scan = json.loads(ir.scan_models(json.dumps({"folders": [folder]})))
            kinds = {Path(m["path"]).name: m["kind"] for m in scan["models"]}
            self.assertEqual(kinds, {"Tiny-Q8_0.gguf": "text", "foreign-Q8_0.gguf": None})
            self.assertIsNone(scan["hf_cache"])

            for call, request in [
                (ir.inspect_model, {"path": str(Path(folder) / "missing.gguf")}),
                (ir.delete_cached_repo, {"repo_id": "../escape"}),
                (ir.download_model, {"repo_id": "acme/tiny", "quant": 4}),
            ]:
                with self.assertRaises(ir.InferenceError) as failure:
                    call(json.dumps(request))
                self.assertEqual(failure.exception.status, ir.Status.INVALID_REQUEST)

    def test_a_cached_download_reports_its_files_and_ends_done(self):
        with tempfile.TemporaryDirectory() as hub:
            repo = Path(hub) / "models--acme--tiny"
            snapshot = repo / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("abc123")
            (snapshot / "config.json").write_text("{}")
            request = {"repo_id": "acme/tiny", "files": ["config.json"], "token_source": "none", "hf_cache_path": hub}
            with ir.download_model(json.dumps(request)) as stream:
                events = list(iter(lambda: stream.next(POLL_SECONDS), None))
            listing = json.loads(ir.scan_models(json.dumps({"hf_cache": True, "hf_cache_path": hub})))
            deleted = json.loads(ir.delete_cached_repo(json.dumps({"repo_id": "acme/tiny", "hf_cache_path": hub})))
            self.assertEqual([e.name for e in events], ["files", "progress", "done"])
            self.assertEqual(events[1].data, {"file": "config.json", "done": 2, "total": 2})
            self.assertTrue(events[2].data["files"][0].endswith("config.json"))
            self.assertEqual([r["repo_id"] for r in listing["hf_cache"]["repos"]], ["acme/tiny"])
            self.assertEqual(deleted["repo_id"], "acme/tiny")
            self.assertFalse(repo.exists())


if __name__ == "__main__":
    unittest.main()
