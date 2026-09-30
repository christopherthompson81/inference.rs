import unittest
from pathlib import Path

from slim_needed import slim_crates, slim_needed

CRATES = [
    (Path("crates/inference-core"), "inference-core"),
    (Path("crates/inference-nn"), "inference-nn"),
    (Path("crates/inference-api"), "inference-api"),
]
SLIM = {"inference-core", "inference-nn"}


class SlimNeededTest(unittest.TestCase):
    def test_a_change_in_core_or_below_needs_the_slim_lint(self):
        self.assertTrue(slim_needed(["crates/inference-nn/src/lib.rs"], CRATES, SLIM))

    def test_changes_above_core_or_outside_crates_skip_it(self):
        paths = ["crates/inference-api/src/engine.rs", "docs/guide.md", "bindings/python/x.py", ""]
        self.assertFalse(slim_needed(paths, CRATES, SLIM))

    def test_workspace_files_and_the_ci_script_always_need_it(self):
        self.assertTrue(slim_needed(["Cargo.lock"], CRATES, SLIM))
        self.assertTrue(slim_needed(["scripts/local_ci.sh"], CRATES, SLIM))

    def test_slim_crates_follow_dependencies_within_the_workspace(self):
        nodes = {"core": ["nn", "serde"], "nn": ["quant"], "quant": [], "api": ["core"]}
        members = {"core", "nn", "quant", "api"}
        names = {"core": "inference-core", "nn": "inference-nn", "quant": "inference-quant", "api": "inference-api"}
        self.assertEqual(
            slim_crates("core", nodes, members, names), {"inference-core", "inference-nn", "inference-quant"}
        )


if __name__ == "__main__":
    unittest.main()
