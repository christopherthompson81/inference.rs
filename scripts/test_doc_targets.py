import unittest
from pathlib import Path

from doc_targets import doc_args

CRATES = [
    (Path("crates/inference-core"), "inference-core"),
    (Path("crates/inference-protocol"), "inference-protocol"),
    (Path("examples/rust"), "inference-examples"),
]


class DocArgsTest(unittest.TestCase):
    def test_changed_files_document_their_crates(self):
        paths = ["crates/inference-core/src/lib.rs", "examples/rust/models/multimodal/main.rs", ""]
        self.assertEqual(doc_args(paths, CRATES, {}), "-p inference-core -p inference-examples")

    def test_files_outside_any_crate_document_nothing(self):
        self.assertEqual(doc_args(["docs/guide.md", "scripts/local_ci.sh"], CRATES, {}), "")

    def test_a_workspace_manifest_change_documents_everything(self):
        self.assertEqual(doc_args(["crates/inference-core/src/lib.rs", "Cargo.lock"], CRATES, {}), "--workspace")

    def test_changed_crates_get_the_features_the_workspace_enables_on_them(self):
        features = {"inference-protocol": ["openai", "utoipa"], "inference-core": []}
        paths = ["crates/inference-protocol/src/openai.rs"]
        self.assertEqual(
            doc_args(paths, CRATES, features),
            "-p inference-protocol --features inference-protocol/openai,inference-protocol/utoipa",
        )


if __name__ == "__main__":
    unittest.main()
