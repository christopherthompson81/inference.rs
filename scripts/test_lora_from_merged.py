import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None

import lora_from_merged

RANK = 4
SHAPES = {
    "model.layers.0.q_proj.weight": (48, 32),
    "model.layers.0.down_proj.weight": (32, 64),
}


@unittest.skipIf(torch is None, "needs torch and safetensors")
class LoraFromMergedTest(unittest.TestCase):
    # a base, and a fine-tune that merged a rank-4 LoRA into it in bf16, changed a norm and one embedding row
    def setUp(self):
        torch.manual_seed(0)
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        base = {key: torch.randn(shape) for key, shape in SHAPES.items()}
        base["model.norm.weight"] = torch.ones(32)
        base["model.embed_tokens.weight"] = torch.randn(10, 32)
        tuned = {key: value.clone() for key, value in base.items()}
        for key, (rows, cols) in SHAPES.items():
            tuned[key] = base[key] + 0.1 * torch.randn(rows, RANK) @ torch.randn(
                RANK, cols
            )
        tuned["model.norm.weight"] += 0.01
        tuned["model.embed_tokens.weight"][3] += 1.0
        tuned["score.weight"] = torch.randn(2, 32)
        for name, tensors in (("base", base), ("tuned", tuned)):
            (root / name).mkdir()
            save_file(
                {k: v.to(torch.bfloat16) for k, v in tensors.items()},
                root / name / "model.safetensors",
            )
        self.base, self.tuned, self.out = (
            root / "base",
            root / "tuned",
            root / "adapter",
        )
        self.weights = {k: v.to(torch.bfloat16).float() for k, v in tuned.items()}
        self.base_weights = {k: v.to(torch.bfloat16).float() for k, v in base.items()}

    def tearDown(self):
        self.dir.cleanup()

    def run_tool(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(lora_from_merged.main([str(a) for a in args]), 0)
        return out.getvalue()

    def test_measure_finds_the_merged_rank(self):
        report = self.run_tool("measure", self.base, self.tuned, "--ranks", "2,4,8,16")
        self.assertIn("suggested rank 4", report)
        # also when the merged rank is the smallest one tried
        report = self.run_tool("measure", self.base, self.tuned, "--ranks", "4,8,16")
        self.assertIn("suggested rank 4", report)

    def test_the_extracted_adapter_rebuilds_the_fine_tune(self):
        self.run_tool(
            "extract",
            self.base,
            self.tuned,
            self.out,
            "--rank",
            RANK,
            "--base-id",
            "org/base",
        )
        config = json.loads((self.out / "adapter_config.json").read_text())
        self.assertEqual((config["r"], config["lora_alpha"]), (RANK, RANK))
        self.assertEqual(
            sorted(config["target_modules"]),
            sorted(k.removesuffix(".weight") for k in SHAPES),
        )
        self.assertEqual(config["base_model_name_or_path"], "org/base")
        adapter = load_file(self.out / "adapter_model.safetensors")
        for key in SHAPES:
            module = "base_model.model." + key.removesuffix(".weight")
            a, b = (
                adapter[module + ".lora_A.weight"].float(),
                adapter[module + ".lora_B.weight"].float(),
            )
            self.assertEqual(a.shape[0], RANK)
            rebuilt = self.base_weights[key] + b @ a
            # within the bf16 rounding of the merged weights and of the factors
            self.assertLess(
                (
                    (rebuilt - self.weights[key]).norm() / self.weights[key].norm()
                ).item(),
                1e-2,
            )

    def test_what_the_adapter_cannot_hold_is_reported(self):
        self.run_tool("extract", self.base, self.tuned, self.out, "--rank", RANK)
        report = json.loads((self.out / "extraction_report.json").read_text())
        self.assertIn("model.norm.weight", report["left_out"])
        self.assertEqual(
            report["left_out"]["model.embed_tokens.weight"]["rows_changed"], 1
        )
        self.assertEqual(report["unmatched"], ["score.weight"])

    def test_rename_prefixes_a_published_adapter_to_the_checkpoint_names(self):
        self.run_tool("extract", self.base, self.tuned, self.out, "--rank", RANK)
        inner = Path(self.dir.name) / "inner"
        inner.mkdir()
        adapter = load_file(self.out / "adapter_model.safetensors")
        save_file(
            {
                k.replace("base_model.model.model.", "base_model.model."): v
                for k, v in adapter.items()
            },
            inner / "adapter_model.safetensors",
        )
        (inner / "adapter_config.json").write_text(
            json.dumps({"r": RANK, "target_modules": ["q_proj", "down_proj"]})
        )
        renamed = Path(self.dir.name) / "renamed"
        self.run_tool("rename", inner, self.base, renamed, "--prefix", "model.")
        self.assertEqual(
            sorted(load_file(renamed / "adapter_model.safetensors")), sorted(adapter)
        )
        with self.assertRaises(SystemExit):
            self.run_tool("rename", inner, self.base, renamed, "--prefix", "wrong.")


if __name__ == "__main__":
    unittest.main()
