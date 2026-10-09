"""Writes the torch.save fixtures the pickle tests read: a bare tensor and a dict of state dicts."""

from pathlib import Path

import torch

here = Path(__file__).parent
torch.save(torch.arange(6, dtype=torch.float32).reshape(2, 3), here / "bare_tensor.pt")
torch.save(
    {"outer": {"module.w": torch.tensor([1.0, -2.0]), "module.b": torch.tensor([0.5])}},
    here / "nested_state_dicts.pth",
)
