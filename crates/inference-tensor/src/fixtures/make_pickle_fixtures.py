"""Writes the torch.save fixtures the pickle tests read: a bare tensor, a dict of state dicts and a pickled module."""

from pathlib import Path

import torch

here = Path(__file__).parent
torch.save(torch.arange(6, dtype=torch.float32).reshape(2, 3), here / "bare_tensor.pt")
torch.save(
    {"outer": {"module.w": torch.tensor([1.0, -2.0]), "module.b": torch.tensor([0.5])}},
    here / "nested_state_dicts.pth",
)
# a whole module, as `torch.save({"model": module})` checkpoints store one; f16 like a halved training checkpoint
m = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1, bias=False), torch.nn.BatchNorm2d(2), torch.nn.SiLU()).eval().half()
with torch.no_grad():
    m[0].weight.copy_(torch.tensor([3.0, -1.0]).view(2, 1, 1, 1))
    m[1].running_mean.copy_(torch.tensor([0.5, 0.25]))
torch.save({"epoch": 3, "model": m}, here / "pickled_module.pt")
