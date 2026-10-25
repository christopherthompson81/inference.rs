# Writes a rasterization fixture from gsplat (with the MaskGaussian gate, e.g. the HY-World 2.0 fork) for
# inference-nn's matches_gsplat_fixture test: one camera inside a trained scene, the forward outputs, and the gradients
# of a fixed random weighting of the render. Usage: python gsplat_parity.py <checkpoint.pt> <out-dir>
import os
import sys

import numpy as np
import torch
from gsplat.rendering import rasterization

WIDTH, HEIGHT, FOCAL = 832, 480, 300.0
NEAR, FAR = 0.01, 1e10
SEED = 0

ckpt, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
dev = torch.device("cuda")
sp = torch.load(ckpt, map_location=dev, weights_only=False)["splats"]

# a camera at the median centre looking along the widest spread, its up the narrowest
means_np = sp["means"].float().cpu().numpy()
centre = np.median(means_np, axis=0)
_, _, axes = np.linalg.svd(means_np[:: max(1, len(means_np) // 200_000)] - centre, full_matrices=False)
z = axes[0] / np.linalg.norm(axes[0])
y = -axes[2] / np.linalg.norm(axes[2])
x = np.cross(y, z)
w2c = np.eye(4, dtype=np.float32)
w2c[:3, :3] = np.stack([x, y, z])
w2c[:3, 3] = -w2c[:3, :3] @ centre
K = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]], dtype=np.float32)

leaf = lambda t: t.detach().clone().float().requires_grad_(True)
means, quats, log_scales, logits, sh0 = (leaf(sp[k]) for k in ("means", "quats", "scales", "opacities", "sh0"))
n = means.shape[0]
if "mask_score" in sp:
    gate = leaf((sp["mask_score"][:, 0] > sp["mask_score"][:, 1]).float())
else:
    gate = leaf((torch.rand(n, generator=torch.Generator(device=dev).manual_seed(SEED), device=dev) > 0.1).float())
viewmat = torch.from_numpy(w2c).to(dev)
Kt = torch.from_numpy(K).to(dev)
render, alpha, info = rasterization(
    means=means, quats=quats, scales=torch.exp(log_scales), opacities=torch.sigmoid(logits), colors=sh0.reshape(n, 1, 3),
    viewmats=viewmat[None], Ks=Kt[None], width=WIDTH, height=HEIGHT, sh_degree=0, packed=False,
    rasterize_mode="antialiased", render_mode="RGB+ED", gauss_masks=gate, near_plane=NEAR, far_plane=FAR)
info["means2d"].retain_grad()
g = torch.Generator(device=dev).manual_seed(SEED)
wr = torch.rand(render.shape, generator=g, device=dev)
wa = torch.rand(alpha.shape, generator=g, device=dev)
((render * wr).sum() + (alpha * wa).sum()).backward()

save = lambda name, t: np.save(f"{out}/{name}.npy", np.ascontiguousarray(np.asarray(t, dtype=np.float32)))
cpu = lambda t: t.detach().float().cpu().numpy()
for name, t in [("means", means), ("quats", quats), ("log_scales", log_scales), ("logit_opacities", logits),
                ("sh0", sh0), ("gate", gate), ("render", render[0]), ("alpha", alpha[0]), ("weight_render", wr[0]),
                ("weight_alpha", wa[0]), ("radii", info["radii"].reshape(n, 2)), ("means2d", info["means2d"].reshape(n, 2)),
                ("v_means2d", info["means2d"].grad.reshape(n, 2))]:
    save(name, cpu(t))
for name, t in [("means", means), ("quats", quats), ("log_scales", log_scales), ("logit_opacities", logits),
                ("sh0", sh0), ("gate", gate)]:
    save(f"grad_{name}", cpu(t.grad))
save("viewmat", w2c)
save("K", K)
save("size", [WIDTH, HEIGHT, NEAR, FAR])
visible = int((info["radii"].reshape(n, 2) > 0).all(-1).sum())
print(f"{n} gaussians, {visible} visible")
sys.exit(0 if visible > 0 else 1)
