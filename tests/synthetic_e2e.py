"""GPU-free end-to-end check: ray-cast a room with one 50 cm box moved between video A and B,
write joint.npz, run check_alignment -> voxel_labels -> diff, print where the changes landed.

Usage: python tests/synthetic_e2e.py OUT_DIR   (~3 min; not collected by pytest)
Expected: removed median x ~ -1.3, added ~ +1.3, z ~ +1.8 (the box faces facing the cameras);
changed near 0 (surfaces carry a fixed world-space checker texture)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
import check_alignment, diff, voxel_labels  # noqa: E402
from geometry import backproject  # noqa: E402

OUT = Path(sys.argv[1])
H, W, F = 320, 240, 200.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
ROOM = (np.array([-3, -1.0, -3]), np.array([3, 1.4, 3]))  # y down: floor at y = 1.4


def box_at(x, z, s=0.5):
    return np.array([x - s / 2, 1.4 - s, z - s / 2]), np.array([x + s / 2, 1.4, z + s / 2])


def yaw(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])


def render(R, C, obj):
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - W / 2 + 0.5) / F, (v - H / 2 + 0.5) / F, np.ones_like(u, float)], -1) @ R
    rays[np.abs(rays) < 1e-9] = 1e-9  # no exact zeros: avoids -inf slab distances
    with np.errstate(divide="ignore", invalid="ignore"):
        t_room = np.min(np.where(rays > 0, (ROOM[1] - C) / rays, (ROOM[0] - C) / rays), -1)
        t1, t2 = (obj[0] - C) / rays, (obj[1] - C) / rays
        tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
    hit = (tn < tf) & (tn > 0)
    return np.where(hit & (tn < t_room), tn, t_room)


def texture(depth, extr):
    """20 cm world-space checker: static surfaces look the same in every frame."""
    p = backproject(depth, K, extr)
    checker = (np.floor(p * 5).astype(int).sum(-1) % 2)[..., None]
    return np.where(checker, [200, 180, 140], [90, 110, 160]).astype(np.uint8)


extr, depth, video = [], [], []
for vid, obj in ((0, box_at(-1.5, 2.0)), (1, box_at(1.5, 2.0))):  # box moves from x=-1.5 to x=+1.5
    for i, a in enumerate(np.linspace(-60, 60, 20)):
        C = np.array([0.3 * np.sin(i / 3), 0.0, -0.5 + 0.03 * i])
        R = yaw(a + 5 * vid)
        extr.append(np.hstack([R, (-R @ C)[:, None]]))
        depth.append(render(R, C, obj))
        video.append(vid)
n = len(depth)
rng = np.random.default_rng(0)
d = dict(extrinsic=np.array(extr), intrinsic=np.repeat(K[None], n, 0), depth=np.array(depth),
         conf=1 + rng.random((n, H, W)), images=np.stack([texture(dp, e) for dp, e in zip(depth, extr)]),
         video=np.array(video), src=np.arange(n))
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / "joint.npz", **d)
for m in (check_alignment, voxel_labels, diff):
    m.OUT = OUT
check_alignment.main()
voxel_labels.main()
diff.main()

g, out = np.load(OUT / "labels.npz"), np.load(OUT / "diff.npz")["diff"].ravel()
c = voxel_labels.grid_centers(g["lo"], g["shape"], float(g["voxel"]))
for name, code, want in (("removed", diff.REMOVED, -1.5), ("added", diff.ADDED, 1.5)):
    p = c[out == code]
    print(f"{name}: {len(p)} voxels, median x {np.median(p[:, 0]):+.2f} (want {want:+.1f}), "
          f"median z {np.median(p[:, 2]):+.2f} (want +2.0)")
print(f"changed: {(out == diff.CHANGED).sum()} voxels (want few: checker-edge residue only)")
