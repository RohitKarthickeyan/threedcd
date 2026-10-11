"""PRD step 5: label voxels occupied / free / unknown, separately for each video.

Usage: python src/voxel_labels.py [NAME=joint]  -> data/out/labels.npz, labels_slice_{a,b}.png
"""
import json
import sys
from pathlib import Path

import numpy as np

from geometry import confident_points, project, valid_mask
from run_vggt import frames_of

OUT = Path("data/out")
UNKNOWN, FREE, OCC = 0, 1, 2


def make_grid(points, voxel):
    lo = np.quantile(points, 0.01, 0) - 2 * voxel
    hi = np.quantile(points, 0.99, 0) + 2 * voxel
    return lo, np.ceil((hi - lo) / voxel).astype(int)


def pick_voxel(points, tau, max_voxels=30e6):
    """Smallest multiple of tau whose grid fits max_voxels (30M ~ 3 GB of temporaries)."""
    k = 1
    while make_grid(points, tau * k)[1].prod() > max_voxels:
        k += 1
    return tau * k


def grid_centers(lo, shape, voxel):
    idx = np.indices(shape).reshape(3, -1).T
    return (lo + (idx + 0.5) * voxel).astype(np.float32)


def label_voxels(centers, depth, ok, K, extr, tau, min_views=2, ratio=0.7, free_k=3.0):
    """Per frame: |z - d| < tau = occupied vote, z < d - free_k*tau = free vote, else none.

    The tau..free_k*tau gap stops grazing views flipping surface voxels (synthetic: 50% -> 99.7% on target)."""
    n_occ = np.zeros(len(centers), np.int16)
    n_free = np.zeros(len(centers), np.int16)
    H, W = depth.shape[1:]
    for i in range(len(depth)):  # ponytail: numpy per frame; move to torch on GPU if this is too slow
        u, v, z = project(centers, K[i], extr[i])
        ui, vi = np.round(u), np.round(v)
        idx = np.flatnonzero((z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H))
        ui, vi = ui[idx].astype(int), vi[idx].astype(int)
        good = ok[i, vi, ui]
        idx, d, zz = idx[good], depth[i, vi[good], ui[good]], z[idx][good]
        n_occ[idx] += np.abs(zz - d) < tau
        n_free[idx] += zz < d - free_k * tau
    total = n_occ + n_free
    lab = np.full(len(centers), UNKNOWN, np.int8)
    lab[(n_free >= min_views) & (n_free >= ratio * total)] = FREE
    lab[(n_occ >= min_views) & (n_occ >= ratio * total)] = OCC
    return lab, n_occ, n_free


def label_video(d, frames, centers, tau):
    ok = valid_mask(d["conf"][frames], d["depth"][frames])
    return label_voxels(centers, d["depth"][frames], ok, d["intrinsic"][frames], d["extrinsic"][frames], tau)[0]


def save_slice_png(path, lab_a, lab_b, centers, up, height, voxel):
    from viz import save_views_png  # reuse the top-down projection

    band = np.abs(centers @ up - height) < voxel / 2
    colors = np.array([[200, 200, 200], [120, 200, 255], [255, 80, 0]], np.uint8)  # unknown, free, occ
    for tag, lab in (("a", lab_a), ("b", lab_b)):
        save_views_png(str(path).replace(".png", f"_{tag}.png"), centers[band], colors[lab[band]], up)


def main(name="joint"):
    d = dict(np.load(OUT / f"{name}.npz"))
    a = json.loads((OUT / "alignment.json").read_text())
    tau, up = a["tau"], np.array(a["up"])
    pts = np.vstack([confident_points(d, frames_of(d, v))[0] for v in (0, 1)])
    voxel = pick_voxel(pts, tau)  # all thresholds are multiples of tau
    lo, shape = make_grid(pts, voxel)
    centers = grid_centers(lo, shape, voxel)
    print(f"grid {shape.tolist()} = {shape.prod() / 1e6:.1f}M voxels of {voxel:.4f}")

    lab_a = label_video(d, frames_of(d, 0), centers, tau)
    lab_b = label_video(d, frames_of(d, 1), centers, tau)
    np.savez_compressed(OUT / "labels.npz", lo=lo, shape=shape, voxel=voxel,
                        label_a=lab_a.reshape(shape), label_b=lab_b.reshape(shape))
    for tag, lab in (("A", lab_a), ("B", lab_b)):
        print(tag, {k: int((lab == v).sum()) for k, v in (("unknown", UNKNOWN), ("free", FREE), ("occ", OCC))})

    floor = -a["floor_d"]  # height of the floor plane along floor_n
    save_slice_png(OUT / "labels_slice.png", lab_a, lab_b, centers, np.array(a["floor_n"]),
                   floor + 0.5 * a["cam_height"], voxel)


if __name__ == "__main__":
    main(*sys.argv[1:])
