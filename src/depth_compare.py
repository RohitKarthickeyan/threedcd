"""Per-pixel depth comparison between videos, with regional offsets removed.

Usage: python src/depth_compare.py [NAME=joint] -> data/out/depth_changes.npz
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter

from geometry import backproject, project, valid_mask
from run_vggt import frames_of

OUT = Path("data/out")


def warp_depth(depth, ok, K_src, E_src, K_dst, E_dst):
    H, W = depth.shape
    p = backproject(depth, K_src, E_src)[ok]
    u, v, z = project(p, K_dst, E_dst)
    keep = (z > 0) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    out = np.full(H * W, np.inf)
    np.minimum.at(out, np.round(v[keep]).astype(int) * W + np.round(u[keep]).astype(int), z[keep])
    return out.reshape(H, W)


def close_holes(w):
    """Nearest surface in each 3x3 window: a magnifying warp leaves holes the background shows through."""
    return minimum_filter(w, 3)


def local_residual(r, ok, block=128):
    out = np.full(r.shape, np.nan)
    for y in range(0, r.shape[0], block):
        for x in range(0, r.shape[1], block):
            sl = (slice(y, y + block), slice(x, x + block))
            m = ok[sl]
            if m.sum() >= m.size // 4:
                out[sl] = r[sl] - np.median(r[sl][m])
    return out


def overlap_pairs(d, i, frames, k=2, step=8):
    """The k frames that see the most of frame i's surface (sampled every step px)."""
    H, W = d["depth"].shape[1:]
    p = backproject(d["depth"][i], d["intrinsic"][i], d["extrinsic"][i])[::step, ::step].reshape(-1, 3)
    hits = []
    for j in frames:
        u, v, z = project(p, d["intrinsic"][j], d["extrinsic"][j])
        hits.append(((z > 0) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)).sum())
    return frames[np.argsort(hits)[::-1][:k]]


def residuals(d, dst_vid, pairs=2):
    """Per dst frame and its best-overlap other-video frames: local residual of (warped other depth - dst depth), valid mask."""
    ok_all = valid_mask(d["conf"], d["depth"])
    src_frames = frames_of(d, 1 - dst_vid)
    out = []
    for i in frames_of(d, dst_vid):
        for j in overlap_pairs(d, i, src_frames, pairs):
            w = close_holes(warp_depth(d["depth"][j], ok_all[j], d["intrinsic"][j], d["extrinsic"][j],
                                       d["intrinsic"][i], d["extrinsic"][i]))
            ok = ok_all[i] & np.isfinite(w)
            out.append((i, local_residual(np.where(ok, w - d["depth"][i], 0.0), ok), ok))
    return out


def flagged_voxels(d, res, lo, voxel, k=4.0, min_pairs=2):
    """Voxel centers of dst points the other video sees > k*noise behind, from >= min_pairs frame pairs."""
    vals = np.concatenate([lr[ok & np.isfinite(lr)] for _, lr, ok in res])
    noise = 1.4826 * np.median(np.abs(vals))
    counts = {}
    for i, lr, ok in res:
        m = ok & np.isfinite(lr) & (lr > k * noise)
        p = backproject(d["depth"][i], d["intrinsic"][i], d["extrinsic"][i])[m]
        for key in map(tuple, np.unique(np.floor((p - lo) / voxel).astype(int), axis=0)):
            counts[key] = counts.get(key, 0) + 1
    keys = np.array([key for key, c in counts.items() if c >= min_pairs]).reshape(-1, 3)
    return lo + (keys + 0.5) * voxel, noise


def main(name="joint"):
    d = dict(np.load(OUT / f"{name}.npz"))
    g = np.load(OUT / "labels.npz")
    removed, na = flagged_voxels(d, residuals(d, 0), g["lo"], float(g["voxel"]))
    added, nb = flagged_voxels(d, residuals(d, 1), g["lo"], float(g["voxel"]))
    np.savez_compressed(OUT / "depth_changes.npz", removed=removed, added=added)
    print(f"noise A {na:.4f} B {nb:.4f}; removed {len(removed)} voxels, added {len(added)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
