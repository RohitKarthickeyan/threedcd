"""PRD step 6: diff per-video voxel labels into added / removed / unobserved.

Usage: python src/diff.py [NAME=joint]  -> data/out/diff.npz, diff_views.png, diff_overlay.png
Codes: 0 same, 1 added, 2 removed, 3 unobserved, 4 changed (occupied in both, colour differs).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter

from geometry import confident_points, project
from run_vggt import frames_of
from viz import draw_points, save_views_png
from voxel_labels import FREE, OCC, UNKNOWN, grid_centers

OUT = Path("data/out")
SAME, ADDED, REMOVED, UNOBSERVED, CHANGED = 0, 1, 2, 3, 4
MIN_DE = 10.0  # ponytail: floor on Lab colour distance (dE 10 is clearly visible); tune on more scenes
COLOUR_EDGE = 40  # /255 per channel, over a 3x3 window


def diff_labels(la, lb, dilate=0):
    # dilate=0: occupied surfaces border occluded-unknown voxels, so dilation erases every change
    out = np.full(la.shape, SAME, np.int8)
    out[(la == FREE) & (lb == OCC)] = ADDED
    out[(la == OCC) & (lb == FREE)] = REMOVED
    unknown = (la == UNKNOWN) | (lb == UNKNOWN)
    if dilate:
        unknown = binary_dilation(unknown, iterations=dilate)
    out[unknown] = UNOBSERVED
    return out


def visible_uv(d, i, pts, margin):
    """Pixel coords of pts that frame i actually sees: in the image, not behind its depth by > margin."""
    H, W = d["depth"].shape[1:]
    u, v, z = project(pts, d["intrinsic"][i], d["extrinsic"][i])
    ok = np.flatnonzero((z > 0) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5))
    seen = z[ok] <= d["depth"][i, np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)] + margin
    return np.c_[u, v][ok[seen]]


def lab_images(images):
    """Per-frame CIELAB (L 0-100) as float32 feature maps."""
    import cv2

    return np.stack([cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_RGB2LAB) for im in images])


def surface_features(d, frames, pts, tau, feats):
    """Median feature, spread and view count per point, from on-surface pixels off colour edges."""
    H, W = d["depth"].shape[1:]
    h, w, C = feats.shape[1:]
    s = np.full((len(frames), len(pts), C), np.nan, np.float32)
    for k, i in enumerate(frames):
        u, v, z = project(pts, d["intrinsic"][i], d["extrinsic"][i])
        ok = np.flatnonzero((z > 0) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5))
        ui, vi = np.round(u[ok]).astype(int), np.round(v[ok]).astype(int)
        img = d["images"][i].astype(np.int16)
        edge = (maximum_filter(img, (3, 3, 1)) - minimum_filter(img, (3, 3, 1))).max(-1) > COLOUR_EDGE
        on = (np.abs(z[ok] - d["depth"][i, vi, ui]) < tau) & ~edge[vi, ui]
        s[k, ok[on]] = feats[i, vi[on] * h // H, ui[on] * w // W]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN median for unseen points
        med = np.nanmedian(s, 0)
        spread = np.nanmedian(np.linalg.norm(s - med, axis=-1), 0)
    return med, spread, np.isfinite(s[..., 0]).sum(0)


def feature_changed(d, pts, tau, feats, min_dist, min_views=2, chunk=2000):
    """Points both videos see whose feature differs beyond the scene's spread, after removing the median A->B offset
    (exposure); the spread gate drops parallax edges."""
    fa, fb, spread, idx = [], [], [], []
    for c in range(0, len(pts), chunk):  # chunked: frames x points x channels
        (ma, sa, na), (mb, sb, nb) = (surface_features(d, frames_of(d, v), pts[c:c + chunk], tau, feats) for v in (0, 1))
        seen = (na >= min_views) & (nb >= min_views)
        fa.append(ma[seen])
        fb.append(mb[seen])
        spread.append(sa[seen] + sb[seen])
        idx.append(c + np.flatnonzero(seen))
    fa, fb, spread, idx = map(np.concatenate, (fa, fb, spread, idx))
    flagged = np.zeros(len(pts), bool)
    if len(idx):
        dist = np.linalg.norm(fa + np.median(fb - fa, 0) - fb, axis=1)
        mad = np.median(np.abs(dist - np.median(dist)))
        flagged[idx] = (dist > max(np.median(dist) + 3 * 1.4826 * mad, min_dist)) & (dist > 2 * spread)
    return flagged


def overlay_png(path, d, layers, margin):
    """One row per (video, points, colour) layer: the 4 frames of that video that see the most of the points."""
    import cv2

    rows = []
    for vid, pts, color in layers:
        frames = frames_of(d, vid)
        uvs = [visible_uv(d, i, pts, margin) for i in frames]
        best = np.argsort([len(uv) for uv in uvs])[::-1][:4]
        rows.append(np.hstack([draw_points(d["images"][frames[j]], uvs[j], color) for j in best]))
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def main(name="joint"):
    d = dict(np.load(OUT / f"{name}.npz"))
    g = np.load(OUT / "labels.npz")
    a = json.loads((OUT / "alignment.json").read_text())
    up = np.array(a["up"])
    out = diff_labels(g["label_a"], g["label_b"])
    centers = grid_centers(g["lo"], g["shape"], float(g["voxel"]))
    both = np.flatnonzero((g["label_a"].ravel() == OCC) & (g["label_b"].ravel() == OCC))
    np.put(out, both[feature_changed(d, centers[both], a["tau"], lab_images(d["images"]), MIN_DE)], CHANGED)
    flat = out.ravel()
    np.savez_compressed(OUT / "diff.npz", diff=out)
    counts = {k: int((out == v).sum()) for k, v in
              (("added", ADDED), ("removed", REMOVED), ("changed", CHANGED), ("unobserved", UNOBSERVED))}
    print(counts)

    added, removed, changed = centers[flat == ADDED], centers[flat == REMOVED], centers[flat == CHANGED]
    bg, _ = confident_points(d, np.arange(len(d["video"])))
    pts = np.vstack([bg, added, removed, changed])
    cols = np.vstack([np.full((len(bg), 3), 170), np.tile([0, 200, 0], (len(added), 1)),
                      np.tile([230, 0, 0], (len(removed), 1)), np.tile([255, 200, 0], (len(changed), 1))]).astype(np.uint8)
    save_views_png(OUT / "diff_views.png", pts, cols, up)
    layers = [(0, removed, (255, 0, 0)), (1, added, (0, 255, 0)), (1, changed, (255, 200, 0))]
    overlay_png(OUT / "diff_overlay.png", d, layers, margin=3 * a["tau"])


if __name__ == "__main__":
    main(*sys.argv[1:])
