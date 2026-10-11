"""PRD steps 3-4: check the joint A/B reconstruction and measure the alignment tolerance tau.

Usage: python src/check_alignment.py [NAME=joint]  -> data/out/alignment.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from geometry import camera_center, confident_points, up_from_cameras
from run_vggt import frames_of

OUT = Path("data/out")
PHONE_HEIGHT_CM = 140  # ponytail: assumed phone height for a rough cm scale; use a measured object if it matters


def fit_floor(points, up):
    """Plane n·x + d = 0 through the lowest 5% of points along up, n pointing up."""
    h = points @ up
    sel = points[h <= np.quantile(h, 0.05)]
    c = sel.mean(0)
    n = np.linalg.svd(sel - c, full_matrices=False)[2][-1]
    if n @ up < 0:
        n = -n
    return n, float(-n @ c)


def subsample(p, n=300_000):
    return p[np.random.default_rng(0).choice(len(p), min(n, len(p)), replace=False)]


def alignment_tau(pa, pb, gate, n_query=300_000):
    """Robust p90 of A->B NN distance: 2.44 x median (half-normal p90/p50), immune to A points B never saw.

    Only A is sampled; thinning B adds its point spacing. Test scene: raw p90 19 cm, robust 3 cm."""
    dist, _ = cKDTree(pb).query(subsample(pa, n_query), distance_upper_bound=gate)
    hit = np.isfinite(dist)
    return float(2.44 * np.median(dist[hit])), float(hit.mean())


def main(name="joint"):
    d = dict(np.load(OUT / f"{name}.npz"))
    up = up_from_cameras(d["extrinsic"])
    pa = confident_points(d, frames_of(d, 0))[0]
    pb = confident_points(d, frames_of(d, 1))[0]

    na, da = fit_floor(subsample(pa), up)
    nb, db = fit_floor(subsample(pb), up)
    n, fd = fit_floor(np.vstack([subsample(pa), subsample(pb)]), up)
    heights = np.array([n @ camera_center(e) + fd for e in d["extrinsic"]])
    cam_height = float(np.median(heights))

    extent = np.linalg.norm(np.quantile(pb, 0.99, 0) - np.quantile(pb, 0.01, 0))
    tau, overlap = alignment_tau(pa, pb, gate=0.05 * extent)
    cm_per_unit = PHONE_HEIGHT_CM / cam_height
    r = {
        "tau": tau, "up": up.tolist(), "floor_n": n.tolist(), "floor_d": fd,
        "floor_angle_deg": float(np.degrees(np.arccos(np.clip(na @ nb, -1, 1)))),
        "floor_offset": abs(da - db), "cam_height": cam_height,
        "cam_height_min_max": [float(heights.min()), float(heights.max())],
        "cm_per_unit": cm_per_unit, "tau_cm": tau * cm_per_unit, "overlap_fraction": overlap,
    }
    (OUT / "alignment.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
