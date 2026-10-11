"""Merge every detector's candidates into one change list.

Usage: python src/fuse.py -> data/out/changes.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from diff import ADDED, CHANGED, REMOVED
from voxel_labels import grid_centers

OUT = Path("data/out")


def clusters(points, voxel, min_size, max_extent=np.inf):
    if not len(points):
        return []
    idx = np.floor(points / voxel + 1e-6).astype(int)
    idx -= idx.min(0)
    grid = np.zeros(idx.max(0) + 1, bool)
    grid[tuple(idx.T)] = True
    lab, n = label(grid, structure=np.ones((3, 3, 3)))
    ids = lab[tuple(idx.T)]
    out = [points[ids == k] for k in range(1, n + 1) if (ids == k).sum() >= min_size]
    return [c for c in out if (c.max(0) - c.min(0)).max() <= max_extent]  # wider = misregistered surface, not an object


def _locs(c):
    return [c[s] for s in ("a", "b") if c[s] is not None]


def _entry(g):
    def mean(key):  # an anchor (labelled) keeps its own location
        vals = [c[key] for c in (g[:1] if g[0]["label"] else g) if c[key] is not None]
        return np.mean(vals, 0) if vals else None

    types = {c["type"] for c in g}
    typ = (g[0]["type"] if g[0]["label"] else "changed" if "changed" in types or {"removed", "added"} <= types
           else types.pop())
    return {"type": typ, "label": next((c["label"] for c in g if c["label"]), None), "a": mean("a"), "b": mean("b"),
            "sources": sorted({s for c in g for s in c["sources"]}), "points": np.vstack([c["points"] for c in g])}


def _near(c, others, radius):
    return any(np.linalg.norm(p - q) < radius for p in _locs(c) for o in others for q in _locs(o))


def fuse(cands, radius):
    """Labelled candidates anchor entries; unlabelled ones attach to the nearest anchor or group among themselves."""
    anchors = [[c] for c in cands if c["label"]]
    loose = []
    for c in (c for c in cands if not c["label"]):
        ends = {"removed": ("a",), "added": ("b",)}.get(c["type"], ("a", "b"))  # a removal can't belong to an arrival
        dist = [min((np.linalg.norm(p - g[0][e]) for p in _locs(c) for e in ends if g[0][e] is not None), default=np.inf)
                for g in anchors]
        if dist and min(dist) < radius:
            anchors[int(np.argmin(dist))].append(c)
        else:
            g = [g for g in loose if g[0]["type"] == c["type"] and _near(c, g, radius)]
            for x in g[1:]:
                g[0] += x
                loose = [l for l in loose if l is not x]
            if g:
                g[0].append(c)
            else:
                loose.append([c])
    entries = [_entry(g) for g in anchors + loose]
    added = [e for e in entries if e["type"] == "added" and e["label"]]
    for r in [e for e in entries if e["type"] == "removed" and e["label"]]:  # same label removed + added = moved
        m = next((a for a in added if a["label"] == r["label"]), None)
        if m is not None:
            added.remove(m)
            entries.remove(m)
            r.update(type="moved", b=m["b"], sources=sorted(set(r["sources"]) | set(m["sources"])),
                     points=np.vstack([r["points"], m["points"]]))
    return entries


def main(min_k=8, fuse_k=20, max_k=50):
    tau = json.loads((OUT / "alignment.json").read_text())["tau"]
    g = np.load(OUT / "labels.npz")
    voxel = float(g["voxel"])
    centers = grid_centers(g["lo"], g["shape"], voxel)
    out = np.load(OUT / "diff.npz")["diff"].ravel()
    dc = np.load(OUT / "depth_changes.npz")
    cands = []
    for src, typ, pts in (("diff", "removed", centers[out == REMOVED]), ("diff", "added", centers[out == ADDED]),
                          ("colour", "changed", centers[out == CHANGED]), ("depth", "removed", dc["removed"]),
                          ("depth", "added", dc["added"])):
        for cl in clusters(pts, voxel, min_k, max_k * tau):
            c = cl.mean(0)
            cands.append({"type": typ, "label": None, "a": None if typ == "added" else c,
                          "b": None if typ == "removed" else c, "sources": [src], "points": cl})
    for c in json.loads((OUT / "objects.json").read_text())["changes"]:
        a = None if c["a"] is None else np.array(c["a"])
        b = None if c["b"] is None else np.array(c["b"])
        cands.append({"type": c["type"], "label": c["label"], "a": a, "b": b, "sources": ["objects"],
                      "points": np.array([p for p in (a, b) if p is not None])})
    entries = fuse(cands, fuse_k * tau)
    js = lambda v: v.tolist() if isinstance(v, np.ndarray) else v
    (OUT / "changes.json").write_text(json.dumps([{k: js(v) for k, v in e.items()} for e in entries]))
    for e in entries:
        print(e["type"], e["label"], e["sources"], len(e["points"]))


if __name__ == "__main__":
    main()
