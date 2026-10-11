"""Score change entries against ground-truth boxes (data/gt.json).

Usage: python src/score.py [CHANGES=data/out/changes.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

from geometry import foreground_points
from run_vggt import frames_of

OUT = Path("data/out")


def gt_boxes(d, gt, tau):
    boxes = {}
    for obj, views in gt.items():
        for tag, v in views.items():
            f = frames_of(d, 0 if tag == "A" else 1)
            i = f[np.argmin(np.abs(d["src"][f] - v["src"]))]
            p = foreground_points(d, i, v["box"], tau, v.get("fg_k", 10))  # fg_k: depth range kept, in tau
            boxes[f"{obj}/{tag}"] = (np.quantile(p, 0.05, 0) - tau, np.quantile(p, 0.95, 0) + tau)
    return boxes


def entry_locations(e):
    return np.array([e[k] for k in ("a", "b") if e[k] is not None])


def score(entries, boxes):
    hits, false = {k: [] for k in boxes}, []
    for name, pts in entries:
        keys = [k for k, (lo, hi) in boxes.items() if ((pts >= lo) & (pts <= hi)).all(-1).any()]
        for k in keys:
            hits[k].append(name)
        if not keys:
            false.append(name)
    return hits, false


def main(path=OUT / "changes.json"):
    d = dict(np.load(OUT / "joint.npz"))
    tau = json.loads((OUT / "alignment.json").read_text())["tau"]
    boxes = gt_boxes(d, json.loads(Path("data/gt.json").read_text()), tau)
    entries = json.loads(Path(path).read_text())
    named = [(f"{k}:{e['type']}:{e.get('label')}:{'+'.join(e['sources'])}", entry_locations(e)) for k, e in enumerate(entries)]
    for src in sorted({s for e in entries for s in e["sources"]}) + ["ALL"]:
        sub = [n for n, e in zip(named, entries) if src == "ALL" or src in e["sources"]]
        hits, false = score(sub, boxes)
        print(f"{src:>10}: boxes hit {sum(bool(v) for v in hits.values())}/{len(boxes)} "
              f"{sorted(k for k, v in hits.items() if v)}, false alarms {len(false)}/{len(sub)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
