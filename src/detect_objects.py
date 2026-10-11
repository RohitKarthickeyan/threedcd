"""OWLv2 open-vocabulary detection per frame, lifted to 3D, compared between videos.

Usage: python src/detect_objects.py [NAME=joint] -> data/out/objects.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from diff import lab_images, visible_uv
from geometry import foreground_points
from run_vggt import frames_of

OUT = Path("data/out")
QUERIES = ["laptop", "pillow", "fan", "chair", "lamp", "bag", "box", "cup", "book", "plant", "bottle",
           "remote control", "shoe", "blanket", "toy"]
SCORE = 0.25  # ponytail: OWLv2 score cut, set by eye on the test scene
MAX_DE = 25.0  # Lab colour distance above which two same-label detections are different objects


def detect(images, queries=QUERIES, threshold=SCORE):
    """Per frame: [(label, score, [x0, y0, x1, y1])] in pixels."""
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda").eval()
    out = []
    for img in images:
        inputs = proc(text=[queries], images=img, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o = model(**inputs)
        s = max(img.shape[:2])  # OWLv2 pads to a square
        r = proc.post_process_grounded_object_detection(o, threshold=threshold, target_sizes=[(s, s)],
                                                        text_labels=[queries])[0]
        out.append([(lab, float(sc), b.tolist()) for lab, sc, b in zip(r["text_labels"], r["scores"], r["boxes"])])
    del model
    torch.cuda.empty_cache()
    return out


def merge(points, labels, frames, radius, min_frames=2, colours=None, max_de=MAX_DE):
    colours = np.zeros((len(labels), 3)) if colours is None else np.asarray(colours, float)
    objs = []
    for lab in sorted(set(labels)):
        groups = []
        for k in (k for k, x in enumerate(labels) if x == lab):
            g = next((g for g in groups if np.linalg.norm(g["sum"] / g["n"] - points[k]) < radius
                      and np.linalg.norm(g["lab"] / g["n"] - colours[k]) < max_de), None)
            if g is None:
                groups.append({"sum": np.array(points[k], float), "lab": colours[k].copy(), "n": 1, "frames": {frames[k]}})
            else:
                g["sum"] += points[k]
                g["lab"] += colours[k]
                g["n"] += 1
                g["frames"].add(frames[k])
        objs += [{"label": lab, "pos": g["sum"] / g["n"], "lab": g["lab"] / g["n"], "views": len(g["frames"])}
                 for g in groups if len(g["frames"]) >= min_frames]
    return objs


def compare(objs_a, objs_b, static_r, max_de=MAX_DE):
    changes = []
    for lab in sorted({o["label"] for o in objs_a + objs_b}):
        a = [o for o in objs_a if o["label"] == lab]
        b = [o for o in objs_b if o["label"] == lab]
        ua, ub = set(), set()
        pairs = [(np.linalg.norm(x["pos"] - y["pos"]), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                 if "lab" not in x or "lab" not in y or np.linalg.norm(x["lab"] - y["lab"]) < max_de]
        for dist, i, j in sorted(pairs):
            if i in ua or j in ub:
                continue
            ua.add(i)
            ub.add(j)
            if dist > static_r:
                changes.append({"type": "moved", "label": lab, "a": a[i]["pos"], "b": b[j]["pos"]})
        changes += [{"type": "removed", "label": lab, "a": a[i]["pos"], "b": None} for i in range(len(a)) if i not in ua]
        changes += [{"type": "added", "label": lab, "a": None, "b": b[j]["pos"]} for j in range(len(b)) if j not in ub]
    return changes


def seen_by(d, vid, p, margin):
    return sum(len(visible_uv(d, i, p[None], margin)) for i in frames_of(d, vid))


def main(name="joint", merge_k=10, static_k=15):
    d = dict(np.load(OUT / f"{name}.npz"))
    tau = json.loads((OUT / "alignment.json").read_text())["tau"]
    dets = detect(d["images"])
    lab_img = lab_images(d["images"])
    objs = []
    for vid in (0, 1):
        pts, labels, frames, colours = [], [], [], []
        for i in frames_of(d, vid):
            for lab, _, (x0, y0, x1, y1) in dets[i]:
                box = [max(int(x0), 0), int(x1), max(int(y0), 0), int(y1)]
                p = foreground_points(d, i, box, tau)
                if len(p):
                    pts.append(np.median(p, 0))
                    labels.append(lab)
                    frames.append(int(i))
                    x0, x1, y0, y1 = box
                    cx, cy, hw, hh = (x0 + x1) // 2, (y0 + y1) // 2, (x1 - x0) // 4 + 1, (y1 - y0) // 4 + 1
                    colours.append(np.median(lab_img[i, max(cy - hh, 0):cy + hh, max(cx - hw, 0):cx + hw].reshape(-1, 3), 0))  # box centre
        objs.append(merge(np.array(pts).reshape(-1, 3), labels, frames, merge_k * tau, colours=colours))
    changes = compare(objs[0], objs[1], static_k * tau)
    changes = [c for c in changes if c["type"] == "moved"
               or seen_by(d, 1 if c["type"] == "removed" else 0, c["a"] if c["a"] is not None else c["b"], 3 * tau) >= 2]
    lists = lambda o: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in o.items()}
    (OUT / "objects.json").write_text(json.dumps(
        {"a": [lists(o) for o in objs[0]], "b": [lists(o) for o in objs[1]], "changes": [lists(c) for c in changes]}, indent=1))
    for c in changes:
        print(c["type"], c["label"])


if __name__ == "__main__":
    main(*sys.argv[1:])
