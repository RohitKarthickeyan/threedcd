"""Describe each verified change in plain English with Claude; write data/out/report.md.

Usage: python src/describe.py -> adds `description` to changes_verified.json, writes report.md, change_N.jpg
"""
import json
from pathlib import Path

import numpy as np

from run_vggt import frame_list
from diff import visible_uv
from verify_vlm import MODEL, PRICE, USAGE, _png, best_views, load_env

OUT = Path("data/out")
FRAMES = ["data/frames/IMG_8289", "data/frames/IMG_8290"]  # the joint run's inputs, A then B
SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "from_place": {"type": "string"},
        "to_place": {"type": "string"},
        "sentence": {"type": "string"},
    },
    "required": ["object", "from_place", "to_place", "sentence"],
    "additionalProperties": False,
}


def full_res_crop(path, uvs, model_hw, min_half, pad=0.2):
    """Crop of the original frame covering points given in model (VGGT input) pixels, plus pad; at least 2*min_half."""
    import cv2

    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    size = np.array(img.shape[1::-1])  # (W, H)
    uv = np.asarray(uvs) * size[0] / model_hw[1]
    lo, hi = uv.min(0), uv.max(0)
    half = np.minimum(np.maximum(min_half, (1 + pad) * (hi - lo) / 2), size // 2).astype(int)
    x0, y0 = np.clip(np.round((lo + hi) / 2 - half).astype(int), 0, size - 2 * half)
    return img[y0:y0 + 2 * half[1], x0:x0 + 2 * half[0]]


def save_pair(path, before, after, height=360, gap=12):
    """Before | after crops side by side at one height, white gap between."""
    import cv2

    tiles = [cv2.resize(c, (round(c.shape[1] * height / c.shape[0]), height)) for c in (before, after)]
    out = np.hstack([tiles[0], np.full((height, gap, 3), 255, np.uint8), tiles[1]])
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def spot_crops(d, e, paths, tau, views=2, min_half=90, spot_k=35):
    """[crops of the old spot in A, crops of the new spot in B], each sized to that spot's object; None if unseen."""
    loc_a, loc_b = np.array(e["a"] if e["a"] else e["b"]), np.array(e["b"] if e["b"] else e["a"])
    va, vb = best_views(d, 0, loc_a, 3 * tau, views), best_views(d, 1, loc_b, 3 * tau, views)
    if not va or not vb:
        return None
    pts = np.array(e["points"])
    crops = []
    for loc, vs in ((loc_a, va), (loc_b, vb)):
        spot = pts[np.linalg.norm(pts - loc, axis=1) < spot_k * tau]  # this spot's object, not the other end
        crops.append([full_res_crop(paths[f], np.vstack([uv[None], visible_uv(d, f, spot, 3 * tau)]),
                                    d["images"].shape[1:3], min_half) for f, uv in vs])
    return crops


def distance_cm(e, cm_per_unit):
    return None if e["type"] != "moved" else float(np.linalg.norm(np.array(e["b"]) - np.array(e["a"])) * cm_per_unit)


def report_line(i, e, cm_per_unit):
    dist = distance_cm(e, cm_per_unit)
    head = f"{e['label'].capitalize()} {e['type']}" + (f" ~{dist / 100:.1f} m" if dist is not None else "")
    return f"{i}. **{head}:** {e['description']['sentence']}"


def prompt(e, dist):
    what = {"moved": f"was moved about {dist / 100:.1f} m (measured)",
            "removed": "was removed", "added": "was added", "changed": "was replaced by something else"}[e["type"]]
    return (f"The first images are from video A (before), the last from video B (after), of the same room. The object at "
            f"the centre of the A images ({e['label']}) {what}; the B images show "
            f"{'where it went' if e['type'] == 'moved' else 'the same spot after'}. "
            "Describe the object briefly (colour, kind, state), say where it was and where it is now in terms of nearby "
            "furniture ('nothing' if not applicable), and write one plain sentence describing the change for a non-expert.")


def describe(client, crops_a, crops_b, text):
    r = client.beta.messages.create(
        model=MODEL, max_tokens=4000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [*map(_png, crops_a), *map(_png, crops_b), {"type": "text", "text": text}]}])
    USAGE[0] += r.usage.input_tokens
    USAGE[1] += r.usage.output_tokens
    return json.loads(next(b.text for b in r.content if b.type == "text")) if r.stop_reason == "end_turn" else None


def main():
    import anthropic

    load_env()
    client = anthropic.Anthropic()
    d = dict(np.load(OUT / "joint.npz"))
    a = json.loads((OUT / "alignment.json").read_text())
    paths, _, src = frame_list(FRAMES)
    assert np.array_equal(src, d["src"]), "frames on disk differ from the joint run"
    path = OUT / "changes_verified.json"
    entries = json.loads(path.read_text())
    lines = ["# Changes between video A (before) and video B (after)", ""]
    for i, e in enumerate(entries, 1):
        crops = spot_crops(d, e, paths, a["tau"])
        if crops is None:
            continue
        save_pair(OUT / f"change_{i}.jpg", crops[0][0], crops[1][0])
        e["description"] = describe(client, *crops, prompt(e, distance_cm(e, a["cm_per_unit"])))
        if e["description"]:
            lines.append(report_line(i, e, a["cm_per_unit"]))
    path.write_text(json.dumps(entries))
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"tokens in {USAGE[0]} out {USAGE[1]}, cost ${(USAGE[0] * PRICE[0] + USAGE[1] * PRICE[1]) / 1e6:.2f}")


if __name__ == "__main__":
    main()
