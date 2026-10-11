"""Confirm and label fused changes with Claude: a before/after crop pair per change location.

Usage: python src/verify_vlm.py -> data/out/changes_verified.json   (ANTHROPIC_API_KEY from env or .env)
"""
import base64
import json
import os
from pathlib import Path

import numpy as np

from diff import visible_uv
from run_vggt import frames_of

OUT = Path("data/out")
MODEL = "claude-sonnet-5-5"
PRICE = (2.0, 10.0)  # $ per 1M input / output tokens for MODEL
USAGE = [0, 0]
SCHEMA = {
    "type": "object",
    "properties": {
        "object_a": {"type": "string"},
        "object_b": {"type": "string"},
        "changed": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["object_a", "object_b", "changed", "confidence"],
    "additionalProperties": False,
}
PROMPT = ("Image 1 is from video A (before), image 2 from video B (after), both of the same room. Both crops are "
          "centred on the same 3D spot. Name the object at the centre of each crop ('nothing' if bare surface), "
          "and say whether the object at that spot changed between A and B (moved away, appeared, or replaced). "
          "Lighting, blur and viewpoint differences are not changes. confidence is 0 to 1.")


def load_env(path=".env"):
    """KEY=value lines into os.environ; variables already set win."""
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip().strip("\"'"):
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def crop(img, uv, half):
    H, W = img.shape[:2]
    x = int(np.clip(round(uv[0]) - half, 0, W - 2 * half))
    y = int(np.clip(round(uv[1]) - half, 0, H - 2 * half))
    return img[y:y + 2 * half, x:x + 2 * half]


def best_views(d, vid, p, margin, k=1):
    """[(frame, uv)] of the video's frames that see p, nearest the image centre first."""
    H, W = d["depth"].shape[1:]
    seen = []
    for i in frames_of(d, vid):
        uv = visible_uv(d, i, p[None], margin)
        if len(uv):
            seen.append((np.hypot(uv[0, 0] - W / 2, uv[0, 1] - H / 2), int(i), uv[0]))
    return [(i, uv) for _, i, uv in sorted(seen, key=lambda t: t[0])[:k]]


def best_view(d, vid, p, margin):
    v = best_views(d, vid, p, margin)
    return v[0] if v else None


def _png(img):
    import cv2

    _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(buf.tobytes()).decode()}}


def ask(client, crop_a, crop_b):
    r = client.beta.messages.create(
        model=MODEL, max_tokens=4000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [_png(crop_a), _png(crop_b), {"type": "text", "text": PROMPT}]}])
    USAGE[0] += r.usage.input_tokens
    USAGE[1] += r.usage.output_tokens
    if r.stop_reason != "end_turn":  # refusal or cut off: no usable JSON
        return None
    return json.loads(next(b.text for b in r.content if b.type == "text"))


def absorb(entries, radius, same_spot=None):
    """Fold fragments into moves: moves sharing an old or a new spot first, then any entry near a move's ends."""
    same_spot = radius if same_spot is None else same_spot
    near = lambda p, q, r=radius: p is not None and q is not None and np.linalg.norm(np.array(p) - np.array(q)) < r

    def fold(m, e):
        m["sources"] = sorted(set(m["sources"]) | set(e["sources"]))
        m["points"] = list(m["points"]) + list(e["points"])

    moves, rest = [], []
    for e in (e for e in entries if e["type"] == "moved"):
        m = next((m for m in moves if near(e["a"], m["a"], same_spot) or near(e["b"], m["b"], same_spot)), None)
        if m:
            fold(m, e)
        else:
            moves.append(e)
    for e in (e for e in entries if e["type"] != "moved"):
        m = next((m for m in moves if any(near(e[k], m[j]) for k in "ab" for j in "ab")), None)
        if m:
            fold(m, e)
        else:
            rest.append(e)
    return moves + rest


def main(half=64, absorb_k=35, same_k=20):
    import anthropic

    load_env()
    client = anthropic.Anthropic()
    d = dict(np.load(OUT / "joint.npz"))
    tau = json.loads((OUT / "alignment.json").read_text())["tau"]
    kept = []
    for e in json.loads((OUT / "changes.json").read_text()):
        answers = []
        for loc in (e["a"], e["b"]):
            if loc is None or any(a["loc"] == loc for a in answers):
                continue
            p = np.array(loc)
            va, vb = best_view(d, 0, p, 3 * tau), best_view(d, 1, p, 3 * tau)
            if va is None or vb is None:
                continue
            ans = ask(client, crop(d["images"][va[0]], va[1], half), crop(d["images"][vb[0]], vb[1], half))
            if ans:
                answers.append({"loc": loc, **ans})
        if any(a["changed"] for a in answers):
            e["vlm"] = answers
            e["label"] = next(a["object_a"] if a["object_a"] != "nothing" else a["object_b"] for a in answers if a["changed"])
            kept.append(e)
        print(e["type"], e.get("label"), [(a["object_a"], a["object_b"], a["changed"]) for a in answers])
    (OUT / "changes_vlm_raw.json").write_text(json.dumps(kept))  # before absorb, so merge rules can be re-tried free
    (OUT / "changes_verified.json").write_text(json.dumps(absorb(kept, absorb_k * tau, same_k * tau)))
    print(f"tokens in {USAGE[0]} out {USAGE[1]}, cost ${(USAGE[0] * PRICE[0] + USAGE[1] * PRICE[1]) / 1e6:.2f}")


if __name__ == "__main__":
    main()
