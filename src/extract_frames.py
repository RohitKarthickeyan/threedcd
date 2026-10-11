"""Extract N evenly spaced frames from a video, choosing the sharpest frame in each window.

Sharpness = variance of the Laplacian (low value = blurry). Instead of deblurring,
we skip blurry frames, which is simpler and more reliable.

Usage: python extract_frames.py VIDEO OUT_DIR [N_FRAMES=40]
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main(video, out_dir, n_frames=40):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    frames, scores = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        scores.append(sharpness(fr))
    scores = np.array(scores)
    total = len(frames)

    # Split the video into n_frames equal windows and keep the sharpest frame of each.
    edges = np.linspace(0, total, n_frames + 1, dtype=int)
    picked = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = lo + int(np.argmax(scores[lo:hi]))
        picked.append(idx)

    for k, idx in enumerate(picked):
        cv2.imwrite(str(out / f"frame_{k:03d}_src{idx:04d}.png"), frames[idx])

    print(f"{video}: {total} frames, kept {len(picked)}")
    print(f"  median sharpness all frames: {np.median(scores):.0f}")
    print(f"  median sharpness kept:       {np.median(scores[picked]):.0f}")
    print(f"  worst kept frame:            {scores[picked].min():.0f}")


if __name__ == "__main__":
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    main(sys.argv[1], sys.argv[2], n)
