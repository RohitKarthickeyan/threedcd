"""Interactive 3D view of the room and its detected changes (Gradio).

Usage: python src/view3d.py [CHANGES=data/out/changes_verified.json] -> http://127.0.0.1:7860, data/out/overview.jpg
"""
import json
import sys
from pathlib import Path

import numpy as np

from geometry import confident_points

OUT = Path("data/out")
COLOURS = {"removed": (230, 0, 0), "added": (0, 200, 0), "moved": (40, 90, 255), "changed": (255, 200, 0)}
FLIP = np.diag([1.0, -1, -1, 1])  # OpenCV (y down, z forward) -> viewer (y up)


def scene_glb(path, points, colours, entries, marker):
    import trimesh

    scene = trimesh.Scene([trimesh.PointCloud(points, colors=colours)])
    for e in entries:
        pts = np.array(e["points"])
        scene.add_geometry(trimesh.PointCloud(pts, colors=np.tile(COLOURS[e["type"]], (len(pts), 1))))
        for key, col in (("a", (230, 0, 0, 255)), ("b", (0, 200, 0, 255))):  # old spot red, new spot green
            if e[key] is not None:
                s = trimesh.creation.icosphere(radius=marker)
                s.apply_translation(e[key])
                s.visual.face_colors = col
                scene.add_geometry(s)
    scene.apply_transform(FLIP)
    scene.export(path)


def splat(points, colours, K, extr, hw, radius=1, background=(20, 20, 26)):
    """Render points into a (H, W) image from camera K/extr; nearer points win; each point is a (2r+1)^2 square."""
    from geometry import project

    H, W = hw
    u, v, z = project(points, K, extr)
    keep = (z > 0) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    u, v, z, c = np.round(u[keep]).astype(int), np.round(v[keep]).astype(int), z[keep], colours[keep]
    order = np.argsort(-z)  # far to near: nearer writes last
    img = np.full((H, W, 3), background, np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            img[np.clip(v[order] + dy, 0, H - 1), np.clip(u[order] + dx, 0, W - 1)] = c[order]
    return img


def overview_png(path, d, up, cam=5, size=(800, 1280)):
    """Room seen from behind and above camera `cam`, looking at the scene's middle (both videos' points)."""
    import cv2

    from geometry import camera_center

    pts, cols = confident_points(d, np.arange(len(d["video"])))
    target = np.median(pts, 0)
    ext = np.linalg.norm(np.quantile(pts, 0.95, 0) - np.quantile(pts, 0.05, 0))
    c = camera_center(d["extrinsic"][cam]) - target
    flat = c - (c @ up) * up
    eye = target + flat / np.linalg.norm(flat) * 0.55 * ext + up * 0.45 * ext
    z = (target - eye) / np.linalg.norm(target - eye)
    y = -up - (-up @ z) * z
    y /= np.linalg.norm(y)
    R = np.stack([np.cross(y, z), y, z])
    H, W = size
    K = np.array([[0.75 * W, 0, W / 2], [0, 0.75 * W, H / 2], [0, 0, 1]])
    img = splat(pts, cols, K, np.hstack([R, (-R @ eye)[:, None]]), size)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main(changes=OUT / "changes_verified.json", max_points=900_000):
    import gradio as gr

    d = dict(np.load(OUT / "joint.npz"))
    a = json.loads((OUT / "alignment.json").read_text())
    tau = a["tau"]
    overview_png(OUT / "overview.jpg", d, np.array(a["floor_n"]))
    pts, cols = confident_points(d, np.arange(len(d["video"])))
    keep = np.random.default_rng(0).choice(len(pts), min(max_points, len(pts)), replace=False)
    entries = json.loads(Path(changes).read_text())
    glb = OUT / "scene.glb"
    scene_glb(glb, pts[keep], cols[keep], entries, marker=3 * tau)
    rows = [[e["type"], e.get("label"), (e.get("description") or {}).get("sentence", ""), ", ".join(e["sources"])]
            for e in entries]
    with gr.Blocks(title="threedcd") as demo:
        gr.Markdown("Changes: **red** removed, **green** added, **blue** moved, **yellow** changed. "
                    "Spheres: red = old spot, green = new spot.")
        gr.Model3D(value=str(glb), clear_color=[0.08, 0.08, 0.1, 1], height=720)
        gr.Dataframe(rows, headers=["type", "label", "what happened", "detectors"], wrap=True)
    demo.launch()


if __name__ == "__main__":
    main(*sys.argv[1:])
