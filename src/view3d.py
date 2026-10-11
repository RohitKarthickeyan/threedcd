"""Interactive 3D view of the room and its detected changes (Gradio).

Usage: python src/view3d.py [CHANGES=data/out/changes_verified.json] -> http://127.0.0.1:7860
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


def main(changes=OUT / "changes_verified.json", max_points=900_000):
    import gradio as gr

    d = dict(np.load(OUT / "joint.npz"))
    tau = json.loads((OUT / "alignment.json").read_text())["tau"]
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
