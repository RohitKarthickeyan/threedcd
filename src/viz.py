"""Debug outputs: .ply point clouds for the user, PNG views for quick inspection."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_ply(path, points, colors):
    v = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = points.T
    v["red"], v["green"], v["blue"] = colors.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(v)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(v.tobytes())


def _basis(up):
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 0, 1.0])
    a /= np.linalg.norm(a)
    return a, np.cross(up, a)


def save_views_png(path, points, colors, up, max_points=200_000):
    """Top-down view (looking along -up) and a side view."""
    idx = np.random.default_rng(0).choice(len(points), min(max_points, len(points)), replace=False)
    p, c = points[idx], colors[idx] / 255.0
    e1, e2 = _basis(up)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].scatter(p @ e1, p @ e2, c=c, s=0.3)
    ax[0].set_title("top-down")
    ax[1].scatter(p @ e1, p @ up, c=c, s=0.3)
    ax[1].set_title("side")
    for a in ax:
        a.set_aspect("equal")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def draw_points(img, uv, color, r=1):
    img = img.copy()
    H, W = img.shape[:2]
    for u, v in np.round(uv[np.isfinite(uv).all(1)]).astype(int):
        if 0 <= u < W and 0 <= v < H:
            img[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1] = color
    return img
