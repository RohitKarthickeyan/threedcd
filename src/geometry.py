"""Camera math shared by every stage. Extrinsics are camera-from-world [R|t], OpenCV axes."""
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def backproject(depth, K, extr):
    """Z-depth (H,W) -> world points (H,W,3)."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)
    return (cam - extr[:, 3]) @ extr[:, :3]  # row form of R^T (p - t)


def project(points, K, extr):
    """World points (M,3) -> pixel u, v and camera z."""
    cam = points @ extr[:, :3].T + extr[:, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return K[0, 0] * cam[:, 0] / z + K[0, 2], K[1, 1] * cam[:, 1] / z + K[1, 2], z


def camera_center(extr):
    return -extr[:, :3].T @ extr[:, 3]


def up_from_cameras(extrs):
    """Average of the cameras' -y axes in world (phones are held roughly upright)."""
    up = -extrs[:, 1, :3].mean(0)
    return up / np.linalg.norm(up)


def valid_mask(conf, depth, drop=0.1, edge_rtol=0.03):
    """Pixels to trust: not in the bottom `drop` confidence of their frame, not on a depth edge."""
    thr = np.quantile(conf.reshape(len(conf), -1), drop, axis=1)
    size = (1, 3, 3)  # 3x3 window per frame
    jump = (maximum_filter(depth, size) - minimum_filter(depth, size)) / np.maximum(depth, 1e-6)
    return (conf >= thr[:, None, None]) & (jump <= edge_rtol)


def confident_points(d, frames):
    """Back-projected points and colors from the given frame indices, invalid pixels dropped."""
    mask = valid_mask(d["conf"][frames], d["depth"][frames])
    pts, cols = [], []
    for m, i in zip(mask, frames):
        pts.append(backproject(d["depth"][i], d["intrinsic"][i], d["extrinsic"][i])[m])
        cols.append(d["images"][i][m])
    return np.concatenate(pts), np.concatenate(cols)


def foreground_points(d, i, box, tau, fg_k=10):
    """Valid points in pixel box [x0, x1, y0, y1] no deeper than the box's 10th-pct depth + fg_k*tau."""
    x0, x1, y0, y1 = box
    ok = valid_mask(d["conf"][[i]], d["depth"][[i]])[0][y0:y1, x0:x1]
    z = d["depth"][i, y0:y1, x0:x1]
    if not ok.any():
        return np.zeros((0, 3))
    fg = ok & (z <= np.quantile(z[ok], 0.1) + fg_k * tau)
    return backproject(d["depth"][i], d["intrinsic"][i], d["extrinsic"][i])[y0:y1, x0:x1][fg]
