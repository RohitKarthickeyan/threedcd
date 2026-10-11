import numpy as np

from geometry import backproject, project


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


K = np.array([[50.0, 0, 3.0], [0, 50.0, 2.0], [0, 0, 1]])
EXTR = np.hstack([rot_y(20), [[0.5], [-0.2], [1.0]]])


def test_backproject_then_project_round_trips_on_non_square_image():
    H, W = 4, 6
    depth = np.linspace(1, 3, H * W).reshape(H, W)
    pts = backproject(depth, K, EXTR)
    u, v, z = project(pts.reshape(-1, 3), K, EXTR)
    vv, uu = np.mgrid[0:H, 0:W]
    assert np.allclose(u, uu.ravel()) and np.allclose(v, vv.ravel())
    assert np.allclose(z, depth.ravel())
