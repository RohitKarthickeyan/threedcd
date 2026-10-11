import numpy as np

from depth_compare import local_residual, warp_depth

K = np.array([[20.0, 0, 20.0], [0, 20.0, 15.0], [0, 0, 1]])  # 40x30, non-square
E = np.hstack([np.eye(3), np.zeros((3, 1))])


def test_local_residual_cancels_offsets_but_keeps_a_small_bump():
    r = np.full((256, 256), 0.3)  # regional offset
    r[100:120, 100:130] += 1.0  # small object
    lr = local_residual(r, np.ones_like(r, bool), block=128)
    assert abs(lr[0, 0]) < 1e-9 and np.isclose(lr[110, 110], 1.0)


def test_overlap_pairs_prefer_a_camera_facing_the_same_way_over_a_closer_one():
    from depth_compare import overlap_pairs

    back = np.hstack([np.diag([-1.0, 1, -1]), np.zeros((3, 1))])  # same spot, looking -z
    far = np.hstack([np.eye(3), [[0], [0], [1.0]]])  # 1 unit behind, looking +z
    d = {"depth": np.full((3, 30, 40), 5.0), "conf": np.ones((3, 30, 40)),
         "intrinsic": np.repeat(K[None], 3, 0), "extrinsic": np.stack([E, back, far])}
    assert overlap_pairs(d, 0, np.array([1, 2]), k=1).tolist() == [2]


def test_warp_into_a_closer_camera_does_not_leak_background_through_holes():
    from geometry import backproject, project

    H, W = 128, 96
    Kb = np.array([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]])
    depth = np.full((H, W), 5.0)
    depth[48:80, 32:64] = 2.0  # object in front of a wall
    dst = np.hstack([np.eye(3), [[-0.2], [0], [-0.8]]])  # 0.8 closer and 0.2 to the side
    from depth_compare import close_holes

    w = close_holes(warp_depth(depth, np.ones((H, W), bool), Kb, E, Kb, dst))
    u, v, _ = project(backproject(depth, Kb, E)[48:80, 32:64].reshape(-1, 3), Kb, dst)
    inner = w[int(v.min()) + 2:int(v.max()) - 1, int(u.min()) + 2:int(u.max()) - 1]
    assert not (np.isfinite(inner) & (inner > 2.0 - 0.8 + 0.1)).any()  # nothing farther than the object inside it
