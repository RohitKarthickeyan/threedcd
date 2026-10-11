import numpy as np

from diff import REMOVED, UNOBSERVED, diff_labels, feature_changed
from voxel_labels import FREE, OCC, UNKNOWN


def grid(fill, **cells):
    g = np.full((5, 5, 5), fill, np.int8)
    for k, v in cells.items():
        g[tuple(int(c) for c in k[1:])] = v
    return g


def test_unknown_in_either_video_is_unobserved_never_removed():
    la = grid(FREE, c222=OCC)
    lb = grid(FREE, c222=UNKNOWN)
    assert diff_labels(la, lb, dilate=0)[2, 2, 2] == UNOBSERVED
    assert diff_labels(lb, la, dilate=0)[2, 2, 2] == UNOBSERVED


def test_default_diff_keeps_removed_shell_that_borders_occluded_interior():
    la = grid(FREE)
    la[1:4, 1:4, 1:4] = OCC
    la[2, 2, 2] = UNKNOWN  # interior hidden behind the object's own surface
    lb = grid(FREE)
    out = diff_labels(la, lb)
    assert out[1, 2, 2] == REMOVED


def two_video_wall(colour_a, colour_b, n_frames=(2, 2)):
    """Videos A and B: identical cameras at the origin looking at a wall at z=5; per-video image colour."""
    K = np.array([[20.0, 0, 20.0], [0, 20.0, 15.0], [0, 0, 1]])
    E = np.hstack([np.eye(3), np.zeros((3, 1))])
    na, nb = n_frames
    imgs = [colour_a] * na + [colour_b] * nb
    return {"depth": np.full((na + nb, 30, 40), 5.0), "intrinsic": np.repeat(K[None], na + nb, 0),
            "extrinsic": np.repeat(E[None], na + nb, 0), "images": np.stack(imgs).astype(np.uint8),
            "video": np.array([0] * na + [1] * nb)}


def fc(d, pts, **kw):
    return feature_changed(d, pts, feats=d["images"].astype(np.float32) / 255, min_dist=0.3, **kw)  # one feature per pixel


WALL = np.array([[x, y, 5.0] for x in np.linspace(-4, 4, 10) for y in np.linspace(-3, 3, 10)])  # ~4 px apart, none on the patch


def test_feature_changed_flags_only_the_recoloured_spot():
    a = np.full((30, 40, 3), 128)
    b = np.full((30, 40, 3), 128)
    a[14:17, 19:22] = (230, 210, 40)  # yellow pillow in A ...
    b[14:17, 19:22] = (20, 20, 20)  # ... dark laptop screen in B, same place
    flagged = fc(two_video_wall(a, b), np.vstack([WALL, [[0, 0, 5.0]]]), tau=0.3)
    assert flagged[-1] and flagged[:-1].sum() == 0


def test_feature_changed_ignores_frames_that_do_not_see_the_point():
    # point visible in 2 of 5 A frames; the 3 that miss it must not count as black
    d = two_video_wall(np.full((30, 40, 3), 128), np.full((30, 40, 3), 128), n_frames=(5, 2))
    d["extrinsic"][2:5, 0, 3] = 0.5  # shifts the edge point (u 39.2 -> 41.2) out of W=40; WALL stays inside
    assert not fc(d, np.vstack([WALL, [[4.8, 0, 5.0]]]), tau=0.3).any()


def test_feature_changed_removes_the_exposure_offset_between_videos():
    # B is darker overall; the spot changed in the opposite direction, so raw distance equals the background's
    a, b = np.full((30, 40, 3), 150), np.full((30, 40, 3), 90)
    a[14:17, 19:22], b[14:17, 19:22] = 90, 150
    pts = np.vstack([WALL, [[0, 0, 5.0]]])
    assert fc(two_video_wall(a, b), pts, tau=0.3).tolist() == [False] * len(WALL) + [True]
