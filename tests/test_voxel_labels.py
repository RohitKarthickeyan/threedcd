import numpy as np

from voxel_labels import FREE, OCC, UNKNOWN, label_voxels

K = np.array([[20.0, 0, 20.0], [0, 20.0, 15.0], [0, 0, 1]])  # 40x30 image, non-square
EXTR = np.hstack([np.eye(3), np.zeros((3, 1))])  # camera at origin looking +z


def run(points, ok=None):
    depth = np.full((1, 30, 40), 5.0)  # a wall at z = 5
    ok = np.ones((1, 30, 40), bool) if ok is None else ok
    lab, _, _ = label_voxels(np.array(points, float), depth, ok, K[None], EXTR[None], tau=0.3, min_views=1)
    return lab.tolist()


def test_wall_is_occupied_in_front_free_behind_unknown():
    assert run([[0, 0, 5.0], [0, 0, 2.0], [0, 0, 7.0]]) == [OCC, FREE, UNKNOWN]


def test_behind_camera_and_outside_image_give_no_evidence():
    assert run([[0, 0, -2.0], [100, 0, 5.0], [0, 100, 5.0]]) == [UNKNOWN] * 3


def test_voxel_just_in_front_of_surface_gives_no_free_evidence():
    # 0.5 in front of the wall: outside the occupied band (tau) but inside the free margin (3 tau),
    # so grazing and head-on views cannot disagree about it.
    assert run([[0, 0, 4.5], [0, 0, 3.9]]) == [UNKNOWN, FREE]
