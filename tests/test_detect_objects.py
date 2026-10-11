import numpy as np

from detect_objects import compare, merge


def test_merge_keeps_same_label_objects_with_different_colours_apart():
    pts = np.zeros((4, 3))  # yellow and brown pillow side by side, two views each
    labs = np.array([[90, 0, 70], [92, 0, 68], [30, 10, 20], [31, 11, 19]], float)
    objs = merge(pts, ["pillow"] * 4, [0, 1, 0, 1], radius=0.3, colours=labs)
    assert len(objs) == 2


def test_compare_pairs_by_colour_so_a_swapped_pillow_is_a_move():
    yellow, brown = np.array([90.0, 0, 70]), np.array([30.0, 10, 20])
    # the brown pillow in B sits nearer the yellow pillow's old spot than the brown one's, so label-only pairing swaps them
    a = [{"label": "pillow", "pos": np.zeros(3), "lab": yellow}, {"label": "pillow", "pos": np.array([0.15, 0, 0]), "lab": brown}]
    b = [{"label": "pillow", "pos": np.array([0.05, 0, 0]), "lab": brown}, {"label": "pillow", "pos": np.array([2.0, 0, 0]), "lab": yellow}]
    ch = compare(a, b, static_r=0.5)
    assert len(ch) == 1 and ch[0]["type"] == "moved" and np.allclose(ch[0]["a"], 0) and np.allclose(ch[0]["b"], [2, 0, 0])
