import numpy as np


def test_entry_locations_use_reported_a_and_b_not_attached_points():
    from score import entry_locations

    e = {"a": [0, 0, 0], "b": None, "points": [[0, 0, 0], [9, 9, 9]]}
    assert np.allclose(entry_locations(e), [[0, 0, 0]])
