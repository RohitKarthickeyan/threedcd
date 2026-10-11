import numpy as np

from fuse import fuse


def cand(src, typ, a=None, b=None, label=None):
    pts = np.array([p for p in (a, b) if p is not None], float)
    return {"type": typ, "label": label, "a": None if a is None else np.array(a, float),
            "b": None if b is None else np.array(b, float), "sources": [src], "points": pts}


def test_fuse_does_not_chain_unlabelled_clusters_onto_an_anchor():
    cs = [cand("objects", "removed", a=[0, 0, 0], label="laptop"),
          cand("depth", "removed", a=[0.4, 0, 0]), cand("depth", "removed", a=[0.8, 0, 0]), cand("depth", "removed", a=[1.2, 0, 0])]
    e = fuse(cs, radius=0.5)
    laptop = next(x for x in e if x["label"] == "laptop")
    assert len(laptop["points"]) == 2 and len(e) == 2  # only the 0.4 cluster attaches; 0.8 + 1.2 stay one loose entry


def test_fuse_joins_two_loose_groups_bridged_by_a_third_cluster():
    cs = [cand("depth", "removed", a=[9, 9, 9]), cand("depth", "removed", a=[0, 0, 0]), cand("depth", "removed", a=[0.8, 0, 0]),
          cand("diff", "removed", a=[0.4, 0, 0])]  # the far group makes list removal compare clusters element-wise
    for c in cs:
        c["points"] = np.vstack([c["points"], c["points"]])  # real clusters hold many points
    e = fuse(cs, radius=0.5)
    assert sorted(len(x["points"]) for x in e) == [2, 6]


def test_removed_clusters_attach_only_to_an_anchors_a_end():
    # a removal at the laptop's NEW spot is a different object (the pillow that left), not the laptop
    cs = [cand("objects", "moved", a=[0, 0, 0], b=[3, 0, 0], label="laptop"), cand("depth", "removed", a=[3.1, 0, 0])]
    assert sorted(x["type"] for x in fuse(cs, radius=0.5)) == ["moved", "removed"]
