import numpy as np



def test_ask_returns_none_when_the_answer_was_cut_off():
    from types import SimpleNamespace

    from verify_vlm import ask

    r = SimpleNamespace(stop_reason="max_tokens", usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                        content=[SimpleNamespace(type="text", text='{"object_a": "lap')])
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: r)))
    img = np.zeros((8, 8, 3), np.uint8)
    assert ask(client, img, img) is None


def test_absorb_merges_a_mispaired_move_that_shares_the_old_spot():
    from verify_vlm import absorb

    mv = lambda lab, a, b: {"type": "moved", "label": lab, "a": a, "b": b, "sources": ["objects"], "points": [a, b]}
    out = absorb([mv("fan", [0, 0, 0], [3, 0, 0]), mv("oscillating fan", [0.1, 0, 0], [0, 5, 0])], radius=1.0)
    assert [e["label"] for e in out] == ["fan"]


def test_absorb_keeps_moves_whose_spots_are_near_but_not_the_same():
    from verify_vlm import absorb

    mv = lambda lab, a, b: {"type": "moved", "label": lab, "a": a, "b": b, "sources": ["objects"], "points": [a, b]}
    out = absorb([mv("laptop", [5, 0, 0], [0, 0, 0]), mv("pillow", [0.1, 3, 0], [0.9, 0, 0])], radius=1.0, same_spot=0.5)
    assert [e["label"] for e in out] == ["laptop", "pillow"]  # new spots 0.9 apart: neighbours, not one object
