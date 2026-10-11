import numpy as np

from check_alignment import alignment_tau, fit_floor

rng = np.random.default_rng(0)


def test_alignment_tau_measures_noise_not_point_spacing():
    def surface(n):  # independent noisy samples of the same plane
        return np.c_[rng.uniform(0, 1, (n, 2)), rng.normal(0, 0.002, n)]

    # Only the queries may be thinned; thinning B to 20k points would add ~0.007 spacing to every distance.
    tau, _ = alignment_tau(surface(200_000), surface(2_000_000), gate=0.05, n_query=20_000)
    assert tau < 0.006  # noise-only 90th percentile is ~2.33 * 0.002


def test_alignment_tau_ignores_a_points_on_surfaces_b_never_saw():
    pa = rng.uniform(0, 1, (20000, 3))
    pb = pa + rng.normal(0, 0.002, pa.shape)
    pa[:5000] += 0.03  # 25%: moved objects, glare, overlap edges -> no B partner, but within the gate
    tau, _ = alignment_tau(pa, pb, gate=0.05)
    assert tau < 0.012  # the raw 90th percentile would be ~0.03


def test_fit_floor_handles_a_million_points():
    # full_matrices SVD would allocate a 50k x 50k matrix (20 GB) for the 5% floor selection
    pts = np.c_[rng.uniform(-3, 3, (1_000_000, 2)), rng.normal(-1.0, 0.001, 1_000_000)]
    n, d = fit_floor(pts, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(n, [0, 0, 1], atol=1e-2) and abs(d - 1.0) < 0.01
