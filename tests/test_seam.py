import numpy as np

from gsop.earth.seam import seam_test


def strip(seam_step=0.0, rows=72, cols=9, seed=0):
    rng = np.random.default_rng(seed)
    lat = np.linspace(57.0, 63.0, rows)  # 6 degrees tall, 1/12 degree per row
    trend = 0.2 * (lat - 60.0)  # natural gradient
    values = trend[:, None] + rng.normal(0, 0.3, size=(rows, cols)) + (lat[:, None] >= 60.0) * seam_step
    return lat, values


def test_no_step_is_reported_as_ok():
    lat, values = strip(0.0)
    res = seam_test(lat, values, 60.0)
    assert res.verdict == "seam OK" and res.n_placebo >= 10


def test_injected_step_is_flagged():
    lat, values = strip(seam_step=2.0)
    res = seam_test(lat, values, 60.0)
    assert res.verdict == "seam step" and res.step > 1.5 and res.z > 3


def test_works_on_the_southern_seam_and_with_nan_and_unsorted_rows():
    lat, values = strip(0.0, seed=2)
    lat = lat - 116.0  # now -59 .. -53, seam -56
    values = values.copy()
    values[5, 3] = np.nan
    order = np.random.default_rng(1).permutation(len(lat))
    res = seam_test(lat[order], values[order], -56.0)
    assert res.verdict == "seam OK"


def test_too_short_a_strip_is_insufficient():
    lat, values = strip(rows=14)
    assert seam_test(lat, values, 60.0).verdict == "insufficient data"
