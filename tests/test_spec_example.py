"""The worked example of docs/SPEC.md (issue C31), recomputed with the real code. If this fails the document and the code disagree."""

from pathlib import Path

import numpy as np
import pytest

from gsop import purposes as P
from gsop import similarity as S

SPEC = (Path(__file__).resolve().parents[1] / "docs" / "SPEC.md").read_text(encoding="utf-8")
FAMILIES = ["slope", "illumination", "cold_dry"]
FAMILY_COLS = [[0], [1, 2], [3]]  # slope_med | sun_max_elev, sun_low_share | water_index
TARGET = np.array([5.0, 2.0, 1.0, 0.8])
SIGMA = np.array([2.0, 10.0, 0.2, 0.4])
TILES = np.array([[5.0, 62.0, 0.20, 0.8], [9.0, 12.0, 0.90, 0.4], [3.0, 2.0, 1.00, 0.0]], dtype="float32")


def score(weights, target=TARGET):
    return S.score_tiles(TILES, target, SIGMA, FAMILY_COLS, np.array(weights, dtype="float32"), return_family=True)


def test_sigma_table():
    def earth(iqr):  # half the tiles at 0, half at iqr: the interquartile range is exactly iqr
        return np.array([0.0] * 50 + [iqr] * 50)

    for col, spread, unc, iqr, expected, winner in [("slope_med", 1.0, 2.0, 6.0, 2.0, "measurement_uncertainty"), ("sun_max_elev", 0.5, 0.0, 40.0, 10.0, "iqr"),
                                                    ("sun_low_share", 0.05, 0.0, 0.8, 0.2, "iqr"), ("water_index", 0.1, 0.0, 1.6, 0.4, "iqr")]:
        t = S.compute_sigma(col, spread, unc, earth(iqr), 0.25)
        assert t.sigma == pytest.approx(expected) and t.winner == winner, col
    assert S.compute_sigma("slope_med", 1.0, 2.0, earth(6.0), 0.25).iqr_term == pytest.approx(1.5)


def test_weight_derivation_and_purpose():
    paper = {"slope": 30, "illumination": 20, "cold_dry": 10}
    match = {"slope": 1.0, "illumination": 0.5, "cold_dry": 1.0}
    raw = {f: paper[f] / sum(paper.values()) * match[f] for f in paper}
    assert sum(raw.values()) == pytest.approx(0.8333, abs=1e-4)
    w = {f: v / sum(raw.values()) for f, v in raw.items()}
    assert [round(w[f], 4) for f in FAMILIES] == [0.6, 0.2, 0.2]
    mult = {"slope": 2.0}
    m = {f: w[f] * mult.get(f, 1.0) for f in FAMILIES}
    assert sum(m.values()) == pytest.approx(1.6)
    assert [round(m[f] / sum(m.values()), 4) for f in FAMILIES] == [0.75, 0.125, 0.125]


def test_column_and_family_similarities():
    s = S.column_similarity(TILES, TARGET, SIGMA)
    assert np.round(s, 4) == pytest.approx(np.array([[1.0, 0.0, 0.0003, 1.0], [0.1353, 0.6065, 0.8825, 0.6065], [0.6065, 1.0, 1.0, 0.1353]]), abs=1e-6)
    assert s[0, 1] == pytest.approx(np.exp(-18), rel=1e-4)  # 1.5e-8
    fam = S.family_similarity(s, FAMILY_COLS)
    assert np.round(fam, 4) == pytest.approx(np.array([[1.0, 0.0002, 1.0], [0.1353, 0.7445, 0.6065], [0.6065, 1.0, 0.1353]]), abs=1e-6)


def test_tile_scores_base_and_landing_mobility():
    base = score([0.6, 0.2, 0.2]).score
    assert np.round(base, 2) == pytest.approx([80.0, 35.14, 59.1], abs=1e-5)
    landing = score([0.75, 0.125, 0.125]).score
    gate = P.parse_gate("slope_med <= 8")
    ok = TILES[:, 0] <= gate.value
    assert ok.tolist() == [True, False, True]  # tile B fails the extra gate
    assert np.round(landing[ok], 2) == pytest.approx([87.5, 59.68], abs=1e-5)
    assert int(np.argmax(base)) == 0 and base[2] > base[1]  # ranking A, C, B


def test_one_monte_carlo_run():
    drawn = np.array([0.55, 0.25, 0.15])
    r1 = drawn / drawn.sum()
    assert np.round(r1, 4).tolist() == [0.5789, 0.2632, 0.1579]
    r2 = r1 * np.array([2.0, 1.0, 1.0])
    assert np.round(r2, 4).tolist() == [1.1579, 0.2632, 0.1579] and r2.sum() == pytest.approx(1.5789, abs=1e-4)
    w = r2 / r2.sum()
    assert np.round(w, 4).tolist() == [0.7333, 0.1667, 0.1]
    shifted = TARGET + np.array([0.5, 0.0, 0.0, -0.1])
    assert shifted.tolist() == [5.5, 2.0, 1.0, pytest.approx(0.7)]
    scores = score(w, shifted).score
    assert np.round(scores, 2) == pytest.approx([80.77, 35.82, 52.4], abs=1e-5)
    import math

    assert math.ceil(0.01 * 3) == 1  # the top 1% of three tiles is one tile


@pytest.mark.parametrize("text", ["**80.00**", "**35.14**", "**59.10**", "**87.50**", "**59.68**", "80.77", "52.40", "0.7333, 0.1667, 0.1000", "0.6, 0.2, 0.2", "0.75, 0.125, 0.125"])
def test_the_numbers_are_in_the_document(text):
    assert text in SPEC
