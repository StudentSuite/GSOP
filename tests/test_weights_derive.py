import ast
from pathlib import Path

import pytest

from gsop import weights as W
from gsop.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config")


def test_derivation_returns_range_published_and_columns_for_every_family():
    for body, n in (("moon", 8), ("mars", 9)):
        d = W.derive_weights(CFG, body)
        assert len(d) == n and abs(sum(w.weight for w in d.values()) - 1) < 1e-12
        for name, w in d.items():
            assert w.range == CFG.weights(body)[name].range and w.published == CFG.weights(body)[name].published
            assert w.columns == CFG.weights(body)[name].columns


def test_column_weights_split_a_family_equally_and_sum_to_one():
    d = W.derive_weights(CFG, "moon")
    fam_cols = {n: w.columns for n, w in d.items()}
    cw = W.column_weights(W.weight_vector(d), fam_cols)
    assert cw["slope_med"] == pytest.approx(cw["slope_p90"]) == pytest.approx(d["slope"].weight / 2)
    assert cw["relief"] == pytest.approx(d["relief"].weight)
    assert sum(cw.values()) == pytest.approx(1.0)


def test_no_weight_literal_appears_in_the_code():
    tree = ast.parse((ROOT / "src" / "gsop" / "weights.py").read_text(encoding="utf-8"))
    numbers = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}
    assert numbers <= {0.0, 1.0}, numbers
