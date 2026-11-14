"""illumination_match: the illumination family alone, shown beside the total (issue C28)."""

from pathlib import Path

import numpy as np
import pandas as pd

import stage_inputs as SI
from gsop import illumination as I
from gsop import similarity as S
from gsop.config import load_config
from gsop.paths import Paths
from gsop.stages import s3, s4, s5

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
ILL = CFG.features.illumination


def moon_family_match(earth_lat: float) -> float:
    """Illumination similarity of an Earth tile at ``earth_lat`` to the Connecting Ridge C1 polar target (lat -89.45), with the real illumination model."""
    cols = ("sun_max_elev", "sun_low_share")
    moon = ILL.bodies["moon"]
    earth = ILL.bodies["earth"]
    target = np.array([I.sun_max_elev(-89.45, moon.obliquity_deg)[0], I.sun_low_share(-89.45, moon.obliquity_deg)[0]])
    lats = np.rad2deg(np.arcsin(np.linspace(-1, 1, 721)[1:-1]))  # equal-area latitudes: the Earth distribution the sigma rule sees
    ref = np.column_stack([I.sun_max_elev(lats, earth.obliquity_deg), I.sun_low_share(lats, earth.obliquity_deg)])
    q75, q25 = np.percentile(ref, [75, 25], axis=0)
    sigma = 0.25 * (q75 - q25)  # the sigma rule of config/run.yaml with no target spread and no measurement uncertainty
    x = np.array([[I.sun_max_elev(earth_lat, earth.obliquity_deg)[0], I.sun_low_share(earth_lat, earth.obliquity_deg)[0]]], dtype="float32")
    s = S.column_similarity(x, target, sigma)
    return float(S.family_similarity(s, [[0, 1]])[0, 0])


def test_an_equatorial_earth_tile_has_a_low_illumination_match_to_the_lunar_pole():
    equator = moon_family_match(0.0)
    assert equator < 0.01  # Sun overhead twice a year versus a Sun that never rises above ~2 degrees
    assert equator < moon_family_match(75.0)  # polar Earth is closer to the Moon pole than the equator is


def test_the_column_is_in_scores_and_in_every_monte_carlo_row(tmp_path):
    root, g, _ = SI.write_inputs(tmp_path, "moon", n=3000, seed=1)
    paths = Paths(root)
    out = paths.results("moon")
    s3.run_gates(CFG, "moon", paths.earth_parquet, paths.targets_parquet("moon"), out, log=lambda m: None)
    s4.run_scores(CFG, "moon", paths.earth_parquet, paths.targets_parquet("moon"), out / "gates_passed.parquet", out, log=lambda m: None)
    s5.run_simulate(CFG, "moon", paths, runs=4, workers=1, block_runs=4, log=lambda m: None)
    scores = pd.read_parquet(out / "scores_base.parquet").set_index(["target", "tile_id"])["illumination_match"]
    mc = pd.read_parquet(next(out.glob("mc_*_base.parquet")))
    assert "illumination_match" in mc.columns and mc.illumination_match.notna().all()
    assert mc.illumination_match.between(0, 1).all()
    expected = scores.loc[list(zip(mc.target, mc.tile_id))].to_numpy()
    assert np.array_equal(mc.illumination_match.to_numpy(), expected)  # copied, not recomputed
    assert list(mc.columns).index("illumination_match") == list(mc.columns).index("tile_id") + 1
