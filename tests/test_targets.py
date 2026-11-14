from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from gsop import targets as TG
from gsop.cli import app
from gsop.config import load_config

CFG = load_config(Path(__file__).resolve().parents[1] / "config")
BUILT = {  # columns the real modules build today (B25-B31); the rest are unavailable
    "moon": {"slope_med", "slope_p90", "relief", "rough_a", "rough_b", "sun_max_elev", "sun_low_share", "water_index"},
    "mars": {"slope_med", "slope_p90", "relief", "rough_a", "rough_b", "water_index", "air_density", "insol_annual", "sun_max_elev"},
}


def fake_rows(cfg, body, *, drop=None):
    rows = []
    for name, t in cfg.targets(body).items():
        if t.lat is None:
            continue
        for col in TG.used_columns(cfg, body) + (["thermal_inertia_themis"] if body == "mars" else []):
            if drop and (name, col) == drop:
                continue
            ok = col in BUILT[body] or col == "thermal_inertia_themis"
            rows.append({
                "target": name, "column": col, "value": 1.5 if ok else float("nan"), "spread": 0.2 if ok else float("nan"),
                "source": "synthetic test source", "evidence_grade": "B", "window": "15 km window", "unavailable": not ok,
                "note": "" if ok else "unavailable: not built in this test",
            })
    return rows


def fake_collect(cfg, body, cache_dir=None, diviner_dir=None):
    return fake_rows(cfg, body)


@pytest.mark.parametrize("body", ["moon", "mars"])
def test_schema_and_files(tmp_path, body):
    df = TG.build_targets(CFG, body, tmp_path, collect=fake_collect)
    parquet, md = tmp_path / f"targets_{body}.parquet", tmp_path / f"targets_{body}.md"
    assert parquet.is_file() and md.is_file()
    table = pq.read_table(parquet)
    assert table.schema.equals(TG.SCHEMA)
    assert [f.name for f in table.schema] == ["target", "column", "value", "spread", "source", "evidence_grade", "window", "unavailable", "unscored", "note"]
    back = table.to_pandas()
    assert len(back) == len(df) and set(back.evidence_grade) <= set("ABC")
    assert back[back.unavailable].value.isna().all()


def test_every_weights_column_is_present_for_every_scored_target_or_flagged(tmp_path):
    for body in ("moon", "mars"):
        df = TG.to_frame(fake_rows(CFG, body), CFG, body)
        assert TG.validate(df, CFG, body) == []
        cols = set(TG.used_columns(CFG, body))
        for target, t in CFG.targets(body).items():
            if t.lat is not None:
                assert cols <= set(df[df.target == target].column)


def test_a_silent_gap_is_an_error_and_so_is_an_available_row_without_a_value(tmp_path):
    df = TG.to_frame(fake_rows(CFG, "moon", drop=("connecting_ridge_c1", "relief")), CFG, "moon")
    assert any("connecting_ridge_c1: no row for column 'relief'" in p for p in TG.validate(df, CFG, "moon"))
    with pytest.raises(TG.TargetError, match="no row for column 'relief'"):
        TG.write_targets(df, CFG, "moon", tmp_path)
    bad = TG.to_frame(fake_rows(CFG, "moon"), CFG, "moon")
    bad.loc[(bad.target == "shackleton_rim") & (bad.column == "slope_med"), "value"] = np.nan
    assert any("marked available but the value is not finite" in p for p in TG.validate(bad, CFG, "moon"))
    nonote = TG.to_frame(fake_rows(CFG, "moon"), CFG, "moon")
    nonote.loc[nonote.unavailable, "note"] = "no reason"
    assert any("unavailable rows must say why" in p for p in TG.validate(nonote, CFG, "moon"))


def test_arcadia_is_written_as_unscored_and_malapert_is_absent():
    df = TG.to_frame(fake_rows(CFG, "mars"), CFG, "mars")
    arc = df[df.target == "arcadia"]
    assert len(arc) == 1 and bool(arc.iloc[0].unscored) and arc.iloc[0].column == "*" and "unscored: no target data" in arc.iloc[0].note
    assert not df[df.target == "jezero"].unscored.any()
    assert not df.target.str.contains("malapert", case=False).any()
    moon = TG.to_frame(fake_rows(CFG, "moon"), CFG, "moon")
    assert set(moon.target) == {"connecting_ridge_c1", "shackleton_rim"} and not moon.unscored.any()


def test_family_coverage_names_what_cannot_be_scored(tmp_path):
    df = TG.to_frame(fake_rows(CFG, "moon"), CFG, "moon")
    cov = TG.family_coverage(df, CFG, "moon")["connecting_ridge_c1"]
    assert cov["slope"]["available"] == ["slope_med", "slope_p90"] and cov["slope"]["missing"] == []
    assert cov["temperature"]["available"] == [] and cov["composition"]["available"] == [] and cov["thermal_inertia"]["available"] == []
    mars = TG.family_coverage(TG.to_frame(fake_rows(CFG, "mars"), CFG, "mars"), CFG, "mars")["jezero"]
    assert mars["composition"]["available"] == [] and mars["thermal_inertia"]["available"] == [] and mars["temperature"]["available"] == []
    assert mars["water"]["available"] == ["water_index"] and mars["sunlight"]["available"] == ["insol_annual", "sun_max_elev"]


def test_markdown_summary_is_readable_and_states_the_blockers(tmp_path):
    TG.build_targets(CFG, "mars", tmp_path, collect=fake_collect)
    md = (tmp_path / "targets_mars.md").read_text(encoding="utf-8")
    assert "## jezero" in md and "## arcadia" in md and "**Unscored.**" in md
    assert "UNAVAILABLE" in md and "unavailable: not built in this test" in md
    assert "Families with no usable target column" in md and "composition (weight 0.23)" in md


def test_cli_reports_a_clean_error_instead_of_a_traceback_on_invalid_output(monkeypatch, tmp_path):
    def broken(cfg, body, cache_dir, diviner_dir=None):
        return fake_rows(cfg, body, drop=("jezero", "relief"))

    monkeypatch.setattr(TG, "collect_rows", broken)
    result = CliRunner().invoke(app, ["s2", "targets", "--body", "mars", "--out-dir", str(tmp_path)])
    assert result.exit_code == 1 and "no row for column 'relief'" in result.output
    assert not (tmp_path / "targets_mars.parquet").exists()  # an invalid file is never written


def test_cli_writes_both_files_with_a_summary_line(monkeypatch, tmp_path):
    monkeypatch.setattr(TG, "collect_rows", lambda cfg, body, cache_dir, diviner_dir=None: fake_rows(cfg, body))
    result = CliRunner().invoke(app, ["s2", "targets", "--body", "moon", "--out-dir", str(tmp_path)])
    assert result.exit_code == 0 and "unavailable" in result.output
    assert (tmp_path / "targets_moon.parquet").is_file() and (tmp_path / "targets_moon.md").is_file()


def test_unknown_body_is_refused():
    with pytest.raises(TG.TargetError):
        TG.collect_rows(CFG, "venus")
