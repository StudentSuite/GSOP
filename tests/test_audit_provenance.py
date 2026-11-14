import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

import test_targets as TT
from gsop import targets as TG
from gsop.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_provenance", ROOT / "scripts" / "audit_provenance.py")
AP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AP)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A copy of config/ and docs/ plus real-format target tables, so the audit can be broken on purpose."""
    shutil.copytree(ROOT / "config", tmp_path / "config")
    shutil.copytree(ROOT / "docs", tmp_path / "docs")
    cfg = load_config(ROOT / "config")
    for body in ("moon", "mars"):
        rows = TT.fake_rows(cfg, body)
        for r in rows:  # give each fake row a source the audit can recognise
            r["source"] = {"moon": "LOLA 5 m DEM, Barker et al. 2021", "mars": "MOLA DEM 463 m (MOLA Team)"}[body]
        TG.write_targets(TG.to_frame(rows, cfg, body), cfg, body, tmp_path / "data" / "processed")
    monkeypatch.setattr(AP, "ROOT", tmp_path)
    return tmp_path


def run():
    rows, skipped = AP.audit()
    return {(w): (s, d) for w, s, d in rows}, skipped


def edit(tree, doc, old, new, count=-1):
    p = tree / "docs" / doc
    p.write_text(p.read_text(encoding="utf-8").replace(old, new, count), encoding="utf-8")


def test_the_repo_as_it_stands_audits_clean():
    rows, skipped = AP.audit()
    assert [r for r in rows if r[1] != "ok"] == []
    assert len(rows) > 60


def test_clean_tree_with_target_tables_is_clean(tree):
    rows, skipped = run()
    assert all(s == "ok" for s, _ in rows.values())
    assert any(k.startswith("targets_moon:") for k in rows) and any(k.startswith("targets_mars:") for k in rows)
    assert any("earth_features_6km.parquet" in s for s in skipped)


def test_a_dataset_missing_from_provenance_is_caught(tree):
    edit(tree, "PROVENANCE.md", "MOLA", "XXXX")
    edit(tree, "UNVERIFIED.md", "MOLA", "XXXX")
    rows, _ = run()
    assert rows["relief.mars"][0] == "MISSING" and "MOLA" in rows["relief.mars"][1]
    assert any(k.startswith("targets_mars:") and v[0] == "MISSING" for k, v in rows.items())


def test_an_unexplained_unavailable_value_is_caught(tree):
    for doc in ("UNVERIFIED.md", "SPEC.md", "PROVENANCE.md"):
        edit(tree, doc, "silica_index", "xxx_index")
    rows, _ = run()
    assert rows["targets_moon: connecting_ridge_c1.silica_index"][0] == "MISSING"
    assert rows["silica_index (column documented)"][0] == "MISSING"


def test_an_undocumented_earth_column_and_a_source_without_a_dataset_are_caught(tree):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"tile_id": [1], "mystery_feature": [1.0]}), tree / "data" / "processed" / "earth_features_6km.parquet")
    rows, _ = run()
    assert rows["earth_features_6km.mystery_feature"][0] == "MISSING" and rows["earth_features_6km.tile_id"][0] == "ok"


def test_unscored_target_must_be_explained(tree):
    for doc in ("UNVERIFIED.md",):
        edit(tree, doc, "Arcadia", "A-r-c")
        edit(tree, doc, "arcadia", "a-r-c")
        edit(tree, doc, "unscored", "u-n-s-c-o-r-e-d")
        edit(tree, doc, "Unscored", "U-n-s-c-o-r-e-d")
    rows, _ = run()
    assert rows["targets_mars: arcadia.*"][0] == "MISSING"


def test_main_exit_code_and_ci_wiring(tree, capsys):
    assert AP.main([]) == 0
    assert "0 missing" in capsys.readouterr().out
    edit(tree, "PROVENANCE.md", "NASADEM", "XXXX")
    edit(tree, "UNVERIFIED.md", "NASADEM", "XXXX")
    assert AP.main([]) == 1
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
    step = next(s for s in wf["jobs"]["test"]["steps"] if "audit_provenance" in s.get("run", ""))
    assert step["continue-on-error"] is True  # warning-only
