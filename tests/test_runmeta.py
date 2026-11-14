import json
from pathlib import Path

from gsop import runmeta
from gsop.atomic import atomic_write_text
from gsop.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
REQUIRED = {
    "schema_version", "gsop_version", "body", "git", "config_hash", "config_file_hashes", "packages",
    "machine", "seeds", "inputs", "stage_timings", "written_at",
}


def test_all_fields_present_and_no_missing_values(tmp_path):
    cfg = load_config(CONFIG_DIR)
    meta = runmeta.build_run_meta(cfg, body="moon", stage_timings={"s3": 1.5})
    assert REQUIRED <= set(meta)
    assert meta["seeds"]["master_seed"] == 20261114
    assert meta["seeds"]["runs"] == 2000
    assert set(meta["git"]) == {"commit", "dirty", "source"}
    assert meta["git"]["commit"]
    assert set(meta["config_file_hashes"]) == {
        "weights", "gates", "purposes", "targets", "known_sites", "ledger", "features", "run",
    }
    for key in ("os", "python", "cores_physical", "cores_logical", "ram_total_gb"):
        assert meta["machine"][key] not in (None, "")
    assert "numpy" in meta["packages"]
    json.dumps(meta)  # serialisable


def test_identical_inputs_give_identical_hashes_and_seeds_but_timings_differ():
    cfg = load_config(CONFIG_DIR)
    a = runmeta.build_run_meta(cfg, "moon", stage_timings={"s3": 1.0})
    b = runmeta.build_run_meta(cfg, "moon", stage_timings={"s3": 9.0})
    assert runmeta.stable_part(a) == runmeta.stable_part(b)
    assert a["stage_timings"] != b["stage_timings"]


def test_seed_fingerprint_depends_on_master_seed_only_through_the_seed():
    assert runmeta.seed_fingerprint(1, 100) == runmeta.seed_fingerprint(1, 100)
    assert runmeta.seed_fingerprint(1, 100) != runmeta.seed_fingerprint(2, 100)


def test_input_hashes_come_from_manifest(tmp_path):
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"files": {"a.parquet": {"sha256": "ab" * 32}}}), encoding="utf-8")
    got = runmeta.input_hashes(manifest)
    assert got["files"] == {"a.parquet": "ab" * 32}
    assert len(got["manifest"]) == 64
    assert runmeta.input_hashes(tmp_path / "nope.json") == {"manifest": None, "files": {}}


def test_record_stage_merges_timings_and_writes_file(tmp_path):
    cfg = load_config(CONFIG_DIR)
    out = tmp_path / "results" / "moon"
    runmeta.record_stage(out, cfg, "s3", 1.25, body="moon")
    runmeta.record_stage(out, cfg, "s4", 2.5, body="moon")
    data = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert data["stage_timings"] == {"s3": 1.25, "s4": 2.5}
    assert REQUIRED <= set(data)


def test_git_info_falls_back_to_stamp_file(tmp_path):
    (tmp_path / "GIT_COMMIT").write_text("abc123\n", encoding="utf-8")
    assert runmeta.git_info(tmp_path) == {"commit": "abc123", "dirty": None, "source": "GIT_COMMIT file"}
    empty = tmp_path / "empty"
    empty.mkdir()
    assert runmeta.git_info(empty)["commit"] == "unknown"


def test_atomic_write_leaves_no_temp_files_and_replaces(tmp_path):
    target = tmp_path / "x.txt"
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
    assert [p.name for p in tmp_path.iterdir()] == ["x.txt"]
