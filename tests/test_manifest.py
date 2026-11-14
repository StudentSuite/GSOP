import json

import pandas as pd
from typer.testing import CliRunner

from gsop import manifest
from gsop.cli import app
from gsop.hashing import sha256_file

runner = CliRunner()


def make_data(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"a": range(100), "b": range(100)}).to_parquet(processed / "earth_features_6km.parquet")
    (processed / "grid.json").write_text('{"crs": "EPSG:6933"}', encoding="utf-8")
    (processed / ".gitkeep").write_text("", encoding="utf-8")
    return tmp_path / "data"


def test_manifest_lists_every_file_with_size_sha_rows(tmp_path):
    data = make_data(tmp_path)
    path = manifest.write_manifest(data)
    m = json.loads(path.read_text(encoding="utf-8"))
    assert set(m["files"]) == {"earth_features_6km.parquet", "grid.json"}
    entry = m["files"]["earth_features_6km.parquet"]
    assert entry["rows"] == 100
    assert entry["size"] == (data / "processed" / "earth_features_6km.parquet").stat().st_size
    assert entry["sha256"] == sha256_file(data / "processed" / "earth_features_6km.parquet")
    assert entry["created"]
    assert m["files"]["grid.json"]["rows"] is None


def test_verify_passes_on_intact_data(tmp_path):
    data = make_data(tmp_path)
    manifest.write_manifest(data)
    result = manifest.verify_data(data)
    assert result.ok and result.checked == 2


def test_one_corrupt_byte_fails_and_names_the_file(tmp_path):
    data = make_data(tmp_path)
    manifest.write_manifest(data)
    target = data / "processed" / "earth_features_6km.parquet"
    raw = bytearray(target.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    target.write_bytes(bytes(raw))
    result = manifest.verify_data(data)
    assert not result.ok
    assert [p.file for p in result.problems] == ["earth_features_6km.parquet"]
    assert result.problems[0].kind == "sha256"

    cli = runner.invoke(app, ["verify-data", "--data-dir", str(data)])
    assert cli.exit_code == 1
    assert "earth_features_6km.parquet" in cli.output


def test_missing_file_and_missing_manifest(tmp_path):
    data = make_data(tmp_path)
    assert not manifest.verify_data(data).ok  # no manifest yet
    manifest.write_manifest(data)
    (data / "processed" / "grid.json").unlink()
    result = manifest.verify_data(data)
    assert [(p.file, p.kind) for p in result.problems] == [("grid.json", "missing")]


def test_manifest_inside_processed_is_found_and_not_self_listed(tmp_path):
    data = make_data(tmp_path)
    path = manifest.write_manifest(data)
    moved = data / "processed" / "MANIFEST.json"
    path.replace(moved)
    assert manifest.find_manifest(data) == moved
    assert manifest.verify_data(data).ok
    again = manifest.build_manifest(data / "processed")
    assert "MANIFEST.json" not in again["files"]


def test_cli_ok_and_extra_file_note(tmp_path):
    data = make_data(tmp_path)
    manifest.write_manifest(data)
    (data / "processed" / "extra.csv").write_text("x", encoding="utf-8")
    cli = runner.invoke(app, ["verify-data", "--data-dir", str(data)])
    assert cli.exit_code == 0
    assert "extra.csv" in cli.output


def test_hashing_is_chunked_and_matches_hashlib(tmp_path):
    import hashlib

    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * 100_000)
    assert sha256_file(f, chunk_bytes=777) == hashlib.sha256(b"x" * 100_000).hexdigest()
