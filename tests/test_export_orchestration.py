import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from gsop.cli import app
from gsop.config import load_config
from gsop.earth import gee_export as G

CFG = load_config(Path(__file__).resolve().parents[1] / "config")


def test_chunks_tile_the_globe_without_gaps_or_overlaps():
    chunks = G.plan_chunks(CFG)
    assert len(chunks) == 12 * 24  # 15 degree blocks
    tile = 6000
    for c in chunks:
        assert c.x0 % tile == c.x1 % tile == c.y0 % tile == c.y1 % tile == 0
        assert c.width_px > 0 and c.height_px > 0
    # neighbours share exactly one edge: sort by row/col and compare
    by = {c.id: c for c in chunks}
    for j in range(12):
        for i in range(23):
            assert by[f"g_r{j:02d}_c{i:02d}"].x1 == by[f"g_r{j:02d}_c{i + 1:02d}"].x0
    for j in range(11):
        for i in range(24):
            assert by[f"g_r{j:02d}_c{i:02d}"].y1 == by[f"g_r{j + 1:02d}_c{i:02d}"].y0
    total_px = sum(c.width_px * c.height_px for c in chunks)
    x0, x1 = min(c.x0 for c in chunks), max(c.x1 for c in chunks)
    y0, y1 = min(c.y0 for c in chunks), max(c.y1 for c in chunks)
    assert total_px == ((x1 - x0) // tile) * ((y1 - y0) // tile)  # an exact partition
    assert 5000 < (x1 - x0) // tile < 6500 and 2000 < (y1 - y0) // tile < 2900  # about 5.8k x 2.4k tiles


def test_region_chunk_and_bad_region():
    (c,) = G.plan_chunks(CFG, [73.0, 16.5, 80.0, 23.5])
    assert c.id.startswith("region") and c.width_px > 0 and c.height_px > 0
    with pytest.raises(ValueError):
        G.plan_chunks(CFG, [80, 10, 70, 20])


def test_expected_size_and_band_list_without_earth_engine():
    bands, skipped = G.planned_bands(CFG)
    assert "slope_med" in bands and "silica_index" in bands and "gate_ndvi_med" in bands
    assert "water_index" not in bands and "water_index" in skipped  # unavailable until Anay decides
    chunks = G.plan_chunks(CFG)
    size = G.expected_size_bytes(chunks, len(bands))
    assert 0.5e9 < size < 4e9  # honest number: float32 uncompressed, about 14M tiles x bands x 4 bytes


def test_dry_run_lists_everything_and_starts_nothing(tmp_path):
    ee = MagicMock()
    lines = []
    report = G.run_export(ee, CFG, tmp_path / "ledger.json", dry_run=True, log=lines.append)
    assert report.dry_run and report.total == 288
    assert sum("dry run: would export" in ln for ln in lines) == 288
    assert any("water_index" in ln and "NOT be exported" in ln for ln in lines)
    ee.batch.Export.image.toDrive.assert_not_called()
    assert not (tmp_path / "ledger.json").exists()


class FakeEE:
    """Earth Engine task API: a task finishes after `polls_to_finish` status calls; some fail on the first attempt."""

    def __init__(self, fail_first=(), always_fail=(), polls_to_finish=2):
        self.fail_first, self.always_fail, self.polls_to_finish = set(fail_first), set(always_fail), polls_to_finish
        self.started, self.status_calls, self._n = [], {}, 0
        self.attempts = {}
        self.batch = SimpleNamespace(Export=SimpleNamespace(image=SimpleNamespace(toDrive=self._to_drive, toCloudStorage=self._to_drive)))
        self.data = SimpleNamespace(getTaskStatus=self._status)
        self.Geometry = SimpleNamespace(Rectangle=lambda *a, **k: ("rect", a))
        self._desc = {}

    def _to_drive(self, **kw):
        self._n += 1
        tid, desc = f"T{self._n}", kw["description"]
        self.attempts[desc] = self.attempts.get(desc, 0) + 1
        self._desc[tid] = (desc, self.attempts[desc])
        assert kw["crsTransform"][0] == 6000 and kw["crs"] == "EPSG:6933" and kw["folder"] == "gsop_earth"
        return SimpleNamespace(id=tid, start=lambda: self.started.append(desc))

    def _status(self, ids):
        tid = ids[0]
        desc, attempt = self._desc[tid]
        self.status_calls[tid] = self.status_calls.get(tid, 0) + 1
        if desc in self.always_fail or (desc in self.fail_first and attempt == 1):
            return [{"state": "FAILED", "error_message": "boom"}]
        return [{"state": "COMPLETED" if self.status_calls[tid] >= self.polls_to_finish else "RUNNING"}]


def image_stub():
    img = MagicMock()
    img.select.return_value.toFloat.return_value = "image"
    return img


def test_run_completes_every_chunk_and_writes_the_ledger(tmp_path):
    ee, ledger = FakeEE(), tmp_path / "ledger.json"
    report = G.run_export(ee, CFG, ledger, region=[73, 16.5, 80, 23.5], image=image_stub(), sleep=lambda s: None, log=lambda m: None)
    assert report.total == 1 and len(report.completed) == 1 and not report.failed
    data = json.loads(ledger.read_text(encoding="utf-8"))["chunks"]
    assert [v["state"] for v in data.values()] == ["COMPLETED"] and [v["attempts"] for v in data.values()] == [1]


def test_failed_chunks_are_retried_then_reported(tmp_path):
    ee = FakeEE(fail_first=["gsop_earth_region_r00_c00"])
    report = G.run_export(ee, CFG, tmp_path / "l.json", region=[73, 16.5, 80, 23.5], image=image_stub(), sleep=lambda s: None, log=lambda m: None)
    assert ee.attempts["gsop_earth_region_r00_c00"] == 2 and report.completed

    ee = FakeEE(always_fail=["gsop_earth_region_r00_c00"])
    report = G.run_export(ee, CFG, tmp_path / "l2.json", region=[73, 16.5, 80, 23.5], image=image_stub(), sleep=lambda s: None, log=lambda m: None)
    assert ee.attempts["gsop_earth_region_r00_c00"] == 1 + CFG.features.earth_export["max_retries"]
    assert report.failed == ["region_r00_c00"] and not report.completed


def test_restart_does_not_duplicate_completed_chunks(tmp_path):
    ledger = tmp_path / "ledger.json"
    region = [73.0, 16.5, 100.0, 23.5]  # one chunk, so make it two with a custom config below
    from dataclasses import replace

    ex = dict(CFG.features.earth_export, chunk_lon_deg=15.0)
    cfg = SimpleNamespace(features=replace(CFG.features, earth_export=ex))
    # two chunks over a 30-degree-wide strip
    region = [60.0, 0.0, 90.0, 15.0]
    ee1 = FakeEE(polls_to_finish=2)
    G.run_export(ee1, cfg, ledger, region=region, image=image_stub(), sleep=lambda s: None, log=lambda m: None, max_polls=1)
    # killed after one poll: tasks are in the ledger, not completed
    state = json.loads(ledger.read_text(encoding="utf-8"))["chunks"]
    assert all(v["task_id"] for v in state.values())
    ee2 = FakeEE(polls_to_finish=1)
    ee2._desc = dict(ee1._desc)  # the same Earth Engine task ids still exist after the restart
    ee2._n = ee1._n
    report = G.run_export(ee2, cfg, ledger, region=region, image=image_stub(), sleep=lambda s: None, log=lambda m: None)
    assert report.started == []  # nothing is re-exported: in-flight tasks were only polled
    assert len(report.completed) == report.total
    ee3 = FakeEE()
    report = G.run_export(ee3, cfg, ledger, region=region, image=image_stub(), sleep=lambda s: None, log=lambda m: None)
    assert ee3.started == [] and report.started == []  # completed chunks are skipped on the next run


def test_cli_dry_run_needs_no_login():
    result = CliRunner().invoke(app, ["s2", "earth", "--source", "gee", "--dry-run", "--region", "73,16.5,80,23.5"])
    assert result.exit_code == 0, result.output
    assert "dry run: would export" in result.output
    bad = CliRunner().invoke(app, ["s2", "earth", "--dry-run", "--region", "1,2,3"])
    assert bad.exit_code == 2
