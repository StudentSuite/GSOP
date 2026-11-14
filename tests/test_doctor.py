import json

from gsop import doctor


def test_chunk_cells_is_within_budget():
    free = 8 * 10**9
    cells = doctor.chunk_cells(free, n_columns=15, fraction=0.4)
    assert cells * 15 * 4 <= 0.4 * free
    assert (cells + 1) * 15 * 4 > 0.4 * free


def test_collect_has_all_fields_without_earth_engine_auth():
    info = doctor.collect()
    for key in ("os", "python", "cores_logical", "cores_physical", "ram_total_gb",
                "ram_free_gb", "disk_free_gb", "packages", "earthengine", "chunk"):
        assert key in info
    assert info["chunk"]["cells"] > 0
    assert "authenticated" in info["earthengine"]
    assert "Chunk size" in doctor.render(info)


def test_json_output(tmp_path):
    out = tmp_path / "doctor.json"
    doctor.run_doctor(out)
    assert json.loads(out.read_text(encoding="utf-8"))["chunk"]["cells"] > 0
