import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from gsop.atomic import atomic_write_text
from gsop.config import load_config
from gsop.runner import Runner, RunnerError, Stage

FIXTURE = Path(__file__).parent / "fixtures" / "config"


def make_stages(root: Path, calls: list, fail_in: str = ""):
    """s3 -> s4 -> s5 chain. s3 depends on weights, s4 on gates, s5 on run config."""
    a, b, c = root / "a.txt", root / "b.txt", root / "c.txt"

    def s3(ctx):
        calls.append("s3")
        atomic_write_text(a, "A")

    def s4(ctx):
        calls.append("s4")
        if fail_in == "s4":
            raise KeyboardInterrupt("simulated kill")
        atomic_write_text(b, a.read_text() + "B")

    def s5(ctx):
        calls.append("s5")
        atomic_write_text(c, b.read_text() + "C")

    return [
        Stage("s3", s3, [], [a], ["weights"], "v1"),
        Stage("s4", s4, [a], [b], ["gates"], "v1"),
        Stage("s5", s5, [b], [c], ["run"], "v1"),
    ]


@pytest.fixture
def setup(tmp_path):
    cfg_dir = tmp_path / "config"
    shutil.copytree(FIXTURE, cfg_dir)
    root = tmp_path / "work"
    root.mkdir()
    runner = lambda cfg: Runner(root / "results", cfg, base_dir=root, log=lambda m: None)  # noqa: E731
    return cfg_dir, root, runner


def names(results, action):
    return [r.name for r in results if r.action == action]


def test_second_run_skips_everything(setup):
    cfg_dir, root, runner = setup
    cfg, calls = load_config(cfg_dir), []
    first = runner(cfg).run(make_stages(root, calls))
    assert names(first, "ran") == ["s3", "s4", "s5"]
    second = runner(cfg).run(make_stages(root, calls))
    assert names(second, "skipped") == ["s3", "s4", "s5"]
    assert calls == ["s3", "s4", "s5"]
    assert all(r.reason for r in second)


def test_interrupted_stage_resumes_and_earlier_stages_are_skipped(setup):
    cfg_dir, root, runner = setup
    cfg, calls = load_config(cfg_dir), []
    with pytest.raises(KeyboardInterrupt):
        runner(cfg).run(make_stages(root, calls, fail_in="s4"))
    calls.clear()
    results = runner(cfg).run(make_stages(root, calls))
    assert names(results, "skipped") == ["s3"]
    assert names(results, "ran") == ["s4", "s5"]
    assert calls == ["s4", "s5"]


def test_changing_one_config_value_reruns_exactly_the_dependent_stages(setup):
    cfg_dir, root, runner = setup
    calls = []
    runner(load_config(cfg_dir)).run(make_stages(root, calls))
    run_yaml = cfg_dir / "run.yaml"
    run_yaml.write_text(run_yaml.read_text().replace("runs: 2000", "runs: 77"), encoding="utf-8")
    calls.clear()
    results = runner(load_config(cfg_dir)).run(make_stages(root, calls))
    assert names(results, "ran") == ["s5"]  # only the stage that reads run.yaml
    assert names(results, "skipped") == ["s3", "s4"]


def test_changed_upstream_output_propagates_by_hash(setup):
    cfg_dir, root, runner = setup
    cfg, calls = load_config(cfg_dir), []
    runner(cfg).run(make_stages(root, calls))
    w = cfg_dir / "weights.yaml"
    w.write_text(w.read_text().replace("paper_score: 16", "paper_score: 17", 1), encoding="utf-8")
    # s3 reruns (weights changed) but writes identical content, so s4/s5 stay skipped.
    calls.clear()
    results = runner(load_config(cfg_dir)).run(make_stages(root, calls))
    assert names(results, "ran") == ["s3"]
    # If s3's output really changes, everything downstream reruns.
    a = root / "a.txt"
    stages = make_stages(root, calls)

    def s3_new(ctx):
        calls.append("s3")
        atomic_write_text(a, "A2")

    stages[0] = Stage("s3", s3_new, [], [a], ["weights"], "v2")
    calls.clear()
    results = runner(load_config(cfg_dir)).run(stages)
    assert names(results, "ran") == ["s3", "s4", "s5"]


def test_missing_output_triggers_rerun(setup):
    cfg_dir, root, runner = setup
    cfg, calls = load_config(cfg_dir), []
    runner(cfg).run(make_stages(root, calls))
    (root / "b.txt").unlink()
    calls.clear()
    results = runner(cfg).run(make_stages(root, calls))
    assert names(results, "ran") == ["s4"]
    assert "output missing" in [r for r in results if r.name == "s4"][0].reason


def test_force_and_from(setup):
    cfg_dir, root, runner = setup
    cfg, calls = load_config(cfg_dir), []
    runner(cfg).run(make_stages(root, calls))
    assert names(runner(cfg).run(make_stages(root, calls), force=["s4"]), "ran") == ["s4"]
    assert names(runner(cfg).run(make_stages(root, calls), from_stage="s4"), "ran") == ["s4", "s5"]
    with pytest.raises(RunnerError, match="unknown stage"):
        runner(cfg).run(make_stages(root, calls), force=["s9"])


def test_missing_input_and_missing_output_are_errors(setup):
    cfg_dir, root, runner = setup
    cfg = load_config(cfg_dir)
    bad_in = Stage("sx", lambda ctx: None, [root / "nope"], [], [], "v1")
    with pytest.raises(RunnerError, match="missing input"):
        runner(cfg).run([bad_in])
    bad_out = Stage("sy", lambda ctx: None, [], [root / "never.txt"], [], "v1")
    with pytest.raises(RunnerError, match="did not write"):
        runner(cfg).run([bad_out])


def test_run_meta_written_after_each_stage(setup):
    cfg_dir, root, runner = setup
    runner(load_config(cfg_dir)).run(make_stages(root, []))
    import json

    meta = json.loads((root / "results" / "run_meta.json").read_text(encoding="utf-8"))
    assert set(meta["stage_timings"]) == {"s3", "s4", "s5"}


def test_hard_kill_mid_stage_leaves_no_marker_and_resumes(setup, tmp_path):
    """A real process death (os._exit) in the middle of a stage."""
    cfg_dir, root, runner = setup
    script = tmp_path / "crash.py"
    script.write_text(textwrap.dedent(f"""
        import os, sys
        from pathlib import Path
        from gsop.atomic import atomic_write_text
        from gsop.config import load_config
        from gsop.runner import Runner, Stage

        root = Path({str(root)!r})
        a = root / "a.txt"; b = root / "b.txt"
        def s3(ctx): atomic_write_text(a, "A")
        def s4(ctx):
            (root / "b.txt.partial").write_text("half")
            os._exit(3)
        stages = [Stage("s3", s3, [], [a], [], "v1"), Stage("s4", s4, [a], [b], [], "v1")]
        Runner(root / "results", None, base_dir=root, log=lambda m: None).run(stages)
    """), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr
    assert (root / "results" / ".stages" / "s3.json").exists()
    assert not (root / "results" / ".stages" / "s4.json").exists()
    calls = []

    def s3(ctx):
        calls.append("s3")

    def s4(ctx):
        calls.append("s4")
        atomic_write_text(root / "b.txt", "B")

    stages = [Stage("s3", s3, [], [root / "a.txt"], [], "v1"), Stage("s4", s4, [root / "a.txt"], [root / "b.txt"], [], "v1")]
    results = Runner(root / "results", None, base_dir=root, log=lambda m: None).run(stages)
    assert calls == ["s4"]
    assert names(results, "skipped") == ["s3"]
