"""Single command-line entry point: ``uv run gsop ...``.

Every stage is a stub for now. A stub prints ``not implemented yet`` and exits
with code 2. Later issues replace the body of one command at a time, each in
its own stage module, so pull requests touch different files.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="gsop",
    help="Geospatial Similarity Observation Platform.",
    no_args_is_help=True,
    add_completion=False,
)

NOT_IMPLEMENTED_EXIT_CODE = 2


class Body(str, Enum):
    moon = "moon"
    mars = "mars"


class EarthSource(str, Enum):
    gee = "gee"
    local = "local"


def _default_runs() -> int:
    """Default Monte Carlo runs: ``runs`` in config/run.yaml (decision D2: 2000)."""
    from gsop.config import load_config

    return load_config().run.runs


def _stub(name: str) -> None:
    typer.echo(f"gsop {name}: not implemented yet")
    raise typer.Exit(NOT_IMPLEMENTED_EXIT_CODE)


BodyOpt = typer.Option(..., "--body", help="Target body: moon or mars.")
TargetCheckOpt = typer.Option(
    None, "--target-check", help="Mars target check (decision D6): warn (default, the failing target is still scored and flagged) or strict (not scored)."
)
RunsOpt = typer.Option(
    None, "--runs", help="Monte Carlo runs (default: runs in config/run.yaml)."
)


def _runs(runs: Optional[int]) -> int:
    return _default_runs() if runs is None else runs


@app.command()
def doctor(
    json_path: Optional[Path] = typer.Option(
        None, "--json", help="Also write the report to this JSON file."
    ),
) -> None:
    """Report OS, cores, RAM, disk, package versions and the safe chunk size."""
    from gsop.doctor import run_doctor

    run_doctor(json_path)


# --- s1 -------------------------------------------------------------------
s1 = typer.Typer(help="s1: grid and land mask.", no_args_is_help=True)
app.add_typer(s1, name="s1")


@s1.command("tiles")
def s1_tiles(force: bool = typer.Option(False, "--force", help="Rerun even if nothing changed.")) -> None:
    """Grid + land mask -> data/interim/tiles.parquet (skipped by the runner when inputs and config are unchanged)."""
    from gsop.config import load_config
    from gsop.runner import Runner, RunnerError
    from gsop.stages import s1

    root = Path(__file__).resolve().parents[2] / "data"
    cfg = load_config()
    earth = root / "processed" / "earth_features_6km.parquet"
    stage = s1.stage(cfg, earth, root / "interim" / "tiles.parquet", log=typer.echo)
    if not earth.is_file():
        typer.echo(f"{earth} does not exist: build the Earth file first (`gsop s2 earth`, issues B16 to B18)", err=True)
        raise typer.Exit(1)
    try:
        Runner(root / "interim", cfg, base_dir=root, log=typer.echo, write_meta=False).run([stage], force=["s1"] if force else ())
    except (s1.TilesError, RunnerError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)


# --- s2 -------------------------------------------------------------------
s2 = typer.Typer(help="s2: Earth features and target profiles.", no_args_is_help=True)
app.add_typer(s2, name="s2")


@s2.command("earth")
def s2_earth(
    source: EarthSource = typer.Option(
        EarthSource.gee, "--source", help="Where Earth features come from."
    ),
    region: Optional[str] = typer.Option(
        None, "--region", help="Test region 'west,south,east,north' in degrees instead of the whole globe."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="List chunks and bands without starting any export."),
    local_dir: Optional[Path] = typer.Option(None, "--local-dir", help="With --source local: folder holding local_sources.yaml and the rasters."),
    build_from: Optional[Path] = typer.Option(
        None, "--build-from", help="Folder of downloaded chunk GeoTIFFs: build data/processed/earth_features_6km.parquet and grid.json from them."
    ),
) -> None:
    """Build data/processed/earth_features_6km.parquet (Earth Engine route: start/poll the chunk exports)."""
    from gsop.config import load_config
    from gsop.earth import gee_export

    cfg = load_config()
    if source is EarthSource.local:
        if local_dir is None:
            typer.echo("--source local needs --local-dir with local_sources.yaml and your rasters (see docs/SPEC.md)", err=True)
            raise typer.Exit(2)
        from gsop.earth import build_parquet, local_loaders

        root = Path(__file__).resolve().parents[2] / "data"
        try:
            report = local_loaders.build_local_chunk(local_dir, cfg, root / "interim" / "local")
        except local_loaders.LocalSourceError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2)
        typer.echo("LOCAL ROUTE: coarser than Earth Engine. Unavailable bands:")
        for name, why in report["unavailable"].items():
            typer.echo(f"  {name}: {why}")
        build_parquet.build_earth_parquet([root / "interim" / "local" / "local_chunk.tif"], cfg, root / "processed", log=typer.echo)
        return
    if build_from is not None:
        from gsop.earth import build_parquet

        rasters = sorted(Path(build_from).glob("*.tif"))
        if not rasters:
            typer.echo(f"no .tif files in {build_from}", err=True)
            raise typer.Exit(1)
        out = Path(__file__).resolve().parents[2] / "data" / "processed"
        build_parquet.build_earth_parquet(rasters, cfg, out, log=typer.echo)
        typer.echo(f"round-trip check: {len(build_parquet.roundtrip_check(out / 'earth_features_6km.parquet', rasters, cfg))} random tiles match their rasters")
        return
    box = None
    if region is not None:
        try:
            box = [float(v) for v in region.split(",")]
            assert len(box) == 4
        except (ValueError, AssertionError):
            typer.echo("--region must be 'west,south,east,north' in degrees", err=True)
            raise typer.Exit(2)
    ledger = Path(__file__).resolve().parents[2] / "data" / "interim" / ("ee_export_ledger_region.json" if box else "ee_export_ledger.json")
    ee = None
    if not dry_run:
        from gsop.earth.gee_auth import ensure_ee

        ee = ensure_ee(cfg=cfg)
    report = gee_export.run_export(ee, cfg, ledger, region=box, dry_run=dry_run)
    if not dry_run and report.failed:
        typer.echo(f"{len(report.failed)} chunk(s) failed after retries: {', '.join(report.failed)}", err=True)
        raise typer.Exit(1)


@s2.command("targets")
def s2_targets(
    body: Body = BodyOpt,
    diviner_dir: Optional[Path] = typer.Option(
        None, "--diviner-dir", help="Moon only: folder with the 144 downloaded Diviner polar seasonal .tab files (otherwise those rows are unavailable)."
    ),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Where to write targets_{body}.parquet/.md (default data/processed)."),
) -> None:
    """Build data/processed/targets_{body}.parquet and targets_{body}.md (windowed reads over HTTP, cached in data/raw/)."""
    from gsop.config import load_config
    from gsop.targets import TargetError, build_targets

    root = Path(__file__).resolve().parents[2] / "data"
    cfg = load_config()
    try:
        target_dir = out_dir or root / "processed"
        df = build_targets(cfg, body.value, target_dir, cache_dir=root / "raw" / "targets_cache", diviner_dir=diviner_dir)
    except TargetError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo(f"wrote {target_dir / f'targets_{body.value}.parquet'} ({len(df)} rows, {int(df.unavailable.sum())} unavailable) and the .md summary")


# --- s3 .. s9 ---------------------------------------------------------------
s3 = typer.Typer(help="s3: gates.", no_args_is_help=True)
app.add_typer(s3, name="s3")


@s3.command("gates")
def s3_gates(body: Body = BodyOpt, target_check: Optional[str] = TargetCheckOpt,
             force: bool = typer.Option(False, "--force", help="Rerun even if nothing changed.")) -> None:
    """Passed and rejected tables with reasons (skipped by the runner when inputs and config are unchanged)."""
    from gsop.config import load_config
    from gsop.paths import Paths
    from gsop.runner import Runner, RunnerError
    from gsop.similarity import SetupError
    from gsop.stages import s3

    cfg, paths = load_config(), Paths()
    try:
        stage = s3.stage(cfg, body.value, paths, target_check, log=typer.echo)
        Runner(paths.results(body.value), cfg, body=body.value, base_dir=paths.root, log=typer.echo).run([stage], force=["s3"] if force else ())
    except (SetupError, RunnerError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)


s4 = typer.Typer(help="s4: base similarity score.", no_args_is_help=True)
app.add_typer(s4, name="s4")


@s4.command("score")
def s4_score(body: Body = BodyOpt, target_check: Optional[str] = TargetCheckOpt,
             force: bool = typer.Option(False, "--force", help="Rerun even if nothing changed.")) -> None:
    """Base similarity 0-100 per target x profile (skipped by the runner when inputs and config are unchanged)."""
    from gsop.config import load_config
    from gsop.paths import Paths
    from gsop.runner import Runner, RunnerError
    from gsop.similarity import SetupError
    from gsop.stages import s4

    cfg, paths = load_config(), Paths()
    try:
        stage = s4.stage(cfg, body.value, paths, target_check, log=typer.echo)
        Runner(paths.results(body.value), cfg, body=body.value, base_dir=paths.root, log=typer.echo).run([stage], force=["s4"] if force else ())
    except (SetupError, RunnerError, s4.ScoreError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)


s5 = typer.Typer(help="s5: Monte Carlo.", no_args_is_help=True)
app.add_typer(s5, name="s5")


@s5.command("simulate")
def s5_simulate(body: Body = BodyOpt, runs: Optional[int] = RunsOpt, target_check: Optional[str] = TargetCheckOpt,
                workers: Optional[int] = typer.Option(None, "--workers", help="Worker processes (default: physical cores). The result does not depend on it."),
                force: bool = typer.Option(False, "--force", help="Rerun even if nothing changed.")) -> None:
    """Monte Carlo over weights and data for every target x profile (resumable: per-profile markers and block-level partial results)."""
    from gsop.config import load_config
    from gsop.paths import Paths
    from gsop.runner import Runner, RunnerError
    from gsop.similarity import SetupError
    from gsop.stages import s4, s5

    cfg, paths = load_config(), Paths()
    try:
        stage = s5.stage(cfg, body.value, paths, _runs(runs), workers, target_check, log=typer.echo)
        Runner(paths.results(body.value), cfg, body=body.value, runs=_runs(runs), base_dir=paths.root, log=typer.echo).run([stage], force=["s5"] if force else ())
    except (SetupError, RunnerError, s4.ScoreError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)


s6 = typer.Typer(help="s6: ML second opinion.", no_args_is_help=True)
app.add_typer(s6, name="s6")


@s6.command("ml")
def s6_ml(body: Body = BodyOpt) -> None:
    """Archetypes and classifier (never produces the score)."""
    _stub("s6 ml")


s7 = typer.Typer(help="s7: validation.", no_args_is_help=True)
app.add_typer(s7, name="s7")


@s7.command("validate")
def s7_validate(body: Body = BodyOpt) -> None:
    """Validate against known analog sites."""
    _stub("s7 validate")


s8 = typer.Typer(help="s8: export.", no_args_is_help=True)
app.add_typer(s8, name="s8")


@s8.command("export")
def s8_export(body: Body = BodyOpt) -> None:
    """Export parquet/JSON per view for the site."""
    _stub("s8 export")


s9 = typer.Typer(help="s9: discovery.", no_args_is_help=True)
app.add_typer(s9, name="s9")


@s9.command("discover")
def s9_discover(body: Body = BodyOpt) -> None:
    """Robust and novel candidate list."""
    _stub("s9 discover")


# --- whole-pipeline and utility commands ------------------------------------
@app.command()
def run(body: Body = BodyOpt, runs: Optional[int] = RunsOpt, target_check: Optional[str] = TargetCheckOpt) -> None:
    """Run s3..s9 in order (resumable)."""
    _runs(runs)
    _stub("run")


@app.command()
def ledger(body: Body = BodyOpt) -> None:
    """Write fidelity.json (limits ledger and Fidelity 0-100 per target x purpose; never used in the score)."""
    from gsop.ledger import write_fidelity

    cfg, paths = load_config(), Paths()
    path = write_fidelity(cfg, body.value, paths.results(body.value))
    typer.echo(f"wrote {path}")


@app.command()
def golden() -> None:
    """Run the tiny fixed slice end to end and write golden.json."""
    _stub("golden")


@app.command("verify-data")
def verify_data(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Folder holding processed/ and MANIFEST.json."),
) -> None:
    """Check sha256 of data/processed files against MANIFEST.json."""
    from gsop import manifest

    result = manifest.verify_data(data_dir or manifest.DEFAULT_DATA_DIR)
    for problem in result.problems:
        typer.echo(f"FAIL {problem}", err=True)
    for name in result.extra_files:
        typer.echo(f"note: {name} is in data/processed but not in the manifest")
    if not result.ok:
        typer.echo(f"verify-data FAILED ({len(result.problems)} problem(s))", err=True)
        raise typer.Exit(1)
    typer.echo(f"verify-data OK: {result.checked} file(s) match {result.manifest}")


@app.command()
def compare(dir_a: Path, dir_b: Path) -> None:
    """Compare two result folders with tolerances."""
    _stub("compare")


@app.command()
def merge(results_moon: Path, results_mars: Path) -> None:
    """Merge Moon and Mars results into results/all/."""
    _stub("merge")


@app.command("check-operators")
def check_operators() -> None:
    """Compare Earth Engine and numpy operators on one test window (writes docs/OPERATOR_CHECK.md; exit 1 if > 10% apart)."""
    from gsop.config import load_config
    from gsop.earth import check_operators as co
    from gsop.earth.gee_auth import ensure_ee

    cfg = load_config()
    ee = ensure_ee(cfg=cfg)
    result = co.run_check(ee, cfg, Path(__file__).resolve().parents[2] / "docs" / "OPERATOR_CHECK.md", log=typer.echo)
    if not result.passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
