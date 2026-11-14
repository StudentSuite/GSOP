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
def s1_tiles() -> None:
    """Grid + land mask -> data/interim/tiles.parquet."""
    _stub("s1 tiles")


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
def s2_targets(body: Body = BodyOpt) -> None:
    """Build data/processed/targets_{body}.parquet."""
    _stub("s2 targets")


# --- s3 .. s9 ---------------------------------------------------------------
s3 = typer.Typer(help="s3: gates.", no_args_is_help=True)
app.add_typer(s3, name="s3")


@s3.command("gates")
def s3_gates(body: Body = BodyOpt) -> None:
    """Passed and rejected tables with reasons."""
    _stub("s3 gates")


s4 = typer.Typer(help="s4: base similarity score.", no_args_is_help=True)
app.add_typer(s4, name="s4")


@s4.command("score")
def s4_score(body: Body = BodyOpt) -> None:
    """Base similarity 0-100 per target x profile."""
    _stub("s4 score")


s5 = typer.Typer(help="s5: Monte Carlo.", no_args_is_help=True)
app.add_typer(s5, name="s5")


@s5.command("simulate")
def s5_simulate(body: Body = BodyOpt, runs: Optional[int] = RunsOpt) -> None:
    """Monte Carlo over weights and data."""
    _runs(runs)
    _stub("s5 simulate")


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
def run(body: Body = BodyOpt, runs: Optional[int] = RunsOpt) -> None:
    """Run s3..s9 in order (resumable)."""
    _runs(runs)
    _stub("run")


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
    """Compare Earth Engine and numpy operators on one test window."""
    _stub("check-operators")


if __name__ == "__main__":
    app()
