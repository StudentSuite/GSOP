from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_workflow_matrix_and_steps():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
    job = wf["jobs"]["test"]
    assert job["strategy"]["matrix"]["os"] == ["windows-latest", "macos-14", "ubuntu-latest"]
    steps = job["steps"]
    runs = [s.get("run", "") for s in steps]
    assert "uv lock --check" in runs
    assert "uv sync --frozen" in runs
    assert "uv run pytest -q" in runs
    uv_step = next(s for s in steps if str(s.get("uses", "")).startswith("astral-sh/setup-uv"))
    assert uv_step["with"]["enable-cache"] is True


def test_readme_has_the_badge():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "actions/workflows/test.yml/badge.svg" in readme


def test_repo_hygiene_files():
    pr = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    for needle in ("Closes #", "uv run pytest -q", "docs/SPEC.md", "docs/UNVERIFIED.md", "Windows and macOS", "uv.lock"):
        assert needle in pr, needle
    templates = sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.md"))
    assert {t.name for t in templates} == {"task.md", "bug.md"}
    for t in templates:
        text = t.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        front = yaml.safe_load(text.split("---")[1])
        assert front["name"] and front["about"]
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for needle in ("Never push to `main`", "<phase>-<key>-<slug>", "tag Anay", "branches/main/protection"):
        assert needle in contributing, needle
    assert "CONTRIBUTING.md" in (ROOT / "README.md").read_text(encoding="utf-8")
