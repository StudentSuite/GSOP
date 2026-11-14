"""A fresh `gsop golden` agrees with the committed one (issue C30): scores within the tolerance of config/run.yaml, top-1% sets with Jaccard at least the minimum there.

Takes about a minute (both bodies, 50 runs). On a failure the message lists which check failed, where, and by how much.
"""

from gsop import golden as G
from gsop.config import load_config

CFG = load_config(G.REPO_ROOT / "config")


def test_a_fresh_golden_run_matches_the_committed_expectation():
    fresh = G.run_golden(CFG, workers=1, log=lambda m: None)
    fails = G.compare(G.read_golden(), fresh, CFG.run.golden_score_tolerance, CFG.run.golden_top1_jaccard_min)
    assert not fails, "golden differs from tests/golden/golden.json:\n" + "\n".join(f"  {f}" for f in fails[:30])


def test_the_tolerances_come_from_run_yaml_not_from_code():
    run = CFG.run
    assert (run.golden_score_tolerance, run.golden_top1_jaccard_min) == (1.0e-4, 0.99)
    src = (G.REPO_ROOT / "src" / "gsop" / "golden.py").read_text(encoding="utf-8")
    assert "1e-4" not in src and "1.0e-4" not in src and "0.99" not in src  # compare() takes them as arguments
