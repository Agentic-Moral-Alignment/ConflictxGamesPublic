"""Live smoke test for the balanced generate/filter/top-up loop.

Runs :mod:`conflictxgame.games.generate_balanced` against the real gateway at
the smallest useful scale (one accepted scenario per value pair) and checks the
mechanics: the loop terminates, the per-pair coverage report is written with
the right schema, resume recounts from the CSV, and ``--status`` reports
progress without spending any calls.

Whether the quota is actually *met* is asserted softly. The judge is
stochastic, and a tiny corpus can legitimately have a pair with nothing kept
yet; failing the suite for that would report data noise as breakage — the same
reasoning behind the ``kept_scenarios`` fixture in ``conftest``.
"""

import json

import pandas as pd
import pytest

from conftest import GEN_MODEL, GEN_SHORT, JUDGE_MODEL, VALUE_SET, run_cli

pytestmark = pytest.mark.live

#: Accepted scenarios per value pair. 1 is the smallest target that still
#: exercises the deficit-aware sampler and the top-up loop.
SMOKE_KEPT_PER_PAIR = 1


@pytest.fixture(scope="module")
def balanced_run(artifacts):
    """Run the balanced loop once into its own artifact directory."""
    base = artifacts / "balanced" / VALUE_SET
    games_dir = base / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    result = run_cli(
        "conflictxgame.games.generate_balanced",
        [
            "-v", VALUE_SET,
            "-m", GEN_MODEL,
            "--judge-model", JUDGE_MODEL,
            "-o", str(games_dir),
            "--csv-dir", str(base),
            "-k", str(SMOKE_KEPT_PER_PAIR),
            "--round-batch", "3",
            "--max-rounds", "4",
            "--max-stall-rounds", "4",
            "--assistant-seat-prob", "1.0",
            "--max-tokens", "8192",
        ],
        stage_index=2,
        timeout=3600,
    )
    assert result.returncode == 0, "generate_balanced CLI failed"
    return {
        "base": base,
        "games_dir": games_dir,
        "csv": base / f"{GEN_SHORT}_games.csv",
        "pair_coverage": games_dir / "pair_coverage.json",
        "stdout": result.stdout or "",
    }


def test_corpus_and_reports_are_written(balanced_run):
    assert balanced_run["csv"].exists(), "no filtered corpus CSV"
    assert balanced_run["pair_coverage"].exists(), "no pair_coverage.json"
    assert (balanced_run["games_dir"] / "pair_coverage.csv").exists()
    assert (balanced_run["games_dir"] / "coverage_report.json").exists()


def test_pair_coverage_schema(balanced_run):
    with open(balanced_run["pair_coverage"], encoding="utf-8") as f:
        coverage = json.load(f)

    assert coverage["target_per_pair"] == SMOKE_KEPT_PER_PAIR
    assert isinstance(coverage["complete"], bool)
    # HHH has three unordered value pairs, and every one must be tracked even
    # if nothing was kept for it yet.
    assert len(coverage["pairs"]) == 3
    for row in coverage["pairs"]:
        assert {"pair", "value1", "value2", "target", "generated", "kept", "deficit"} <= set(row)
        assert row["kept"] <= row["generated"]
        assert row["deficit"] == max(0, row["target"] - row["kept"])


def test_recount_matches_the_report(balanced_run):
    """The CSV is the resume state, so it must agree with what was reported."""
    from conflictxgame.games.pair_quota import kept_pair_counts_from_csv, pair_label

    with open(balanced_run["pair_coverage"], encoding="utf-8") as f:
        reported = {row["pair"]: row["kept"] for row in json.load(f)["pairs"]}
    kept, _ = kept_pair_counts_from_csv(str(balanced_run["csv"]))
    for pair, count in kept.items():
        assert reported.get(pair_label(pair), 0) == count


def test_every_kept_scenario_has_a_tracked_value_pair(balanced_run):
    """A kept 2x2 game must carry exactly two distinct tags per seat."""
    from conflictxgame.games.pair_quota import canonical_pair

    frame = pd.read_csv(balanced_run["csv"])
    kept = frame[frame["keep_scenario"].astype(str).str.lower() == "true"]
    if kept.empty:
        pytest.skip("judge kept nothing in this run; nothing to check")
    for _, row in kept.iterrows():
        for seat_tags in json.loads(row["value_tags"]):
            assert canonical_pair(seat_tags) is not None


def test_status_reports_without_spending_calls(balanced_run):
    result = run_cli(
        "conflictxgame.games.generate_balanced",
        [
            "-v", VALUE_SET,
            "-m", GEN_MODEL,
            "-o", str(balanced_run["games_dir"]),
            "--csv-dir", str(balanced_run["base"]),
            "-k", str(SMOKE_KEPT_PER_PAIR),
            "--status",
        ],
        stage_index=0,
        timeout=120,
    )
    assert result.returncode == 0
    assert "Pair quota progress" in result.stdout


def test_resume_is_idempotent_once_the_quota_is_met(balanced_run):
    """Re-running a met quota must not generate anything new."""
    with open(balanced_run["pair_coverage"], encoding="utf-8") as f:
        if not json.load(f)["complete"]:
            pytest.skip("quota not met in this run; resume would legitimately generate")

    before = len(pd.read_csv(balanced_run["csv"]))
    result = run_cli(
        "conflictxgame.games.generate_balanced",
        [
            "-v", VALUE_SET,
            "-m", GEN_MODEL,
            "--judge-model", JUDGE_MODEL,
            "-o", str(balanced_run["games_dir"]),
            "--csv-dir", str(balanced_run["base"]),
            "-k", str(SMOKE_KEPT_PER_PAIR),
            "--round-batch", "3",
            "--max-rounds", "2",
        ],
        stage_index=1,
        timeout=600,
    )
    assert result.returncode == 0
    assert len(pd.read_csv(balanced_run["csv"])) == before


def test_quota_progress_is_reported(balanced_run):
    """Soft check: report the outcome rather than failing on judge stochasticity."""
    with open(balanced_run["pair_coverage"], encoding="utf-8") as f:
        coverage = json.load(f)
    short = [row["pair"] for row in coverage["pairs"] if row["deficit"] > 0]
    if short:
        print(
            f"\nNote: quota not met for {short} within the round cap "
            f"(acceptance rate {coverage['overall_acceptance_rate']:.2f}). "
            "Expected at smoke scale; not a code failure."
        )
    assert coverage["overall_acceptance_rate"] >= 0.0


def test_cost_block_reports_spend_per_accepted_scenario(balanced_run):
    """Cost per accepted scenario is what says whether a mechanism paid off.

    An acceptance rate can be raised by attempting fewer, safer scenarios;
    calls per kept cannot be moved that way, so it is the number to compare
    arms on. The ledger counts real gateway requests here, so this is also the
    check that it agrees with what the dashboard bills.
    """
    with open(balanced_run["pair_coverage"], encoding="utf-8") as f:
        cost = json.load(f)["cost"]

    assert cost["by_role"]["generation"]["requests"] > 0
    assert cost["by_role"]["judge"]["requests"] > 0
    # Both denominators, because the pre-screen moves them apart.
    assert "kept_per_filtered_scenario" in cost
    assert "kept_per_attempted_outline" in cost
    print(
        f"\ncost: {cost['requests_total']} request(s) for {cost['kept_this_run']} "
        f"accepted scenario(s); "
        f"{cost['by_role']['judge']['requests']} judge, "
        f"{cost['by_role']['generation']['requests']} generation"
    )
    if cost.get("requests_per_kept"):
        print(f"      {cost['requests_per_kept']:.1f} call(s) per accepted scenario")


def test_run_records_which_mechanisms_produced_the_corpus(balanced_run):
    """An appended corpus has to stay self-describing across default changes."""
    with open(balanced_run["pair_coverage"], encoding="utf-8") as f:
        config = json.load(f)["config"]
    assert config["prescreen"] in ("off", "shadow", "enforce")
    assert isinstance(config["guidance"], list)

    frame = pd.read_csv(balanced_run["csv"])
    spec = json.loads(frame.iloc[0]["game_spec"])
    generation = spec["metadata"]["generation"]
    assert generation["judge_model"]
    # The fields scenario_id and backlog recovery depend on must survive.
    assert spec["metadata"]["generating_model"]
    assert spec["metadata"]["context"]


def test_prescreen_runs_in_shadow_and_can_be_calibrated(balanced_run):
    """Shadow mode must record an opinion without discarding anything."""
    from conflictxgame.common.generation_quality import tabulate_prescreen

    with open(balanced_run["pair_coverage"], encoding="utf-8") as f:
        coverage = json.load(f)
    prescreen = coverage.get("prescreen")
    if prescreen is None:
        pytest.skip("pre-screen was disabled for this run")
    assert prescreen["dropped"] == 0, "shadow mode must never discard an outline"
    assert prescreen["screened"] > 0

    frame = pd.read_csv(balanced_run["csv"])
    verdicts = [
        json.loads(row["game_spec"])["metadata"].get("prescreen")
        for _, row in frame.iterrows()
    ]
    assert any(v is not None for v in verdicts), "no screening verdict reached the corpus"

    summary = tabulate_prescreen([str(balanced_run["csv"])])
    if summary.empty:
        print("\nNote: the screen flagged nothing at smoke scale; nothing to calibrate yet.")
        return
    print("\npre-screen vs judge (precision must beat base_reject_rate to be worth enforcing):")
    print(summary.to_string(index=False))
