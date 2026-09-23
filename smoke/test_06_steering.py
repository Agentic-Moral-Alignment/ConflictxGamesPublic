"""Steering ("leading") and the steering-impact analysis, end to end.

Runs the same MCQ evaluation with the paper's Appendix J HHH system prompt,
into a separate results directory, then measures alignment with the target
ranking against the unsteered run from ``test_03_mcq``.
"""

import pandas as pd
import pytest

from conftest import (
    REPO_ROOT,
    TARGET_MODEL,
    VALUE_SET,
    run_cli,
)

pytestmark = pytest.mark.live

#: The paper's HHH target ranking (§5.3).
TARGET_RANKING = "harmlessness>honesty>helpfulness"

STEER_PROMPT = REPO_ROOT / "value_sets" / "steering" / f"{VALUE_SET}.txt"


@pytest.fixture(scope="module")
def unsteered(corpus, data_dir, results_dir):
    """The unsteered MCQ run (reused if test_03 already produced it)."""
    existing = list(results_dir.glob("*_mcq.csv"))
    if existing:
        return pd.read_csv(existing[0])
    result = run_cli(
        "conflictxgame.games.evaluate_games",
        [
            "-m", TARGET_MODEL, "--mode", "mcq",
            "-d", str(data_dir), "-o", str(results_dir),
            "--max-tokens", "1024",
        ],
        stage_index=2,
        timeout=1800,
    )
    assert result.returncode == 0
    return pd.read_csv(list(results_dir.glob("*_mcq.csv"))[0])


@pytest.fixture(scope="module")
def steered(corpus, data_dir, steered_results_dir):
    """The steered MCQ run, into its own results directory."""
    result = run_cli(
        "conflictxgame.games.evaluate_games",
        [
            "-m", TARGET_MODEL, "--mode", "mcq",
            "-d", str(data_dir), "-o", str(steered_results_dir),
            "--max-tokens", "1024",
            "--steer-prompt", str(STEER_PROMPT),
        ],
        stage_index=7,
        timeout=1800,
    )
    assert result.returncode == 0, "steered evaluation failed"
    csv_files = list(steered_results_dir.glob("*_mcq.csv"))
    assert csv_files, f"no steered *_mcq.csv in {steered_results_dir}"
    return pd.read_csv(csv_files[0])


class TestSteeringPromptShipped:
    def test_appendix_j_prompt_exists_for_this_value_set(self):
        assert STEER_PROMPT.exists(), STEER_PROMPT
        text = STEER_PROMPT.read_text(encoding="utf-8")
        assert text.index("HARMLESSNESS FIRST") < text.index("HONESTY SECOND")
        assert text.index("HONESTY SECOND") < text.index("HELPFULNESS THIRD")


class TestSteeredRun:
    def test_rows_produced_and_labelled(self, steered):
        assert not steered.empty
        assert set(steered["mode"]) == {"mcq"}
        # The steering column records which prompt was in force.
        assert set(steered["steering"].dropna()) == {VALUE_SET}

    def test_choices_still_parse_under_steering(self, steered):
        parsed = steered["parsed_ok"].astype(str).str.lower().eq("true")
        print(f"\nsteered parse rate: {parsed.mean():.0%}")
        assert parsed.mean() > 0.5


class TestTargetRankingFlag:
    def test_generated_steering_prompt_runs(self, corpus, data_dir, steered_results_dir):
        """--target-ranking builds a prompt when no file is given."""
        result = run_cli(
            "conflictxgame.games.evaluate_games",
            [
                "-m", TARGET_MODEL, "--mode", "mcq",
                "-d", str(data_dir), "-o", str(steered_results_dir),
                "--output-name", "generated_ranking_mcq.csv",
                "--max-tokens", "1024",
                "--target-ranking", TARGET_RANKING,
                "-v", VALUE_SET,
            ],
            stage_index=8,
            timeout=1800,
        )
        assert result.returncode == 0, "--target-ranking run failed"
        df = pd.read_csv(steered_results_dir / "generated_ranking_mcq.csv")
        assert not df.empty
        assert set(df["steering"].dropna()) == {f"ranking:{TARGET_RANKING}"}


class TestSteeringImpactAnalysis:
    def test_analysis_writes_steering_impact(
        self, unsteered, steered, data_dir, results_dir, steered_results_dir, artifacts
    ):
        analysis_dir = artifacts / "analysis"
        result = run_cli(
            "conflictxgame.games.analyze_games",
            [
                "--results-dir", str(results_dir),
                "--steered-results-dir", str(steered_results_dir),
                "--target-ranking", TARGET_RANKING,
                "--scenarios-dir", str(data_dir),
                "--output-dir", str(analysis_dir),
            ],
            stage_index=0,
            timeout=900,
        )
        assert result.returncode == 0, "analysis failed"

        impact_path = analysis_dir / "steering_impact.csv"
        assert impact_path.exists(), "steering_impact.csv was not written"
        impact = pd.read_csv(impact_path)
        assert not impact.empty

        expected = {
            "model", "mode", "target_ranking", "a_default", "a_steered",
            "normalized_effect", "n_events_default", "n_events_steered",
        }
        assert expected <= set(impact.columns), expected - set(impact.columns)
        assert (impact["a_default"].between(0, 1)).all()
        assert (impact["a_steered"].between(0, 1)).all()
        assert "overall" in set(impact["mode"])
        print("\n" + impact.to_string(index=False))

    def test_value_rankings_were_also_produced(self, unsteered, artifacts):
        analysis_dir = artifacts / "analysis"
        bt = analysis_dir / "value_rankings_bradley_terry.csv"
        if bt.exists():
            frame = pd.read_csv(bt)
            assert {"value", "ability"} <= set(frame.columns)
            print("\n" + frame.to_string(index=False))
        else:
            pytest.skip("Bradley-Terry fit needs more decision events than a smoke run has")
