"""Static MCQ evaluation against the live gateway."""

import json

import pandas as pd
import pytest

from conftest import TARGET_MODEL, VALUE_SET, run_cli

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def mcq_results(corpus, data_dir, results_dir):
    """Run --mode mcq once for this module."""
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
    assert result.returncode == 0, "mcq evaluation failed"
    csv_files = list(results_dir.glob("*_mcq.csv"))
    assert csv_files, f"no *_mcq.csv written to {results_dir}"
    return pd.read_csv(csv_files[0])


class TestMcqResults:
    def test_rows_were_produced_for_every_seat(self, mcq_results):
        assert not mcq_results.empty
        assert set(mcq_results["mode"]) == {"mcq"}
        assert set(mcq_results["seat"]) == {0, 1}
        print(f"\nmcq rows: {len(mcq_results)}")

    def test_choices_parsed(self, mcq_results):
        parsed = mcq_results["parsed_ok"].astype(str).str.lower().eq("true")
        rate = parsed.mean()
        print(f"mcq parse rate: {rate:.0%}")
        assert rate > 0.5, "most MCQ answers were unparseable"

    def test_chosen_actions_carry_value_tags(self, mcq_results):
        valid = mcq_results[mcq_results["choice_value_tag"] != "INVALID"]
        assert not valid.empty
        assert valid["choice_value_tag"].notna().all()

    def test_likert_and_outcome_rankings_were_elicited(self, mcq_results):
        plain = mcq_results[mcq_results["probe"].fillna("") == ""]
        likerts = plain["likert_by_action"].dropna()
        assert not likerts.empty
        for raw in likerts:
            parsed = json.loads(raw)
            assert parsed, "empty Likert map"
        rankings = plain["outcome_ranking"].dropna()
        good = [r for r in rankings if r not in ("", "INVALID")]
        assert good, "no valid outcome ranking was produced"
        for raw in good:
            assert isinstance(json.loads(raw), list)

    def test_scenario_metadata_joined_and_steering_column_present(self, mcq_results):
        for column in ("cell_key", "structure", "n_players", "value_set", "model"):
            assert column in mcq_results.columns, column
        assert set(mcq_results["value_set"]) == {VALUE_SET}
        # Unsteered run: the steering column exists but is blank.
        assert "steering" in mcq_results.columns
        assert mcq_results["steering"].fillna("").eq("").all()
