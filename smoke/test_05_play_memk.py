"""Live play with mem-k memory, in both repeat modes.

One-shot 2x2 scenarios are played as repeated sequences at evaluation time;
the acting model sees only the last k rounds.
"""

import json

import pandas as pd
import pytest

from conftest import TARGET_MODEL, run_cli

pytestmark = pytest.mark.live

ROUNDS = 3


@pytest.fixture(scope="module")
def same_mode(corpus, data_dir, results_dir):
    """--repeat-mode same: one scenario replayed for ROUNDS rounds, mem-1."""
    result = run_cli(
        "conflictxgame.games.evaluate_games",
        [
            "-m", TARGET_MODEL, "--mode", "play",
            "-d", str(data_dir), "-o", str(results_dir),
            "--output-name", "play_same.csv",
            "--opponents", "scripted",
            "--strategies", "tit_for_tat",
            "--repetitions", "1",
            "--rounds", str(ROUNDS), "--memory-k", "1",
            "--max-tokens", "1024",
        ],
        stage_index=5,
        timeout=2400,
    )
    assert result.returncode == 0, "play (same) evaluation failed"
    return pd.read_csv(results_dir / "play_same.csv")


@pytest.fixture(scope="module")
def sequence_mode(corpus, data_dir, results_dir):
    """--repeat-mode sequence: distinct scenarios chained with mem-1 carryover."""
    result = run_cli(
        "conflictxgame.games.evaluate_games",
        [
            "-m", TARGET_MODEL, "--mode", "play",
            "-d", str(data_dir), "-o", str(results_dir),
            "--output-name", "play_sequence.csv",
            "--opponents", "scripted",
            "--strategies", "always_cooperate",
            "--repetitions", "1",
            "--rounds", "2", "--memory-k", "1",
            "--repeat-mode", "sequence", "--sequence-grouping", "shuffle",
            "--max-tokens", "1024",
        ],
        stage_index=6,
        timeout=2400,
    )
    assert result.returncode == 0, "play (sequence) evaluation failed"
    return pd.read_csv(results_dir / "play_sequence.csv")


class TestRepeatModeSame:
    def test_scenario_was_played_for_every_round(self, same_mode):
        assert not same_mode.empty
        assert set(same_mode["mode"]) == {"play"}
        rounds_seen = sorted(same_mode["round"].unique())
        assert rounds_seen == list(range(ROUNDS)), rounds_seen
        assert set(same_mode["rounds_played"]) == {ROUNDS}
        print(f"\nplay(same) rows: {len(same_mode)} over {ROUNDS} rounds")

    def test_memory_k_recorded_and_actions_parsed(self, same_mode):
        assert set(same_mode["memory_k"]) == {1}
        parsed = same_mode["parsed_ok"].astype(str).str.lower().eq("true")
        print(f"play(same) parse rate: {parsed.mean():.0%}")
        assert parsed.mean() > 0.5

    def test_payoffs_and_profiles_recorded(self, same_mode):
        assert same_mode["round_payoff"].notna().all()
        for raw in same_mode["round_profile"].dropna():
            profile = json.loads(raw)
            assert len(profile) == 2
            assert all(isinstance(a, int) for a in profile)

    def test_opponent_was_the_scripted_strategy(self, same_mode):
        assert set(same_mode["opponent_config"]) == {"scripted:tit_for_tat"}


class TestRepeatModeSequence:
    def test_sequence_positions_recorded(self, sequence_mode):
        assert not sequence_mode.empty
        assert "sequence_id" in sequence_mode.columns
        assert "sequence_pos" in sequence_mode.columns
        positions = sorted(sequence_mode["sequence_pos"].unique())
        assert positions[0] == 0
        print(f"\nplay(sequence) rows: {len(sequence_mode)}, positions {positions}")

    def test_each_position_is_a_one_shot_game(self, sequence_mode):
        # Every game in a sequence is one round; repetition is across games.
        assert set(sequence_mode["round"]) == {0}

    def test_distinct_scenarios_appear_in_a_sequence(self, sequence_mode):
        for sequence_id, group in sequence_mode.groupby("sequence_id"):
            ids = group["scenario_id"].tolist()
            assert len(set(ids)) == len(ids), (
                f"sequence {sequence_id} repeated a scenario: {ids}"
            )
