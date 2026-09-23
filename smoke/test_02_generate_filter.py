"""Generation and filtering against the live gateway.

Exercises the defaults introduced in the refactor: one-shot 2-player 2x2
games, output paths derived from the value set, and the filtered CSV that is
a superset of the upstream ConflictScope scenario schema.
"""

import json

import pandas as pd
import pytest

from conftest import GEN_SHORT, SMOKE_SCENARIOS, VALUE_SET

pytestmark = pytest.mark.live

#: The eleven columns of the upstream ConflictScope scenario CSV.
LEGACY_COLUMNS = {
    "scenario_id", "context", "description", "user_prompt",
    "value1", "value2", "action1", "action2",
    "keep_scenario", "check_results", "generating_model",
}

#: Columns the game pipeline adds on top.
GAME_COLUMNS = {
    "game_id", "cell_key", "structure", "n_players", "arities", "horizon",
    "value_set", "value_tags", "player2_values", "payoff_matrix",
    "agent_kinds", "agent_mix", "game_spec",
}


class TestGeneratedArtifacts:
    def test_expected_files_exist(self, corpus, data_dir):
        games_dir = data_dir / "games"
        assert (games_dir / f"{GEN_SHORT}_games.json").exists()
        assert (games_dir / f"{GEN_SHORT}_game_prompts.json").exists()
        assert (games_dir / "coverage_report.json").exists()
        assert corpus.exists()

    def test_generated_specs_are_one_shot_2x2(self, data_dir):
        games = json.loads(
            (data_dir / "games" / f"{GEN_SHORT}_games.json").read_text(encoding="utf-8")
        )
        assert games, "no games generated"
        for game_id, spec in games.items():
            config = spec["config"]
            assert config["structure"] == "simultaneous", game_id
            assert config["n_players"] == 2, game_id
            assert config["arities"] == [2, 2], game_id
            assert config["horizon"] == 1, game_id

    def test_specs_validate_and_carry_value_tags(self, data_dir):
        from conflictxgame.games.game_structures import GameSpec

        games = json.loads(
            (data_dir / "games" / f"{GEN_SHORT}_games.json").read_text(encoding="utf-8")
        )
        for game_id, raw in games.items():
            spec = GameSpec.from_dict(raw)
            assert spec.validate() == [], (game_id, spec.validate())
            for player in spec.players:
                tags = {a.value_tag for a in player.actions}
                assert len(tags) == 2, f"{game_id}: menu does not encode a conflict"
            # All-assistant seats were requested.
            assert spec.agent_mix == "all_assistant", game_id


class TestFilteredCsv:
    def test_schema_is_a_superset_of_the_upstream_scenario_csv(self, corpus):
        df = pd.read_csv(corpus)
        assert LEGACY_COLUMNS <= set(df.columns), LEGACY_COLUMNS - set(df.columns)
        assert GAME_COLUMNS <= set(df.columns), GAME_COLUMNS - set(df.columns)
        assert len(df) >= 1

    def test_solo_compatible_fields_are_populated(self, corpus):
        df = pd.read_csv(corpus)
        for _, row in df.iterrows():
            assert isinstance(row["description"], str) and row["description"].strip()
            prompt = json.loads(row["user_prompt"])
            assert set(prompt) == {"persona", "background", "goal"}
            # Seat-0 value tags are short names, as in the upstream CSV.
            assert row["value1"] and row["value2"]
            assert row["value1"] != row["value2"]
            assert row["action1"] and row["action2"]

    def test_game_columns_round_trip(self, corpus):
        from conflictxgame.games.game_structures import GameSpec

        df = pd.read_csv(corpus)
        for _, row in df.iterrows():
            spec = GameSpec.from_json(row["game_spec"])
            assert spec.validate() == []
            assert json.loads(row["payoff_matrix"]) == spec.payoffs
            assert len(json.loads(row["player2_values"])) == spec.config.n_players - 1
            assert row["n_players"] == 2 and row["arities"] == "2x2"

    def test_judge_recorded_every_check(self, corpus):
        df = pd.read_csv(corpus)
        expected = {
            "scenario_realism", "scenario_groundedness", "action_feasibility",
            "value_guidedness", "payoff_consistency", "scenario_ambiguity",
        }
        for _, row in df.iterrows():
            checks = json.loads(row["check_results"])
            assert set(checks) == expected, set(checks) ^ expected
        kept = int(df["keep_scenario"].sum())
        print(f"\nfiltered corpus: {kept}/{len(df)} scenarios kept")

    def test_keep_rate_is_reported_not_asserted(self, corpus, kept_scenarios):
        """Report the keep rate; do not fail on it.

        Rejection is the filter doing its job (a scenario with an obvious
        right answer, or whose value tags the judge disputes), and on a
        3-scenario corpus the rate is dominated by noise. Evaluation tests run
        unfiltered so they never depend on this verdict.
        """
        total = len(pd.read_csv(corpus))
        print(f"\nkeep rate: {kept_scenarios}/{total}")
        assert total >= 1, "the filter produced no rows at all"

    def test_filter_flag_selects_only_kept_rows(self, corpus, kept_scenarios):
        """``--filter`` must hand downstream code exactly the kept scenarios."""
        from conflictxgame.games.evaluate_games import load_game_scenarios

        if kept_scenarios == 0:
            pytest.skip("no scenario survived filtering in this corpus")
        filtered = load_game_scenarios(str(corpus), True)
        unfiltered = load_game_scenarios(str(corpus), False)
        assert len(filtered) == kept_scenarios
        assert len(unfiltered) == len(pd.read_csv(corpus))


class TestDataSeparation:
    def test_smoke_data_never_lands_in_the_real_data_directory(self, corpus):
        """Artifacts must stay under smoke/_artifacts/, not data/."""
        from conftest import REPO_ROOT

        assert str(corpus).startswith(str(REPO_ROOT / "smoke" / "_artifacts"))
        real_value_set_dir = REPO_ROOT / "data" / VALUE_SET
        stray = list(real_value_set_dir.glob("*_games.csv"))
        assert not stray, f"smoke run polluted the real data directory: {stray}"
