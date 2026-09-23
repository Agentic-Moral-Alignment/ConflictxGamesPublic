"""Tests that the games filter CSV is a superset of the solo pipeline schema."""

import json

from conftest import make_spec
from conflictxgame.games.filter_game_scenarios import results_to_dataframe

#: The original solo pipeline's columns (generating_model is appended in main()).
LEGACY_COLUMNS = {
    "scenario_id", "context", "description", "user_prompt",
    "value1", "value2", "action1", "action2",
    "keep_scenario", "check_results",
}

GAME_COLUMNS = {
    "game_id", "cell_key", "structure", "n_players", "arities", "horizon",
    "value_set", "value_tags", "player2_values", "payoff_matrix",
    "agent_kinds", "agent_mix", "game_spec", "filter_error",
}


def make_results():
    spec = make_spec(
        "simultaneous", [2, 2],
        {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
        game_id="simultaneous_n2_a2x2-0",
        tags=[["v1", "v2"], ["v2", "v3"]],
    )
    spec.metadata["generating_model"] = "dummy-generator"
    spec.metadata["context"] = "A test context."
    checks = {"scenario_realism": True}
    return {spec.game_id: {"spec": spec, "checks": checks, "keep_scenario": True}}, spec


class TestGamesCsvSchema:
    def test_superset_of_solo_columns(self):
        results, _ = make_results()
        df = results_to_dataframe(results)
        assert LEGACY_COLUMNS <= set(df.columns)
        assert GAME_COLUMNS <= set(df.columns)

    def test_seat0_perspective_fields(self):
        results, spec = make_results()
        row = results_to_dataframe(results).iloc[0]
        assert row["scenario_id"] == "dummy-generator_simultaneous_n2_a2x2-0"
        assert row["context"] == "A test context."
        assert row["value1"] == "v1" and row["value2"] == "v2"
        assert row["action1"] == spec.players[0].actions[0].description
        assert row["action2"] == spec.players[0].actions[1].description
        user_prompt = json.loads(row["user_prompt"])
        assert set(user_prompt) == {"persona", "background", "goal"}
        assert user_prompt["persona"] == spec.players[0].persona
        assert user_prompt["background"] == spec.players[0].role_description

    def test_game_columns_round_trip(self):
        results, spec = make_results()
        row = results_to_dataframe(results).iloc[0]
        assert row["n_players"] == 2
        assert row["arities"] == "2x2"
        assert row["structure"] == "simultaneous"
        assert row["horizon"] == 1
        assert json.loads(row["player2_values"]) == [["v2", "v3"]]
        assert json.loads(row["payoff_matrix"]) == spec.payoffs
        assert json.loads(row["value_tags"]) == [["v1", "v2"], ["v2", "v3"]]

    def test_wide_menus_leave_solo_fields_empty(self):
        import conflictxgame.games.game_structures as gs

        spec = make_spec(
            "simultaneous", [3, 2],
            {gs.profile_key(p): [1, 1] for p in gs.enumerate_joint_profiles((3, 2))},
        )
        spec.metadata["generating_model"] = "dummy-generator"
        results = {spec.game_id: {"spec": spec, "checks": {}, "keep_scenario": True}}
        row = results_to_dataframe(results).iloc[0]
        assert row["value1"] == "" and row["action2"] == ""
        assert row["n_players"] == 2 and row["arities"] == "3x2"
