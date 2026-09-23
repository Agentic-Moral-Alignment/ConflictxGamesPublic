"""Tests for steering prompts and the steering-impact metric (paper §4.2/§5.3)."""

import os

import pandas as pd
import pytest

from conftest import TEST_VALUE_DICT, make_spec
from conflictxgame.games.analyze_games import (
    compute_steering_impact,
    target_alignment_events,
)
from conflictxgame.games.steering import build_steering_prompt, parse_target_ranking

RANKING = ["v1", "v2", "v3"]


class TestParseTargetRanking:
    def test_both_separator_styles(self):
        assert parse_target_ranking("v1>v2>v3") == RANKING
        assert parse_target_ranking("v1, v2 , v3") == RANKING

    def test_rejects_too_short_or_duplicated(self):
        with pytest.raises(ValueError):
            parse_target_ranking("v1")
        with pytest.raises(ValueError):
            parse_target_ranking("v1>v1>v2")


class TestBuildSteeringPrompt:
    def test_contains_ranking_definitions_and_pairwise_rules(self):
        prompt = build_steering_prompt(TEST_VALUE_DICT, RANKING)
        assert prompt.index("V1 FIRST") < prompt.index("V2 SECOND") < prompt.index("V3 THIRD")
        for definition in (TEST_VALUE_DICT[v] for v in RANKING):
            assert definition in prompt
        # One rule per non-final value, each naming the values it outranks.
        assert "When v1 conflicts with v2, v3" in prompt
        assert "When v2 conflicts with v3" in prompt
        assert "DECISION FRAMEWORK" in prompt

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError):
            build_steering_prompt(TEST_VALUE_DICT, ["v1", "nonexistent"])


class TestShippedAppendixJPrompts:
    def test_all_three_value_sets_have_a_steering_prompt(self):
        root = os.path.join(os.path.dirname(__file__), "..", "value_sets", "steering")
        for name in ("HHH", "modelspec", "personalprotective"):
            path = os.path.join(root, f"{name}.txt")
            assert os.path.exists(path), path
            text = open(path, encoding="utf-8").read()
            assert "strict order" in text
            assert "DECISION FRAMEWORK" in text

    def test_hhh_prompt_states_the_papers_target_ranking(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "value_sets", "steering", "HHH.txt"
        )
        text = open(path, encoding="utf-8").read()
        assert text.index("HARMLESSNESS FIRST") < text.index("HONESTY SECOND")
        assert text.index("HONESTY SECOND") < text.index("HELPFULNESS THIRD")


def spec_with(tags, scenario_id):
    """A 2x2 spec whose seat-0 menu carries the given value tags."""
    return make_spec(
        "simultaneous", [2, 2],
        {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
        game_id=scenario_id,
        tags=[list(tags), list(tags)],
    )


def rows(choices, mode="mcq", model="m"):
    """Build result rows choosing the given action index per scenario."""
    return pd.DataFrame(
        [
            {
                "scenario_id": f"s{i}", "seat": 0, "choice_action": choice,
                "mode": mode, "probe": "", "model": model,
            }
            for i, choice in enumerate(choices)
        ]
    )


class TestAlignmentEvents:
    def specs(self, n=4):
        # Seat-0 menu is [v2, v1]: action 1 (v1) is the target-aligned choice.
        return {f"s{i}": spec_with(("v2", "v1"), f"s{i}") for i in range(n)}

    def test_alignment_counts_highest_ranked_tag(self):
        events = target_alignment_events(rows([1, 1, 0, 0]), self.specs(), RANKING)
        assert list(events["aligned"]) == [True, True, False, False]

    def test_probe_rows_and_invalid_choices_are_excluded(self):
        data = rows([1, 1, 1, 1])
        data["choice_action"] = data["choice_action"].astype(object)
        data.loc[0, "probe"] = "coop_history"
        data.loc[1, "choice_action"] = "INVALID"
        events = target_alignment_events(data, self.specs(), RANKING)
        assert len(events) == 2

    def test_menu_outside_the_ranking_is_skipped(self):
        specs = {"s0": spec_with(("v1", "unranked"), "s0")}
        assert target_alignment_events(rows([0]), specs, RANKING).empty

    def test_three_action_menu_needs_the_top_ranked_tag(self):
        spec = make_spec(
            "simultaneous", [3, 2],
            {f"{a},{b}": [1, 1] for a in range(3) for b in range(2)},
            game_id="s0",
            tags=[["v3", "v2", "v1"], ["v1", "v2"]],
        )
        specs = {"s0": spec}
        # Only action 2 (v1) is the highest-ranked tag in the menu.
        assert list(target_alignment_events(rows([2]), specs, RANKING)["aligned"]) == [True]
        assert list(target_alignment_events(rows([1]), specs, RANKING)["aligned"]) == [False]


class TestSteeringImpact:
    def specs(self, n=4):
        return {f"s{i}": spec_with(("v2", "v1"), f"s{i}") for i in range(n)}

    def test_normalized_effect_size(self):
        default = rows([1, 0, 0, 0])   # 1/4 aligned
        steered = rows([1, 1, 1, 0])   # 3/4 aligned
        impact = compute_steering_impact(default, steered, self.specs(), RANKING)
        overall = impact[impact["mode"] == "overall"].iloc[0]
        assert overall["a_default"] == pytest.approx(0.25)
        assert overall["a_steered"] == pytest.approx(0.75)
        # (0.75 - 0.25) / (1 - 0.25) = 2/3
        assert overall["normalized_effect"] == pytest.approx(2 / 3)
        assert overall["n_events_default"] == 4
        assert overall["target_ranking"] == "v1>v2>v3"

    def test_already_aligned_default_gives_nan_effect(self):
        impact = compute_steering_impact(
            rows([1, 1]), rows([1, 1]), self.specs(2), RANKING
        )
        overall = impact[impact["mode"] == "overall"].iloc[0]
        assert overall["a_default"] == 1.0
        assert pd.isna(overall["normalized_effect"])

    def test_per_mode_rows(self):
        default = pd.concat([rows([0, 0]), rows([0, 0], mode="open")], ignore_index=True)
        steered = pd.concat([rows([1, 1]), rows([0, 1], mode="open")], ignore_index=True)
        impact = compute_steering_impact(default, steered, self.specs(2), RANKING)
        by_mode = impact.set_index("mode")
        assert by_mode.loc["mcq", "a_steered"] == pytest.approx(1.0)
        assert by_mode.loc["open", "a_steered"] == pytest.approx(0.5)
        assert by_mode.loc["overall", "a_steered"] == pytest.approx(0.75)

    def test_empty_when_no_valid_events(self):
        assert compute_steering_impact(rows([]), rows([]), self.specs(), RANKING).empty
