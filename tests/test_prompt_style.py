"""Paper-style game outlines and seat-kind sampling."""

import json
import random

from conflictxgame.games.game_structures import GameConfig
from conflictxgame.games.generate_game_scenarios import (
    PAPER_TENSION_FRAMES,
    generate_stage_one_games,
)
from tests.fakes import DummyGenerationClient


VALUE_DICT = {
    "harmlessness": "avoid causing material or social harm",
    "helpfulness": "carry out the person's stated request",
    "honesty": "do not materially mislead anyone",
}

CONFIG = GameConfig(
    structure="simultaneous", n_players=2, arities=[2, 2], horizon=1
)


def test_paper_prompt_is_small_and_has_no_slot_assignments():
    log = {}
    generate_stage_one_games(
        CONFIG, DummyGenerationClient(), 1, VALUE_DICT, random.Random(0),
        assistant_seat_prob=1.0, prompt_log=log, prompt_style="paper",
    )
    prompt = log["stage_one"][0]["prompt"]
    assert "Value assignments" not in prompt
    assert "must favor" not in prompt
    assert "START FROM ONE SHARED OUTCOME" not in prompt
    assert "MUTUAL EXCLUSIVITY" not in prompt
    assert '"favors"' in prompt
    assert "decision-maker 1: AI assistant, 2 options" in prompt
    for value, definition in VALUE_DICT.items():
        assert f"- {value}: {definition}" in prompt


def test_guided_is_the_explicit_default_prompt_style():
    """The control arm must stay comparable to pre-paper runs."""
    default_log, guided_log = {}, {}
    generate_stage_one_games(
        CONFIG, DummyGenerationClient(), 1, VALUE_DICT, random.Random(4),
        prompt_log=default_log,
    )
    generate_stage_one_games(
        CONFIG, DummyGenerationClient(), 1, VALUE_DICT, random.Random(4),
        prompt_log=guided_log, prompt_style="guided",
    )
    assert default_log["stage_one"][0]["prompt"] == guided_log["stage_one"][0]["prompt"]
    assert "Value assignments" in default_log["stage_one"][0]["prompt"]


def test_paper_favors_become_seed_value_tags_and_pairs():
    seeds = generate_stage_one_games(
        CONFIG, DummyGenerationClient(), 1, VALUE_DICT, random.Random(0),
        prompt_style="paper",
    )
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed["metadata"]["prompt_style"] == "paper"
    for player, assignment in zip(seed["players"], seed["value_assignment"]):
        tags = [action["value_tag"] for action in player["actions"]]
        assert tags == assignment
        assert player["value_pair"] == sorted(set(tags))
        assert len(set(tags)) == 2


def test_targeted_paper_uses_unordered_pairs_and_exact_favors():
    requested = [
        ["harmlessness", "honesty"],
        ["harmlessness", "honesty"],
    ]

    def sampler(config, value_dict, rng):
        return [list(values) for values in requested]

    log = {}
    seeds = generate_stage_one_games(
        CONFIG,
        DummyGenerationClient(),
        1,
        VALUE_DICT,
        random.Random(0),
        assistant_seat_prob=1.0,
        prompt_log=log,
        assignment_sampler=sampler,
        prompt_style="paper",
        paper_quota_targeting=True,
    )

    assert len(seeds) == 1
    prompt = log["stage_one"][0]["prompt"]
    assert "required values (one option each, either order)" in prompt
    assert "action 1 ->" not in prompt
    assert "- helpfulness:" not in prompt
    assert seeds[0]["metadata"]["paper_quota_targeting"] is True
    assert seeds[0]["metadata"]["requested_value_pairs"] == requested
    for player in seeds[0]["players"]:
        assert sorted(action["value_tag"] for action in player["actions"]) == sorted(
            requested[0]
        )


def test_targeted_paper_drops_an_off_target_but_valid_pair():
    def sampler(config, value_dict, rng):
        return [
            ["harmlessness", "honesty"],
            ["harmlessness", "honesty"],
        ]

    dropped = []
    seeds = generate_stage_one_games(
        CONFIG,
        BadPaperClient(("helpfulness", "honesty")),
        1,
        VALUE_DICT,
        random.Random(0),
        assignment_sampler=sampler,
        prompt_style="paper",
        paper_quota_targeting=True,
        dropped=dropped,
    )
    assert seeds == []
    assert len(dropped) == 1
    assert dropped[0]['metadata']['requested_value_pairs'] == [
        ['harmlessness', 'honesty'], ['harmlessness', 'honesty'],
    ]
    mismatch = dropped[0]['metadata']['paper_quota_mismatch']
    assert mismatch['seat'] == 0
    assert mismatch['returned'] == ['helpfulness', 'honesty']


class BadPaperClient(DummyGenerationClient):
    """Returns one invalid paper-style seed for validation tests."""

    def __init__(self, favors):
        super().__init__()
        self.favors = favors

    def _stage_one_paper(self, text):
        return json.dumps(
            {
                "scenario_1": {
                    "context": "Two coordinators make connected decisions.",
                    "players": [
                        {
                            "name": "Dana Reyes",
                            "role_description": "Coordinates the response.",
                            "actions": [
                                {"label": "First option", "description": "Do the first thing.", "favors": self.favors[0]},
                                {"label": "Second option", "description": "Do the second thing.", "favors": self.favors[1]},
                            ],
                        },
                        {
                            "name": "Sam Okafor",
                            "role_description": "Coordinates the record.",
                            "actions": [
                                {"label": "Third option", "description": "Do the third thing.", "favors": "helpfulness"},
                                {"label": "Fourth option", "description": "Do the fourth thing.", "favors": "honesty"},
                            ],
                        },
                    ],
                }
            }
        )


def test_paper_drops_unknown_and_single_value_menus():
    for favors in (("unknown", "honesty"), ("honesty", "honesty")):
        seeds = generate_stage_one_games(
            CONFIG, BadPaperClient(favors), 1, VALUE_DICT, random.Random(0),
            prompt_style="paper",
        )
        assert seeds == []


def test_tension_frame_uses_its_own_rng_not_generation_rng():
    class Picker:
        def __init__(self, value):
            self.value = value

        def randrange(self, size):
            assert size == len(PAPER_TENSION_FRAMES)
            return self.value

    left_log, right_log = {}, {}
    left = generate_stage_one_games(
        CONFIG, DummyGenerationClient(), 1, VALUE_DICT, random.Random(9),
        prompt_log=left_log, prompt_style="paper", guidance_rng=Picker(0),
    )
    right = generate_stage_one_games(
        CONFIG, DummyGenerationClient(), 1, VALUE_DICT, random.Random(9),
        prompt_log=right_log, prompt_style="paper", guidance_rng=Picker(1),
    )
    assert left[0]["agent_kinds"] == right[0]["agent_kinds"]
    assert left_log["stage_one"][0]["tension_frame"] != right_log["stage_one"][0]["tension_frame"]
    assert left_log["stage_one"][0]["tension_frame"] == PAPER_TENSION_FRAMES[0]
