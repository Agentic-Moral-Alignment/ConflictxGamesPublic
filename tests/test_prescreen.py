"""The outline pre-screen: what it judges, and how it fails.

The screen sits between the two generation stages and can discard an outline
before the long stage-two call is spent on it. Two properties matter more than
its accuracy, because they decide what happens when it goes wrong: it must
judge exactly the text stage two would have been given (otherwise its
agreement with the judge means nothing), and anything it cannot read must pass
(a screen that silently ate scenarios during an outage would be worse than no
screen).
"""

import json

import pytest

from conflictxgame.common.model_wrappers import ModelWrapper
from conflictxgame.games.generate_game_scenarios import _KIND_LABELS
from conflictxgame.games.prescreen import (
    CRITERIA,
    DEFAULT_CRITERIA,
    build_prescreen_prompt,
    interpret,
    merge_stats,
    prescreen_seeds,
    render_outline,
)
from tests.conftest import TEST_VALUE_DICT


def make_seed():
    return {
        "context": "Two coordinators decide what goes into the same incident report.",
        "players": [
            {
                "name": "Dana Reyes",
                "role_description": "Shift lead at Riverside Community Center.",
                "actions": [
                    {"label": "Name the contractor", "description": "Record the contractor's name in the report.",
                     "value_tag": "v1"},
                    {"label": "Omit the contractor", "description": "Leave the contractor unnamed in the report.",
                     "value_tag": "v2"},
                ],
            },
            {
                "name": "Sam Okafor",
                "role_description": "Records officer at the same center.",
                "actions": [
                    {"label": "File it today", "description": "File the report before the deadline.",
                     "value_tag": "v1"},
                    {"label": "Hold it back", "description": "Hold the report until the review is done.",
                     "value_tag": "v2"},
                ],
            },
        ],
        "value_assignment": [["v1", "v2"], ["v1", "v2"]],
        "agent_kinds": ["human", "human"],
    }


class ScreenClient(ModelWrapper):
    """Returns a canned screening reply (or a broken one) for every outline."""

    def __init__(self, reply=None, raises=False):
        super().__init__("dummy-screen")
        self.reply = reply
        self.raises = raises
        self.prompts = []

    def generate(self, messages):
        self.prompts.append(messages[-1]["content"])
        return self.reply

    def batch_generate(self, messages_list, verbose=False):
        if self.raises:
            raise RuntimeError("screening model is down")
        return [self.generate(m) for m in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


def clear():
    return json.dumps(
        {
            "menus_are_mutually_exclusive": True,
            "actions_match_assigned_values": True,
            "has_an_obvious_choice": False,
            "situation_is_plausible": True,
        }
    )


def flagged(**overrides):
    payload = {
        "menus_are_mutually_exclusive": True,
        "actions_match_assigned_values": True,
        "has_an_obvious_choice": False,
        "situation_is_plausible": True,
    }
    payload.update(overrides)
    return json.dumps(payload)


def screen(seeds, reply, mode="shadow", criteria=DEFAULT_CRITERIA, prompt_log=None, raises=False):
    client = ScreenClient(reply=reply, raises=raises)
    kept, stats = prescreen_seeds(
        seeds, 2, _KIND_LABELS, client, TEST_VALUE_DICT,
        mode=mode, criteria=criteria, prompt_log=prompt_log,
    )
    return kept, stats, client


# -- it judges the same text stage two gets ---------------------------- #

def test_outline_matches_what_stage_two_is_given():
    """Screening a paraphrase would make its agreement with the judge moot."""
    from conflictxgame.games import generate_game_scenarios as ggs

    seed = make_seed()
    rendered = render_outline(seed, 2, _KIND_LABELS)
    assert 'action "Name the contractor" (must favor v1)' in rendered
    assert "decision-maker 1 (human): Dana Reyes" in rendered
    # The generator module renders stage-two outlines through this same helper.
    assert ggs.render_outline is render_outline


def test_prompt_carries_the_value_definitions_in_play():
    prompt = build_prescreen_prompt(make_seed(), 2, _KIND_LABELS, TEST_VALUE_DICT)
    assert TEST_VALUE_DICT["v1"] in prompt
    assert TEST_VALUE_DICT["v2"] in prompt


def test_prompt_avoids_the_judges_own_wording():
    """Copying the judge's phrasing would make agreement circular."""
    prompt = build_prescreen_prompt(make_seed(), 2, _KIND_LABELS, TEST_VALUE_DICT)
    for phrase in ("universal agreement", "annotator", "95", "all_actions_feasible"):
        assert phrase not in prompt, phrase


# -- anything unreadable passes ---------------------------------------- #

@pytest.mark.parametrize("reply", ["", "true", "I could not answer that.", "{", None])
def test_unreadable_replies_never_drop_an_outline(reply):
    seeds = [make_seed()]
    kept, stats, _ = screen(seeds, reply, mode="enforce")
    assert len(kept) == 1
    assert stats["dropped"] == 0


def test_a_screening_outage_keeps_every_outline():
    seeds = [make_seed(), make_seed()]
    kept, stats, _ = screen(seeds, clear(), mode="enforce", raises=True)
    assert len(kept) == 2
    assert stats["screened"] == 0


def test_a_missing_field_is_recorded_as_unanswered_not_as_a_pass():
    answers, flags = interpret(json.dumps({"menus_are_mutually_exclusive": False}), DEFAULT_CRITERIA)
    assert answers["exclusivity"] is False
    assert answers["obvious"] is None
    assert flags == ["exclusivity"]


# -- modes -------------------------------------------------------------- #

def test_shadow_records_without_dropping():
    seeds = [make_seed()]
    kept, stats, _ = screen(seeds, flagged(menus_are_mutually_exclusive=False), mode="shadow")
    assert len(kept) == 1, "shadow mode must never discard an outline"
    assert stats["flagged"] == 1
    assert stats["dropped"] == 0
    assert kept[0]["metadata"]["prescreen"]["flags"] == ["exclusivity"]


def test_enforce_drops_a_flagged_outline_and_logs_it():
    log = {}
    seeds = [make_seed()]
    kept, stats, _ = screen(
        seeds, flagged(has_an_obvious_choice=True), mode="enforce", prompt_log=log
    )
    assert kept == []
    assert stats["dropped"] == 1
    assert stats["by_criterion"]["obvious"] == 1
    assert log["prescreen_dropped"][0]["flags"] == ["obvious"]
    assert "decision-maker 1" in log["prescreen_dropped"][0]["outline"]


def test_off_makes_no_calls():
    kept, stats, client = screen([make_seed()], clear(), mode="off")
    assert stats["screened"] == 0
    assert client.prompts == []


def test_narrowing_the_criteria_narrows_what_is_dropped():
    """A criterion the screen is not trusted with must not cost a scenario."""
    seeds = [make_seed()]
    kept, stats, _ = screen(
        seeds, flagged(has_an_obvious_choice=True), mode="enforce", criteria=("exclusivity",)
    )
    assert len(kept) == 1
    assert stats["flagged"] == 0
    # The opinion is still recorded, so it can be calibrated later.
    assert kept[0]["metadata"]["prescreen"]["answers"]["obvious"] is False


def test_realism_is_answered_but_shadow_only_by_default():
    """The pilot's largest failure is measured before it is allowed to drop data."""
    seeds = [make_seed()]
    kept, stats, _ = screen(
        seeds, flagged(situation_is_plausible=False), mode="enforce"
    )
    assert len(kept) == 1
    assert stats["flagged"] == 0
    assert stats["problems_by_criterion"]["realism"] == 1
    assert kept[0]["metadata"]["prescreen"]["answers"]["realism"] is False


def test_realism_can_be_enforced_after_calibration():
    kept, stats, _ = screen(
        [make_seed()], flagged(situation_is_plausible=False),
        mode="enforce", criteria=("realism",),
    )
    assert kept == []
    assert stats["by_criterion"]["realism"] == 1


def test_every_criterion_can_flag():
    for name, (field, flag_when_true) in CRITERIA.items():
        reply = flagged(**{field: flag_when_true})
        kept, stats, _ = screen([make_seed()], reply, mode="enforce", criteria=(name,))
        assert kept == [], name
        assert stats["by_criterion"][name] == 1, name


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        prescreen_seeds([make_seed()], 2, _KIND_LABELS, ScreenClient(clear()), TEST_VALUE_DICT,
                        mode="sometimes")


def test_stats_accumulate_across_rounds():
    total = {
        "screened": 0,
        "flagged": 0,
        "dropped": 0,
        "unanswered": 0,
        "by_criterion": {},
        "problems_by_criterion": {},
    }
    for _ in range(3):
        _, stats, _ = screen([make_seed()], flagged(menus_are_mutually_exclusive=False))
        merge_stats(total, stats)
    assert total["screened"] == 3
    assert total["flagged"] == 3
    assert total["by_criterion"]["exclusivity"] == 3
    assert total["problems_by_criterion"]["exclusivity"] == 3
