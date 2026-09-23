import random

from conftest import make_spec
from conflictxgame.games.evaluate_games import create_game_mcq_prompt, state_probe_note
from conflictxgame.games.game_engine import (
    LLMPlayer,
    build_likert_messages,
    build_outcome_ranking_messages,
)
from conflictxgame.games.open_ended import (
    build_open_target_messages,
    build_open_user_messages,
)
from conflictxgame.solo.evaluate_models import create_scenario_prompt_mcq
from test_game_engine import CannedClient, first_option_responder


HIDDEN_VALUES = ("hidden_value_alpha", "hidden_value_beta")
PAYOFFS = {"0,0": [1, 1], "0,1": [0, 2], "1,0": [2, 0], "1,1": [1, 1]}


def _blind_spec():
    spec = make_spec(
        "simultaneous", [2, 2], PAYOFFS,
        tags=[[HIDDEN_VALUES[0], HIDDEN_VALUES[1]]] * 2,
    )
    for player in spec.players:
        for index, action in enumerate(player.actions):
            action.label = f"Concrete option {index + 1}"
            action.description = f"Take concrete action {index + 1}."
    return spec


def test_solo_mcq_does_not_insert_value_metadata():
    row = {
        "description": "A person must make a concrete decision.",
        "action1": "Send the first version.",
        "action2": "Send the second version.",
        "value1": HIDDEN_VALUES[0],
        "value2": HIDDEN_VALUES[1],
    }
    prompt, _ = create_scenario_prompt_mcq(row)
    assert all(value not in prompt for value in HIDDEN_VALUES)


def test_game_mcq_and_play_do_not_insert_value_tags():
    spec = _blind_spec()
    mcq_prompt, _ = create_game_mcq_prompt(spec, 0, random.Random(0))
    play_messages = LLMPlayer(CannedClient(first_option_responder)).build_messages(
        spec, 0, [], None, [0, 1]
    )
    rendered = mcq_prompt + "\n" + "\n".join(m["content"] for m in play_messages)
    assert all(value not in rendered for value in HIDDEN_VALUES)


def test_game_likert_and_outcome_ranking_do_not_insert_value_tags():
    spec = _blind_spec()
    likert = build_likert_messages(spec, 0, 0)
    ranking = build_outcome_ranking_messages(
        spec, 0, list(spec.outcome_descriptions)
    )
    rendered = str(likert) + str(ranking)
    assert all(value not in rendered for value in HIDDEN_VALUES)


def test_state_probe_note_uses_action_labels_not_value_tags():
    spec = _blind_spec()
    rendered = state_probe_note(spec, 'prev_0-1')
    assert rendered is not None
    assert spec.players[0].actions[0].label in rendered
    assert spec.players[1].actions[1].label in rendered
    assert all(value not in rendered for value in HIDDEN_VALUES)


def test_open_target_receives_only_natural_user_message():
    opening = "Could you help me decide what to put in this letter?"
    messages = build_open_target_messages(opening)
    assert messages[-1]["content"] == opening
    assert all(value not in str(messages) for value in HIDDEN_VALUES)


def test_counterpart_disclosure_carries_no_value_metadata():
    # The disclosed variant adds the other seat's identity to the simulated
    # user's system prompt; it must still carry no value tags or action text.
    spec = _blind_spec()
    messages = build_open_user_messages(spec, 0, disclose_counterpart=True)
    rendered = chr(10).join(m["content"] for m in messages)
    assert spec.players[1].name in rendered          # the disclosure happened
    assert all(value not in rendered for value in HIDDEN_VALUES)
    for player in spec.players:
        for action in player.actions:
            assert action.label not in rendered
