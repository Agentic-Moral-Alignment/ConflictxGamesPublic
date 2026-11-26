"""Predict-then-choose probes: the counterpart prediction precedes the choice."""

import random

from conftest import make_spec
from conflictxgame.common.model_wrappers import ModelWrapper
from conflictxgame.games.evaluate_games import (
    create_game_mcq_prompt,
    create_prediction_prompt,
    evaluate_mcq,
    parse_prediction,
    split_mcq_prompt,
)
from conflictxgame.games.game_structures import as_repeated

PAYOFFS = {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]}
TAGS = ("hidden_tag_one", "hidden_tag_two")


def spec():
    game = make_spec("simultaneous", [2, 2], dict(PAYOFFS), game_id="pred-0", tags=[list(TAGS)] * 2)
    for seat, player in enumerate(game.players):
        player.agent_kind = "assistant"
        for i, action in enumerate(player.actions):
            # The fixture's default descriptions name the tag; overwrite both, as
            # test_value_blind_prompts does, so the check below is meaningful.
            action.label = f"seat{seat} option {i}"
            action.description = f"Take concrete action {i + 1}."
    return as_repeated(game, 5)


class Scripted(ModelWrapper):
    """Answers prediction turns and choice turns differently; records conversations."""

    def __init__(self, prediction="B, Yes", choice="A"):
        super().__init__("scripted")
        self.prediction, self.choice, self.conversations = prediction, choice, []

    def generate(self, messages):
        self.conversations.append(messages)
        last = messages[-1]["content"]
        if "Before " in last and "answer two questions" in last:
            return self.prediction
        if "Likert" in last or "rank" in last.lower():
            return "4"
        return self.choice

    def batch_generate(self, messages_list, verbose=False):
        return [self.generate(m) for m in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


def test_prediction_prompt_lists_the_counterparts_menu_without_value_tags():
    game = spec()
    prompt, _ = create_game_mcq_prompt(game, 0, random.Random(0))
    scenario, question = split_mcq_prompt(prompt, game.players[0].name)
    pred, letters = create_prediction_prompt(game, 0, scenario, [1, 0])
    assert question.startswith(f"What should {game.players[0].name} do")
    assert f"A) seat1 option 1" in pred and f"B) seat1 option 0" in pred
    assert f"Will {game.players[1].name} learn what {game.players[0].name} chooses?" in pred
    assert letters == {"A": 1, "B": 0}
    assert all(tag not in pred for tag in TAGS)


def test_parse_prediction():
    letters = {"A": 1, "B": 0}
    assert parse_prediction("B, Yes", letters) == (0, "yes")
    assert parse_prediction("(A) no", letters) == (1, "no")
    assert parse_prediction("I cannot say", letters) == ("INVALID", "INVALID")


def test_choice_follows_the_models_own_prediction_in_one_conversation():
    client = Scripted(prediction="B, Yes", choice="A")
    rows = evaluate_mcq(client, [("pred-0", spec())], random.Random(0),
                        state_probes=True, role_cue="explicit", predict_first=True)
    choice_turns = [c for c in client.conversations
                    if len(c) == 4 and c[2]["role"] == "assistant"]
    assert len(choice_turns) == 10                       # 2 seats x 5 history states
    for conv in choice_turns:
        assert conv[2]["content"] == "B, Yes"
        assert "answer two questions" in conv[1]["content"]
        assert conv[3]["content"].startswith("What should")
        assert "WHAT YOU REMEMBER" in conv[1]["content"]     # history is in the shared scenario turn
    mcq_rows = [r for r in rows if r["probe"]]
    assert all(r["predicted_cp_learns"] == "yes" for r in mcq_rows)
    assert all(r["predicted_cp_action"] in (0, 1) for r in mcq_rows)
    assert all(r["predicted_cp_value_tag"] in TAGS for r in mcq_rows)


def test_predict_turn_choice_tolerates_a_carried_over_prefix_but_default_does_not():
    rows = evaluate_mcq(Scripted(prediction="A, No", choice=": B"), [("pred-0", spec())],
                        random.Random(0), state_probes=True, role_cue="explicit", predict_first=True)
    assert all(r["parsed_ok"] and r["choice_raw"] == ": B" for r in rows if r["probe"])
    strict = evaluate_mcq(Scripted(choice=": B"), [("pred-0", spec())], random.Random(0),
                          state_probes=True)
    assert not any(r["parsed_ok"] for r in strict if r["probe"])


def test_default_mode_sends_no_prediction_turn_and_adds_no_columns():
    client = Scripted()
    rows = evaluate_mcq(client, [("pred-0", spec())], random.Random(0), state_probes=True)
    assert not any("answer two questions" in c[-1]["content"] for c in client.conversations)
    assert all("predicted_cp_action" not in r for r in rows)
