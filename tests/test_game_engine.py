"""Tests for the game engine: structures, batching, parsing, elicitation."""

import json
import random
from typing import Dict, List

import pytest

from conflictxgame.games import game_structures as gs
from conftest import make_spec
from conflictxgame.common.model_wrappers import ModelWrapper
from conflictxgame.games.scripted_strategies import make_strategy
from conflictxgame.games.game_engine import (
    GameEngine,
    GameSetup,
    LLMPlayer,
    ScriptedPlayer,
    build_likert_messages,
    build_outcome_ranking_messages,
    normalize_likert,
    parse_ranking,
    process_likert_response,
    shuffled_outcome_order,
)


class CannedClient(ModelWrapper):
    """A ModelWrapper returning canned responses (no API access needed)."""

    def __init__(self, responder, model_name="canned"):
        super().__init__(model_name)
        self.responder = responder
        self.calls: List[List[dict]] = []

    def generate(self, messages):
        self.calls.append(messages)
        return self.responder(messages)

    def batch_generate(self, messages_list, verbose=False):
        return [self.generate(messages) for messages in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


def first_option_responder(messages):
    """Pick the first listed option label from an action prompt."""
    text = messages[-1]["content"]
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('- "'):
            label = line.split('"')[1]
            return json.dumps({"reasoning": "canned", "action": label})
    return "no options found"


def scripted_setup(spec, names):
    return GameSetup(
        spec=spec, agents=[ScriptedPlayer(make_strategy(n)) for n in names]
    )


class TestEngineStructures:
    def test_repeated_horizon_and_payoffs(self, pd_spec):
        engine = GameEngine(rng=random.Random(0))
        result = engine.play_batch(
            [scripted_setup(pd_spec, ["tit_for_tat", "always_defect"])]
        )[0]
        profiles = result.profiles()
        assert len(profiles) == pd_spec.config.horizon
        assert profiles[0] == (0, 1)  # TFT opens cooperatively, gets defected on
        assert all(p == (1, 1) for p in profiles[1:])
        assert result.total_payoffs() == [0 + 4 * 1, 5 + 4 * 1]

    def test_sequential_within_round_visibility(self, pd_spec):
        seq = make_spec(
            "sequential", [2, 2],
            {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
        )
        engine = GameEngine(rng=random.Random(0))
        # Grim trigger moves second and sees the first mover's defection.
        result = engine.play_batch(
            [scripted_setup(seq, ["always_defect", "grim_trigger"])]
        )[0]
        assert result.profiles() == [(1, 1)]

    def test_single_decision(self):
        spec = make_spec("single", [3], {"0": [1.0], "1": [2.0], "2": [3.0]})
        engine = GameEngine(rng=random.Random(0))
        result = engine.play_batch([scripted_setup(spec, ["always_last"])])[0]
        assert result.profiles() == [(2,)]

    def test_mixed_structure_batch(self, pd_spec):
        one_shot = make_spec(
            "simultaneous", [2, 2],
            {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
        )
        engine = GameEngine(rng=random.Random(0))
        results = engine.play_batch(
            [
                scripted_setup(pd_spec, ["always_cooperate", "always_cooperate"]),
                scripted_setup(one_shot, ["always_defect", "always_cooperate"]),
            ]
        )
        assert len(results[0].profiles()) == pd_spec.config.horizon
        assert results[1].profiles() == [(1, 0)]

    def test_determinism_with_seed(self, pd_spec):
        def run():
            engine = GameEngine(rng=random.Random(7))
            return engine.play_batch(
                [scripted_setup(pd_spec, ["random", "win_stay_lose_shift"])]
            )[0].profiles()

        assert run() == run()


class TestLLMPlayer:
    def test_llm_player_plays_via_labels(self, pd_spec):
        client = CannedClient(first_option_responder)
        engine = GameEngine(rng=random.Random(0))
        setup = GameSetup(
            spec=pd_spec,
            agents=[LLMPlayer(client), ScriptedPlayer(make_strategy("always_cooperate"))],
            metadata={"focal_seats": [0]},
        )
        result = engine.play_batch([setup])[0]
        assert len(result.rounds) == pd_spec.config.horizon
        for round_decisions in result.rounds:
            assert round_decisions[0].parsed_ok
            assert round_decisions[0].reasoning == "canned"
        # The chosen label always matches the seat's actual menu.
        labels = {a.label for a in pd_spec.players[0].actions}
        assert all(rnd[0].label in labels for rnd in result.rounds)

    def test_unparseable_falls_back_to_random_flagged(self, pd_spec):
        client = CannedClient(lambda messages: "I refuse to answer in the requested format.")
        engine = GameEngine(rng=random.Random(0))
        setup = GameSetup(
            spec=pd_spec,
            agents=[LLMPlayer(client), ScriptedPlayer(make_strategy("always_cooperate"))],
        )
        result = engine.play_batch([setup])[0]
        assert all(not rnd[0].parsed_ok for rnd in result.rounds)
        assert all(rnd[0].action in (0, 1) for rnd in result.rounds)

    def test_judge_fallback(self, pd_spec):
        player_client = CannedClient(lambda messages: "Well, I suppose sharing seems right.")
        label = pd_spec.players[0].actions[1].label
        judge_client = CannedClient(lambda messages: label)
        engine = GameEngine(judge_client=judge_client, rng=random.Random(0))
        setup = GameSetup(
            spec=pd_spec,
            agents=[LLMPlayer(player_client), ScriptedPlayer(make_strategy("always_cooperate"))],
        )
        result = engine.play_batch([setup])[0]
        assert all(rnd[0].action == 1 for rnd in result.rounds)

    def test_parse_action_exact_and_substring(self, pd_spec):
        player = LLMPlayer(CannedClient(lambda m: ""))
        label = pd_spec.players[0].actions[0].label
        assert player.parse_action(json.dumps({"action": label}), pd_spec, 0) == 0
        assert player.parse_action(json.dumps({"action": label.upper()}), pd_spec, 0) == 0
        assert player.parse_action(f"I choose {label}, definitely.", pd_spec, 0) == 0
        assert player.parse_action("gibberish", pd_spec, 0) is None

    def test_history_and_partial_round_in_prompt(self, pd_spec):
        player = LLMPlayer(CannedClient(lambda m: ""))
        messages = player.build_messages(pd_spec, 0, [(0, 1)], None, [0, 1])
        text = messages[-1]["content"]
        assert "WHAT YOU REMEMBER OF EARLIER ROUNDS" in text
        assert pd_spec.players[1].actions[1].label in text

        seq = make_spec(
            "sequential", [2, 2],
            {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
        )
        messages = player.build_messages(seq, 1, [], {0: 1}, [0, 1])
        assert "chose:" in messages[-1]["content"]


class TestElicitation:
    def test_likert_parsing_and_normalization(self):
        assert process_likert_response("5") == 5
        assert process_likert_response("Rating: 3.") == 3  # '3.' parses as an integer float
        assert process_likert_response("I rate it 7 out of 7") == 7
        assert process_likert_response("8") == "INVALID"  # out of the 1..7 range
        assert process_likert_response("nine") == "INVALID"
        assert normalize_likert(1) == -1.0
        assert normalize_likert(7) == 1.0
        assert normalize_likert(4) == 0.0

    def test_likert_messages_mention_action(self, pd_spec):
        messages = build_likert_messages(pd_spec, 0, 1)
        assert pd_spec.players[0].actions[1].label in messages[-1]["content"]

    def test_outcome_ranking_roundtrip(self, pd_spec):
        rng = random.Random(3)
        order = shuffled_outcome_order(pd_spec, rng)
        assert sorted(order) == sorted(pd_spec.payoffs.keys())
        messages = build_outcome_ranking_messages(pd_spec, 0, order)
        assert "O1:" in messages[-1]["content"]

        response = json.dumps({"ranking": [f"O{i + 1}" for i in range(len(order))]})
        ranked = parse_ranking(response, order)
        assert ranked == list(order)

    def test_parse_ranking_rejects_bad_inputs(self, pd_spec):
        order = list(pd_spec.payoffs.keys())
        assert parse_ranking("not json", order) is None
        assert parse_ranking(json.dumps({"ranking": ["O1", "O1", "O2", "O3"]}), order) is None
        assert parse_ranking(json.dumps({"ranking": ["O1"]}), order) is None
        assert parse_ranking(json.dumps({"ranking": ["O1", "O2", "O9", "O4"]}), order) is None
