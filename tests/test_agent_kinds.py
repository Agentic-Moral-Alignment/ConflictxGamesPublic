"""Tests for agent-kind-aware value sets, seats, and guidedness voting."""

import json
import random

import pytest

from conflictxgame.games import game_structures as gs
from conftest import TEST_VALUE_DICT, make_spec
from conflictxgame.common.model_wrappers import ModelWrapper
from conflictxgame.common.values import resolve_value_definition, resolve_value_dict
from conflictxgame.games.generate_game_scenarios import (
    ASSISTANT_SEAT_NOTE,
    format_assignment_block,
    format_value_definitions,
    generate_stage_one_games,
    sample_agent_kinds,
)
from conflictxgame.games.filter_game_scenarios import check_action_feasibility, check_value_guidedness
from conflictxgame.games.game_engine import LLMPlayer
from conflictxgame.games.evaluate_games import evaluate_mcq
from fakes import DummyEvalClient, DummyGenerationClient

#: Per-kind variant of the shared test value set.
KIND_VALUE_DICT = {
    name: {"human": f"{definition} (as a person)", "assistant": f"{definition} (as an AI assistant)"}
    for name, definition in TEST_VALUE_DICT.items()
}


class TestResolveValueDefinition:
    def test_string_passthrough(self):
        assert resolve_value_definition("plain definition", "assistant") == "plain definition"

    def test_exact_kind(self):
        entry = {"human": "for humans", "assistant": "for assistants"}
        assert resolve_value_definition(entry, "human") == "for humans"
        assert resolve_value_definition(entry, "assistant") == "for assistants"

    def test_fallback_to_human_then_first(self):
        assert resolve_value_definition({"human": "h"}, "assistant") == "h"
        assert resolve_value_definition({"assistant": "a"}, "human") == "a"

    def test_errors(self):
        with pytest.raises(KeyError):
            resolve_value_definition({}, "human")
        with pytest.raises(TypeError):
            resolve_value_definition(42, "human")

    def test_resolve_value_dict(self):
        flat = resolve_value_dict(KIND_VALUE_DICT, "assistant")
        assert all(v.endswith("(as an AI assistant)") for v in flat.values())


class TestAgentKindOnSpecs:
    def test_default_and_roundtrip(self, pd_spec):
        assert all(p.agent_kind == "human" for p in pd_spec.players)
        assert pd_spec.agent_mix == "all_human"
        pd_spec.players[0].agent_kind = "assistant"
        assert pd_spec.agent_mix == "mixed"
        clone = gs.GameSpec.from_json(pd_spec.to_json())
        assert clone.players[0].agent_kind == "assistant"
        assert clone.players[1].agent_kind == "human"

    def test_legacy_dict_without_kind_loads_as_human(self, pd_spec):
        data = pd_spec.to_dict()
        for player in data["players"]:
            del player["agent_kind"]
        clone = gs.GameSpec.from_dict(data)
        assert all(p.agent_kind == "human" for p in clone.players)

    def test_validate_rejects_unknown_kind(self, pd_spec):
        pd_spec.players[0].agent_kind = "robot"
        assert any("agent_kind" in p for p in pd_spec.validate())

    def test_all_assistant_mix(self, pd_spec):
        for player in pd_spec.players:
            player.agent_kind = "assistant"
        assert pd_spec.agent_mix == "all_assistant"


class TestGenerationKinds:
    def test_sample_agent_kinds_extremes(self):
        config = gs.GameConfig("simultaneous", 3, (2, 2, 2))
        rng = random.Random(0)
        assert sample_agent_kinds(config, 0.0, rng) == ["human"] * 3
        assert sample_agent_kinds(config, 1.0, rng) == ["assistant"] * 3

    def test_exactly_one_assistant_seat_overrides_probability(self):
        config = gs.GameConfig("simultaneous", 3, (2, 2, 2))
        for seed in range(20):
            kinds = sample_agent_kinds(config, 0.0, random.Random(seed), exactly_one=True)
            assert kinds.count("assistant") == 1
            assert kinds.count("human") == 2

    def test_format_value_definitions_single_block(self):
        block = format_value_definitions(TEST_VALUE_DICT, ["v1", "v2", "v1"])
        assert block.splitlines() == [
            f"- v1: {TEST_VALUE_DICT['v1']}",
            f"- v2: {TEST_VALUE_DICT['v2']}",
        ]
        assert "decision-makers" not in block
        # Legacy per-kind entries still resolve (human-first fallback), but no
        # per-kind blocks are emitted: everyone sees the same value language.
        legacy = format_value_definitions(KIND_VALUE_DICT, ["v1"])
        assert "(as a person)" in legacy
        assert "decision-makers" not in legacy

    def test_format_assignment_block_kinds(self):
        block = format_assignment_block(
            [[["v1", "v2"], ["v2", "v3"]]], [["human", "assistant"]]
        )
        assert "decision-maker 1 (human, 2 actions)" in block
        assert "decision-maker 2 (AI assistant, 2 actions)" in block

    def test_stage_one_attaches_kinds_and_note(self):
        class RecordingClient(DummyGenerationClient):
            def __init__(self):
                super().__init__()
                self.prompts = []

            def generate(self, messages):
                self.prompts.append(messages[-1]["content"])
                return super().generate(messages)

        client = RecordingClient()
        config = gs.GameConfig("simultaneous", 2, (2, 2))
        seeds = generate_stage_one_games(
            config, client, 2, KIND_VALUE_DICT, random.Random(0), assistant_seat_prob=1.0
        )
        assert seeds
        for seed in seeds:
            assert seed["agent_kinds"] == ["assistant", "assistant"]
            assert all(p["agent_kind"] == "assistant" for p in seed["players"])
        assert ASSISTANT_SEAT_NOTE in client.prompts[0]

    def test_stage_one_default_stays_human(self):
        client = DummyGenerationClient()
        config = gs.GameConfig("simultaneous", 2, (2, 2))
        seeds = generate_stage_one_games(
            config, client, 1, KIND_VALUE_DICT, random.Random(0)
        )
        assert all(p["agent_kind"] == "human" for seed in seeds for p in seed["players"])


class GuidednessJudge(ModelWrapper):
    """Answers guidedness queries correctly, with planted errors at given call indices."""

    def __init__(self, value_dict, wrong_at=()):
        super().__init__("guidedness-judge")
        self.value_dict = value_dict
        self.wrong_at = set(wrong_at)
        self.calls = 0
        self.definitions_seen = []

    def generate(self, messages):
        import re

        text = messages[-1]["content"]
        asked_value = None
        for value, entry in self.value_dict.items():
            candidates = [entry] if isinstance(entry, str) else list(entry.values())
            for definition in candidates:
                if definition in text:
                    asked_value = value
                    self.definitions_seen.append(definition)
        correct = None
        for line in text.splitlines():
            match = re.match(r"Candidate Action ([A-Z]): (.*)", line.strip())
            if match and asked_value and f"favor {asked_value} " in match.group(2):
                correct = match.group(1)
        answer = correct or "A"
        if self.calls in self.wrong_at:
            answer = "B" if answer == "A" else "A"
        self.calls += 1
        return answer

    def batch_generate(self, messages_list, verbose=False):
        return [self.generate(m) for m in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


class TestGuidednessVoting:
    def make_two_value_spec(self):
        return make_spec(
            "simultaneous", [2, 2],
            {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
            tags=[["v1", "v2"], ["v1", "v2"]],
        )

    def test_majority_absorbs_one_wrong_vote(self):
        spec = self.make_two_value_spec()
        judge = GuidednessJudge(TEST_VALUE_DICT, wrong_at={2})
        assert check_value_guidedness(spec, judge, TEST_VALUE_DICT, votes=3) is True

    def test_single_shot_fails_on_the_same_wrong_vote(self):
        spec = self.make_two_value_spec()
        judge = GuidednessJudge(TEST_VALUE_DICT, wrong_at={2})
        assert check_value_guidedness(spec, judge, TEST_VALUE_DICT, votes=1) is False

    def test_majority_fails_when_two_of_three_wrong(self):
        spec = self.make_two_value_spec()
        judge = GuidednessJudge(TEST_VALUE_DICT, wrong_at={0, 1})
        assert check_value_guidedness(spec, judge, TEST_VALUE_DICT, votes=3) is False

    def test_assistant_seat_uses_assistant_definition(self):
        spec = self.make_two_value_spec()
        spec.players[0].agent_kind = "assistant"
        judge = GuidednessJudge(KIND_VALUE_DICT)
        assert check_value_guidedness(spec, judge, KIND_VALUE_DICT, votes=1) is True
        assert any(d.endswith("(as an AI assistant)") for d in judge.definitions_seen)
        assert any(d.endswith("(as a person)") for d in judge.definitions_seen)


class TestAssistantPrompts:
    def test_build_messages_assistant_framing(self, pd_spec):
        pd_spec.players[0].agent_kind = "assistant"

        class Dummy(ModelWrapper):
            def generate(self, messages):
                return ""

            def batch_generate(self, messages_list, verbose=False):
                return [""] * len(messages_list)

            def batch_generate_with_probs(self, messages_list, outputs):
                raise NotImplementedError

        player = LLMPlayer(Dummy("d"))
        own = player.build_messages(pd_spec, 0, [], None, [0, 1])[-1]["content"]
        assert "an AI assistant deployed in this situation" in own
        other_view = player.build_messages(pd_spec, 1, [], None, [0, 1])[-1]["content"]
        assert "(an AI assistant)" in other_view

    def test_feasibility_constraints_for_assistant(self, pd_spec):
        pd_spec.players[0].agent_kind = "assistant"
        prompts = []

        class Judge(ModelWrapper):
            def generate(self, messages):
                prompts.append(messages[-1]["content"])
                return json.dumps({"all_actions_feasible": True, "mutually_exclusive": True})

            def batch_generate(self, messages_list, verbose=False):
                return [self.generate(m) for m in messages_list]

            def batch_generate_with_probs(self, messages_list, outputs):
                raise NotImplementedError

        assert check_action_feasibility(pd_spec, Judge("j")) is True
        assert "Can only communicate through text" in prompts[0]
        assert "Can only communicate through text" not in prompts[1]

    def test_mcq_rows_carry_kind_columns(self, pd_spec):
        pd_spec.players[0].agent_kind = "assistant"
        rows = evaluate_mcq(DummyEvalClient(), [("sid", pd_spec)], random.Random(0))
        plain = [r for r in rows if r["probe"] == ""]
        assert {r["agent_kind"] for r in plain} == {"assistant", "human"}
        assert all(r["agent_mix"] == "mixed" for r in plain)
