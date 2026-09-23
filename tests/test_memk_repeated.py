"""Tests for mem-k bounded memory, one-shot -> repeated conversion, and
cross-scenario sequence evaluation."""

import json
import random

import pytest

from conflictxgame.games import game_structures as gs
from conftest import make_spec
from conflictxgame.games.game_engine import GameEngine, LLMPlayer
from conflictxgame.games.evaluate_games import (
    build_play_setups,
    compose_sequences,
    evaluate_play,
    evaluate_play_sequences,
    format_carryover_note,
)
from test_game_engine import CannedClient, first_option_responder

PD_PAYOFFS = {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]}


def one_shot_spec(game_id=None, tags=None):
    return make_spec("simultaneous", [2, 2], dict(PD_PAYOFFS), game_id=game_id, tags=tags)


class TestAsRepeated:
    def test_roundtrip_preserves_players_and_payoffs(self):
        spec = one_shot_spec()
        clone = gs.as_repeated(spec, 5)
        assert clone.config.structure == "repeated"
        assert clone.config.horizon == 5
        assert clone.metadata["derived_from_structure"] == "simultaneous"
        assert clone.payoffs == spec.payoffs
        assert [p.name for p in clone.players] == [p.name for p in spec.players]
        assert clone.validate() == []
        # The original spec is untouched.
        assert spec.config.structure == "simultaneous"
        assert spec.config.horizon == 1

    def test_rejects_non_simultaneous_and_short_horizons(self, pd_spec):
        seq = make_spec("sequential", [2, 2], dict(PD_PAYOFFS))
        with pytest.raises(ValueError):
            gs.as_repeated(seq, 5)
        with pytest.raises(ValueError):
            gs.as_repeated(pd_spec, 5)  # already repeated
        with pytest.raises(ValueError):
            gs.as_repeated(one_shot_spec(), 1)


class TestMemKPromptRendering:
    def history(self):
        return [(0, 1), (1, 1), (0, 0)]

    def test_full_history_by_default(self, pd_spec):
        player = LLMPlayer(CannedClient(lambda m: ""))
        text = player.build_messages(pd_spec, 0, self.history(), None, [0, 1])[-1]["content"]
        assert "Previous round" in text and "2 rounds ago" in text and "3 rounds ago" in text
        # The horizon and the elapsed count are deliberately withheld.
        assert "total round(s) played" not in text
        assert "5 rounds" not in text

    def test_mem_1_shows_only_the_previous_round(self, pd_spec):
        player = LLMPlayer(CannedClient(lambda m: ""), memory_k=1)
        text = player.build_messages(pd_spec, 0, self.history(), None, [0, 1])[-1]["content"]
        assert "Previous round" in text
        assert "2 rounds ago" not in text and "3 rounds ago" not in text
        assert "record only goes back 1 round(s)" in text
        assert "total round(s) played" not in text

    def test_mem_0_shows_no_history_at_all(self, pd_spec):
        player = LLMPlayer(CannedClient(lambda m: ""), memory_k=0)
        text = player.build_messages(pd_spec, 0, self.history(), None, [0, 1])[-1]["content"]
        assert "no record of what was chosen" in text
        assert "Previous round" not in text
        assert "total round(s) played" not in text

    def test_first_round_says_it_just_started(self, pd_spec):
        player = LLMPlayer(CannedClient(lambda m: ""), memory_k=1)
        text = player.build_messages(pd_spec, 0, [], None, [0, 1])[-1]["content"]
        assert "the interaction has just started" in text

    def test_horizon_is_never_disclosed(self, pd_spec):
        player = LLMPlayer(CannedClient(lambda m: ""), memory_k=1)
        for hist in ([], self.history()):
            text = player.build_messages(pd_spec, 0, hist, None, [0, 1])[-1]["content"]
            assert "repeats for" not in text
            assert str(pd_spec.config.horizon) + " round" not in text
            assert "running online" in text

    def test_carryover_note_is_injected(self, pd_spec):
        note = "PRIOR INTERACTIONS: canned note."
        player = LLMPlayer(CannedClient(lambda m: ""), carryover_note=note)
        text = player.build_messages(pd_spec, 0, [], None, [0, 1])[-1]["content"]
        assert note in text


class TestBuildPlaySetupsThreading:
    def test_memory_k_reaches_every_llm_player(self):
        spec = gs.as_repeated(one_shot_spec(), 3)
        client = CannedClient(first_option_responder)
        setups = build_play_setups(
            "sid", spec, client, "scripted", ["tit_for_tat"], [], 2, None, memory_k=1
        )
        assert setups
        for setup in setups:
            for agent in setup.agents:
                if isinstance(agent, LLMPlayer):
                    assert agent.memory_k == 1

    def test_play_rows_carry_rounds_played(self):
        spec = gs.as_repeated(one_shot_spec(game_id="rp-0"), 3)
        client = CannedClient(first_option_responder)
        engine = GameEngine(rng=random.Random(0))
        setups = build_play_setups(
            "sid", spec, client, "scripted", ["always_cooperate"], [], 1, None, memory_k=1
        )
        rows = evaluate_play(engine, setups, batch_size=4)
        assert rows
        assert all(row["rounds_played"] == 3 for row in rows)


class TestCarryoverNote:
    def records(self, n):
        return [
            {
                "own_label": f"label-{i}",
                "own_value_tag": "v1",
                "others": [("Riley Chen", f"other-{i}")],
                "own_payoff": float(i),
            }
            for i in range(n)
        ]

    def test_mem_1_keeps_exactly_last_record(self):
        note = format_carryover_note(self.records(4), 1)
        assert "4 earlier situation(s)" in note
        assert "label-3" in note and "label-2" not in note
        assert "Situation 4:" in note
        assert "most recent 1" in note

    def test_full_and_zero_memory(self):
        full = format_carryover_note(self.records(3), None)
        assert all(f"label-{i}" in full for i in range(3))
        blind = format_carryover_note(self.records(3), 0)
        assert "no record" in blind and "label-0" not in blind


class TestSequences:
    def make_pool(self, n=6):
        pool = []
        tag_cycle = [["v1", "v2"], ["v2", "v3"], ["v1", "v3"]]
        for i in range(n):
            tags = [tag_cycle[i % 3], tag_cycle[i % 3]]
            pool.append((f"s{i}", one_shot_spec(game_id=f"g-{i}", tags=tags)))
        return pool

    def test_composition_is_deterministic_per_seed(self):
        pool = self.make_pool()
        a = compose_sequences(pool, 3, "shuffle", random.Random(5))
        b = compose_sequences(pool, 3, "shuffle", random.Random(5))
        assert [[sid for sid, _ in seq] for seq in a] == [[sid for sid, _ in seq] for seq in b]
        assert sum(len(seq) for seq in a) == len(pool)

    def test_value_pair_grouping_is_homogeneous(self):
        pool = self.make_pool(9)
        sequences = compose_sequences(pool, 2, "value-pair", random.Random(0))
        for seq in sequences:
            pairs = {
                tuple(sorted({a.value_tag for a in spec.players[0].actions}))
                for _, spec in seq
            }
            assert len(pairs) == 1

    def test_sequence_rows_carry_positions_and_carryover(self):
        pool = self.make_pool(4)
        client = CannedClient(first_option_responder)
        engine = GameEngine(rng=random.Random(0))
        rows = evaluate_play_sequences(
            engine, pool, client, "scripted", ["always_cooperate"],
            repetitions=1, steering_prompt=None, memory_k=1,
            length=4, grouping="shuffle", rng=random.Random(0), batch_size=8,
        )
        assert len(rows) == 4
        assert [row["sequence_pos"] for row in rows] == [0, 1, 2, 3]
        assert all(row["sequence_id"] == 0 for row in rows)
        assert all(row["rounds_played"] == 4 for row in rows)
        assert all(row["parsed_ok"] for row in rows)
        # From position 1 on, the acting model's prompt carried a mem-1 note
        # describing exactly one prior situation.
        note_prompts = [
            call[-1]["content"] for call in client.calls
            if "PRIOR INTERACTIONS" in call[-1]["content"]
        ]
        assert len(note_prompts) == 3
        for text in note_prompts:
            assert text.count("Situation ") == 1

    def test_zd_strategies_are_skipped_in_sequence_mode(self):
        pool = self.make_pool(2)
        client = CannedClient(first_option_responder)
        engine = GameEngine(rng=random.Random(0))
        rows = evaluate_play_sequences(
            engine, pool, client, "scripted", ["zd_extort", "always_defect"],
            repetitions=1, steering_prompt=None, memory_k=1,
            length=2, grouping="shuffle", rng=random.Random(0), batch_size=8,
        )
        assert rows
        assert {row["opponent_config"] for row in rows} == {"scripted:always_defect"}


class TestStateProbes:
    """The mem-1 state space enumerated directly, instead of sampled by play."""

    def spec(self):
        return gs.as_repeated(one_shot_spec(), 5)

    def test_two_by_two_has_exactly_five_states(self):
        from conflictxgame.games.evaluate_games import state_probe_labels
        labels = state_probe_labels(self.spec())
        assert labels == ["fresh", "prev_0-0", "prev_0-1", "prev_1-0", "prev_1-1"]

    def test_every_state_renders_and_none_leaks_the_horizon(self):
        from conflictxgame.games.evaluate_games import state_probe_labels, state_probe_note
        spec = self.spec()
        for label in state_probe_labels(spec):
            note = state_probe_note(spec, label)
            assert note, label
            assert "repeats for" not in note
            assert "total round(s) played" not in note
            assert str(spec.config.horizon) + " round" not in note

    def test_fresh_state_says_it_just_started(self):
        from conflictxgame.games.evaluate_games import state_probe_note
        assert "just started" in state_probe_note(self.spec(), "fresh")

    def test_previous_round_names_both_players_actions(self):
        from conflictxgame.games.evaluate_games import state_probe_note
        spec = self.spec()
        note = state_probe_note(spec, "prev_0-1")
        assert "Previous round" in note
        assert spec.players[0].actions[0].label in note
        assert spec.players[1].actions[1].label in note

    def test_rejects_malformed_or_out_of_range_labels(self):
        from conflictxgame.games.evaluate_games import state_probe_note
        spec = self.spec()
        for bad in ["prev_9-0", "prev_0", "prev_x-y", "nonsense"]:
            assert state_probe_note(spec, bad) is None
