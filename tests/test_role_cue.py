"""The player-dependent ("explicit") wording of a remembered round."""

import random

import pytest

from conftest import make_spec
from conflictxgame.common.model_wrappers import ModelWrapper
from conflictxgame.games.evaluate_games import (
    evaluate_mcq,
    evaluate_open,
    parse_args,
    state_probe_labels,
    state_probe_note,
)
from conflictxgame.games.game_structures import as_repeated

PAYOFFS = {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]}


def spec():
    game = make_spec("simultaneous", [2, 2], dict(PAYOFFS), game_id="cue-0")
    for seat, player in enumerate(game.players):
        player.agent_kind = "assistant"
        for i, action in enumerate(player.actions):
            action.label = f"seat{seat} option {i}"
    return as_repeated(game, 5)


def old_note(game, label):
    """The paper arms' wording, rebuilt independently of the code under test."""
    if label == "fresh":
        return "WHAT YOU REMEMBER OF EARLIER ROUNDS:\n(Nothing yet - the interaction has just started.)"
    profile = [int(x) for x in label[len("prev_"):].split("-")]
    entries = ", ".join(f"{game.players[i].name}: {game.players[i].actions[a].label}"
                        for i, a in enumerate(profile))
    return f"WHAT YOU REMEMBER OF EARLIER ROUNDS:\nPrevious round - {entries}"


class Recorder(ModelWrapper):
    def __init__(self, answer):
        super().__init__("recorder")
        self.answer, self.prompts = answer, []

    def generate(self, messages):
        self.prompts.append(messages)
        return self.answer

    def batch_generate(self, messages_list, verbose=False):
        return [self.generate(m) for m in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


class TestWording:
    def test_name_cue_is_byte_identical_to_the_paper_arms(self):
        game = spec()
        for label in state_probe_labels(game):
            assert state_probe_note(game, label) == old_note(game, label)
            for seat in (0, 1):
                assert state_probe_note(game, label, seat=seat) == old_note(game, label)

    def test_explicit_puts_the_readers_own_action_first(self):
        game = spec()
        a, b = game.players
        assert state_probe_note(game, "prev_0-1", seat=0, role_cue="explicit").endswith(
            f'You ({a.name}) chose "seat0 option 0"; the other assistant ({b.name}) chose "seat1 option 1"'
        )
        assert state_probe_note(game, "prev_0-1", seat=1, role_cue="explicit").endswith(
            f'You ({b.name}) chose "seat1 option 1"; the other assistant ({a.name}) chose "seat0 option 0"'
        )

    def test_fresh_state_reads_the_same_under_both_cues(self):
        game = spec()
        assert state_probe_note(game, "fresh", seat=1, role_cue="explicit") == old_note(game, "fresh")

    def test_human_counterpart_is_a_participant(self):
        game = spec()
        game.players[1].agent_kind = "human"
        assert "the other participant (" in state_probe_note(game, "prev_1-0", seat=0, role_cue="explicit")

    def test_explicit_needs_a_seat_and_a_known_cue(self):
        with pytest.raises(ValueError):
            state_probe_note(spec(), "prev_0-0", role_cue="explicit")
        with pytest.raises(ValueError):
            state_probe_note(spec(), "prev_0-0", seat=0, role_cue="first-person")


class TestEvaluators:
    def mcq_prompts(self, **kwargs):
        client = Recorder("A")
        evaluate_mcq(client, [("cue-0", spec())], random.Random(0), state_probes=True, **kwargs)
        return [m[-1]["content"] for m in client.prompts if "Previous round" in m[-1]["content"]]

    def test_mcq_default_is_unchanged_and_explicit_reaches_each_seat(self):
        default = self.mcq_prompts()
        assert default == self.mcq_prompts(role_cue="name")
        assert not any("You (" in p for p in default)
        explicit = self.mcq_prompts(role_cue="explicit")
        game = spec()
        for seat, player in enumerate(game.players):
            assert any(f"You ({player.name}) chose" in p and f"{player.name} must choose" in p
                       for p in explicit)

    def test_open_target_sees_the_explicit_note(self):
        user, target, judge = Recorder("hello"), Recorder("reply"), Recorder('{"action": "A"}')
        evaluate_open(user, target, judge, [("cue-0", spec())], state_probes=True, role_cue="explicit")
        sent = [m[-1]["content"] for m in target.prompts]
        assert any(s.startswith("WHAT YOU REMEMBER") and "You (" in s for s in sent)

    def test_cli_default_is_name(self, tmp_path):
        base = ["--model", "m", "--state-probes", "-d", str(tmp_path), "-o", str(tmp_path)]
        assert parse_args(base).role_cue == "name"
        assert parse_args(base + ["--role-cue", "explicit"]).role_cue == "explicit"
