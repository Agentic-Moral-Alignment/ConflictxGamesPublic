"""Tests for the open-ended evaluation mode (paper §3.3, Appendix C.5/C.6)."""

import json

import pytest

from conftest import make_spec
from conflictxgame.common.model_wrappers import ModelWrapper
from conflictxgame.games import open_ended
from conflictxgame.games.evaluate_games import (
    evaluate_open,
    evaluate_open_repeated,
    resolve_open_prompt_path,
)
from conflictxgame.games.open_ended import (
    build_open_followup_user_messages,
    build_open_judge_messages,
    build_open_target_messages,
    build_open_target_messages_with_history,
    build_open_user_messages,
    build_open_user_retry_messages,
    counterpart_named,
    parse_open_judge,
)

PD_PAYOFFS = {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]}


def two_by_two(game_id="open-0"):
    return make_spec("simultaneous", [2, 2], dict(PD_PAYOFFS), game_id=game_id)


class RecordingClient(ModelWrapper):
    """Returns a canned response and records every prompt it received."""

    def __init__(self, response, name="canned"):
        super().__init__(name)
        self.response = response
        self.prompts = []

    def generate(self, messages):
        self.prompts.append(messages)
        return self.response(messages) if callable(self.response) else self.response

    def batch_generate(self, messages_list, verbose=False):
        return [self.generate(m) for m in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


def judge_picking(letter, likert=None):
    payload = {"reasoning": "canned", "action": letter}
    if likert is not None:
        payload["likert"] = likert
    return json.dumps(payload)


class TestPromptConstruction:
    def test_assistant_seat_framing(self):
        spec = two_by_two()
        spec.players[0].agent_kind = "assistant"
        system = build_open_user_messages(spec, 0)[0]["content"]
        assert "You are the person interacting with" in system
        assert "an AI assistant" in system
        assert spec.description in system

    def test_human_seat_is_recast_as_consulting_a_user(self):
        spec = two_by_two()
        system = build_open_user_messages(spec, 0)[0]["content"]
        assert f"You are {spec.players[0].name}" in system
        assert "message an AI assistant" in system

    def test_user_turn_is_the_appendix_c5_block(self):
        from conflictxgame.solo.prompts import OPENING_MESSAGE_INSTRUCTIONS

        messages = build_open_user_messages(two_by_two(), 0)
        assert messages[-1]["content"] == OPENING_MESSAGE_INSTRUCTIONS

    def test_target_sees_only_the_user_message(self):
        spec = two_by_two()
        messages = build_open_target_messages("Please help me decide.", None)
        assert len(messages) == 2
        assert messages[-1]["content"] == "Please help me decide."
        joined = " ".join(m["content"] for m in messages)
        assert spec.description not in joined
        assert spec.players[0].actions[0].label not in joined

    def test_steering_prompt_becomes_the_target_system_prompt(self):
        messages = build_open_target_messages("hi", "BE HARMLESS FIRST")
        assert messages[0]["content"] == "BE HARMLESS FIRST"

    def test_judge_lists_actions_and_asks_for_likert_only_when_binary(self):
        spec = two_by_two()
        text = build_open_judge_messages(spec, 0, "user msg", "model reply")[-1]["content"]
        assert "Action A:" in text and "Action B:" in text
        assert "user msg" in text and "model reply" in text
        assert '"likert"' in text

        wide = make_spec(
            "simultaneous", [3, 2],
            {f"{a},{b}": [1, 1] for a in range(3) for b in range(2)},
        )
        wide_text = build_open_judge_messages(wide, 0, "u", "c")[-1]["content"]
        assert "Action C:" in wide_text
        assert '"likert"' not in wide_text


class TestJudgeParsing:
    def test_letter_forms(self):
        assert parse_open_judge(judge_picking("B"), 2)[0] == 1
        assert parse_open_judge(judge_picking("Action B"), 2)[0] == 1
        assert parse_open_judge(judge_picking("a"), 2)[0] == 0

    def test_likert_only_for_binary_menus_and_in_range(self):
        assert parse_open_judge(judge_picking("A", 6), 2)[1] == 6
        assert parse_open_judge(judge_picking("A", 99), 2)[1] == ""
        assert parse_open_judge(judge_picking("A", 6), 3)[1] == ""

    def test_invalid_and_out_of_range_letters(self):
        assert parse_open_judge("no json here", 2)[0] == "INVALID"
        # 'C' is out of range for a 2-action menu.
        assert parse_open_judge(judge_picking("C"), 2)[0] == "INVALID"


class TestEvaluateOpen:
    def run(self, spec, judge_response, scenario_id="sid"):
        user = RecordingClient("I need help right now.", name="user-model")
        target = RecordingClient("Here is what I would do.", name="target-model")
        judge = RecordingClient(judge_response, name="judge-model")
        rows, _ = evaluate_open(user, target, judge, [(scenario_id, spec)])
        return rows, user, target, judge

    def test_rows_carry_choice_and_provenance(self):
        spec = two_by_two()
        rows, user, target, judge = self.run(spec, judge_picking("B", 6))
        assert len(rows) == spec.config.n_players
        row = rows[0]
        assert row["mode"] == "open"
        assert row["choice_action"] == 1
        assert row["choice_label"] == spec.players[0].actions[1].label
        assert row["choice_value_tag"] == spec.players[0].actions[1].value_tag
        assert row["parsed_ok"] is True
        assert row["open_user_prompt"] == "I need help right now."
        assert row["open_completion"] == "Here is what I would do."
        assert row["open_judge_likert"] == 6
        assert row["user_model"] == "user-model"
        assert row["judge_model"] == "judge-model"
        assert row["reasoning"] == "canned"

    def test_one_call_per_seat_per_stage(self):
        spec = two_by_two()
        _, user, target, judge = self.run(spec, judge_picking("A"))
        assert len(user.prompts) == 2      # one opening per seat
        assert len(target.prompts) == 2
        assert len(judge.prompts) == 3     # one pre-flight call, then one per seat

    def test_unreachable_judge_stops_before_any_target_call(self):
        target = RecordingClient("reply", name="target-model")
        judge = RecordingClient("", name="dead-judge")   # '' is what a failed call returns
        with pytest.raises(RuntimeError, match="pre-flight"):
            evaluate_open(RecordingClient("hi", name="u"), target, judge, [("sid", two_by_two())])
        assert target.prompts == []

    def test_unparseable_judge_marks_row_invalid(self):
        rows, _, _, _ = self.run(two_by_two(), "I cannot tell.")
        assert all(row["choice_action"] == "INVALID" for row in rows)
        assert all(row["parsed_ok"] is False for row in rows)

    def test_steering_prompt_reaches_only_the_target(self):
        spec = two_by_two()
        user = RecordingClient("msg", name="u")
        target = RecordingClient("reply", name="t")
        judge = RecordingClient(judge_picking("A"), name="j")
        evaluate_open(user, target, judge, [("sid", spec)], steering_prompt="STEER ME")
        assert all(p[0]["content"] == "STEER ME" for p in target.prompts)
        assert all("STEER ME" not in p[0]["content"] for p in user.prompts)
        assert all("STEER ME" not in p[-1]["content"] for p in judge.prompts)

    @pytest.mark.parametrize('prompt_cache', [{}, None])
    def test_state_probes_generate_one_opening_per_seat(self, prompt_cache):
        spec = two_by_two()
        user = RecordingClient(counting_response('u'), name='user-model')
        target = RecordingClient('reply', name='target-model')
        judge = RecordingClient(judge_picking('A'), name='judge-model')
        rows, _ = evaluate_open(
            user, target, judge, [('sid', spec)], state_probes=True,
            prompt_cache=prompt_cache,
        )
        assert len(user.prompts) == 2
        for seat in range(2):
            seat_rows = [row for row in rows if row['seat'] == seat]
            assert len(seat_rows) == 5
            assert len({row['open_user_prompt'] for row in seat_rows}) == 1

    def test_state_probe_openings_are_reused_on_second_cached_run(self):
        spec = two_by_two()
        cache = {}
        first_user = RecordingClient(counting_response('u'), name='user-model')
        evaluate_open(
            first_user, RecordingClient('reply'), RecordingClient(judge_picking('A')),
            [('sid', spec)], state_probes=True, prompt_cache=cache,
        )
        second_user = RecordingClient('unused', name='user-model')
        rows, _ = evaluate_open(
            second_user, RecordingClient('reply'), RecordingClient(judge_picking('A')),
            [('sid', spec)], state_probes=True, prompt_cache=cache,
        )
        assert second_user.prompts == []
        assert len(rows) == 10


def counting_response(prefix):
    """A callable RecordingClient response yielding prefix0, prefix1, ... in call order."""
    counter = {"n": 0}

    def _respond(messages):
        counter["n"] += 1
        return f"{prefix}{counter['n'] - 1}"

    return _respond


class TestOpenFollowupAndHistoryPrompts:
    def test_followup_sees_full_history_with_roles_swapped(self):
        spec = two_by_two()
        history = [
            {"sent": "u0", "received": "t0"},
            {"sent": "u1", "received": "t1"},
        ]
        messages = build_open_followup_user_messages(spec, 0, history)
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "assistant", "content": "u0"}
        assert messages[2] == {"role": "user", "content": "t0"}
        assert messages[3] == {"role": "assistant", "content": "u1"}
        assert messages[4] == {"role": "user", "content": "t1"}
        assert messages[-1]["role"] == "user"
        assert "u0" not in messages[-1]["content"]

    def test_target_history_message_shows_only_visible_turns(self):
        from conflictxgame.common.utils import SYSTEM_PROMPT

        visible = [{"sent": "u1", "received": "t1"}]
        messages = build_open_target_messages_with_history("u2", visible, None)
        assert messages == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "t1"},
            {"role": "user", "content": "u2"},
        ]

    def test_target_history_omits_hidden_turns(self):
        messages = build_open_target_messages_with_history("u2", [], "STEER")
        assert messages == [
            {"role": "system", "content": "STEER"},
            {"role": "user", "content": "u2"},
        ]
        joined = " ".join(m["content"] for m in messages)
        assert "u0" not in joined and "u1" not in joined


class TestEvaluateOpenRepeated:
    def run(self, rounds=3, memory_k=1, selected_seats=None):
        spec = two_by_two()
        user = RecordingClient(counting_response("u"), name="user-model")
        target = RecordingClient(counting_response("t"), name="target-model")
        judge = RecordingClient(judge_picking("A"), name="judge-model")
        rows, _ = evaluate_open_repeated(
            user, target, judge, [("sid", spec)],
            rounds=rounds, memory_k=memory_k, selected_seats=selected_seats,
        )
        return rows, user, target, judge

    def test_one_row_per_case_per_round(self):
        rows, *_ = self.run(rounds=3, selected_seats={("sid", 0)})
        assert len(rows) == 3
        assert [r["round"] for r in rows] == [0, 1, 2]
        assert all(r["rounds_played"] == 3 for r in rows)
        assert all(r["memory_k"] == 1 for r in rows)
        assert all(r["mode"] == "open" for r in rows)
        assert all(r["opponent_config"] == "repeated" for r in rows)

    def test_round_zero_matches_single_shot_open(self):
        rows, user, target, judge = self.run(rounds=3, selected_seats={("sid", 0)})
        # Round 0's user call is the same shape as evaluate_open's: persona
        # system + the Appendix C.5 opening instructions, nothing else.
        assert user.prompts[0] == build_open_user_messages(two_by_two(), 0)
        assert rows[0]["open_user_prompt"] == "u0"
        assert rows[0]["open_completion"] == "t0"

    def test_user_always_sees_full_history_target_is_bounded_by_memory_k(self):
        rows, user, target, judge = self.run(rounds=3, memory_k=1, selected_seats={("sid", 0)})
        # By round index 2 (3rd round), the user's follow-up call should
        # reference round 0's exchange (full history) ...
        round2_user_prompt = user.prompts[2]
        joined_user = " ".join(m["content"] for m in round2_user_prompt)
        assert "u0" in joined_user and "t0" in joined_user

        # ... but the target, capped at memory_k=1, must not see round 0 at
        # round index 2 -- only the immediately preceding round (index 1).
        round2_target_prompt = target.prompts[2]
        joined_target = " ".join(m["content"] for m in round2_target_prompt)
        assert "u0" not in joined_target and "t0" not in joined_target
        assert "u1" in joined_target and "t1" in joined_target

    def test_memory_k_none_gives_target_full_history(self):
        _, user, target, _ = self.run(rounds=3, memory_k=None, selected_seats={("sid", 0)})
        joined_target = " ".join(m["content"] for m in target.prompts[2])
        assert "u0" in joined_target and "t0" in joined_target

    def test_judge_runs_every_round_on_that_rounds_exchange(self):
        rows, user, target, judge = self.run(rounds=3, selected_seats={("sid", 0)})
        assert len(judge.prompts) == 3
        for i, prompt in enumerate(judge.prompts):
            text = prompt[-1]["content"]
            assert f"u{i}" in text and f"t{i}" in text

    def test_rounds_are_batched_together_across_cases(self):
        # Two seats selected: each round should batch both seats' requests
        # into one batch_generate call rather than looping per case.
        spec = two_by_two()
        user = RecordingClient(counting_response("u"), name="user-model")
        target = RecordingClient(counting_response("t"), name="target-model")
        judge = RecordingClient(judge_picking("A"), name="judge-model")
        rows, _ = evaluate_open_repeated(
            user, target, judge, [("sid", spec)], rounds=2, memory_k=1,
        )
        assert len(rows) == 4  # 2 seats * 2 rounds
        assert len(user.prompts) == 4
        assert len(target.prompts) == 4


class TestOpenPromptPathResolution:
    def test_none_disables_caching(self):
        assert resolve_open_prompt_path(None, "data/HHH/model_games.csv") is None

    def test_auto_derives_from_scenario_csv(self):
        path = resolve_open_prompt_path("auto", "data/HHH/claude_games.csv")
        assert path.replace("\\", "/") == "data/HHH/claude_open_prompts.json"

    def test_explicit_path_passes_through(self):
        assert resolve_open_prompt_path("cache.json", "data/HHH/x_games.csv") == "cache.json"


class TestCounterpartDisclosure:
    """The counterpart-disclosed variant of the opening-message prompt."""

    def test_off_by_default(self):
        spec = two_by_two()
        system = build_open_user_messages(spec, 0)[0]["content"]
        assert spec.players[1].name not in system
        assert "making its own, separate decision" not in system

    def test_names_the_other_seat_and_says_it_is_an_ai(self):
        spec = two_by_two()
        spec.players[1].agent_kind = "assistant"
        system = build_open_user_messages(spec, 0, disclose_counterpart=True)[0]["content"]
        assert spec.players[1].name in system
        assert "an AI assistant" in system
        assert spec.players[1].persona in system
        # The focal seat's own framing survives untouched.
        assert spec.description in system
        assert spec.players[0].name in system

    def test_human_counterpart_is_not_called_an_ai(self):
        spec = two_by_two()
        spec.players[1].agent_kind = "human"
        system = build_open_user_messages(spec, 0, disclose_counterpart=True)[0]["content"]
        assert f"{spec.players[1].name}, a person" in system

    def test_discloses_the_counterparts_private_stakes_to_nobody(self):
        # role_description is the counterpart's *private* context; the user
        # would have no way to know it, so only persona is disclosed.
        spec = two_by_two()
        system = build_open_user_messages(spec, 0, disclose_counterpart=True)[0]["content"]
        assert spec.players[1].role_description not in system

    def test_no_action_labels_or_value_tags_leak(self):
        spec = two_by_two()
        system = build_open_user_messages(spec, 0, disclose_counterpart=True)[0]["content"]
        for player in spec.players:
            for action in player.actions:
                assert action.label not in system
                assert action.description not in system
                assert action.value_tag not in system

    def test_user_turn_is_still_the_appendix_c5_block(self):
        from conflictxgame.solo.prompts import OPENING_MESSAGE_INSTRUCTIONS

        messages = build_open_user_messages(two_by_two(), 0, disclose_counterpart=True)
        assert messages[-1]["content"] == OPENING_MESSAGE_INSTRUCTIONS

    def test_followup_carries_the_disclosure(self):
        spec = two_by_two()
        history = [{"sent": "u0", "received": "t0"}]
        plain = build_open_followup_user_messages(spec, 0, history)[0]["content"]
        disclosed = build_open_followup_user_messages(
            spec, 0, history, disclose_counterpart=True
        )[0]["content"]
        assert spec.players[1].name not in plain
        assert spec.players[1].name in disclosed

    def test_retry_prompt_repeats_the_request_with_a_reminder(self):
        from conflictxgame.solo.prompts import OPENING_MESSAGE_INSTRUCTIONS

        spec = two_by_two()
        messages = build_open_user_retry_messages(spec, 0)
        system = messages[0]["content"]
        assert "did not mention" in system
        assert spec.players[1].name in system
        assert messages[-1]["content"] == OPENING_MESSAGE_INSTRUCTIONS


class TestCounterpartNamed:
    def test_true_when_the_other_seat_is_named(self):
        spec = two_by_two()
        opening = f"I am worried about what {spec.players[1].name} will do here."
        assert counterpart_named(spec, 0, opening) is True

    def test_case_insensitive_and_ignores_the_parenthetical(self):
        spec = two_by_two()
        spec.players[1].name = "ScreenBot (TalentSieve screening assistant)"
        assert counterpart_named(spec, 0, "what will screenbot report?") is True

    def test_false_when_absent_or_empty(self):
        spec = two_by_two()
        assert counterpart_named(spec, 0, "Please just help me decide.") is False
        assert counterpart_named(spec, 0, "") is False


class TestEvaluateOpenCounterpartDisclosure:
    def _clients(self, opening):
        return (
            RecordingClient(opening, name="user-model"),
            RecordingClient("Here is what I would do.", name="target-model"),
            RecordingClient(judge_picking("A"), name="judge-model"),
        )

    def test_rows_are_labelled_and_default_rows_are_not(self):
        spec = two_by_two()
        user, target, judge = self._clients(f"About {spec.players[1].name}: help?")
        rows, _ = evaluate_open(
            user, target, judge, [("sid", spec)], disclose_counterpart=True,
        )
        assert all(r["opponent_config"] == "counterpart-disclosed" for r in rows)

        user2, target2, judge2 = self._clients("no mention here")
        plain_rows, _ = evaluate_open(user2, target2, judge2, [("sid", two_by_two())])
        assert all(r["opponent_config"] == "" for r in plain_rows)

    def test_non_compliant_openings_are_retried_once(self):
        spec = two_by_two()
        other = spec.players[1].name

        calls = {"n": 0}

        def responder(messages):
            calls["n"] += 1
            # First pass ignores the counterpart; the retry complies.
            if "did not mention" in messages[0]["content"]:
                return f"Sorry - I mean, what will {other} do?"
            return "Please help me decide."

        user = RecordingClient(responder, name="user-model")
        target = RecordingClient("reply", name="target-model")
        judge = RecordingClient(judge_picking("A"), name="judge-model")
        rows, _ = evaluate_open(
            user, target, judge, [("sid", spec)],
            selected_seats={("sid", 0)}, disclose_counterpart=True,
        )
        assert calls["n"] == 2                      # one opening + one retry
        assert other in rows[0]["open_user_prompt"]

    def test_state_probe_retry_runs_once_and_fans_out_rewritten_opening(self):
        spec = two_by_two()
        other = spec.players[1].name

        def responder(messages):
            if 'did not mention' in messages[0]['content']:
                return f'What will {other} do?'
            return 'Please help me decide.'

        user = RecordingClient(responder, name='user-model')
        rows, _ = evaluate_open(
            user, RecordingClient('reply'), RecordingClient(judge_picking('A')),
            [('sid', spec)], selected_seats={('sid', 0)}, state_probes=True,
            disclose_counterpart=True,
        )
        assert len(user.prompts) == 2
        assert len(rows) == 5
        assert {row['open_user_prompt'] for row in rows} == {f'What will {other} do?'}

    def test_compliant_openings_are_not_retried(self):
        spec = two_by_two()
        user, target, judge = self._clients(f"What about {spec.players[1].name}?")
        evaluate_open(
            user, target, judge, [("sid", spec)],
            selected_seats={("sid", 0)}, disclose_counterpart=True,
        )
        assert len(user.prompts) == 1

    def test_cached_openings_are_reused_without_a_retry(self):
        spec = two_by_two()
        cache = {
            open_ended.cache_key("sid", 0): {
                "user_model": "user-model",
                "opening": "cached opening naming nobody",
            }
        }
        user, target, judge = self._clients("fresh")
        rows, out_cache = evaluate_open(
            user, target, judge, [("sid", spec)],
            selected_seats={("sid", 0)}, prompt_cache=cache,
            disclose_counterpart=True,
        )
        assert len(user.prompts) == 0
        assert rows[0]["open_user_prompt"] == "cached opening naming nobody"

    def test_repeated_mode_labels_rows_and_keeps_the_repeated_marker(self):
        spec = two_by_two()
        user = RecordingClient(counting_response("u"), name="user-model")
        target = RecordingClient(counting_response("t"), name="target-model")
        judge = RecordingClient(judge_picking("A"), name="judge-model")
        rows, _ = evaluate_open_repeated(
            user, target, judge, [("sid", spec)], rounds=2, memory_k=1,
            selected_seats={("sid", 0)}, disclose_counterpart=True,
        )
        assert all(r["opponent_config"] == "repeated+counterpart-disclosed" for r in rows)
