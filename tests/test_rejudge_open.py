"""Tests for re-judging stored open-ended replies with a second judge."""

import json
import os

import pandas as pd
import pytest

from conftest import make_spec
from conflictxgame.common.model_wrappers import ModelWrapper
from conflictxgame.games import rejudge_open
from conflictxgame.games.open_ended import build_open_judge_messages
from conflictxgame.games.rejudge_open import (
    DONE,
    PENDING,
    find_input_files,
    map_verdict,
    presentation_order,
    rejudge_file,
)

PAYOFFS = {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]}


def spec(game_id="g0"):
    return make_spec("simultaneous", [2, 2], dict(PAYOFFS), game_id=game_id)


class ScriptedJudge(ModelWrapper):
    """Answers each prompt from a callable and records what it saw."""

    def __init__(self, answer):
        super().__init__("judge-under-test")
        self.answer = answer
        self.prompts = []

    def generate(self, messages):
        self.prompts.append(messages)
        return self.answer(messages)

    def batch_generate(self, messages_list, verbose=False):
        return [self.generate(m) for m in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


def verdict(letter, likert=None):
    payload = {"reasoning": "because", "action": letter}
    if likert is not None:
        payload["likert"] = likert
    return json.dumps(payload)


def write_results(path, specs_by_id, original_letter="A"):
    rows = []
    for scenario_id, game in specs_by_id.items():
        for seat in range(2):
            action = 0 if original_letter == "A" else 1
            chosen = game.players[seat].actions[action]
            rows.append({
                "scenario_id": scenario_id, "mode": "open", "seat": seat, "probe": "",
                "choice_action": action, "choice_label": chosen.label,
                "choice_value_tag": chosen.value_tag, "parsed_ok": True,
                "reasoning": "original", "open_user_prompt": f"opening {scenario_id}/{seat}",
                "open_completion": f"reply {scenario_id}/{seat}", "open_judge_likert": 2,
                "judge_model": "original-judge",
            })
    pd.DataFrame(rows).to_csv(path, index=False)


class TestPromptOrder:
    def test_default_order_is_byte_identical_to_menu_order(self):
        game = spec()
        assert build_open_judge_messages(game, 0, "u", "c") == \
            build_open_judge_messages(game, 0, "u", "c", order=[0, 1])

    def test_swap_lists_the_second_action_under_letter_a(self):
        game = spec()
        text = build_open_judge_messages(game, 0, "u", "c", order=[1, 0])[-1]["content"]
        second = game.players[0].actions[1].label
        assert f"Action A: {second} -" in text

    def test_rejects_an_order_that_is_not_a_permutation(self):
        with pytest.raises(ValueError):
            build_open_judge_messages(spec(), 0, "u", "c", order=[0, 0])


class TestVerdictMapping:
    def test_verbatim_is_identity(self):
        assert presentation_order("verbatim", 2) == [0, 1]
        assert map_verdict(1, 6, [0, 1]) == (1, 6)

    def test_swap_maps_letter_and_reflects_likert(self):
        order = presentation_order("swap", 2)
        assert map_verdict(0, 2, order) == (1, 6)
        assert map_verdict(1, "", order) == (0, "")

    def test_invalid_passes_through(self):
        assert map_verdict("INVALID", "", [1, 0]) == ("INVALID", "")


class TestFindInputFiles:
    TAIL = "_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv"

    def make(self, tmp_path, *names):
        # Real result names overflow Windows MAX_PATH under a deep tmp dir,
        # which is the case long_path exists for.
        for name in names:
            with open(rejudge_open.long_path(str(tmp_path / name)), "w") as handle:
                handle.write("x\n")

    def test_baseline_is_not_taken_from_an_arm_that_sorts_first(self, tmp_path):
        self.make(tmp_path,
                  "m_open_balanced50" + self.TAIL,
                  "m_open_balanced50_observed" + self.TAIL,
                  "m_open_balanced50_stakes-alignment" + self.TAIL,
                  "m_mcq_balanced50.csv")
        found = find_input_files(str(tmp_path), ["m"], ["", "_stakes-alignment"])
        assert os.path.basename(found[("m", "")]) == "m_open_balanced50" + self.TAIL
        assert "stakes-alignment" in found[("m", "_stakes-alignment")]

    def test_missing_arm_raises(self, tmp_path):
        self.make(tmp_path, "m_open_balanced50" + self.TAIL)
        with pytest.raises(FileNotFoundError):
            find_input_files(str(tmp_path), ["m"], ["_stakes-payoff"])

    def test_duplicate_arm_raises(self, tmp_path):
        self.make(tmp_path, "m_open_balanced50" + self.TAIL, "m_open_balanced50.csv")
        with pytest.raises(ValueError):
            find_input_files(str(tmp_path), ["m"], [""])


class TestRejudgeFile:
    def setup_method(self):
        self.specs = {"g0": spec("g0"), "g1": spec("g1")}

    def run(self, tmp_path, answer, condition="verbatim", **kwargs):
        src = tmp_path / "in.csv"
        if not src.exists():
            write_results(src, self.specs)
        out = tmp_path / "out.csv"
        judge = ScriptedJudge(answer)
        counts = rejudge_file(str(src), str(out), self.specs, judge, condition, **kwargs)
        return counts, pd.read_csv(out, dtype=str, keep_default_na=False), judge

    def test_judge_sees_the_stored_opening_and_reply(self, tmp_path):
        _, _, judge = self.run(tmp_path, lambda m: verdict("B"))
        text = judge.prompts[0][-1]["content"]
        assert "USER: opening g0/0" in text and "ASSISTANT: reply g0/0" in text

    def test_new_verdict_replaces_choice_and_original_is_kept(self, tmp_path):
        counts, out, _ = self.run(tmp_path, lambda m: verdict("B", 6))
        assert counts == {"done": 4, "pending": 0, "invalid": 0, "calls": 4}
        row = out.iloc[0]
        assert row["choice_action"] == "1" and row["orig_choice_action"] == "0"
        assert row["choice_value_tag"] == self.specs["g0"].players[0].actions[1].value_tag
        assert row["orig_judge_model"] == "original-judge"
        assert row["judge_model"] == "judge-under-test"
        assert row["open_judge_likert"] == "6"

    def test_swap_maps_back_to_menu_order(self, tmp_path):
        _, out, _ = self.run(tmp_path, lambda m: verdict("A", 2), condition="swap")
        row = out.iloc[0]
        assert row["choice_action"] == "1"          # presented A was menu action 1
        assert row["open_judge_likert"] == "6"       # scale reflected
        assert row["rejudge_order"] == "1,0"

    def test_failed_call_stays_pending_and_resume_completes_it(self, tmp_path):
        calls = {"n": 0}

        def flaky(messages):
            calls["n"] += 1
            return "" if calls["n"] == 2 else verdict("A")

        counts, out, _ = self.run(tmp_path, flaky)
        assert counts["pending"] == 1 and counts["done"] == 3
        assert (out["rejudge_status"] == PENDING).sum() == 1

        counts, out, judge = self.run(tmp_path, lambda m: verdict("B"))
        assert len(judge.prompts) == 1               # only the pending row is re-asked
        assert counts["pending"] == 0
        assert (out["rejudge_status"] == DONE).all()
        assert (out["orig_judge_model"] == "original-judge").all()

    def test_unparseable_response_is_a_verdict_not_a_retry(self, tmp_path):
        counts, out, _ = self.run(tmp_path, lambda m: "no letter here")
        assert counts["invalid"] == 4 and counts["pending"] == 0
        assert (out["parsed_ok"] == "False").all()

    def test_limit_bounds_a_smoke_run(self, tmp_path):
        counts, _, judge = self.run(tmp_path, lambda m: verdict("A"), limit=2)
        assert len(judge.prompts) == 2 and counts["pending"] == 2

    def test_breaker_trip_saves_progress_then_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rejudge_open.time, "sleep", lambda s: None)

        def broken(messages):
            raise Exception("circuit breaker")

        with pytest.raises(RuntimeError):
            self.run(tmp_path, broken, chunk_attempts=2)
        saved = pd.read_csv(tmp_path / "out.csv", dtype=str, keep_default_na=False)
        assert (saved["rejudge_status"] == PENDING).all()
