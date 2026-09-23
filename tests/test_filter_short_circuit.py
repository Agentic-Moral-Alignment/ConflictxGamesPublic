"""The short-circuited filter pass must be a pure cost saving.

``short_circuit`` changes which judge calls are issued, never which scenarios
are kept: ``keep_scenario`` is a conjunction over the fatal checks, so the
order they run in and whether the already-doomed ones run at all cannot move
it. These tests pin both halves of that claim — the verdicts are identical
scenario for scenario, and the calls really are saved — plus the bookkeeping
that makes a skipped check readable afterwards (``None``, never a silent
``False``).
"""

import json

import pandas as pd
import pytest

from conflictxgame.common.generation_quality import (
    failed_check_names,
    tabulate_check_failures,
)
from conflictxgame.common.model_wrappers import ModelWrapper
from conflictxgame.games.filter_game_scenarios import (
    ALL_CHECKS,
    CHECK_ORDER,
    FATAL_CHECKS,
    check_value_guidedness,
    filter_game_scenarios,
    results_to_dataframe,
)
from tests.conftest import TEST_VALUE_DICT, make_spec

#: Substring identifying each check's prompt, and the answer that passes it.
PROMPT_MARKERS = {
    "scenario_realism": ("how realistic it is", "true"),
    "scenario_groundedness": ("sufficiently grounded", "true"),
    "action_feasibility": ("all_actions_feasible", None),
    "value_guidedness": ("ONLY one letter", None),
    "payoff_consistency": ("ORDINALLY consistent", "true"),
    # The ambiguity check passes when the judge says there is NO consensus.
    "scenario_ambiguity": ("universal agreement", "false"),
}


class CountingJudge(ModelWrapper):
    """Passes everything except the named check, and counts its calls."""

    def __init__(self, fail=None):
        super().__init__("counting-judge")
        self.fail = fail
        self.calls = 0
        self.checks_seen = []

    def generate(self, messages):
        text = messages[-1]["content"]
        self.calls += 1
        for name, (marker, _) in PROMPT_MARKERS.items():
            if marker in text:
                self.checks_seen.append(name)
                return self._answer(name, text)
        raise AssertionError(f"Unrecognized judge prompt: {text[:120]}")

    def _answer(self, name, text):
        failing = name == self.fail
        if name == "action_feasibility":
            return json.dumps(
                {
                    "all_actions_feasible": not failing,
                    "mutually_exclusive": not failing,
                }
            )
        if name == "value_guidedness":
            # "A" is correct for the first slot's value and wrong for the
            # second, so answering "A" throughout fails exactly one
            # sub-judgment -- which is all a fatal failure needs.
            if failing:
                return "A"
            return self._guidedness_letter(text)
        if name == "scenario_ambiguity":
            # true == everyone agrees == the dilemma is too easy == fails.
            return "true" if failing else "false"
        return "false" if failing else "true"

    @staticmethod
    def _guidedness_letter(text):
        import re

        menu = re.findall(r"Candidate Action ([A-Z]): (.+?) - ", text)
        for value, definition in TEST_VALUE_DICT.items():
            if isinstance(definition, str) and definition in text:
                for letter, label in menu:
                    if value in label:
                        return letter
        return menu[0][0] if menu else "A"

    def batch_generate(self, messages_list, verbose=False):
        return [self.generate(m) for m in messages_list]

    def batch_generate_with_probs(self, messages_list, outputs):
        raise NotImplementedError


def two_by_two():
    return make_spec(
        "simultaneous", [2, 2],
        {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
        tags=[["v1", "v2"], ["v1", "v2"]],
    )


def run(fail, short_circuit):
    judge = CountingJudge(fail=fail)
    results = filter_game_scenarios(
        {"g0": two_by_two()}, judge, TEST_VALUE_DICT,
        guidedness_votes=3, short_circuit=short_circuit,
    )
    return results["g0"], judge


# -- the corpus is identical ------------------------------------------- #

@pytest.mark.parametrize("fail", [None, *FATAL_CHECKS, "payoff_consistency"])
def test_verdicts_are_identical_in_both_modes(fail):
    """Every planted failure must decide the same way with and without."""
    exhaustive, _ = run(fail, short_circuit=False)
    short, _ = run(fail, short_circuit=True)
    assert exhaustive["keep_scenario"] == short["keep_scenario"]


def test_payoff_failure_never_rejects_in_either_mode():
    """Payoffs annotate the stakes; an inconsistency is recorded, not fatal."""
    for short_circuit in (False, True):
        record, _ = run("payoff_consistency", short_circuit)
        assert record["keep_scenario"] is True


# -- the calls really are saved ---------------------------------------- #

def test_ambiguity_failure_costs_one_call():
    """The check that rejects the most scenarios is asked first."""
    assert CHECK_ORDER[0] == "scenario_ambiguity"
    _, judge = run("scenario_ambiguity", short_circuit=True)
    assert judge.calls == 1
    exhaustive_record, exhaustive_judge = run("scenario_ambiguity", short_circuit=False)
    assert exhaustive_judge.calls > 10, "the exhaustive pass should be the expensive one"


def test_guidedness_is_asked_last():
    """The twelve-call check runs only for scenarios nothing cheaper rejected."""
    _, judge = run("action_feasibility", short_circuit=True)
    assert "value_guidedness" not in judge.checks_seen


def test_a_kept_scenario_still_pays_for_every_check():
    """Nothing is skipped when every answer matters."""
    record, judge = run(None, short_circuit=True)
    assert record["keep_scenario"] is True
    assert set(judge.checks_seen) == set(ALL_CHECKS)


def test_payoff_is_skipped_for_a_rejected_scenario():
    """A non-fatal answer nobody will read is not worth a call."""
    _, judge = run("action_feasibility", short_circuit=True)
    assert "payoff_consistency" not in judge.checks_seen


# -- skipped checks are legible ---------------------------------------- #

def test_skipped_checks_are_recorded_as_none():
    """Every key is present, and None means 'not asked', not 'failed'."""
    record, _ = run("scenario_ambiguity", short_circuit=True)
    checks = record["checks"]
    assert set(checks) == set(ALL_CHECKS), "a row must always carry every check key"
    assert checks["scenario_ambiguity"] is False
    assert checks["value_guidedness"] is None


def test_revision_is_never_told_about_a_check_that_did_not_run():
    """Naming an unevaluated check would send a repair after a phantom fault."""
    record, _ = run("scenario_ambiguity", short_circuit=True)
    assert failed_check_names(record["checks"], pipeline="games") == ["scenario_ambiguity"]


def test_diagnostics_separate_unevaluated_from_failed(tmp_path):
    """A short-circuited row must not be counted as five extra failures."""
    record, _ = run("scenario_ambiguity", short_circuit=True)
    frame = results_to_dataframe({"g0": record})
    path = tmp_path / "corpus.csv"
    frame.to_csv(path, index=False)

    summary = tabulate_check_failures([str(path)])
    by_check = {row["check"]: row for _, row in summary.iterrows()}
    assert by_check["scenario_ambiguity"]["n_failed"] == 1
    assert by_check["value_guidedness"]["n_failed"] == 0
    assert by_check["value_guidedness"]["n_evaluated"] == 0
    assert by_check["value_guidedness"]["n_unevaluated"] == 1
    # Sole-cause attribution is meaningless when checks were skipped, so the
    # row must not claim one.
    assert by_check["scenario_ambiguity"]["n_sole_cause"] == 0


def test_full_checks_still_attribute_a_sole_cause(tmp_path):
    """The measuring mode keeps the diagnostic that tuning depends on."""
    record, _ = run("scenario_ambiguity", short_circuit=False)
    frame = results_to_dataframe({"g0": record})
    path = tmp_path / "corpus.csv"
    frame.to_csv(path, index=False)

    summary = tabulate_check_failures([str(path)])
    by_check = {row["check"]: row for _, row in summary.iterrows()}
    assert by_check["scenario_ambiguity"]["n_sole_cause"] == 1


# -- early-stopped voting ---------------------------------------------- #

class VotingJudge(CountingJudge):
    """Answers guidedness correctly except at the given call indices."""

    def __init__(self, wrong_at=()):
        super().__init__()
        self.wrong_at = set(wrong_at)
        self.vote = 0

    def _answer(self, name, text):
        if name != "value_guidedness":
            return super()._answer(name, text)
        letter = self._guidedness_letter(text)
        wrong = self.vote in self.wrong_at
        self.vote += 1
        if not wrong:
            return letter
        return "B" if letter == "A" else "A"


def test_unanimous_votes_stop_at_the_majority():
    """Two agreeing votes settle a best-of-three; the third cannot move it."""
    judge = VotingJudge()
    assert check_value_guidedness(two_by_two(), judge, TEST_VALUE_DICT, votes=3) is True
    # 2 seats x 2 values x 2 votes, instead of x 3.
    assert judge.calls == 8


def test_a_lost_majority_stops_early_too():
    """Once the majority cannot be reached, the remaining votes are pointless."""
    judge = VotingJudge(wrong_at={0, 1})
    assert check_value_guidedness(two_by_two(), judge, TEST_VALUE_DICT, votes=3) is False
    assert judge.calls == 2


def test_one_wrong_vote_is_still_absorbed():
    """Early stopping must not cost the de-noising the votes were added for."""
    judge = VotingJudge(wrong_at={0})
    assert check_value_guidedness(two_by_two(), judge, TEST_VALUE_DICT, votes=3) is True


def test_single_shot_voting_is_unchanged():
    judge = VotingJudge()
    assert check_value_guidedness(two_by_two(), judge, TEST_VALUE_DICT, votes=1) is True
    assert judge.calls == 4
