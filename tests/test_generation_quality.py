"""Tests for the acceptance-rate helpers shared by both generation pipelines."""

import json
import random

import re

import pandas as pd
import pytest

from conflictxgame.common.generation_quality import (
    CHECKLIST_CLOSING,
    FILTER_CHECKLISTS,
    build_revision_prompt,
    checklist_block,
    failed_check_names,
    guidance_block,
    parse_check_results,
    parse_guidance,
    tabulate_check_failures,
    tabulate_prescreen,
)


# -- prompt construction ------------------------------------------------ #

def test_checklist_block_is_empty_when_disabled():
    assert checklist_block("games", criteria_in_context=False) == ""


def test_checklist_block_includes_the_pipeline_checklist():
    block = checklist_block("solo", criteria_in_context=True)
    assert FILTER_CHECKLISTS["solo"] in block


def test_checklist_block_ends_by_restating_the_output_instruction():
    """The block is appended after the template's own 'return only JSON' line,
    so without a closing restatement the prompt stops describing its format."""
    for pipeline in ("solo", "games"):
        block = checklist_block(pipeline, criteria_in_context=True)
        assert block.rstrip().endswith(CHECKLIST_CLOSING)


def test_checklist_closing_does_not_invite_narration():
    """Narrated self-checking is output tokens, which is what truncates replies."""
    assert "no other text" in CHECKLIST_CLOSING
    assert len(CHECKLIST_CLOSING.split()) < 40


def test_checklists_lead_with_the_highest_impact_requirements():
    """Ordered by measured sole-cause rejection rate, not by filter order.

    Exclusivity and genuine-dilemma together account for most rejections in
    the shipped corpora; realism for almost none. Leading with the rare ones
    would spend the model's attention where there is nothing to gain.
    """
    for pipeline in ("solo", "games"):
        checklist = FILTER_CHECKLISTS[pipeline]
        order = [
            checklist.index(marker)
            for marker in ("MUTUAL EXCLUSIVITY", "GENUINE DILEMMA", "VALUE-GUIDEDNESS",
                           "FEASIBILITY", "CONCRETENESS", "REALISM")
        ]
        assert order == sorted(order), f"{pipeline} checklist is out of impact order"


def test_checklists_cover_every_filter_dimension():
    """A checklist that omits a dimension cannot raise that check's pass rate."""
    games = FILTER_CHECKLISTS["games"].lower()
    for keyword in ("realis", "placeholder", "text", "exclusiv", "payoff", "disagree"):
        assert keyword in games, keyword
    solo = FILTER_CHECKLISTS["solo"].lower()
    for keyword in ("realis", "placeholder", "text", "exclusiv", "disagree"):
        assert keyword in solo, keyword


def test_unknown_pipeline_is_rejected():
    with pytest.raises(KeyError):
        checklist_block("nonsense", criteria_in_context=True)


# -- generation prompts actually carry the block ------------------------ #

def test_games_stage_one_prompt_gains_the_checklist_only_when_enabled():
    from conflictxgame.games.game_structures import GameConfig
    from conflictxgame.games.generate_game_scenarios import generate_stage_one_games
    from tests.fakes import DummyGenerationClient

    config = GameConfig(structure="simultaneous", n_players=2, arities=[2, 2], horizon=1)
    value_dict = {name: f"definition of {name}" for name in ("a_value", "b_value", "c_value")}

    for enabled in (False, True):
        log = {}
        generate_stage_one_games(
            config, DummyGenerationClient(), 1, value_dict, random.Random(0),
            prompt_log=log, criteria_in_context=enabled,
        )
        prompt = log["stage_one"][0]["prompt"]
        assert (FILTER_CHECKLISTS["games"] in prompt) is enabled


def test_solo_stage_one_prompt_gains_the_checklist(monkeypatch):
    from conflictxgame.solo import generate_scenarios

    captured = {}

    class Recorder:
        def generate(self, messages):
            captured["prompt"] = messages[-1]["content"]
            return json.dumps({})

    generate_scenarios.generate_stage_one(
        "honesty", "helpfulness", Recorder(), 4, 0,
        {"honesty": "being honest", "helpfulness": "being helpful"},
        criteria_in_context=True,
    )
    assert FILTER_CHECKLISTS["solo"] in captured["prompt"]


def test_solo_stage_one_prompt_accepts_full_construction_guidance():
    from conflictxgame.solo import generate_scenarios

    captured = {}

    class Recorder:
        def generate(self, messages):
            captured["prompt"] = messages[-1]["content"]
            return json.dumps({})

    generate_scenarios.generate_stage_one(
        "honesty", "helpfulness", Recorder(), 4, 0,
        {"honesty": "being honest", "helpfulness": "being helpful"},
        guidance_parts=("recipes", "nearmiss", "checklist", "stage2"),
    )
    assert "START FROM ONE LEVER" in captured["prompt"]
    assert "does NOT work" in captured["prompt"]
    assert FILTER_CHECKLISTS["solo"] in captured["prompt"]


# -- revision prompts --------------------------------------------------- #

def test_failed_check_names_lists_only_fatal_failures():
    checks = {
        "scenario_realism": True,
        "scenario_ambiguity": False,
        "payoff_consistency": False,  # non-fatal for games
        "value_guidedness": False,
    }
    failed = failed_check_names(checks, pipeline="games")
    assert failed == ["value_guidedness", "scenario_ambiguity"]
    assert "payoff_consistency" not in failed


def test_failed_check_names_treats_impossibility_as_fatal_for_solo():
    checks = {"scenario_impossibility": False}
    assert failed_check_names(checks, pipeline="solo") == ["scenario_impossibility"]


def test_revision_prompt_names_only_the_failed_checks():
    prompt = build_revision_prompt(
        "the scenario", ["scenario_ambiguity"], pipeline="games"
    )
    assert "the scenario" in prompt
    assert "obviously correct choice" in prompt  # ambiguity guidance
    assert "placeholder" not in prompt.split("Problems found:")[1].split("\n\n")[0]


def test_revision_prompt_includes_output_format_when_given():
    prompt = build_revision_prompt(
        "s", ["scenario_realism"], pipeline="games", output_format="RETURN JSON"
    )
    assert prompt.rstrip().endswith("RETURN JSON")


def test_revision_prompt_handles_an_empty_failure_list():
    prompt = build_revision_prompt("s", [], pipeline="games")
    assert "did not meet the requirements" in prompt


# -- check-result parsing ----------------------------------------------- #

def test_parse_check_results_reads_json_and_python_repr():
    assert parse_check_results('{"a": true}') == {"a": True}
    assert parse_check_results("{'a': True}") == {"a": True}


def test_parse_check_results_rejects_junk():
    assert parse_check_results("") is None
    assert parse_check_results(None) is None
    assert parse_check_results("not a dict") is None
    assert parse_check_results("[1, 2]") is None


# -- failure diagnostics ------------------------------------------------ #

def make_csv(tmp_path, rows, games=True):
    frame = pd.DataFrame(rows)
    if games:
        frame["game_spec"] = "{}"
    path = tmp_path / "corpus.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_tabulate_counts_failures_and_sole_causes(tmp_path):
    rows = [
        {
            "keep_scenario": False,
            "check_results": json.dumps(
                {"scenario_realism": True, "scenario_ambiguity": False}
            ),
        },
        {
            "keep_scenario": False,
            "check_results": json.dumps(
                {"scenario_realism": False, "scenario_ambiguity": False}
            ),
        },
        {
            "keep_scenario": True,
            "check_results": json.dumps(
                {"scenario_realism": True, "scenario_ambiguity": True}
            ),
        },
    ]
    summary = tabulate_check_failures([make_csv(tmp_path, rows)])
    by_check = {r["check"]: r for _, r in summary.iterrows()}

    assert by_check["scenario_ambiguity"]["n_failed"] == 2
    # Only the first row fails ambiguity alone; the second fails two checks.
    assert by_check["scenario_ambiguity"]["n_sole_cause"] == 1
    assert by_check["scenario_realism"]["n_sole_cause"] == 0
    assert by_check["ALL"]["n_failed"] == 2
    assert by_check["ALL"]["failure_rate"] == pytest.approx(2 / 3)


def test_tabulate_marks_payoff_consistency_non_fatal(tmp_path):
    rows = [
        {
            "keep_scenario": True,
            "check_results": json.dumps(
                {"scenario_realism": True, "payoff_consistency": False}
            ),
        }
    ]
    summary = tabulate_check_failures([make_csv(tmp_path, rows)])
    by_check = {r["check"]: r for _, r in summary.iterrows()}
    assert bool(by_check["payoff_consistency"]["fatal"]) is False
    assert by_check["payoff_consistency"]["n_sole_cause"] == 0


def test_tabulate_infers_the_solo_pipeline_without_game_spec(tmp_path):
    rows = [
        {
            "keep_scenario": False,
            "check_results": json.dumps({"scenario_impossibility": False}),
        }
    ]
    summary = tabulate_check_failures([make_csv(tmp_path, rows, games=False)])
    by_check = {r["check"]: r for _, r in summary.iterrows()}
    assert summary["pipeline"].iloc[0] == "solo"
    assert bool(by_check["scenario_impossibility"]["fatal"]) is True


def test_tabulate_skips_missing_files(tmp_path):
    summary = tabulate_check_failures([str(tmp_path / "nope.csv")])
    assert summary.empty


def test_tabulate_survives_unparseable_check_results(tmp_path):
    rows = [
        {"keep_scenario": True, "check_results": "garbage"},
        {"keep_scenario": True, "check_results": json.dumps({"scenario_realism": True})},
    ]
    summary = tabulate_check_failures([make_csv(tmp_path, rows)])
    by_check = {r["check"]: r for _, r in summary.iterrows()}
    assert by_check["scenario_realism"]["n_evaluated"] == 1
    assert by_check["ALL"]["n_evaluated"] == 2


# -- construction guidance ---------------------------------------------- #

class TestGuidanceBlock:
    """The guidance is prompt text, so its risks are textual ones."""

    def test_no_parts_leaves_the_prompt_untouched(self):
        assert guidance_block("games", "stage_one", ()) == ""
        assert guidance_block("games", "stage_one", None) == ""

    def test_checklist_only_matches_the_original_block(self):
        """The old behavior has to survive byte for byte to stay comparable."""
        assert guidance_block("games", "stage_one", ("checklist",)) == checklist_block("games", True)

    def test_output_instruction_is_restated_exactly_once(self):
        """The block is appended after the prompt's own format instruction."""
        for parts in (("recipes",), ("checklist",), ("recipes", "checklist"),
                      ("recipes", "nearmiss", "checklist")):
            block = guidance_block("games", "stage_one", parts)
            assert block.endswith(CHECKLIST_CLOSING), parts
            assert block.count(CHECKLIST_CLOSING) == 1, parts

    def test_parts_are_scoped_to_their_stage(self):
        """One part list can be passed for the whole run without leaking."""
        stage_one = guidance_block("games", "stage_one", ("recipes", "stage2"))
        stage_two = guidance_block("games", "stage_two", ("recipes", "stage2"))
        assert "START FROM ONE SHARED OUTCOME" in stage_one
        assert "150-220 words" not in stage_one
        assert "150-220 words" in stage_two
        assert "START FROM ONE SHARED OUTCOME" not in stage_two

    def test_construction_order_precedes_the_rules(self):
        """Telling the model how to build comes before what will be checked."""
        block = guidance_block("games", "stage_one", ("recipes", "checklist"))
        assert block.index("START FROM ONE SHARED OUTCOME") < block.index("MUTUAL EXCLUSIVITY")

    def test_single_decision_scenarios_drop_the_shared_outcome_step(self):
        """There is nothing to share when only one person decides."""
        block = guidance_block("games", "stage_one", ("recipes",), n_players=1)
        assert "START FROM ONE SHARED OUTCOME" not in block
        assert "ONE LEVER, INCOMPATIBLE SETTINGS" in block

    def test_guidance_never_quotes_the_judge(self):
        """Optimizing for the judge's wording is not the same as passing it."""
        for pipeline in ("solo", "games"):
            for stage, parts in (("stage_one", ("recipes", "nearmiss")), ("stage_two", ("stage2",))):
                block = guidance_block(pipeline, stage, parts)
                for phrase in ("universal agreement", "annotator", "95+", "95%",
                               "all_actions_feasible", "ONLY one letter"):
                    assert phrase not in block, (pipeline, stage, phrase)

    def test_guidance_cannot_be_mistaken_for_generator_output(self):
        """A guidance line that parsed as an assignment row would corrupt a seed."""
        patterns = [
            re.compile(r"scenario_(\d+):"),
            re.compile(r"decision-maker (\d+) \((?:human|AI assistant), (\d+) actions\):"),
            re.compile(r"^P\d+:"),
            re.compile(r"^decision-maker \d+ \((?:human|AI assistant)\):"),
        ]
        for pipeline in ("solo", "games"):
            for stage, parts in (("stage_one", ("recipes", "nearmiss", "checklist")),
                                 ("stage_two", ("stage2", "checklist"))):
                for line in guidance_block(pipeline, stage, parts).splitlines():
                    for pattern in patterns:
                        assert not pattern.match(line.strip()), line

    def test_only_near_misses_are_shown(self):
        """A worked good example pulls every later scenario toward its shape."""
        block = guidance_block("games", "stage_one", ("nearmiss",))
        assert "does NOT work" in block

    def test_examples_rotate_across_calls(self):
        """Every call showing the same example is a template, not guidance."""
        rng = random.Random(0)
        seen = {
            guidance_block("games", "stage_one", ("recipes", "nearmiss"), rng=rng)
            for _ in range(60)
        }
        assert len(seen) > 1

    def test_rotation_is_deterministic_without_an_rng(self):
        first = guidance_block("games", "stage_one", ("recipes", "nearmiss"))
        second = guidance_block("games", "stage_one", ("recipes", "nearmiss"))
        assert first == second

    def test_unknown_pipeline_and_stage_are_rejected(self):
        with pytest.raises(KeyError):
            guidance_block("nonsense", "stage_one", ("checklist",))
        with pytest.raises(ValueError):
            guidance_block("games", "stage_three", ("checklist",))


class TestParseGuidance:
    def test_parses_and_orders_parts(self):
        assert parse_guidance("stage2,checklist") == ("checklist", "stage2")

    def test_none_and_empty_mean_no_guidance(self):
        assert parse_guidance("none") == ()
        assert parse_guidance("") == ()

    def test_a_typo_is_refused_rather_than_ignored(self):
        """Silently dropping a part would silently disable what is measured."""
        with pytest.raises(ValueError):
            parse_guidance("checklist,recipies")


# -- pre-screen calibration --------------------------------------------- #

def _prescreen_row(flags, kept, answers=None):
    return {
        "keep_scenario": kept,
        "check_results": json.dumps({"scenario_ambiguity": kept}),
        "game_spec": json.dumps(
            {"metadata": {"prescreen": {"flags": list(flags), "answers": answers or {}}}}
        ),
    }


class TestTabulatePrescreen:
    def test_reports_precision_and_the_damage_side_by_side(self, tmp_path):
        """Both numbers, because they trade off against each other."""
        rows = [
            _prescreen_row(["exclusivity"], False),   # flagged, judge agreed
            _prescreen_row(["exclusivity"], False),   # flagged, judge agreed
            _prescreen_row(["exclusivity"], True),    # flagged, but kept: damage
            _prescreen_row([], True),
            _prescreen_row([], True),
        ]
        path = tmp_path / "corpus.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

        summary = tabulate_prescreen([str(path)])
        by_criterion = {r["criterion"]: r for _, r in summary.iterrows()}
        excl = by_criterion["exclusivity"]
        assert excl["n_flagged"] == 3
        assert excl["precision"] == pytest.approx(2 / 3)
        assert excl["precision_lower_95"] < excl["precision"]
        # One of the three kept scenarios would have been thrown away.
        assert excl["kept_lost_rate"] == pytest.approx(1 / 3)
        assert excl["kept_lost_upper_95"] > excl["kept_lost_rate"]
        # A random flagger would have scored this, so precision must beat it.
        assert excl["base_reject_rate"] == pytest.approx(2 / 5)

    def test_any_row_counts_a_scenario_once(self, tmp_path):
        path = tmp_path / "corpus.csv"
        pd.DataFrame([_prescreen_row(["exclusivity", "obvious"], False)]).to_csv(path, index=False)
        summary = tabulate_prescreen([str(path)])
        by_criterion = {r["criterion"]: r for _, r in summary.iterrows()}
        assert by_criterion["ANY"]["n_flagged"] == 1
        assert by_criterion["exclusivity"]["n_flagged"] == 1
        assert by_criterion["obvious"]["n_flagged"] == 1

    def test_shadowed_realism_is_calibrated_from_answers(self, tmp_path):
        """A criterion need not flag before we can measure its false positives."""
        rows = [
            _prescreen_row([], False, {"realism": False}),
            _prescreen_row([], True, {"realism": True}),
        ]
        path = tmp_path / "corpus.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

        summary = tabulate_prescreen([str(path)])
        by_criterion = {r["criterion"]: r for _, r in summary.iterrows()}
        assert by_criterion["realism"]["n_flagged"] == 1
        assert by_criterion["realism"]["precision"] == 1.0
        assert by_criterion["realism"]["kept_lost_upper_95"] is not None
        assert "ANY" not in by_criterion

    def test_a_corpus_without_verdicts_is_empty_not_an_error(self, tmp_path):
        path = tmp_path / "corpus.csv"
        pd.DataFrame([{"keep_scenario": True, "game_spec": "{}"}]).to_csv(path, index=False)
        assert tabulate_prescreen([str(path)]).empty

    def test_missing_files_are_skipped(self, tmp_path):
        assert tabulate_prescreen([str(tmp_path / "nope.csv")]).empty
