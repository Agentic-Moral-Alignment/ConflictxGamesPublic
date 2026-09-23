"""End-to-end tests for the balanced generate/filter/top-up loop.

Everything runs against :mod:`tests.fakes`, so the loop's control flow —
quota termination, resume, backlog recovery, revision salvage, and the stall
abort — is exercised without any API calls.
"""

import json
import os
import sys

import pandas as pd
import pytest

from conflictxgame.common.generation_quality import tabulate_prescreen
from conflictxgame.common.model_wrappers import ModelWrapper
from conflictxgame.games import generate_balanced
from conflictxgame.games.pair_quota import (
    PairQuotaTracker,
    all_value_pairs,
    kept_pair_counts_from_csv,
)
from tests.fakes import DummyGenerationClient, DummyJudgeClient

HHH = ["harmlessness", "helpfulness", "honesty"]
VALUE_DICT = {name: f"the principle of {name}, defined for tests" for name in HHH}


@pytest.fixture
def workspace(tmp_path):
    """A run directory plus the paths the orchestrator writes."""
    out = tmp_path / "games"
    csv_dir = tmp_path / "data"
    out.mkdir()
    csv_dir.mkdir()
    return {
        "out": str(out),
        "csv_dir": str(csv_dir),
        "csv": str(csv_dir / "dummy-generator_games.csv"),
        "games_json": str(out / "dummy-generator_games.json"),
        "pair_coverage": str(out / "pair_coverage.json"),
    }


def run_balanced(monkeypatch, workspace, judge, extra_args=(), generator=None):
    """Invoke the orchestrator's CLI entry point with fake clients.

    No revision flag is passed, so every test runs on the shipped default and
    the ones that want revision opt in through ``extra_args``.
    """
    generator = generator or DummyGenerationClient()
    clients = {"gen": generator, "judge": judge}

    def fake_create(name, **kwargs):
        selected = clients["judge"] if name == "judge-model" else clients["gen"]
        if selected is clients["gen"]:
            selected.create_kwargs = kwargs
        return selected

    monkeypatch.setattr(ModelWrapper, "create", staticmethod(fake_create))
    monkeypatch.setattr(generate_balanced, "load_value_set", lambda name: VALUE_DICT)
    monkeypatch.setattr(
        generate_balanced, "short_model_name", lambda name: "dummy-generator"
    )

    argv = [
        "generate_balanced",
        "-v", "HHH",
        "-m", "gen-model",
        "--judge-model", "judge-model",
        "-o", workspace["out"],
        "--csv-dir", workspace["csv_dir"],
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    generate_balanced.main()
    return generator, judge


def read_pair_coverage(workspace):
    with open(workspace["pair_coverage"], encoding="utf-8") as f:
        return json.load(f)


# -- the core loop ------------------------------------------------------ #

def test_loop_fills_every_pair_quota(monkeypatch, workspace):
    run_balanced(
        monkeypatch, workspace,
        DummyJudgeClient(VALUE_DICT),
        ["-k", "2", "--round-batch", "6", "--max-rounds", "10"],
    )
    kept, _ = kept_pair_counts_from_csv(workspace["csv"], HHH)
    for pair in all_value_pairs(HHH):
        assert kept.get(pair, 0) >= 2, f"{pair} short: {kept}"

    coverage = read_pair_coverage(workspace)
    assert coverage["complete"] is True
    assert coverage["target_per_pair"] == 2
    assert len(coverage["pairs"]) == 3
    assert all(row["deficit"] == 0 for row in coverage["pairs"])
    assert all(row['pending'] == 0 for row in coverage['pairs'])


def test_loop_stops_once_the_quota_is_met(monkeypatch, workspace):
    """Termination is on kept counts, so a met quota generates nothing more."""
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "6", "--max-rounds", "10"],
    )
    first = len(pd.read_csv(workspace["csv"]))

    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "6", "--max-rounds", "10"],
    )
    assert len(pd.read_csv(workspace["csv"])) == first


def test_resume_recounts_from_the_csv_and_tops_up(monkeypatch, workspace):
    """A raised target continues from the existing corpus rather than restarting."""
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "3", "--max-rounds", "10"],
    )
    after_first = len(pd.read_csv(workspace["csv"]))

    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "6", "--round-batch", "3", "--max-rounds", "20"],
    )
    kept, _ = kept_pair_counts_from_csv(workspace["csv"], HHH)
    assert all(kept.get(p, 0) >= 6 for p in all_value_pairs(HHH))
    # Topped up, not regenerated from scratch.
    assert len(pd.read_csv(workspace["csv"])) > after_first


def test_rejections_do_not_count_toward_the_quota(monkeypatch, workspace):
    """Rejected scenarios must not satisfy a quota defined on accepted ones."""
    judge = DummyJudgeClient(VALUE_DICT, reject_first=4)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "12"],
    )
    frame = pd.read_csv(workspace["csv"])
    assert (~frame["keep_scenario"].astype(str).str.lower().eq("true")).sum() >= 4
    kept, _ = kept_pair_counts_from_csv(workspace["csv"], HHH)
    assert all(kept.get(p, 0) >= 1 for p in all_value_pairs(HHH))


def test_generated_corpus_is_balanced_across_pairs(monkeypatch, workspace):
    """The point of the exercise: accepted counts even out across pairs."""
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "3", "--round-batch", "6", "--max-rounds", "15"],
    )
    kept, _ = kept_pair_counts_from_csv(workspace["csv"], HHH)
    counts = [kept.get(p, 0) for p in all_value_pairs(HHH)]
    assert min(counts) >= 3
    # No pair is wildly over-generated relative to the one that lagged.
    assert max(counts) - min(counts) <= 4


# -- interruption recovery ---------------------------------------------- #

def test_round_batch_larger_than_batch_size_is_split_across_calls(monkeypatch, workspace):
    """A big round must issue several calls, not silently shrink to one batch.

    One oversized stage-1 call overruns the model's output budget; reasoning
    models then return no content at all and the round yields nothing.
    """
    generator = DummyGenerationClient()
    calls = []
    original = generator.generate

    def counting_generate(messages):
        calls.append(messages[-1]["content"])
        return original(messages)

    monkeypatch.setattr(generator, "generate", counting_generate)
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "99", "--round-batch", "6", "--batch-size", "2", "--max-rounds", "1"],
        generator=generator,
    )
    assert len(calls) == 3, f"expected 3 stage-1 calls of 2, got {len(calls)}"
    for prompt in calls:
        assert "Generate 2 scenario outlines" in prompt
    assert len(pd.read_csv(workspace["csv"])) == 6


def test_default_batch_size_is_one_scenario_per_call(monkeypatch, workspace):
    """One scenario per reply caps truncation damage at a single scenario."""
    generator = DummyGenerationClient()
    calls = []
    original = generator.generate

    def counting_generate(messages):
        calls.append(messages[-1]["content"])
        return original(messages)

    monkeypatch.setattr(generator, "generate", counting_generate)
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "99", "--round-batch", "3", "--max-rounds", "1"],
        generator=generator,
    )
    assert len(calls) == 3, f"expected one call per scenario, got {len(calls)}"
    for prompt in calls:
        assert "Generate 1 scenario outline." in prompt  # singular, not "outlines"
    assert len(pd.read_csv(workspace["csv"])) == 3


def test_round_batch_is_not_overshot(monkeypatch, workspace):
    """A round must generate --round-batch scenarios, not ceil-per-cell more."""
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "99", "--round-batch", "5", "--batch-size", "2", "--max-rounds", "1"],
    )
    assert len(pd.read_csv(workspace["csv"])) == 5


def test_backlog_is_filtered_before_new_generation(monkeypatch, workspace):
    """Specs generated but never filtered (an interrupt) are not wasted."""
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "6", "--max-rounds", "10"],
    )
    with open(workspace["games_json"], encoding="utf-8") as f:
        games = json.load(f)
    # Simulate the interrupt: drop the CSV, keep the generated specs.
    os.remove(workspace["csv"])

    judge = DummyJudgeClient(VALUE_DICT)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "6", "--max-rounds", "10"],
    )
    frame = pd.read_csv(workspace["csv"])
    backlog_ids = {f"dummy-generator_{gid}" for gid in games}
    assert backlog_ids & set(frame["scenario_id"]), "backlog was never filtered"


def test_pair_coverage_files_are_written(monkeypatch, workspace):
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "6", "--max-rounds", "10"],
    )
    coverage = read_pair_coverage(workspace)
    assert set(coverage["pairs"][0]) >= {
        "pair", "value1", "value2", "target", "generated", "kept", "deficit"
    }
    frame = pd.read_csv(os.path.join(workspace["out"], "pair_coverage.csv"))
    assert len(frame) == 3
    assert os.path.exists(os.path.join(workspace["out"], "coverage_report.json"))


# -- caps --------------------------------------------------------------- #

def test_api_failures_are_diagnosed_as_such(monkeypatch, workspace, capsys):
    """An exhausted key must not be reported as a token-budget problem."""
    generator = DummyGenerationClient()

    def failing_generate(messages):
        # What ModelWrapper does after its retries are exhausted: log the
        # failure, bump the counter, and hand back an empty completion.
        generator.consecutive_failures += 1
        return ""

    monkeypatch.setattr(generator, "generate", failing_generate)
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "2", "--max-rounds", "1", "--max-stall-rounds", "1"],
        generator=generator,
    )
    out = capsys.readouterr().out
    assert "API calls are failing" in out
    assert "not a token-budget problem" in out
    assert "--max-tokens" not in out.split("generated nothing this round")[1][:400]


def test_unparseable_replies_suggest_the_token_budget(monkeypatch, workspace, capsys):
    """With no API failures, an empty reply really is a budget problem."""
    generator = DummyGenerationClient()
    monkeypatch.setattr(generator, "generate", lambda messages: "")
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "2", "--max-rounds", "1", "--max-stall-rounds", "1"],
        generator=generator,
    )
    out = capsys.readouterr().out
    assert "--max-tokens" in out
    assert "API calls are failing" not in out


def test_stall_abort_when_nothing_is_ever_kept(monkeypatch, workspace):
    """An always-rejecting judge must terminate, not spin to the round cap."""
    judge = DummyJudgeClient(VALUE_DICT, reject_all=True)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "5", "--round-batch", "3", "--max-rounds", "50", "--max-stall-rounds", "2"],
    )
    coverage = read_pair_coverage(workspace)
    assert coverage["complete"] is False
    assert any("kept nothing" in note for note in coverage["notes"])
    assert all(row["kept"] == 0 for row in coverage["pairs"])


def test_max_rounds_cap_is_reported(monkeypatch, workspace):
    judge = DummyJudgeClient(VALUE_DICT, reject_all=True)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "5", "--round-batch", "3", "--max-rounds", "1", "--max-stall-rounds", "99"],
    )
    coverage = read_pair_coverage(workspace)
    assert coverage["complete"] is False
    assert any("max-rounds" in note for note in coverage["notes"])


def test_status_makes_no_api_calls(monkeypatch, workspace, capsys):
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "6", "--max-rounds", "10"],
    )
    judge = DummyJudgeClient(VALUE_DICT)
    generator = DummyGenerationClient()
    capsys.readouterr()
    run_balanced(
        monkeypatch, workspace, judge, ["-k", "1", "--status"], generator=generator
    )
    out = capsys.readouterr().out
    assert judge.scenarios_seen == 0
    assert "Pair quota progress" in out


# -- revision salvage --------------------------------------------------- #

def test_revision_salvages_rejected_scenarios(monkeypatch, workspace):
    """Repaired near-misses are re-filtered and counted, with provenance."""
    judge = DummyJudgeClient(VALUE_DICT, reject_first=3)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "6", "--revise-failures", "1"],
    )
    frame = pd.read_csv(workspace["csv"])
    revisions = frame[frame["scenario_id"].str.endswith("-r1")]
    assert not revisions.empty, "no revision was attempted"
    assert revisions["keep_scenario"].astype(str).str.lower().eq("true").any()

    spec = json.loads(revisions.iloc[0]["game_spec"])
    assert spec["metadata"]["revision_of"]


def test_revision_preserves_value_tags(monkeypatch, workspace):
    """A revision must not change which value pair a scenario tests."""
    judge = DummyJudgeClient(VALUE_DICT, reject_first=3)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "6", "--revise-failures", "1"],
    )
    frame = pd.read_csv(workspace["csv"])
    by_id = {r["scenario_id"]: r for _, r in frame.iterrows()}
    for scenario_id, revision in by_id.items():
        if not str(scenario_id).endswith("-r1"):
            continue
        original = by_id.get(str(scenario_id)[: -len("-r1")])
        if original is not None:
            assert json.loads(revision["value_tags"]) == json.loads(original["value_tags"])


def test_malformed_revision_response_costs_one_scenario_not_the_run(monkeypatch, workspace):
    """A model that returns outcomes as a list must not abort the whole run."""
    generator = DummyGenerationClient()
    original_batch = generator.batch_generate

    def batch(messages_list, verbose=False):
        if any("was reviewed and rejected" in m[-1]["content"] for m in messages_list):
            # Outcomes as an array, a very common malformed shape.
            return [
                json.dumps(
                    {"description": "revised", "players": [],
                     "outcomes": [{"description": "z", "payoffs": [1, 2]}]}
                )
                for _ in messages_list
            ]
        return original_batch(messages_list, verbose)

    monkeypatch.setattr(generator, "batch_generate", batch)
    judge = DummyJudgeClient(VALUE_DICT, reject_first=3)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "3", "--revise-failures", "1"],
        generator=generator,
    )
    # The run completed and wrote its corpus despite every revision being junk.
    assert os.path.exists(workspace["csv"])
    coverage = read_pair_coverage(workspace)
    assert len(coverage["pairs"]) == 3


def test_revision_is_not_reapplied_to_an_existing_revision(monkeypatch, workspace):
    """Backlog recovery must not mint a second row with the same revision id."""
    judge = DummyJudgeClient(VALUE_DICT, reject_first=3)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "3", "--revise-failures", "1"],
    )
    # Simulate the interrupt the backlog path exists for: the CSV is lost while
    # games.json (which now holds revisions too) survives.
    os.remove(workspace["csv"])
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "3", "--max-rounds", "2", "--revise-failures", "1"],
    )
    frame = pd.read_csv(workspace["csv"])
    duplicated = frame["scenario_id"][frame["scenario_id"].duplicated()]
    assert duplicated.empty, f"duplicate scenario_ids: {list(duplicated)}"


def test_truncated_corpus_csv_does_not_crash_resume(monkeypatch, workspace):
    """A zero-byte CSV (a write cut short) must be treated as no corpus."""
    open(workspace["csv"], "w").close()
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "3", "--max-rounds", "3"],
    )
    kept, _ = kept_pair_counts_from_csv(workspace["csv"], HHH)
    assert all(kept.get(p, 0) >= 1 for p in all_value_pairs(HHH))


def test_foreign_schema_csv_is_reported_not_crashed(monkeypatch, workspace, capsys):
    """--csv-dir pointed at a solo corpus must warn, not raise KeyError."""
    pd.DataFrame([{"value1": "harmlessness", "value2": "honesty", "keep_scenario": True}]).to_csv(
        workspace["csv"], index=False
    )
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "3", "--max-rounds", "2"],
    )
    assert "no scenario_id column" in capsys.readouterr().out


def test_zero_batch_size_does_not_hang(monkeypatch, workspace):
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "2", "--batch-size", "0", "--max-rounds", "2"],
    )
    assert os.path.exists(workspace["csv"])


def test_revision_disabled_by_default_flag(monkeypatch, workspace):
    judge = DummyJudgeClient(VALUE_DICT, reject_first=3)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "6"],  # --revise-failures 0
    )
    frame = pd.read_csv(workspace["csv"])
    assert frame[frame["scenario_id"].str.endswith("-r1")].empty


# -- cost accounting ---------------------------------------------------- #

def test_cost_block_reports_spend_per_accepted_scenario(monkeypatch, workspace):
    """Acceptance rate alone cannot say whether a mechanism paid for itself."""
    judge = DummyJudgeClient(VALUE_DICT)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "3"],
    )
    cost = read_pair_coverage(workspace)["cost"]
    assert cost["kept_this_run"] > 0
    assert cost["by_role"]["judge"]["requests"] > 0
    assert cost["by_role"]["generation"]["requests"] > 0
    assert 'temperature' in cost['by_role']['judge']
    assert cost["judge_requests_per_kept"] > 0
    # Both denominators are reported, because a pre-screen moves them apart.
    assert cost["kept_per_filtered_scenario"] > 0
    assert cost["kept_per_attempted_outline"] > 0


def test_cost_block_survives_a_run_that_kept_nothing(monkeypatch, workspace):
    """Division by an empty corpus must not take the run down with it."""
    judge = DummyJudgeClient(VALUE_DICT, reject_all=True)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "2", "--max-rounds", "1", "--max-stall-rounds", "1"],
    )
    cost = read_pair_coverage(workspace)["cost"]
    assert cost["kept_this_run"] == 0
    assert cost["requests_per_kept"] is None


def test_short_circuit_is_on_by_default_and_costs_less(monkeypatch, workspace, tmp_path):
    """The saving is on rejections, so it takes a rejecting judge to see it.

    A scenario that passes everything is asked every question in both modes;
    what short-circuiting removes is the dozen questions still put to a
    scenario that has already failed one.
    """
    def spend(ws, judge, extra):
        run_balanced(
            monkeypatch, ws, judge,
            ["-k", "1", "--round-batch", "2", "--max-rounds", "1",
             "--max-stall-rounds", "1", *extra],
        )
        return judge.usage.requests

    default_calls = spend(workspace, DummyJudgeClient(VALUE_DICT, reject_all=True), [])
    assert read_pair_coverage(workspace)["config"]["short_circuit"] is True

    full = {
        "out": str(tmp_path / "games2"),
        "csv_dir": str(tmp_path / "data2"),
        "csv": str(tmp_path / "data2" / "dummy-generator_games.csv"),
        "games_json": str(tmp_path / "games2" / "dummy-generator_games.json"),
        "pair_coverage": str(tmp_path / "games2" / "pair_coverage.json"),
    }
    os.makedirs(full["out"])
    os.makedirs(full["csv_dir"])
    full_calls = spend(full, DummyJudgeClient(VALUE_DICT, reject_all=True), ["--full-checks"])
    assert read_pair_coverage(full)["config"]["short_circuit"] is False
    assert default_calls < full_calls, (
        f"short-circuiting spent {default_calls} judge calls, exhaustive {full_calls}"
    )


def test_revision_forces_full_checks(monkeypatch, workspace, capsys):
    """A repair has to be told every check that failed, not just the first."""
    judge = DummyJudgeClient(VALUE_DICT, reject_first=3)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "3", "--revise-failures", "1"],
    )
    assert "--full-checks" in capsys.readouterr().out
    assert read_pair_coverage(workspace)["config"]["short_circuit"] is False


def test_revision_salvage_is_measured_per_attempt(monkeypatch, workspace):
    """Salvage against attempts, not against the repairs that parsed."""
    judge = DummyJudgeClient(VALUE_DICT, reject_first=3)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "3", "--revise-failures", "1"],
    )
    cost = read_pair_coverage(workspace)["cost"]
    assert cost["revision_attempts"] >= 1
    assert 0.0 <= cost["salvage_per_attempt"] <= 1.0


# -- provenance --------------------------------------------------------- #

def test_every_scenario_records_how_it_was_generated(monkeypatch, workspace):
    """An appended corpus must stay self-describing across mechanism changes."""
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "3", "--max-rounds", "3"],
    )
    frame = pd.read_csv(workspace["csv"])
    for _, row in frame.iterrows():
        spec = json.loads(row["game_spec"])
        generation = spec["metadata"]["generation"]
        assert generation["prescreen"] == "shadow"
        assert "checklist" in generation["guidance"]
        assert 'judge_temperature_effective' in generation
        assert 'screen_temperature_effective' in generation
        # The fields scenario_id and backlog matching depend on must survive.
        assert spec["metadata"]["generating_model"] == "dummy-generator"
        assert spec["metadata"]["context"]


# -- the outline pre-screen --------------------------------------------- #

def test_prescreen_defaults_to_shadow_and_drops_nothing(monkeypatch, workspace):
    """Shipping it in enforce mode would change the corpus before anyone looked."""
    judge = DummyJudgeClient(VALUE_DICT, flag_outlines=99)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "3"],
    )
    coverage = read_pair_coverage(workspace)
    assert coverage["prescreen"]["flagged"] > 0
    assert coverage["prescreen"]["dropped"] == 0
    frame = pd.read_csv(workspace["csv"])
    assert not frame.empty, "shadow mode must not stop scenarios reaching the judge"


def test_prescreen_verdict_travels_into_the_corpus(monkeypatch, workspace):
    """Calibration needs the screen's opinion on rows the judge also ruled on."""
    judge = DummyJudgeClient(VALUE_DICT, flag_outlines=99, flag_criterion="obvious")
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "3", "--max-rounds", "3"],
    )
    frame = pd.read_csv(workspace["csv"])
    verdicts = [json.loads(r["game_spec"])["metadata"]["prescreen"] for _, r in frame.iterrows()]
    assert all(v["flags"] == ["obvious"] for v in verdicts)

    summary = tabulate_prescreen([workspace["csv"]])
    row = summary[summary["criterion"] == "obvious"].iloc[0]
    assert row["n_flagged"] == len(frame)
    assert 0.0 <= row["precision"] <= 1.0
    assert 0.0 <= row["kept_lost_rate"] <= 1.0


def test_enforce_drops_flagged_outlines_before_stage_two(monkeypatch, workspace):
    """The saving is the long stage-two reply that is never requested."""
    judge = DummyJudgeClient(VALUE_DICT, flag_outlines=2)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "9", "--round-batch", "3", "--max-rounds", "1", "--max-stall-rounds", "1",
         "--prescreen", "enforce"],
    )
    with open(workspace["games_json"], encoding="utf-8") as f:
        games = json.load(f)
    # Three outlines, two flagged: only one may reach stage two.
    assert len(games) == 1
    coverage = read_pair_coverage(workspace)
    assert coverage["prescreen"]["screened"] == 3
    assert coverage["prescreen"]["dropped"] == 2


def test_a_round_the_screen_ate_is_diagnosed_as_such(monkeypatch, workspace, capsys):
    """Blaming --max-tokens here would send someone tuning the wrong knob."""
    judge = DummyJudgeClient(VALUE_DICT, flag_outlines=99)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "2", "--max-rounds", "1", "--max-stall-rounds", "1",
         "--prescreen", "enforce"],
    )
    out = capsys.readouterr().out
    assert "pre-screen flagged all" in out
    assert "--max-tokens" not in out


def test_prescreen_off_makes_no_screening_calls(monkeypatch, workspace):
    judge = DummyJudgeClient(VALUE_DICT)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "2", "--max-rounds", "3", "--prescreen", "off"],
    )
    assert judge.outlines_screened == 0
    assert "prescreen" not in read_pair_coverage(workspace)


def test_unknown_prescreen_criterion_is_refused(monkeypatch, workspace):
    """A typo must not silently disable the mechanism being measured."""
    with pytest.raises(SystemExit):
        run_balanced(
            monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
            ["-k", "1", "--round-batch", "2", "--prescreen-criteria", "exclusivity,nonsense"],
        )


# -- paper-style generation and seat mixes -------------------------------- #

def test_paper_style_passively_fills_pairs_and_records_provenance(monkeypatch, workspace):
    """No pair is assigned in the prompt; the generator's ``favors`` labels drive coverage."""
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        [
            "-k", "1", "--round-batch", "3", "--max-rounds", "3",
            "--prompt-style", "paper", "--generation-guidance", "checklist,recipes,stage2",
        ],
    )
    kept, _ = kept_pair_counts_from_csv(workspace["csv"], HHH)
    assert all(kept.get(pair, 0) >= 1 for pair in all_value_pairs(HHH))

    coverage = read_pair_coverage(workspace)
    assert coverage["config"]["prompt_style"] == "paper"
    assert coverage["config"]["guidance"] == []
    with open(str(workspace["games_json"]).replace("_games.json", "_game_prompts.json"),
              encoding="utf-8") as f:
        log = json.load(f)
    assert "Value assignments" not in log["stage_one"][0]["prompt"]
    assert "START FROM ONE SHARED OUTCOME" not in log["stage_one"][0]["prompt"]


def test_paper_quota_targeting_fills_deficits_and_records_provenance(monkeypatch, workspace):
    run_balanced(
        monkeypatch,
        workspace,
        DummyJudgeClient(VALUE_DICT),
        [
            "-k", "1", "--round-batch", "3", "--max-rounds", "3",
            "--prompt-style", "paper", "--paper-quota-targeting",
            "--generation-guidance", "none",
        ],
    )

    coverage = read_pair_coverage(workspace)
    assert coverage["complete"] is True
    assert coverage["config"]["prompt_style"] == "paper"
    assert coverage["config"]["paper_quota_targeting"] is True
    with open(
        str(workspace["games_json"]).replace("_games.json", "_game_prompts.json"),
        encoding="utf-8",
    ) as handle:
        log = json.load(handle)
    assert "required values (one option each, either order)" in log["stage_one"][0]["prompt"]

    frame = pd.read_csv(workspace["csv"])
    for raw_spec in frame.loc[frame["keep_scenario"].astype(bool), "game_spec"]:
        spec = json.loads(raw_spec)
        assert spec["metadata"]["generation"]["paper_quota_targeting"] is True


def test_targeted_paper_resume_does_not_generate_an_already_full_pair(
    monkeypatch, workspace
):
    common = [
        "-k", "1", "--round-batch", "1", "--max-rounds", "1",
        "--prompt-style", "paper", "--generation-guidance", "none",
    ]
    run_balanced(
        monkeypatch,
        workspace,
        DummyJudgeClient(VALUE_DICT),
        common,
    )
    before = read_pair_coverage(workspace)
    before_generated = {row["pair"]: row["generated"] for row in before["pairs"]}
    before_kept = {row["pair"]: row["kept"] for row in before["pairs"]}
    full_pairs = {label for label, kept in before_kept.items() if kept >= 1}
    missing_pairs = {label for label, kept in before_kept.items() if kept == 0}
    assert len(full_pairs) == 2
    assert len(missing_pairs) == 1

    run_balanced(
        monkeypatch,
        workspace,
        DummyJudgeClient(VALUE_DICT),
        [*common, "--paper-quota-targeting"],
    )
    after = read_pair_coverage(workspace)
    after_generated = {row["pair"]: row["generated"] for row in after["pairs"]}
    assert after["complete"] is True
    for label in full_pairs:
        assert after_generated[label] == before_generated[label]
    for label in missing_pairs:
        assert after_generated[label] > before_generated[label]


def test_paper_quota_targeting_rejects_nonbinary_menus(monkeypatch, workspace):
    with pytest.raises(SystemExit, match="requires two actions"):
        run_balanced(
            monkeypatch,
            workspace,
            DummyJudgeClient(VALUE_DICT),
            [
                "-k", "1", "--prompt-style", "paper",
                "--paper-quota-targeting", "--min-actions", "3", "--max-actions", "3",
            ],
        )


def test_one_assistant_seat_reaches_each_generated_spec(monkeypatch, workspace):
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "3", "--max-rounds", "3", "--one-assistant-seat"],
    )
    frame = pd.read_csv(workspace["csv"])
    assert not frame.empty
    for _, row in frame.iterrows():
        spec = json.loads(row["game_spec"])
        assert [p["agent_kind"] for p in spec["players"]].count("assistant") == 1
        assert spec["metadata"]["generation"]["assistant_seats"] == "one"


def test_realism_is_a_valid_opt_in_prescreen_criterion(monkeypatch, workspace):
    """Shadow-only is a default policy, not an inability to request the check."""
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        [
            "-k", "1", "--round-batch", "2", "--max-rounds", "2",
            "--prescreen-criteria", "realism",
        ],
    )
    assert read_pair_coverage(workspace)["config"]["prescreen_criteria"] == ["realism"]


# -- generation guidance ------------------------------------------------ #

def test_guidance_reaches_both_generation_stages(monkeypatch, workspace):
    generator = DummyGenerationClient()
    prompts = []
    original = generator.generate

    def record(messages):
        prompts.append(messages[-1]["content"])
        return original(messages)

    monkeypatch.setattr(generator, "generate", record)
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "2", "--max-rounds", "2"],
        generator=generator,
    )
    assert any("START FROM ONE SHARED OUTCOME" in p for p in prompts)
    with open(str(workspace["games_json"]).replace("_games.json", "_game_prompts.json"),
              encoding="utf-8") as f:
        log = json.load(f)
    assert "recipes" in log["stage_one"][0]["guidance"]
    stage_two = next(iter(log["stage_two"].values()))
    assert "roughly 150-220 words" in stage_two["prompt"]


def test_guidance_none_restores_the_original_prompts(monkeypatch, workspace):
    generator = DummyGenerationClient()
    prompts = []
    original = generator.generate

    def record(messages):
        prompts.append(messages[-1]["content"])
        return original(messages)

    monkeypatch.setattr(generator, "generate", record)
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "2", "--max-rounds", "2",
         "--generation-guidance", "none"],
        generator=generator,
    )
    assert prompts
    for prompt in prompts:
        assert "START FROM ONE SHARED OUTCOME" not in prompt
        assert "MUTUAL EXCLUSIVITY" not in prompt


def test_no_criteria_removes_only_the_checklist(monkeypatch, workspace):
    generator = DummyGenerationClient()
    prompts = []
    original = generator.generate

    def record(messages):
        prompts.append(messages[-1]["content"])
        return original(messages)

    monkeypatch.setattr(generator, "generate", record)
    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "2", "--max-rounds", "2", "--no-criteria"],
        generator=generator,
    )
    assert any("START FROM ONE SHARED OUTCOME" in p for p in prompts)
    assert not any("MUTUAL EXCLUSIVITY" in p for p in prompts)


def test_guidance_does_not_change_which_scenarios_are_generated(monkeypatch, workspace, tmp_path):
    """Two arms must differ in their prompts alone, or they cannot be compared."""
    def value_tags(ws):
        frame = pd.read_csv(ws["csv"])
        return [json.loads(r["value_tags"]) for _, r in frame.iterrows()]

    run_balanced(
        monkeypatch, workspace, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "3", "--max-rounds", "1", "--seed", "7"],
    )
    other = {
        "out": str(tmp_path / "g2"),
        "csv_dir": str(tmp_path / "d2"),
        "csv": str(tmp_path / "d2" / "dummy-generator_games.csv"),
        "games_json": str(tmp_path / "g2" / "dummy-generator_games.json"),
        "pair_coverage": str(tmp_path / "g2" / "pair_coverage.json"),
    }
    os.makedirs(other["out"])
    os.makedirs(other["csv_dir"])
    run_balanced(
        monkeypatch, other, DummyJudgeClient(VALUE_DICT),
        ["-k", "1", "--round-batch", "3", "--max-rounds", "1", "--seed", "7",
         "--generation-guidance", "none"],
    )
    assert value_tags(workspace) == value_tags(other)


def test_cost_does_not_double_count_a_shared_client(monkeypatch, workspace):
    """Two roles can share one client object; its ledger is one ledger."""
    judge = DummyJudgeClient(VALUE_DICT)
    run_balanced(
        monkeypatch, workspace, judge,
        ["-k", "1", "--round-batch", "2", "--max-rounds", "2"],
    )
    cost = read_pair_coverage(workspace)["cost"]
    # The harness routes both the judge and the screen to the same fake.
    assert cost["by_role"]["judge"] == cost["by_role"]["screen"]
    assert cost["requests_total"] == (
        cost["by_role"]["judge"]["requests"] + cost["by_role"]["generation"]["requests"]
    )


# -- JSON config and Anthropic Message Batch orchestration ------------------ #


class DummyMessageBatchGenerationClient(DummyGenerationClient):
    """Generation fake that records and services stable provider batch keys."""

    def __init__(self):
        super().__init__()
        self.provider_batches = []

    def batch_generate(self, messages_list, verbose=False, batch_key=None):
        if batch_key is None:
            return super().batch_generate(messages_list, verbose=verbose)
        self.provider_batches.append(
            {"key": batch_key, "count": len(messages_list), "messages": messages_list}
        )
        responses = []
        for messages in messages_list:
            self._handle_api_success()
            prompt = messages[-1]["content"]
            if "Your task is to generate realistic scenario outlines" in prompt:
                responses.append(self._stage_one(prompt))
            else:
                responses.append(self._stage_two(prompt))
        return responses


def test_config_run_root_derives_paths_and_cli_overrides(tmp_path):
    run_root = tmp_path / "configured-run"
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "value_set": "HHH",
                "run_root": str(run_root),
                "prompt_style": "guided",
                "round_batch": 300,
                "anthropic_message_batches": True,
            }
        ),
        encoding="utf-8",
    )

    args = generate_balanced.parse_args(
        ["--config", str(config_path), "--prompt-style", "paper", "--round-batch", "7"]
    )

    assert args.value_set == "HHH"
    assert args.prompt_style == "paper"
    assert args.round_batch == 7
    assert args.output_dir == os.path.join(str(run_root), "games")
    assert args.csv_dir == str(run_root)
    assert args.message_batch_state_dir == os.path.join(str(run_root), "message_batches")


def test_config_only_status_run_needs_no_cli_value_set_or_api(
    monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "status.json"
    config_path.write_text(
        json.dumps(
            {
                "value_set": "HHH",
                "run_root": str(tmp_path / "run"),
                "model": "anthropic:claude-test",
                "status": True,
                "anthropic_message_batches": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(generate_balanced, "load_value_set", lambda name: VALUE_DICT)
    monkeypatch.setattr(
        ModelWrapper,
        "create",
        staticmethod(lambda *args, **kwargs: pytest.fail("status made an API client")),
    )
    monkeypatch.setattr(
        sys, "argv", ["generate_balanced", "--config", str(config_path)]
    )

    generate_balanced.main()

    assert "Remaining deficit" in capsys.readouterr().out


def test_cli_run_root_overrides_config_paths_except_explicit_cli_path(tmp_path):
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "value_set": "HHH",
                "run_root": "from-config",
                "output_dir": "configured-games",
                "csv_dir": "configured-csv",
            }
        ),
        encoding="utf-8",
    )

    args = generate_balanced.parse_args(
        [
            "--config", str(config_path),
            "--run-root", "from-cli",
            "--csv-dir", "explicit-cli-csv",
        ]
    )

    assert args.output_dir == os.path.join("from-cli", "games")
    assert args.csv_dir == "explicit-cli-csv"
    assert args.message_batch_state_dir == os.path.join("from-cli", "message_batches")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"value_set": "HHH", "unknown_knob": 1},
        {"value_set": "HHH", "round_batch": 1.5},
        {"value_set": "HHH", "prompt_style": "almost-paper"},
        {"value_set": "HHH", "deduplicate": "yes"},
    ],
)
def test_config_rejects_non_object_unknown_untyped_or_bad_choice(tmp_path, payload):
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit):
        generate_balanced.parse_args(["--config", str(config_path)])


def test_message_batch_mode_batches_stage_one_and_uses_distinct_stage_keys(
    monkeypatch, workspace, tmp_path
):
    generator = DummyMessageBatchGenerationClient()
    state_dir = tmp_path / "batch-state"
    run_balanced(
        monkeypatch,
        workspace,
        DummyJudgeClient(VALUE_DICT),
        [
            "--model", "anthropic:claude-test",
            "-k", "1",
            "--round-batch", "3",
            "--max-rounds", "1",
            "--batch-size", "1",
            "--prescreen", "off",
            "--anthropic-message-batches",
            "--archive-stage-one-outlines",
            "--message-batch-state-dir", str(state_dir),
            "--message-batch-poll-seconds", "0.25",
        ],
        generator=generator,
    )

    assert len(generator.provider_batches) == 2
    stage_one, stage_two = generator.provider_batches
    assert stage_one["count"] == 3
    assert "-stage1-" in stage_one["key"]
    assert "-stage2-" in stage_two["key"]
    assert stage_one["key"] != stage_two["key"]
    assert generator.create_kwargs["use_message_batches"] is True
    assert generator.create_kwargs["batch_state_dir"] == os.path.abspath(state_dir)
    assert generator.create_kwargs["batch_poll_interval"] == 0.25

    coverage = read_pair_coverage(workspace)
    assert coverage["config"]["anthropic_message_batches"] is True
    assert coverage["config"]["archive_stage_one_outlines"] is True
    assert coverage["config"]["message_batch_poll_seconds"] == 0.25
    with open(
        os.path.join(workspace["out"], "dummy-generator_outlines.json"),
        encoding="utf-8",
    ) as handle:
        outline_archive = json.load(handle)
    assert len(outline_archive["outlines"]) == 3
    assert {
        record["disposition"] for record in outline_archive["outlines"].values()
    } == {"eligible"}
    journal = json.loads(
        (state_dir / "balanced_message_batch_rounds.json").read_text(encoding="utf-8")
    )
    assert journal["active"] is None
    assert journal["last_completed_round_id"] == 1


def test_unfinished_batch_round_reuses_key_only_for_identical_inputs(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    journal_path = str(state_dir / "balanced_message_batch_rounds.json")
    messages = [[{"role": "user", "content": "same"}]]
    first_state, first_id = generate_balanced._claim_batch_round(
        journal_path, "run-signature", messages
    )
    resumed_state, resumed_id = generate_balanced._claim_batch_round(
        journal_path, "run-signature", messages
    )
    assert first_id == resumed_id == 1
    assert first_state["active"] == resumed_state["active"]

    with pytest.raises(SystemExit, match="different stage-one prompts"):
        generate_balanced._claim_batch_round(
            journal_path,
            "run-signature",
            [[{"role": "user", "content": "changed"}]],
        )


def test_generated_persisted_batch_round_is_committed_on_restart(
    monkeypatch, workspace, tmp_path
):
    state_dir = tmp_path / "batch-state"
    generator = DummyMessageBatchGenerationClient()
    original_complete = generate_balanced._complete_batch_round
    calls = {"count": 0}

    def crash_before_commit(path, state, round_id):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated crash after local persistence")
        return original_complete(path, state, round_id)

    monkeypatch.setattr(generate_balanced, "_complete_batch_round", crash_before_commit)
    common = [
        "--model", "anthropic:claude-test",
        "-k", "1",
        "--round-batch", "3",
        "--max-rounds", "1",
        "--batch-size", "1",
        "--prescreen", "off",
        "--anthropic-message-batches",
        "--message-batch-state-dir", str(state_dir),
        "--message-batch-poll-seconds", "0.01",
    ]
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_balanced(
            monkeypatch,
            workspace,
            DummyJudgeClient(VALUE_DICT),
            common,
            generator=generator,
        )

    pending = json.loads(
        (state_dir / "balanced_message_batch_rounds.json").read_text(encoding="utf-8")
    )
    assert pending["active"]["phase"] == "generated_persisted"

    monkeypatch.setattr(generate_balanced, "_complete_batch_round", original_complete)
    run_balanced(
        monkeypatch,
        workspace,
        DummyJudgeClient(VALUE_DICT),
        common,
        generator=DummyMessageBatchGenerationClient(),
    )
    recovered = json.loads(
        (state_dir / "balanced_message_batch_rounds.json").read_text(encoding="utf-8")
    )
    assert recovered["active"] is None
    assert recovered["last_completed_round_id"] == 1


def _outline_seed(*seat_pairs):
    return {
        "context": "archived outline",
        "players": [],
        "value_assignment": [list(pair) for pair in seat_pairs],
        "agent_kinds": ["assistant"] * len(seat_pairs),
    }


def test_completed_pair_pruning_drops_all_complete_but_retains_mixed_seats():
    pairs = all_value_pairs(HHH)
    complete_pair = pairs[0]
    live_pair = pairs[1]
    tracker = PairQuotaTracker(pairs, 1, kept={complete_pair: 1})

    assert not generate_balanced.seed_has_incomplete_pair(
        _outline_seed(complete_pair, complete_pair), tracker
    )
    assert generate_balanced.seed_has_incomplete_pair(
        _outline_seed(complete_pair, live_pair), tracker
    )


def test_release_seeds_releases_every_requested_seat_pair():
    pairs = all_value_pairs(HHH)
    tracker = PairQuotaTracker(pairs, 5)
    seed = _outline_seed(pairs[0], pairs[1])
    for pair in (pairs[0], pairs[1]):
        tracker.record_pending(pair)
    drops = {}
    generate_balanced.release_seeds(tracker, [seed], 'prescreen', drops)
    rows = {row['pair']: row for row in tracker.summary_rows()}
    assert rows['harmlessness|helpfulness']['pending'] == 0
    assert rows['harmlessness|honesty']['pending'] == 0
    assert drops == {'prescreen': 1}


def test_stage_one_archive_is_idempotent_and_records_pruned_disposition(tmp_path):
    pairs = all_value_pairs(HHH)
    complete_pair = pairs[0]
    live_pair = pairs[1]
    tracker = PairQuotaTracker(pairs, 1, kept={complete_pair: 1})
    complete = _outline_seed(complete_pair, complete_pair)
    mixed = _outline_seed(complete_pair, live_pair)
    mixed["metadata"] = {
        "requested_value_pairs": [list(complete_pair), list(live_pair)]
    }
    archive_path = tmp_path / "outlines.json"
    archive = {
        "schema_version": generate_balanced.STAGE_ONE_ARCHIVE_VERSION,
        "run_signature": "stable-run-signature",
        "value_set": "HHH",
        "generating_model": "dummy",
        "outlines": {},
    }

    generate_balanced._archive_stage_one_seeds(
        str(archive_path),
        archive,
        "stable-run-signature",
        4,
        [(7, [complete, mixed])],
        tracker,
        True,
    )
    first_bytes = archive_path.read_bytes()
    records = list(archive["outlines"].values())
    assert [record["disposition"] for record in records] == [
        "pruned_completed_pairs",
        "eligible",
    ]
    assert records[0]["requested_seat_pairs"] is None
    assert records[1]["requested_seat_pairs"] == [
        "harmlessness|helpfulness",
        "harmlessness|honesty",
    ]
    assert records[1]["actual_seat_pairs"] == records[1]["requested_seat_pairs"]

    # Cached provider replay uses the same logical round/request/seed IDs and
    # rewrites the exact same archive instead of appending duplicates.
    generate_balanced._archive_stage_one_seeds(
        str(archive_path),
        archive,
        "stable-run-signature",
        4,
        [(7, [complete, mixed])],
        tracker,
        True,
    )
    assert len(archive["outlines"]) == 2
    assert archive_path.read_bytes() == first_bytes
