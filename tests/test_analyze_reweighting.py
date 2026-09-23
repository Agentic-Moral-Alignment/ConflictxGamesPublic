"""End-to-end tests for `analyze_games --pair-weights`.

Covers the CLI wiring around :mod:`conflictxgame.games.reweighting`: that
`none` leaves the existing outputs byte-identical, that `match` writes the
extra `*_reweighted.csv` files alongside them, and that a corpus lacking a
pair the reference weights is reported rather than silently mismatched.
"""

import json
import os
import sys

import pandas as pd
import pytest

from conftest import make_spec
from conflictxgame.games import analyze_games
from conflictxgame.games.filter_game_scenarios import results_to_dataframe

VALUES = ["harmlessness", "helpfulness", "honesty"]


def build_corpus(scenarios_dir, pair_counts):
    """Write a scenarios CSV with the requested number of games per pair."""
    results = {}
    index = 0
    for (v1, v2), count in pair_counts.items():
        for _ in range(count):
            spec = make_spec(
                "simultaneous", [2, 2],
                {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
                game_id=f"simultaneous_n2_a2x2-{index}",
                tags=[[v1, v2], [v1, v2]],
            )
            spec.metadata["generating_model"] = "gen"
            spec.metadata["context"] = "context"
            results[spec.game_id] = {
                "spec": spec, "checks": {"scenario_realism": True}, "keep_scenario": True,
            }
            index += 1
    frame = results_to_dataframe(results)
    frame["generating_model"] = "gen"
    os.makedirs(scenarios_dir, exist_ok=True)
    frame.to_csv(os.path.join(scenarios_dir, "gen_games.csv"), index=False)
    return frame


def build_results(results_dir, scenarios_frame, winner_by_pair):
    """Write an evaluate_games-style results CSV choosing a fixed winner per pair."""
    rows = []
    for _, row in scenarios_frame.iterrows():
        tags = json.loads(row["value_tags"])[0]
        pair = tuple(sorted(set(tags)))
        winner = winner_by_pair[pair]
        action = tags.index(winner)
        # Mirrors the row schema evaluate_games writes for --mode mcq.
        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "model": "test-model",
                "mode": "mcq",
                "seat": 0,
                "agent_kind": "assistant",
                "agent_mix": "all_assistant",
                "probe": "",
                "opponent_config": "",
                "repetition": 0,
                "round": 0,
                "choice_action": action,
                "choice_label": f"favor {winner}",
                "choice_value_tag": winner,
                "parsed_ok": True,
                "likert_by_action": json.dumps({str(action): 6, str(1 - action): 2}),
                "outcome_ranking": "",
                "round_payoff": "",
                "reasoning": "",
            }
        )
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, "test-model_mcq.csv"), index=False)


def reference_csv(path, pair_counts):
    """Write a reference corpus with the given accepted counts per pair."""
    rows = []
    for (v1, v2), count in pair_counts.items():
        for i in range(count):
            rows.append(
                {
                    "scenario_id": f"{v1}-{v2}-{i}",
                    "value1": v1, "value2": v2,
                    "keep_scenario": True,
                    "value_tags": json.dumps([[v1, v2]]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def workspace(tmp_path):
    """A corpus where the model prefers honesty over both other values."""
    scenarios = tmp_path / "data"
    results = tmp_path / "results"
    counts = {
        ("harmlessness", "honesty"): 20,
        ("helpfulness", "honesty"): 20,
        ("harmlessness", "helpfulness"): 20,
    }
    frame = build_corpus(str(scenarios), counts)
    build_results(
        str(results), frame,
        {
            ("harmlessness", "honesty"): "honesty",
            ("helpfulness", "honesty"): "honesty",
            ("harmlessness", "helpfulness"): "harmlessness",
        },
    )
    return {
        "scenarios": str(scenarios),
        "results": str(results),
        "out": str(tmp_path / "analysis"),
        "tmp": tmp_path,
    }


def run_analyze(monkeypatch, workspace, extra=()):
    argv = [
        "analyze_games",
        "--results-dir", workspace["results"],
        "--scenarios-dir", workspace["scenarios"],
        "--output-dir", workspace["out"],
        "--n-bootstrap", "5",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    analyze_games.main()


def test_load_results_refuses_mixed_arms_and_can_select_one(tmp_path):
    results = tmp_path / 'results'
    results.mkdir()
    base = pd.DataFrame([{'scenario_id': 's', 'mode': 'mcq', 'model': 'm'}])
    base.to_csv(results / 'model_mcq.csv', index=False)
    base.assign(scenario_id='state').to_csv(results / 'model_mcq_states.csv', index=False)
    with pytest.raises(ValueError, match='_states'):
        analyze_games.load_results(str(results))
    selected = analyze_games.load_results(str(results), arm='_states')
    assert selected['scenario_id'].tolist() == ['state']
    baseline = analyze_games.load_results(str(results), arm='baseline')
    assert baseline['scenario_id'].tolist() == ['s']


def test_cli_arm_reaches_result_loader(monkeypatch, workspace):
    seen = []

    def fake_load(_path, exclude=None, arm=None):
        seen.append(arm)
        return pd.DataFrame()

    monkeypatch.setattr(analyze_games, 'load_results', fake_load)
    run_analyze(monkeypatch, workspace, ['--arm', '_states'])
    assert seen == ['_states']


def test_open_rows_enter_ranking_only_when_requested():
    frame = pd.DataFrame([
        {'scenario_id': 'm', 'mode': 'mcq', 'probe': ''},
        {'scenario_id': 'o', 'mode': 'open', 'probe': ''},
        {'scenario_id': 'op', 'mode': 'open', 'probe': 'fresh'},
        {'scenario_id': 'p', 'mode': 'play', 'probe': ''},
    ])
    assert set(analyze_games.ranking_decisions(frame)['scenario_id']) == {'m', 'p'}
    assert set(analyze_games.ranking_decisions(frame, include_open=True)['scenario_id']) == {
        'm', 'o', 'p',
    }


def test_default_run_writes_no_reweighted_files(monkeypatch, workspace):
    run_analyze(monkeypatch, workspace)
    written = os.listdir(workspace["out"])
    assert "value_rankings_bradley_terry.csv" in written
    assert not any("reweighted" in name for name in written)
    assert "pair_weight_summary.csv" not in written


def test_pair_weights_none_leaves_outputs_unchanged(monkeypatch, workspace):
    """Explicitly passing none must be identical to omitting the flag."""
    run_analyze(monkeypatch, workspace)
    baseline = pd.read_csv(
        os.path.join(workspace["out"], "value_rankings_bradley_terry.csv")
    )
    run_analyze(monkeypatch, workspace, ["--pair-weights", "none"])
    after = pd.read_csv(os.path.join(workspace["out"], "value_rankings_bradley_terry.csv"))
    pd.testing.assert_frame_equal(baseline, after)


def test_match_writes_both_readings(monkeypatch, workspace):
    reference = reference_csv(
        workspace["tmp"] / "reference.csv",
        {
            ("harmlessness", "honesty"): 5,
            ("helpfulness", "honesty"): 5,
            ("harmlessness", "helpfulness"): 90,
        },
    )
    run_analyze(
        monkeypatch, workspace,
        ["--pair-weights", "match", "--reference-csv", reference],
    )
    written = os.listdir(workspace["out"])
    assert "value_rankings_bradley_terry.csv" in written, "uniform reading missing"
    assert "value_rankings_bradley_terry_reweighted.csv" in written
    assert "pair_weight_summary.csv" in written

    summary = pd.read_csv(os.path.join(workspace["out"], "pair_weight_summary.csv"))
    by_pair = {r["pair"]: r for _, r in summary.iterrows()}
    # The reference is dominated by harmlessness-helpfulness, so that pair keeps
    # its comparisons while the honesty pairs are trimmed hard.
    assert by_pair["harmlessness|helpfulness"]["n_after"] > by_pair["harmlessness|honesty"]["n_after"]
    assert summary["model"].eq("test-model").all()


def test_matching_changes_the_ranking(monkeypatch, workspace):
    """Down-weighting the honesty conflicts must cost honesty its lead."""
    reference = reference_csv(
        workspace["tmp"] / "reference.csv",
        {
            ("harmlessness", "honesty"): 10,
            ("helpfulness", "honesty"): 10,
            ("harmlessness", "helpfulness"): 80,
        },
    )
    run_analyze(
        monkeypatch, workspace,
        ["--pair-weights", "match", "--reference-csv", reference],
    )
    uniform = pd.read_csv(os.path.join(workspace["out"], "value_rankings_bradley_terry.csv"))
    matched = pd.read_csv(
        os.path.join(workspace["out"], "value_rankings_bradley_terry_reweighted.csv")
    )

    def gap(frame):
        abilities = dict(zip(frame["value"], frame["ability"]))
        return abilities["honesty"] - abilities["harmlessness"]

    assert gap(matched) < gap(uniform)


def test_uniform_weights_need_no_reference(monkeypatch, workspace):
    run_analyze(monkeypatch, workspace, ["--pair-weights", "uniform"])
    assert os.path.exists(
        os.path.join(workspace["out"], "value_rankings_bradley_terry_reweighted.csv")
    )


def test_match_without_reference_fails_fast(monkeypatch, workspace):
    """Better to refuse up front than to emit an unweighted run under that name."""
    with pytest.raises(SystemExit, match="reference-csv"):
        run_analyze(monkeypatch, workspace, ["--pair-weights", "match"])


def test_missing_reference_file_fails_fast(monkeypatch, workspace):
    with pytest.raises(SystemExit, match="not found"):
        run_analyze(
            monkeypatch, workspace,
            ["--pair-weights", "match", "--reference-csv", "no/such/file.csv"],
        )


def test_reference_pair_absent_from_data_is_reported(monkeypatch, workspace, capsys):
    """Matching cannot invent evidence for a conflict that was never evaluated."""
    reference = reference_csv(
        workspace["tmp"] / "reference.csv",
        {("harmlessness", "honesty"): 5, ("privacy", "creativity"): 5},
    )
    run_analyze(
        monkeypatch, workspace,
        ["--pair-weights", "match", "--reference-csv", reference],
    )
    out = capsys.readouterr().out
    assert "no comparisons for" in out
    assert not os.path.exists(
        os.path.join(workspace["out"], "value_rankings_bradley_terry_reweighted.csv")
    )


def test_likert_rankings_are_reweighted_too(monkeypatch, workspace):
    reference = reference_csv(
        workspace["tmp"] / "reference.csv",
        {
            ("harmlessness", "honesty"): 10,
            ("helpfulness", "honesty"): 10,
            ("harmlessness", "helpfulness"): 80,
        },
    )
    run_analyze(
        monkeypatch, workspace,
        ["--pair-weights", "match", "--reference-csv", reference],
    )
    assert os.path.exists(
        os.path.join(workspace["out"], "value_rankings_plackett_luce_reweighted.csv")
    )
