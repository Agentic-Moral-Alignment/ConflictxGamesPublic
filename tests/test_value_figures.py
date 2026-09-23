"""Tests for :mod:`conflictxgame.games.value_figures`.

Builds a tiny corpus with a known winner per value pair, writes mcq/open/play
results in the evaluate_games row schema, and checks that the pairwise
preference tables recover the planted preferences and that every figure and
table is written.
"""

import json
import os

import pandas as pd
import pytest

from conftest import make_spec
from conflictxgame.games import value_figures
from conflictxgame.games.filter_game_scenarios import results_to_dataframe

pytest.importorskip("matplotlib")
pytest.importorskip("choix")

PAIRS = [("harmlessness", "honesty"), ("harmlessness", "helpfulness"), ("honesty", "helpfulness")]


def build_corpus(scenarios_dir, per_pair=6):
    results = {}
    index = 0
    for v1, v2 in PAIRS:
        for _ in range(per_pair):
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


def base_row(scenario_id, seat, mode, action, tags):
    return {
        "scenario_id": scenario_id, "mode": mode, "seat": seat, "agent_kind": "assistant",
        "agent_mix": "all_assistant", "probe": "", "opponent_config": "", "repetition": seat,
        "round": 0, "choice_action": action, "choice_label": f"action {action}",
        "choice_value_tag": tags[action], "parsed_ok": True, "likert_by_action": "",
        "outcome_ranking": "", "round_payoff": "", "reasoning": "", "model": "test-model",
    }


def build_results(results_dir, scenarios, rounds=3):
    """MCQ and open always pick value A; play picks A in round 0 and B afterwards."""
    mcq, opened, play = [], [], []
    for _, row in scenarios.iterrows():
        tags = json.loads(row["value_tags"])[0]
        for seat in (0, 1):
            m = base_row(row["scenario_id"], seat, "mcq", 0, tags)
            m["likert_by_action"] = json.dumps({"0": 1.0, "1": -1.0})
            mcq.append(m)
            opened.append(base_row(row["scenario_id"], seat, "open", 0, tags))
            for opponent in ("scripted:tit_for_tat", "scripted:always_defect"):
                for rnd in range(rounds):
                    p = base_row(row["scenario_id"], seat, "play", 0 if rnd == 0 else 1, tags)
                    p.update({
                        "opponent_config": opponent, "round": rnd,
                        "round_profile": json.dumps([p["choice_action"], 0]), "round_payoff": 3.0,
                    })
                    play.append(p)
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(mcq).to_csv(os.path.join(results_dir, "test-model_mcq.csv"), index=False)
    pd.DataFrame(opened).to_csv(os.path.join(results_dir, "test-model_open.csv"), index=False)
    pd.DataFrame(play).to_csv(os.path.join(results_dir, "test-model_play.csv"), index=False)


def test_run_writes_figures_and_recovers_planted_preferences(tmp_path):
    scenarios_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "analysis"
    scenarios = build_corpus(str(scenarios_dir))
    build_results(str(results_dir), scenarios)

    tables = value_figures.run(str(results_dir), str(scenarios_dir), str(output_dir), n_bootstrap=5)

    for name in tables:
        assert (output_dir / f"{name}.png").exists(), name
        assert (output_dir / f"{name}.csv").exists(), name

    # The planted design: A always wins in MCQ (choice and Likert) and open, A
    # wins round 1 of play, B wins every later round.
    pairwise = tables["pairwise_preference_by_condition"]
    by_condition = pairwise.groupby("condition")["p"].mean()
    assert by_condition["MCQ choice"] == pytest.approx(1.0)
    assert by_condition["MCQ Likert"] == pytest.approx(1.0)
    assert by_condition["Open-ended"] == pytest.approx(1.0)
    assert by_condition["Play r1"] == pytest.approx(1.0)
    assert by_condition["Play r2"] == pytest.approx(0.0)
    assert by_condition["Play r3"] == pytest.approx(0.0)

    over_rounds = tables["pairwise_preference_over_rounds"]
    # Both opponents exceed the per-opponent minimum (18 scenarios x 2 seats x 3 rounds).
    assert set(over_rounds["opponent"]) == {"scripted:tit_for_tat", "scripted:always_defect"}
    assert (over_rounds[over_rounds["round"] == 1]["p"] == 1.0).all()
    assert (over_rounds[over_rounds["round"] > 1]["p"] == 0.0).all()

    share = tables["value_share_by_condition"].set_index("condition")
    assert share.loc["MCQ choice", "n_decisions"] == len(scenarios) * 2

    agreement = tables["cross_condition_agreement"].set_index(["condition_a", "condition_b"])
    assert agreement.loc[("MCQ choice", "Open-ended"), "agreement"] == pytest.approx(1.0)
    assert agreement.loc[("MCQ choice", "Play r3"), "agreement"] == pytest.approx(0.0)


def test_multiple_models_require_model_flag(tmp_path):
    scenarios_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    scenarios = build_corpus(str(scenarios_dir), per_pair=2)
    build_results(str(results_dir), scenarios)
    extra = pd.read_csv(results_dir / "test-model_mcq.csv")
    extra["model"] = "other-model"
    extra.to_csv(results_dir / "other-model_mcq.csv", index=False)

    with pytest.raises(SystemExit):
        value_figures.run(str(results_dir), str(scenarios_dir), str(tmp_path / "out"), n_bootstrap=2)
    tables = value_figures.run(
        str(results_dir), str(scenarios_dir), str(tmp_path / "out"), model="other-model", n_bootstrap=2
    )
    assert set(tables["value_share_by_condition"]["condition"]) == {"MCQ choice"}


def test_results_exclude_keeps_a_separate_arm_out_of_the_baseline(tmp_path):
    """A same-mode arm in the same directory must not pool into the baseline.

    One results directory holds every arm of an experiment, and the ``mode``
    column cannot tell them apart -- a memory-state probe and the baseline are
    both ``mode == "mcq"``. Without a filename filter the two concatenate into a
    single "MCQ choice" condition and the baseline silently reports the average
    of an experiment and its control.
    """
    scenarios_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    scenarios = build_corpus(str(scenarios_dir), per_pair=4)
    build_results(str(results_dir), scenarios)

    # Same model, same mode, opposite choice -- the arm we do not want pooled in.
    arm = pd.read_csv(results_dir / "test-model_mcq.csv")
    arm["choice_action"] = 1
    arm["choice_value_tag"] = [
        json.loads(r["value_tags"])[0][1]
        for _, r in scenarios.iterrows() for _ in (0, 1)
    ]
    arm.to_csv(results_dir / "test-model_mcq_states.csv", index=False)

    with pytest.raises(ValueError, match='_states'):
        value_figures.run(
            str(results_dir), str(scenarios_dir), str(tmp_path / "pooled"), n_bootstrap=5
        )
    filtered = value_figures.run(
        str(results_dir), str(scenarios_dir), str(tmp_path / "filtered"), n_bootstrap=5,
        results_exclude=["_states"],
    )["pairwise_preference_by_condition"]

    filtered_mcq = filtered[filtered["condition"] == "MCQ choice"]["p"].mean()
    assert filtered_mcq == pytest.approx(1.0), "excluding the arm should restore the baseline"


@pytest.mark.parametrize('probe_column', [pd.Series([float('nan')]), None])
def test_likert_events_tolerates_nan_or_missing_probe(probe_column):
    spec = make_spec(
        'simultaneous', [2, 2],
        {'0,0': [3, 3], '0,1': [0, 5], '1,0': [5, 0], '1,1': [1, 1]},
        game_id='sid', tags=[['a', 'b'], ['a', 'b']],
    )
    row = base_row('sid', 0, 'mcq', 0, ['a', 'b'])
    row['likert_by_action'] = json.dumps({'0': 1, '1': -1})
    frame = pd.DataFrame([row])
    if probe_column is None:
        frame = frame.drop(columns='probe')
    else:
        frame['probe'] = probe_column
    assert len(value_figures.likert_events(frame, {'sid': spec})) == 1


@pytest.mark.parametrize(
    ('model', 'expected'),
    [
        ('openai:anthropic.claude-sonnet-5', 'anthropic.claude-sonnet-5'),
        ('meta-llama/llama-3.3-70b', 'llama-3.3-70b'),
        ('llama-3.3-70b', 'llama-3.3-70b'),
    ],
)
def test_short_model_name_uses_canonical_helper(model, expected):
    assert value_figures.short_model_name(model) == expected


def test_results_exclude_reaches_run_through_a_config_file(tmp_path):
    """The exclusion has to survive the config loader, not just the CLI.

    ``results_exclude`` is repeatable, so it arrives from a config file as a
    JSON list; a loader that only accepts scalars rejects it and the arms stay
    pooled with no figure to show for it.
    """
    scenarios_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    scenarios = build_corpus(str(scenarios_dir), per_pair=2)
    build_results(str(results_dir), scenarios)
    config = tmp_path / "figures.yaml"
    config.write_text(
        "results_dir: " + str(results_dir).replace("\\", "/") + "\n"
        "scenarios_dir: " + str(scenarios_dir).replace("\\", "/") + "\n"
        "output_dir: " + str(tmp_path / "out").replace("\\", "/") + "\n"
        "n_bootstrap: 2\n"
        'results_exclude: ["_play", "_open"]\n',
        encoding="utf-8",
    )

    args = value_figures.parse_args(["--config", str(config)])
    assert args.results_exclude == ["_play", "_open"]
    tables = value_figures.run(
        args.results_dir, args.scenarios_dir, args.output_dir,
        args.model, args.n_bootstrap, args.results_exclude,
    )
    assert set(tables["value_share_by_condition"]["condition"]) == {"MCQ choice"}
