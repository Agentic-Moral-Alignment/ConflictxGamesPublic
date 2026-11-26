"""Regression tests for the NeurIPS 2026 offline paper reanalysis."""
from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reproduce" / "reanalysis.py"
FIGURE_SCRIPT = ROOT / "reproduce" / "make_paper_figures.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


paper = _load(SCRIPT, "paper_reanalysis_tests")
figures = _load(FIGURE_SCRIPT, "paper_figure_tests")


@pytest.mark.parametrize(
    ("representation", "condition", "states", "expected"),
    [
        ("menu", "baseline", False, ("mcq", "")),
        ("menu", "payoff", False, ("mcq", "_stakes-payoff")),
        ("menu", "alignment", False, ("mcq", "_stakes-alignment")),
        ("menu", "usefulness", False, ("mcq", "_stakes-usefulness")),
        ("menu", "steering", False, ("mcq", "_steer-HHH")),
        ("menu", "observed", False, ("mcq", "_observed")),
        ("menu", "unobserved", False, ("mcq", "_unobserved")),
        ("menu", "baseline", True, ("mcq", "_states")),
        ("menu", "payoff", True, ("mcq", "_states_stakes-payoff")),
        ("menu", "alignment", True, ("mcq", "_states_stakes-alignment")),
        ("menu", "usefulness", True, ("mcq", "_states_stakes-usefulness")),
        ("open_undisclosed", "baseline", False, ("open", "")),
        ("open_undisclosed", "payoff", False, ("open", "_stakes-payoff")),
        ("open_undisclosed", "alignment", False, ("open", "_stakes-alignment")),
        ("open_undisclosed", "usefulness", False, ("open", "_stakes-usefulness")),
        ("open_undisclosed", "steering", False, ("open", "_steer-HHH")),
        ("open_undisclosed", "observed", False, ("open", "_observed")),
        ("open_undisclosed", "unobserved", False, ("open", "_unobserved")),
        ("open_undisclosed", "baseline", True, ("open", "_states")),
        ("open_undisclosed", "payoff", True, ("open", "_states-stakes-payoff")),
        ("open_undisclosed", "alignment", True, ("open", "_states-stakes-alignment")),
        ("open_undisclosed", "usefulness", True, ("open", "_states-stakes-usefulness")),
        ("open_disclosed", "baseline", False, ("open", "_counterpart")),
        ("open_disclosed", "payoff", False, ("open", "_counterpart_stakes-payoff")),
        ("open_disclosed", "alignment", False, ("open", "_counterpart_stakes-alignment")),
        ("open_disclosed", "usefulness", False, ("open", "_counterpart_stakes-usefulness")),
        ("open_disclosed", "steering", False, ("open", "_counterpart_steer-HHH")),
        ("open_disclosed", "baseline", True, ("open", "_counterpart_states")),
        ("open_disclosed", "payoff", True, ("open", "_counterpart_states-stakes-payoff")),
        ("open_disclosed", "alignment", True, ("open", "_counterpart_states-stakes-alignment")),
        ("open_disclosed", "usefulness", True, ("open", "_counterpart_states-stakes-usefulness")),
    ],
)
def test_exact_arm_mapping(representation, condition, states, expected):
    assert paper.arm_token_for(representation, condition, states=states) == expected


def test_disclosed_representation_rejects_visibility_arm():
    with pytest.raises(ValueError, match="has no"):
        paper.arm_token_for("open_disclosed", "observed")


def test_missing_and_duplicate_arm_rejection(tmp_path):
    with pytest.raises(ValueError, match="missing result arm"):
        paper.resolve_result_file(
            str(tmp_path), "model", "menu", filenames=["model_mcq_balanced50_observed.csv"]
        )
    duplicate = [
        "model_open_balanced50_user-a.csv",
        "model_open_balanced50_user-b.csv",
    ]
    with pytest.raises(ValueError, match="duplicate result arm"):
        paper.resolve_result_file(
            str(tmp_path), "model", "open_undisclosed", filenames=duplicate
        )


def test_direct_game_spec_scenario_parsing(tmp_path):
    spec = {
        "players": [
            {
                "value_pair": ["honesty", "helpfulness"],
                "actions": [{"value_tag": "honesty"}, {"value_tag": "helpfulness"}],
            },
            {
                "value_pair": ["harmlessness", "honesty"],
                "actions": [{"value_tag": "honesty"}, {"value_tag": "harmlessness"}],
            },
        ]
    }
    source = tmp_path / "scenarios.csv"
    pd.DataFrame([{"scenario_id": "s1", "game_spec": json.dumps(spec)}]).to_csv(
        source, index=False
    )
    pairs, actions, provenance = paper.load_pairs_and_actions(str(source))
    assert pairs[("s1", 0)] == ("helpfulness", "honesty")
    assert actions[("s1", 0)] == ("honesty", "helpfulness")
    assert pairs[("s1", 1)] == ("harmlessness", "honesty")
    assert provenance["sha256"]


def test_complete_ten_model_arm_coverage_in_canonical_results():
    resolved = paper.resolve_all_result_files(paper.GAME_DIR, paper.condition_specs())
    assert len(resolved) == 31 * 10
    assert {model for _, model in resolved} == set(paper.MODELS)


def _frame(rows):
    normalized = []
    for item, scenario, seat, chosen, answered in rows:
        normalized.append(
            {
                "item_key": item,
                "scenario_id": scenario,
                "seat": seat,
                "probe": "",
                "state": "",
                "pair": ("harmlessness", "honesty"),
                "chosen": chosen if answered else "",
                "answered": answered,
            }
        )
    return pd.DataFrame(normalized)


def test_paired_answered_item_intersection_and_refusals():
    base = _frame(
        [
            (("s1", 0), "s1", 0, "harmlessness", True),
            (("s2", 0), "s2", 0, "honesty", True),
            (("s3", 0), "s3", 0, "", False),
        ]
    )
    arm = _frame(
        [
            (("s1", 0), "s1", 0, "honesty", True),
            (("s2", 0), "s2", 0, "", False),
            (("s3", 0), "s3", 0, "harmlessness", True),
        ]
    )
    paired = paper.paired_frame(base, arm, "harmlessness")
    assert paired["item_key"].tolist() == [("s1", 0)]
    assert paired["d"].tolist() == [-1]
    # The helper must not mutate answered status while filtering values.
    assert base["answered"].tolist() == [True, True, False]
    assert arm["answered"].tolist() == [True, False, True]


def test_four_condition_interaction_arithmetic():
    def choices(first, second):
        return _frame(
            [
                (("s1", 0), "s1", 0, first, True),
                (("s2", 0), "s2", 0, second, True),
            ]
        )

    base_a = choices("honesty", "harmlessness")       # y: 0, 1
    stake_a = choices("harmlessness", "honesty")      # y: 1, 0 => +1, -1
    base_b = choices("honesty", "honesty")            # y: 0, 0
    stake_b = choices("honesty", "harmlessness")      # y: 0, 1 => 0, +1
    result = paper.interaction_frame(base_a, stake_a, base_b, stake_b, "harmlessness")
    assert result["effect_a"].tolist() == [1, -1]
    assert result["effect_b"].tolist() == [0, 1]
    assert result["d"].tolist() == [-1, 2]
    assert result["d"].mean() == pytest.approx(0.5)


def test_scenario_and_seat_alignment_is_validated():
    base = _frame([(("shared", 0), "s1", 0, "harmlessness", True)])
    arm = _frame([(("shared", 0), "s2", 0, "honesty", True)])
    with pytest.raises(ValueError, match="scenario_id misalignment"):
        paper.paired_frame(base, arm, "harmlessness")


@pytest.mark.parametrize("size", [9, 27])
def test_holm_family_sizes(size):
    records = [
        {"anchor": f"a{index}", "perm_p_scenario_clustered": (index + 1) / (size + 1)}
        for index in range(size)
    ]
    result = paper.apply_holm_family(records, "test", size)
    assert result["size"] == size
    assert all(0.0 <= record["p_holm"] <= 1.0 for record in records)
    with pytest.raises(ValueError, match="expected"):
        paper.apply_holm_family(records[:-1], "wrong", size)


def test_finite_json_and_native_scalar_conversion():
    converted = paper._native_json({"count": np.int64(3), "rate": np.float64(0.25)})
    assert type(converted["count"]) is int
    assert type(converted["rate"]) is float
    paper.assert_finite_json(converted)
    with pytest.raises(ValueError, match="root.bad"):
        paper.assert_finite_json({"bad": float("nan")})
    with pytest.raises(TypeError, match="Non-native"):
        paper.assert_finite_json({"bad": np.int64(1)})


def test_deterministic_small_artifact_writes(tmp_path):
    stats = {"manuscript_numbers": {"StatExample": "0.125"}, "x": 0.125}
    tables = {"small.csv": pd.DataFrame([{"b": 2, "a": 1.25}])}
    provenance = {"inputs": [{"path": "same", "sha256": "abc", "bytes": 1}]}
    first, second = tmp_path / "first", tmp_path / "second"
    paper.write_artifacts(str(first), stats, tables, provenance, ["done"])
    paper.write_artifacts(str(second), stats, tables, provenance, ["done"])
    for name in ("small.csv", "stats.json", "provenance.json", "manuscript_numbers.tex", "reanalysis_log.txt"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def _figure_stats():
    def estimate(**fields):
        return {
            **fields,
            "pooled_delta": 0.01,
            "ci_lower": -0.02,
            "ci_upper": 0.04,
        }

    levels = []
    for representation in figures.REPRESENTATIONS:
        for value in figures.VALUES:
            levels.append(
                {
                    "representation": representation,
                    "value": value,
                    "mean_rate": 0.5,
                    "ci_lower": 0.45,
                    "ci_upper": 0.55,
                    "per_model": [{"rate": 0.45 + index / 100} for index in range(10)],
                }
            )
    stake = [
        estimate(representation=r, stake=s, value=v)
        for r in figures.REPRESENTATIONS
        for s in figures.STAKES
        for v in figures.VALUES
    ]
    steering = [
        estimate(representation=r, value=v)
        for r in figures.REPRESENTATIONS
        for v in figures.VALUES
    ]
    visibility = [
        estimate(representation=r, value=v)
        for r in ("menu", "open_undisclosed")
        for v in figures.VALUES
    ]
    history = [
        estimate(representation=r, stake=s, state=state, value=v)
        for r in figures.REPRESENTATIONS
        for s in figures.STAKES
        for state in figures.STATES
        for v in figures.VALUES
    ]
    ongoing = [
        estimate(representation=r, value=v)
        for r in figures.REPRESENTATIONS
        for v in figures.VALUES
    ]
    # Menu-versus-open contrasts, shaped like reanalysis.py's records; the
    # stakes figure annotates the payoff/helpfulness pair with p_holm.
    interactions = [
        estimate(
            representation_a="menu",
            representation_b=r,
            stake=s,
            value=v,
            p_holm=0.01,
        )
        for r in ("open_undisclosed", "open_disclosed")
        for s in figures.STAKES
        for v in figures.VALUES
    ]
    return {
        "representation_levels": levels,
        "stake_effects": stake,
        "stake_representation_interactions": interactions,
        "steering_effects": steering,
        "visibility_effects": visibility,
        "history_stake_effects": history,
        "ongoing_framing": ongoing,
        "ordering_stability": {
            r: {"steering": {"n_flipped": 2}} for r in figures.REPRESENTATIONS
        },
    }


def test_figure_smoke_generation(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(_figure_stats()), encoding="utf-8")
    output = tmp_path / "figures"
    figures.make_figures(str(stats_path), str(output))
    for stem in ("stakes_by_action_representation", "context_modifiers"):
        assert (output / f"{stem}.pdf").stat().st_size > 1000
        assert (output / f"{stem}.png").stat().st_size > 1000


def test_generated_manuscript_numbers_trace_to_stats(tmp_path):
    stats = {
        "design": {"n_models": 10, "n_scenarios": 102, "n_scenario_seats": 150},
        "representation_levels": [],
        "representation_contrasts": [],
        "stake_effects": [],
        "stake_representation_interactions": [],
        "steering_effects": [],
        "visibility_effects": [],
        "ongoing_framing": [],
        "history_stake_effects": [],
    }
    stats["manuscript_numbers"] = paper.manuscript_numbers(stats)
    output = tmp_path / "trace"
    paper.write_artifacts(str(output), stats, {}, {"inputs": []}, ["done"])
    tex = (output / "manuscript_numbers.tex").read_text(encoding="utf-8")
    for macro, value in stats["manuscript_numbers"].items():
        assert f"\\providecommand{{\\{macro}}}{{{value}}}" in tex
