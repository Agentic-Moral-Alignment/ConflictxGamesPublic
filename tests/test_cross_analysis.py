"""Tests for the solo-vs-games cross-analysis tables."""

import os
import sys

import pandas as pd
import pytest

from conflictxgame import cross_analysis

HHH = ["harmlessness", "helpfulness", "honesty"]


def rankings_frame(order, model="model-a"):
    """Build a Bradley-Terry ranking frame with the given best-to-worst order."""
    return pd.DataFrame(
        {
            "value": order,
            "ability": [1.0 - 0.5 * i for i in range(len(order))],
            "ci_lower": [0.5 - 0.5 * i for i in range(len(order))],
            "ci_upper": [1.5 - 0.5 * i for i in range(len(order))],
            "model": model,
        }
    )


@pytest.fixture
def dirs(tmp_path):
    solo = tmp_path / "solo"
    games = tmp_path / "games"
    out = tmp_path / "cross"
    solo.mkdir()
    games.mkdir()
    return {"solo": str(solo), "games": str(games), "out": str(out)}


def write_solo(dirs, order, variant="filtered", model="model-a"):
    rankings_frame(order, model).to_csv(
        os.path.join(dirs["solo"], f"bradley_terry_rankings_{variant}.csv"), index=False
    )


def write_games(dirs, order, name="value_rankings_bradley_terry.csv", model="model-a"):
    rankings_frame(order, model).to_csv(os.path.join(dirs["games"], name), index=False)


def run_cross(monkeypatch, dirs, extra=()):
    argv = [
        "cross_analysis",
        "--solo-analysis-dir", dirs["solo"],
        "--games-analysis-dir", dirs["games"],
        "--output-dir", dirs["out"],
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cross_analysis.main()


# -- loading and alignment ---------------------------------------------- #

def test_ranking_comparison_aligns_settings_on_model_and_value(monkeypatch, dirs):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"])
    write_games(dirs, ["helpfulness", "honesty", "harmlessness"])
    run_cross(monkeypatch, dirs)

    frame = pd.read_csv(os.path.join(dirs["out"], "ranking_comparison.csv"))
    assert set(frame["setting"]) == {"solo_filtered", "games_uniform"}
    solo = frame[frame["setting"] == "solo_filtered"].set_index("value")["rank"]
    games = frame[frame["setting"] == "games_uniform"].set_index("value")["rank"]
    assert solo["harmlessness"] == 0
    assert games["helpfulness"] == 0
    assert len(frame) == 6


def test_uniform_and_reweighted_appear_side_by_side(monkeypatch, dirs):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"])
    write_games(dirs, ["helpfulness", "honesty", "harmlessness"])
    write_games(
        dirs, ["honesty", "harmlessness", "helpfulness"],
        name="value_rankings_bradley_terry_reweighted.csv",
    )
    run_cross(monkeypatch, dirs)

    frame = pd.read_csv(os.path.join(dirs["out"], "ranking_comparison.csv"))
    assert {"games_uniform", "games_matched"} <= set(frame["setting"])
    matched = frame[frame["setting"] == "games_matched"].set_index("value")["rank"]
    assert matched["honesty"] == 0


def test_ranks_are_assigned_per_model(monkeypatch, dirs):
    solo = pd.concat(
        [
            rankings_frame(["harmlessness", "honesty", "helpfulness"], "model-a"),
            rankings_frame(["helpfulness", "honesty", "harmlessness"], "model-b"),
        ],
        ignore_index=True,
    )
    solo.to_csv(os.path.join(dirs["solo"], "bradley_terry_rankings_filtered.csv"), index=False)
    write_games(dirs, ["honesty", "harmlessness", "helpfulness"])
    run_cross(monkeypatch, dirs)

    frame = pd.read_csv(os.path.join(dirs["out"], "ranking_comparison.csv"))
    for model in ("model-a", "model-b"):
        ranks = frame[(frame["model"] == model) & (frame["setting"] == "solo_filtered")]["rank"]
        assert sorted(ranks) == [0, 1, 2]


# -- agreement statistics ----------------------------------------------- #

def test_identical_orders_agree_perfectly(monkeypatch, dirs):
    order = ["harmlessness", "honesty", "helpfulness"]
    write_solo(dirs, order)
    write_games(dirs, order)
    run_cross(monkeypatch, dirs)

    agreement = pd.read_csv(os.path.join(dirs["out"], "rank_agreement.csv"))
    row = agreement.iloc[0]
    assert row["spearman"] == pytest.approx(1.0)
    assert row["kendall"] == pytest.approx(1.0)
    assert bool(row["top_value_agrees"]) is True
    assert bool(row["order_identical"]) is True


def test_reversed_orders_are_perfectly_anticorrelated(monkeypatch, dirs):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"])
    write_games(dirs, ["helpfulness", "honesty", "harmlessness"])
    run_cross(monkeypatch, dirs)

    agreement = pd.read_csv(os.path.join(dirs["out"], "rank_agreement.csv"))
    row = agreement.iloc[0]
    assert row["spearman"] == pytest.approx(-1.0)
    assert bool(row["top_value_agrees"]) is False
    assert bool(row["order_identical"]) is False


def test_agreement_uses_only_shared_values(monkeypatch, dirs):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"])
    write_games(dirs, ["harmlessness", "honesty"])
    run_cross(monkeypatch, dirs)

    agreement = pd.read_csv(os.path.join(dirs["out"], "rank_agreement.csv"))
    assert agreement.iloc[0]["n_values"] == 2


def test_agreement_covers_every_solo_variant(monkeypatch, dirs):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"], variant="filtered")
    write_solo(dirs, ["honesty", "harmlessness", "helpfulness"], variant="unfiltered")
    write_games(dirs, ["harmlessness", "honesty", "helpfulness"])
    run_cross(monkeypatch, dirs)

    agreement = pd.read_csv(os.path.join(dirs["out"], "rank_agreement.csv"))
    assert set(agreement["baseline"]) == {"solo_filtered", "solo_unfiltered"}


# -- pair preference comparison ----------------------------------------- #

def test_pair_preference_comparison_canonicalizes_pairs(monkeypatch, dirs):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"])
    write_games(dirs, ["harmlessness", "honesty", "helpfulness"])
    pd.DataFrame(
        [
            {"model": "model-a", "normalized_value1": "honesty",
             "normalized_value2": "harmlessness", "preference_rate": 0.3, "total": 10},
        ]
    ).to_csv(os.path.join(dirs["solo"], "preference_rates_filtered.csv"), index=False)
    pd.DataFrame(
        [{"model": "model-a", "pair": "harmlessness|honesty", "n_before": 42, "n_after": 20}]
    ).to_csv(os.path.join(dirs["games"], "pair_weight_summary.csv"), index=False)

    run_cross(monkeypatch, dirs)
    frame = pd.read_csv(os.path.join(dirs["out"], "pair_preference_comparison.csv"))
    assert frame.iloc[0]["pair"] == "harmlessness|honesty"
    assert frame.iloc[0]["solo_preference_rate"] == pytest.approx(0.3)
    assert frame.iloc[0]["games_n_comparisons"] == 42


def test_pair_preference_skipped_without_solo_rates(monkeypatch, dirs, capsys):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"])
    write_games(dirs, ["harmlessness", "honesty", "helpfulness"])
    run_cross(monkeypatch, dirs)
    assert not os.path.exists(os.path.join(dirs["out"], "pair_preference_comparison.csv"))
    assert "preference_rates" in capsys.readouterr().out


# -- graceful degradation ----------------------------------------------- #

def test_missing_game_rankings_are_noted_not_fatal(monkeypatch, dirs, capsys):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"])
    run_cross(monkeypatch, dirs)
    out = capsys.readouterr().out
    assert "skipped games_uniform" in out
    # The solo side alone still produces a comparison table.
    assert os.path.exists(os.path.join(dirs["out"], "ranking_comparison.csv"))


def test_no_inputs_at_all_is_reported_cleanly(monkeypatch, dirs, capsys):
    run_cross(monkeypatch, dirs)
    out = capsys.readouterr().out
    assert "(none)" in out
    assert not os.path.exists(os.path.join(dirs["out"], "ranking_comparison.csv"))


def test_empty_ranking_file_is_skipped(monkeypatch, dirs, capsys):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"])
    open(os.path.join(dirs["games"], "value_rankings_bradley_terry.csv"), "w").close()
    run_cross(monkeypatch, dirs)
    assert "skipped games_uniform" in capsys.readouterr().out


def test_mode_comparison_skipped_without_results_dir(monkeypatch, dirs, capsys):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"])
    write_games(dirs, ["harmlessness", "honesty", "helpfulness"])
    run_cross(monkeypatch, dirs)
    assert "skipped mode_comparison" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(dirs["out"], "mode_comparison.csv"))


def test_mode_comparison_drops_probe_rows_from_open(monkeypatch, tmp_path):
    from conflictxgame.games import analyze_games

    results = tmp_path / 'results'
    specs = tmp_path / 'specs'
    results.mkdir()
    specs.mkdir()
    data = pd.DataFrame([
        {'scenario_id': 'base', 'mode': 'open', 'probe': '', 'model': 'm'},
        {'scenario_id': 'probe', 'mode': 'open', 'probe': 'fresh', 'model': 'm'},
    ])
    monkeypatch.setattr(analyze_games, 'load_results', lambda *a, **k: data)
    monkeypatch.setattr(analyze_games, 'load_specs', lambda *_: {'base': object(), 'probe': object()})
    seen = []

    def collect(group, _specs):
        seen.extend(group['scenario_id'])
        return [('a', 'b')]

    monkeypatch.setattr(analyze_games, 'collect_pairwise_value_wins', collect)
    monkeypatch.setattr(
        analyze_games, 'fit_bradley_terry',
        lambda *_a, **_k: pd.DataFrame({'value': ['a', 'b'], 'ability': [1.0, 0.0]}),
    )
    result = cross_analysis.build_mode_comparison(str(results), str(specs), [])
    assert not result.empty
    assert seen == ['base']


def test_plots_are_written_when_requested(monkeypatch, dirs):
    write_solo(dirs, ["harmlessness", "honesty", "helpfulness"])
    write_games(dirs, ["helpfulness", "honesty", "harmlessness"])
    run_cross(monkeypatch, dirs, extra=["--plots"])
    import glob

    assert glob.glob(os.path.join(dirs["out"], "ranking_heatmap_*.png"))
