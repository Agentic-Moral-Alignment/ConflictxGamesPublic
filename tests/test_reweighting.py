"""Tests for matching a uniform corpus's comparisons to a reference distribution."""

import json

import pandas as pd
import pytest

from conflictxgame.games.reweighting import (
    ReweightingError,
    build_pair_weights,
    reference_pair_distribution,
    resample_comparisons,
    resample_rankings,
    uniform_pair_weights,
)

HHH = ["harmlessness", "helpfulness", "honesty"]
HARM_HELP = ("harmlessness", "helpfulness")
HARM_HON = ("harmlessness", "honesty")
HELP_HON = ("helpfulness", "honesty")


def reference_csv(tmp_path, counts):
    """Write a reference corpus with the given accepted count per pair."""
    rows = []
    for (v1, v2), count in counts.items():
        for i in range(count):
            rows.append(
                {
                    "scenario_id": f"{v1}-{v2}-{i}",
                    "value1": v1,
                    "value2": v2,
                    "keep_scenario": True,
                    "value_tags": json.dumps([[v1, v2]]),
                }
            )
    path = tmp_path / "reference.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def comparisons_for(counts):
    """Build a comparison list with the given number per pair."""
    out = []
    for (v1, v2), count in counts.items():
        out.extend([(v1, v2)] * count)
    return out


# -- reference distribution --------------------------------------------- #

def test_reference_distribution_matches_accepted_counts(tmp_path):
    path = reference_csv(tmp_path, {HARM_HELP: 6, HARM_HON: 3, HELP_HON: 1})
    weights = reference_pair_distribution(path, HHH)
    assert weights[HARM_HELP] == pytest.approx(0.6)
    assert weights[HARM_HON] == pytest.approx(0.3)
    assert weights[HELP_HON] == pytest.approx(0.1)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_reference_distribution_ignores_rejected_rows(tmp_path):
    path = tmp_path / "ref.csv"
    pd.DataFrame(
        [
            {"scenario_id": "a", "value1": "harmlessness", "value2": "honesty",
             "keep_scenario": True, "value_tags": json.dumps([["harmlessness", "honesty"]])},
            {"scenario_id": "b", "value1": "helpfulness", "value2": "honesty",
             "keep_scenario": False, "value_tags": json.dumps([["helpfulness", "honesty"]])},
        ]
    ).to_csv(path, index=False)
    weights = reference_pair_distribution(str(path), HHH)
    assert weights == {HARM_HON: 1.0}


def test_reference_distribution_requires_accepted_scenarios(tmp_path):
    path = reference_csv(tmp_path, {})
    with pytest.raises(ReweightingError):
        reference_pair_distribution(path, HHH)


def test_uniform_weights_are_equal():
    weights = uniform_pair_weights([HARM_HELP, HARM_HON, HELP_HON, HARM_HON])
    assert len(weights) == 3
    assert all(w == pytest.approx(1 / 3) for w in weights.values())


# -- resampling --------------------------------------------------------- #

def test_resampling_reproduces_the_reference_proportions():
    comparisons = comparisons_for({HARM_HELP: 100, HARM_HON: 100, HELP_HON: 100})
    weights = {HARM_HELP: 0.6, HARM_HON: 0.3, HELP_HON: 0.1}
    resampled, summary = resample_comparisons(comparisons, weights, seed=0)

    counts = {}
    for winner, loser in resampled:
        counts[tuple(sorted((winner, loser)))] = counts.get(tuple(sorted((winner, loser))), 0) + 1
    total = sum(counts.values())
    for pair, weight in weights.items():
        assert counts[pair] / total == pytest.approx(weight, abs=0.01)
    assert set(summary["pair"]) == {
        "harmlessness|helpfulness", "harmlessness|honesty", "helpfulness|honesty"
    }


def test_resampling_keeps_the_binding_pair_whole():
    """The scarcest pair relative to its target sets the scale and is kept intact."""
    comparisons = comparisons_for({HARM_HELP: 100, HARM_HON: 100, HELP_HON: 10})
    weights = {HARM_HELP: 1 / 3, HARM_HON: 1 / 3, HELP_HON: 1 / 3}
    resampled, summary = resample_comparisons(comparisons, weights, seed=0)
    by_pair = {row["pair"]: row for _, row in summary.iterrows()}
    assert by_pair["helpfulness|honesty"]["n_after"] == 10
    assert len(resampled) == 30


def test_resampling_never_duplicates_comparisons():
    """Downsampling only; duplication would fabricate bootstrap certainty."""
    comparisons = comparisons_for({HARM_HELP: 10, HARM_HON: 10})
    weights = {HARM_HELP: 0.9, HARM_HON: 0.1}
    resampled, _ = resample_comparisons(comparisons, weights, seed=0)
    assert len(resampled) <= len(comparisons)
    counts = {}
    for c in resampled:
        counts[c] = counts.get(c, 0) + 1
    for c, n in counts.items():
        assert n <= comparisons.count(c)


def test_resampling_is_deterministic_under_a_seed():
    comparisons = [("harmlessness", "honesty")] * 50 + [("helpfulness", "honesty")] * 50
    weights = {HARM_HON: 0.5, HELP_HON: 0.5}
    first, _ = resample_comparisons(comparisons, weights, seed=3)
    second, _ = resample_comparisons(comparisons, weights, seed=3)
    third, _ = resample_comparisons(comparisons, weights, seed=4)
    assert first == second
    assert isinstance(third, list)


def test_resampling_errors_when_a_weighted_pair_has_no_data():
    """Matching cannot manufacture evidence for a pair that was never evaluated."""
    comparisons = comparisons_for({HARM_HELP: 10})
    weights = {HARM_HELP: 0.5, HELP_HON: 0.5}
    with pytest.raises(ReweightingError, match="helpfulness|honesty"):
        resample_comparisons(comparisons, weights, seed=0)


def test_non_strict_mode_with_no_overlap_raises_a_catchable_error():
    """Must be a ReweightingError, so callers' handlers actually catch it."""
    comparisons = comparisons_for({HARM_HELP: 10})
    with pytest.raises(ReweightingError):
        resample_comparisons(comparisons, {HELP_HON: 1.0}, seed=0, strict=False)


def test_binding_pair_is_kept_whole_across_many_weightings():
    """Float truncation must not silently shave the pair that sets the scale."""
    for size_a, size_b, weight_a in (
        (254, 390, 0.408), (49, 250, 0.0328), (100, 100, 0.5), (7, 993, 0.007),
    ):
        comparisons = comparisons_for({HARM_HELP: size_a, HARM_HON: size_b})
        weights = {HARM_HELP: weight_a, HARM_HON: 1 - weight_a}
        _, summary = resample_comparisons(comparisons, weights, seed=0)
        by_pair = {row["pair"]: row for _, row in summary.iterrows()}
        # Exactly one pair binds; it must keep every comparison it had.
        assert (
            by_pair["harmlessness|helpfulness"]["n_after"] == size_a
            or by_pair["harmlessness|honesty"]["n_after"] == size_b
        ), f"neither pair kept whole for {(size_a, size_b, weight_a)}"


def test_never_keeps_more_than_available():
    comparisons = comparisons_for({HARM_HELP: 5, HARM_HON: 5})
    _, summary = resample_comparisons(
        comparisons, {HARM_HELP: 0.5, HARM_HON: 0.5}, seed=0
    )
    for _, row in summary.iterrows():
        assert row["n_after"] <= row["n_before"]


def test_non_strict_mode_renormalizes_around_missing_pairs():
    comparisons = comparisons_for({HARM_HELP: 10})
    weights = {HARM_HELP: 0.5, HELP_HON: 0.5}
    resampled, summary = resample_comparisons(comparisons, weights, seed=0, strict=False)
    assert len(resampled) == 10
    assert set(summary["pair"]) == {"harmlessness|helpfulness"}


def test_empty_comparisons_pass_through():
    resampled, summary = resample_comparisons([], {HARM_HELP: 1.0}, seed=0)
    assert resampled == []
    assert summary.empty


def test_zero_weights_drop_a_pair():
    comparisons = comparisons_for({HARM_HELP: 10, HARM_HON: 10})
    resampled, _ = resample_comparisons(
        comparisons, {HARM_HELP: 1.0, HARM_HON: 0.0}, seed=0
    )
    assert all(tuple(sorted(c)) == HARM_HELP for c in resampled)


def test_extreme_weights_can_empty_a_pair_and_warn(caplog):
    """An extreme reference can trim a pair to nothing, dropping its values.

    Arithmetically correct for the requested distribution, but it changes which
    values the ranking covers, so it must be surfaced rather than silent.
    """
    import logging

    comparisons = comparisons_for({HARM_HELP: 20, HARM_HON: 20})
    weights = {HARM_HELP: 0.99, HARM_HON: 0.01}
    with caplog.at_level(logging.WARNING):
        resampled, summary = resample_comparisons(comparisons, weights, seed=0)
    by_pair = {row["pair"]: row for _, row in summary.iterrows()}
    assert by_pair["harmlessness|honesty"]["n_after"] == 0
    assert all(tuple(sorted(c)) == HARM_HELP for c in resampled)
    assert "no comparisons for harmlessness|honesty" in caplog.text


def test_summary_reports_shares_before_and_after():
    comparisons = comparisons_for({HARM_HELP: 90, HARM_HON: 10})
    weights = {HARM_HELP: 0.5, HARM_HON: 0.5}
    _, summary = resample_comparisons(comparisons, weights, seed=0)
    by_pair = {row["pair"]: row for _, row in summary.iterrows()}
    assert by_pair["harmlessness|helpfulness"]["share_before"] == pytest.approx(0.9)
    assert by_pair["harmlessness|helpfulness"]["share_after"] == pytest.approx(0.5)


# -- rankings ----------------------------------------------------------- #

def test_rankings_are_resampled_by_pair():
    rankings = [["harmlessness", "honesty"]] * 50 + [["helpfulness", "honesty"]] * 50
    weights = {HARM_HON: 0.8, HELP_HON: 0.2}
    resampled, _ = resample_rankings(rankings, weights, seed=0)
    counts = {}
    for ranking in resampled:
        key = tuple(sorted(ranking))
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    assert counts[HARM_HON] / total == pytest.approx(0.8, abs=0.02)


def test_rankings_spanning_three_values_pass_through_unweighted():
    rankings = [["harmlessness", "honesty"]] * 10 + [
        ["harmlessness", "honesty", "helpfulness"]
    ] * 3
    resampled, _ = resample_rankings(rankings, {HARM_HON: 1.0}, seed=0)
    wide = [r for r in resampled if len(set(r)) == 3]
    assert len(wide) == 3


def test_rankings_preserve_order_within_a_ranking():
    rankings = [["honesty", "harmlessness"]] * 10
    resampled, _ = resample_rankings(rankings, {HARM_HON: 1.0}, seed=0)
    assert all(r == ["honesty", "harmlessness"] for r in resampled)


# -- weight resolution -------------------------------------------------- #

def test_build_pair_weights_none_disables():
    assert build_pair_weights("none", [HARM_HELP]) is None


def test_build_pair_weights_match_requires_reference():
    with pytest.raises(ReweightingError, match="reference-csv"):
        build_pair_weights("match", [HARM_HELP])


def test_build_pair_weights_match_reads_reference(tmp_path):
    path = reference_csv(tmp_path, {HARM_HELP: 3, HARM_HON: 1})
    weights = build_pair_weights("match", [HARM_HELP, HARM_HON], path, HHH)
    assert weights[HARM_HELP] == pytest.approx(0.75)


def test_build_pair_weights_rejects_unknown_mode():
    with pytest.raises(ReweightingError):
        build_pair_weights("bogus", [HARM_HELP])


# -- integration with the Bradley-Terry fit ----------------------------- #

def test_reweighting_shifts_the_fitted_ranking():
    """The point of the exercise: composition changes what the fit concludes."""
    from conflictxgame.games.analyze_games import fit_bradley_terry

    # honesty beats harmlessness a lot; helpfulness beats honesty a little.
    comparisons = [("honesty", "harmlessness")] * 100 + [("helpfulness", "honesty")] * 100
    weights = {HARM_HON: 0.05, HELP_HON: 0.95}
    resampled, _ = resample_comparisons(comparisons, weights, seed=0)

    base = fit_bradley_terry(comparisons, n_bootstrap=5)
    shifted = fit_bradley_terry(resampled, n_bootstrap=5)
    assert base is not None and shifted is not None
    # Down-weighting the honesty>harmlessness evidence lowers honesty's lead
    # over harmlessness.
    def gap(frame):
        abilities = dict(zip(frame["value"], frame["ability"]))
        return abilities["honesty"] - abilities["harmlessness"]

    assert gap(shifted) < gap(base)
