"""Tests for recomputing per-pair accepted counts from a filtered CSV.

This recount is the balanced generator's entire resume mechanism, so it has to
survive everything the CSVs actually contain: menu-order value pairs, booleans
that round-trip as text, rows with no ``value_tags``, and multi-seat games.
"""

import json

import pandas as pd
import pytest

from conflictxgame.games.pair_quota import kept_pair_counts_from_csv

HHH = ["harmlessness", "helpfulness", "honesty"]
HARM_HON = ("harmlessness", "honesty")
HELP_HON = ("helpfulness", "honesty")


def row(seat_tags, keep, **overrides):
    """Build one filtered-CSV row from per-seat value-tag lists."""
    data = {
        "scenario_id": overrides.pop("scenario_id", "m_simultaneous_n2_a2x2-0"),
        "value1": seat_tags[0][0] if seat_tags else "",
        "value2": seat_tags[0][1] if seat_tags and len(seat_tags[0]) > 1 else "",
        "keep_scenario": keep,
        "value_tags": json.dumps(seat_tags),
    }
    data.update(overrides)
    return data


def write_csv(tmp_path, rows, name="model_games.csv"):
    """Write rows to a CSV and return its path."""
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_missing_file_yields_empty_counts(tmp_path):
    kept, generated = kept_pair_counts_from_csv(str(tmp_path / "absent.csv"), HHH)
    assert kept == {}
    assert generated == {}


def test_empty_csv_yields_empty_counts(tmp_path):
    path = write_csv(tmp_path, [])
    kept, generated = kept_pair_counts_from_csv(path, HHH)
    assert kept == {}
    assert generated == {}


def test_flipped_tag_order_counts_as_one_pair(tmp_path):
    """The CSVs store seat-0 tags in menu order, so both orderings occur."""
    path = write_csv(
        tmp_path,
        [
            row([["harmlessness", "honesty"]], True, scenario_id="a"),
            row([["honesty", "harmlessness"]], True, scenario_id="b"),
        ],
    )
    kept, generated = kept_pair_counts_from_csv(path, HHH)
    assert kept == {HARM_HON: 2}
    assert generated == {HARM_HON: 2}


def test_keep_scenario_accepts_bool_and_text(tmp_path):
    path = write_csv(
        tmp_path,
        [
            row([["harmlessness", "honesty"]], True, scenario_id="a"),
            row([["harmlessness", "honesty"]], "True", scenario_id="b"),
            row([["harmlessness", "honesty"]], "true", scenario_id="c"),
            row([["harmlessness", "honesty"]], False, scenario_id="d"),
            row([["harmlessness", "honesty"]], "False", scenario_id="e"),
        ],
    )
    kept, generated = kept_pair_counts_from_csv(path, HHH)
    assert kept == {HARM_HON: 3}
    assert generated == {HARM_HON: 5}


def test_rejected_rows_count_toward_generated_only(tmp_path):
    path = write_csv(tmp_path, [row([["helpfulness", "honesty"]], False)])
    kept, generated = kept_pair_counts_from_csv(path, HHH)
    assert kept == {}
    assert generated == {HELP_HON: 1}


def test_multi_seat_game_credits_every_seat(tmp_path):
    """Quotas are per-seat marginals, so differing seats credit both pairs."""
    path = write_csv(
        tmp_path,
        [row([["harmlessness", "honesty"], ["helpfulness", "honesty"]], True)],
    )
    kept, generated = kept_pair_counts_from_csv(path, HHH)
    assert kept == {HARM_HON: 1, HELP_HON: 1}


def test_same_pair_on_both_seats_counts_twice(tmp_path):
    path = write_csv(
        tmp_path,
        [row([["harmlessness", "honesty"], ["honesty", "harmlessness"]], True)],
    )
    kept, _ = kept_pair_counts_from_csv(path, HHH)
    assert kept == {HARM_HON: 2}


def test_missing_value_tags_falls_back_to_value_columns(tmp_path):
    path = write_csv(
        tmp_path,
        [
            {
                "scenario_id": "a",
                "value1": "harmlessness",
                "value2": "honesty",
                "keep_scenario": True,
                "value_tags": None,
            }
        ],
    )
    kept, _ = kept_pair_counts_from_csv(path, HHH)
    assert kept == {HARM_HON: 1}


def test_unparseable_value_tags_falls_back_to_value_columns(tmp_path):
    path = write_csv(
        tmp_path,
        [row([["harmlessness", "honesty"]], True, value_tags="not json")],
    )
    kept, _ = kept_pair_counts_from_csv(path, HHH)
    assert kept == {HARM_HON: 1}


def test_rows_without_usable_tags_are_skipped(tmp_path):
    path = write_csv(
        tmp_path,
        [
            {"scenario_id": "a", "value1": "", "value2": "", "keep_scenario": True,
             "value_tags": None},
            row([["harmlessness", "honesty"]], True, scenario_id="b"),
        ],
    )
    kept, generated = kept_pair_counts_from_csv(path, HHH)
    assert kept == {HARM_HON: 1}
    assert generated == {HARM_HON: 1}


def test_single_value_menu_is_not_tracked(tmp_path):
    path = write_csv(tmp_path, [row([["honesty", "honesty"]], True)])
    kept, generated = kept_pair_counts_from_csv(path, HHH)
    assert kept == {}
    assert generated == {}


def test_foreign_pairs_are_excluded_when_a_value_set_is_given(tmp_path):
    """A CSV from another value set must not pollute the quotas."""
    path = write_csv(
        tmp_path,
        [
            row([["privacy", "creativity"]], True, scenario_id="a"),
            row([["harmlessness", "honesty"]], True, scenario_id="b"),
        ],
    )
    kept, _ = kept_pair_counts_from_csv(path, HHH)
    assert kept == {HARM_HON: 1}


def test_without_a_value_set_every_pair_is_counted(tmp_path):
    path = write_csv(tmp_path, [row([["privacy", "creativity"]], True)])
    kept, _ = kept_pair_counts_from_csv(path, None)
    assert kept == {("creativity", "privacy"): 1}


def test_embedded_newlines_do_not_corrupt_counts(tmp_path):
    """These CSVs quote multi-line descriptions; line counting would miscount."""
    path = write_csv(
        tmp_path,
        [
            row(
                [["harmlessness", "honesty"]],
                True,
                scenario_id="a",
                description="line one\nline two\nline three",
            ),
            row([["helpfulness", "honesty"]], True, scenario_id="b", description="x"),
        ],
    )
    kept, _ = kept_pair_counts_from_csv(path, HHH)
    assert kept == {HARM_HON: 1, HELP_HON: 1}


@pytest.mark.parametrize("target,expected_deficit", [(1, 0), (3, 2)])
def test_counts_feed_the_tracker(tmp_path, target, expected_deficit):
    from conflictxgame.games.pair_quota import PairQuotaTracker, all_value_pairs

    path = write_csv(tmp_path, [row([["harmlessness", "honesty"]], True)])
    kept, generated = kept_pair_counts_from_csv(path, HHH)
    tracker = PairQuotaTracker(all_value_pairs(HHH), target, kept=kept, generated=generated)
    assert tracker.deficits()[HARM_HON] == expected_deficit
