"""The memory-state probes must be keyed by VALUE, not by action index.

The probe labels stored on disk are action-index profiles (``prev_0-1``), but
the index is not comparable across scenarios: in this corpus action 0 names
honesty in 98% of harmlessness/honesty seats and roughly half the time for the
other pairs. Tallying on the raw label therefore averages "both chose
helpfulness" together with "both chose harmlessness" -- opposite conditions --
and cancels the very effect the five-state design exists to measure. It did:
the published figure showed flat lines for every model and arm until the states
were recoded.
"""
import pytest

pytest.importorskip("matplotlib")

from conflictxgame.figures.state_effect_figure import state_key


ACTION_VALUES = {
    # Seat 0 lists harmlessness first, seat 1 lists it second. A raw index tally
    # reads "0-0" as one condition; by value it is one player each way.
    ("s1", 0): ["harmlessness", "honesty"],
    ("s1", 1): ["honesty", "harmlessness"],
    # Both seats list harmlessness first.
    ("s2", 0): ["harmlessness", "honesty"],
    ("s2", 1): ["harmlessness", "honesty"],
}
PAIR = ("harmlessness", "honesty")


def test_fresh_state_passes_through():
    assert state_key("fresh", "s1", PAIR, ACTION_VALUES) == "fresh"


def test_same_index_profile_means_different_things_in_different_scenarios():
    """The core of the bug: one label, two opposite conditions."""
    # s1: seat 0 action 0 is harmlessness, seat 1 action 0 is honesty -> one each.
    assert state_key("prev_0-0", "s1", PAIR, ACTION_VALUES) == "one"
    # s2: both seats' action 0 is harmlessness -> both chose it.
    assert state_key("prev_0-0", "s2", PAIR, ACTION_VALUES) == "both"


def test_counts_players_choosing_the_first_named_value():
    assert state_key("prev_1-1", "s2", PAIR, ACTION_VALUES) == "none"
    assert state_key("prev_0-1", "s2", PAIR, ACTION_VALUES) == "one"
    assert state_key("prev_1-0", "s2", PAIR, ACTION_VALUES) == "one"


def test_unknown_scenario_or_malformed_label_is_dropped_not_guessed():
    assert state_key("prev_0-0", "missing", PAIR, ACTION_VALUES) is None
    assert state_key("prev_9-0", "s2", PAIR, ACTION_VALUES) is None
    assert state_key("prev_x-0", "s2", PAIR, ACTION_VALUES) is None
    assert state_key("prev_0", "s2", PAIR, ACTION_VALUES) is None
    assert state_key("something_else", "s2", PAIR, ACTION_VALUES) is None
