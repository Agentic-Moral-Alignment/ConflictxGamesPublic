"""Unit tests for per-value-pair quota tracking and deficit-aware sampling."""

import random

import pytest

from conflictxgame.games.game_structures import GameConfig
from conflictxgame.games.pair_quota import (
    PairQuotaTracker,
    PAIR_RATE_PRIOR_WEIGHT,
    all_value_pairs,
    canonical_pair,
    make_deficit_sampler,
    pair_label,
)

HHH = ["harmlessness", "helpfulness", "honesty"]


def config(arities=(2, 2)):
    """Build a simultaneous config with the given per-seat arities."""
    return GameConfig(
        structure="simultaneous",
        n_players=len(arities),
        arities=list(arities),
        horizon=1,
    )


# -- canonicalization --------------------------------------------------- #

def test_canonical_pair_is_order_insensitive():
    assert canonical_pair(["honesty", "harmlessness"]) == ("harmlessness", "honesty")
    assert canonical_pair(["harmlessness", "honesty"]) == ("harmlessness", "honesty")


def test_canonical_pair_needs_exactly_two_distinct_tags():
    assert canonical_pair(["honesty"]) is None
    assert canonical_pair(["honesty", "honesty"]) is None
    assert canonical_pair(["honesty", "harmlessness", "helpfulness"]) is None
    assert canonical_pair([]) is None


def test_canonical_pair_ignores_empty_tags():
    assert canonical_pair(["honesty", "", "harmlessness"]) == ("harmlessness", "honesty")


def test_canonical_pair_collapses_repeats_in_wider_menus():
    """A 3-slot menu spanning only two values is still a tracked pair."""
    assert canonical_pair(["honesty", "harmlessness", "honesty"]) == ("harmlessness", "honesty")


def test_all_value_pairs_covers_every_combination():
    pairs = all_value_pairs(HHH)
    assert len(pairs) == 3
    assert ("harmlessness", "helpfulness") in pairs
    assert all(p[0] < p[1] for p in pairs)
    # Value-set agnostic: n values -> n*(n-1)/2 pairs.
    assert len(all_value_pairs([f"v{i}" for i in range(6)])) == 15


def test_pair_label_is_stable():
    assert pair_label(("harmlessness", "honesty")) == "harmlessness|honesty"


# -- tracker accounting ------------------------------------------------- #

def test_deficits_and_completion():
    tracker = PairQuotaTracker(all_value_pairs(HHH), target=2)
    assert tracker.total_deficit() == 6
    assert not tracker.is_complete()
    for pair in all_value_pairs(HHH):
        tracker.record_kept(pair)
        tracker.record_kept(pair)
    assert tracker.total_deficit() == 0
    assert tracker.is_complete()


def test_deficit_never_goes_negative():
    pair = ("harmlessness", "honesty")
    tracker = PairQuotaTracker([pair], target=1)
    tracker.record_kept(pair)
    tracker.record_kept(pair)
    assert tracker.deficits()[pair] == 0


def test_seeded_counts_are_applied():
    pairs = all_value_pairs(HHH)
    tracker = PairQuotaTracker(
        pairs, target=5, kept={pairs[0]: 3}, generated={pairs[0]: 9}
    )
    assert tracker.deficits()[pairs[0]] == 2
    assert tracker.generated[pairs[0]] == 9


def test_unknown_pairs_are_ignored():
    """Counts for a pair outside the tracked set must not create entries."""
    tracker = PairQuotaTracker(all_value_pairs(HHH), target=1)
    tracker.record_kept(("bogus", "pair"))
    tracker.record_generated(("bogus", "pair"))
    tracker.record_kept(None)
    assert sum(tracker.kept.values()) == 0
    assert sum(tracker.generated.values()) == 0


def test_pass_rate_uses_prior_then_observations():
    pair = ("harmlessness", "honesty")
    tracker = PairQuotaTracker([pair], target=10)
    assert tracker.pass_rate() == pytest.approx(0.25)  # prior
    for _ in range(4):
        tracker.record_generated(pair)
    tracker.record_kept(pair)
    tracker.record_kept(pair)
    assert tracker.pass_rate() == pytest.approx(0.5)


def test_pass_rate_is_clamped_to_floor():
    pair = ("harmlessness", "honesty")
    tracker = PairQuotaTracker([pair], target=10)
    for _ in range(100):
        tracker.record_generated(pair)
    assert tracker.pass_rate() == pytest.approx(0.1)


def test_dropped_attempt_releases_pending_without_going_negative():
    pair = ('harmlessness', 'honesty')
    tracker = PairQuotaTracker([pair], target=10)
    tracker.record_pending(pair)
    tracker.record_dropped(pair)
    tracker.record_dropped(pair)
    row = tracker.summary_rows()[0]
    assert row['pending'] == 0
    assert row['dropped'] == 2


def test_generated_attempt_releases_pending():
    pair = ('harmlessness', 'honesty')
    tracker = PairQuotaTracker([pair], target=10)
    tracker.record_pending(pair)
    tracker.record_generated(pair)
    assert tracker.summary_rows()[0]['pending'] == 0


def test_pair_pass_rate_shrinks_toward_global_yield():
    first, second = all_value_pairs(HHH)[:2]
    tracker = PairQuotaTracker([first, second], target=20)
    for pair, kept in ((first, 10), (second, 0)):
        for _ in range(10):
            tracker.record_generated(pair)
        for _ in range(kept):
            tracker.record_kept(pair)
    assert PAIR_RATE_PRIOR_WEIGHT == 10
    assert tracker.pass_rate() == pytest.approx(0.5)
    assert tracker.pass_rate(first) == pytest.approx(0.75)
    assert tracker.pass_rate(second) == pytest.approx(0.25)


def test_revision_salvage_rate():
    pair = ("harmlessness", "honesty")
    tracker = PairQuotaTracker([pair], target=10)
    tracker.record_kept(pair)
    tracker.record_kept(pair, revised=True)
    assert tracker.revision_salvage_rate() == pytest.approx(0.5)


# -- deficit-aware selection -------------------------------------------- #

def test_next_pair_prefers_the_largest_deficit():
    pairs = all_value_pairs(HHH)
    behind = pairs[2]
    tracker = PairQuotaTracker(pairs, target=5, kept={pairs[0]: 5, pairs[1]: 4})
    assert tracker.next_pair(random.Random(0)) == behind


def test_next_pair_never_picks_a_completed_pair_while_others_lag():
    pairs = all_value_pairs(HHH)
    done = pairs[0]
    tracker = PairQuotaTracker(pairs, target=3, kept={done: 3})
    rng = random.Random(1)
    assert all(tracker.next_pair(rng) != done for _ in range(25))


def test_pending_discount_never_reactivates_a_completed_pair():
    pairs = all_value_pairs(HHH)
    done = pairs[0]
    tracker = PairQuotaTracker(pairs, target=3, kept={done: 3})
    tracker.begin_round()
    for pair in pairs[1:]:
        for _ in range(100):
            tracker.record_pending(pair)
    assert tracker.next_pair(random.Random(0), pass_rate=1.0) != done


def test_next_pair_is_deterministic_for_a_fixed_seed():
    pairs = all_value_pairs(HHH)
    first = PairQuotaTracker(pairs, target=5)
    second = PairQuotaTracker(pairs, target=5)
    a = [first.next_pair(random.Random(7)) for _ in range(10)]
    b = [second.next_pair(random.Random(7)) for _ in range(10)]
    assert a == b


def test_pending_discount_spreads_a_batch_across_pairs():
    """Without the pending discount a whole round would pile onto one pair."""
    pairs = all_value_pairs(HHH)
    tracker = PairQuotaTracker(pairs, target=10)
    tracker.begin_round()
    rng = random.Random(0)
    drawn = []
    for _ in range(9):
        pair = tracker.next_pair(rng, pass_rate=1.0)
        tracker.record_pending(pair)
        drawn.append(pair)
    assert set(drawn) == set(pairs)
    assert all(drawn.count(p) == 3 for p in pairs)


def test_next_pair_discounts_pending_by_its_own_yield():
    first, second = all_value_pairs(HHH)[:2]
    tracker = PairQuotaTracker([first, second], target=10)
    for pair in (first, second):
        tracker.record_generated(pair)
        tracker.record_kept(pair)
    for _ in range(9):
        tracker.record_dropped(second)
    for _ in range(10):
        tracker.record_pending(first)
        tracker.record_pending(second)
    assert tracker.pass_rate(first) > tracker.pass_rate(second)
    assert tracker.next_pair(random.Random(0)) == second


def test_begin_round_clears_pending():
    pairs = all_value_pairs(HHH)
    tracker = PairQuotaTracker(pairs, target=10)
    tracker.record_pending(pairs[0])
    tracker.begin_round()
    rng = random.Random(0)
    # With pending cleared, every pair is equally behind again.
    assert tracker.next_pair(rng, pass_rate=1.0) in pairs


def test_next_pair_falls_back_when_everything_is_complete():
    pairs = all_value_pairs(HHH)
    tracker = PairQuotaTracker(pairs, target=1, kept={p: 1 for p in pairs})
    assert tracker.next_pair(random.Random(0)) in pairs


def test_next_pair_requires_pairs():
    with pytest.raises(ValueError):
        PairQuotaTracker([], target=1).next_pair(random.Random(0))


# -- sampler ------------------------------------------------------------ #

def value_dict():
    return {name: f"definition of {name}" for name in HHH}


def test_sampler_shape_and_tracked_pairs():
    tracker = PairQuotaTracker(all_value_pairs(HHH), target=10)
    sampler = make_deficit_sampler(tracker)
    assignment = sampler(config((2, 2)), value_dict(), random.Random(0))
    assert len(assignment) == 2
    for seat_values in assignment:
        assert len(seat_values) == 2
        assert canonical_pair(seat_values) in tracker.pairs


def test_sampler_records_pending_per_seat():
    tracker = PairQuotaTracker(all_value_pairs(HHH), target=10)
    tracker.begin_round()
    sampler = make_deficit_sampler(tracker)
    sampler(config((2, 2)), value_dict(), random.Random(0))
    assert sum(tracker._pending.values()) == 2


def test_sampler_shuffles_slot_order():
    """Canonical ordering must not leak into the prompt as a position bias."""
    tracker = PairQuotaTracker(all_value_pairs(HHH), target=1000)
    sampler = make_deficit_sampler(tracker)
    rng = random.Random(0)
    seen = set()
    for _ in range(50):
        for seat_values in sampler(config((2, 2)), value_dict(), rng):
            seen.add(tuple(seat_values))
    # Both orderings of at least one pair were emitted.
    assert any((b, a) in seen for (a, b) in seen)


def test_sampler_falls_back_to_uniform_for_non_binary_menus():
    tracker = PairQuotaTracker(all_value_pairs(HHH), target=10)
    tracker.begin_round()
    sampler = make_deficit_sampler(tracker)
    assignment = sampler(config((3, 2)), value_dict(), random.Random(0))
    assert len(assignment[0]) == 3
    assert len(assignment[1]) == 2
    # Only the arity-2 seat is quota-tracked.
    assert sum(tracker._pending.values()) == 1


def test_sampler_drives_generation_toward_the_lagging_pair():
    pairs = all_value_pairs(HHH)
    behind = pairs[1]
    tracker = PairQuotaTracker(
        pairs, target=5, kept={pairs[0]: 5, pairs[2]: 5}
    )
    tracker.begin_round()
    sampler = make_deficit_sampler(tracker)
    rng = random.Random(0)
    assignment = sampler(config((2, 2)), value_dict(), rng)
    assert all(canonical_pair(seat) == behind for seat in assignment)


def test_summary_rows_expose_progress():
    pairs = all_value_pairs(HHH)
    tracker = PairQuotaTracker(pairs, target=4, kept={pairs[0]: 1}, generated={pairs[0]: 4})
    rows = {row["pair"]: row for row in tracker.summary_rows()}
    row = rows[pair_label(pairs[0])]
    assert row["kept"] == 1
    assert row["deficit"] == 3
    assert row["acceptance_rate"] == pytest.approx(0.25)
    assert {'pending', 'dropped', 'yield_rate'} <= set(row)
    assert len(rows) == 3
