"""Per-value-pair quota tracking for accepted (kept-after-filter) scenarios.

The generation CLIs sample value tags uniformly at random, so the *accepted*
corpus that survives filtering is multinomial noise over value pairs: pairs
whose conflicts are harder to write pass the judge less often and end up
under-represented. Faithful comparison against the upstream ConflictScope
corpora needs a corpus whose per-pair accepted counts are known and even, so
this module provides the accounting that
:mod:`conflictxgame.games.generate_balanced` uses to generate until every pair
reaches a target count of kept scenarios.

Three concerns live here:

* **Canonicalization.** ``value1``/``value2`` in the filtered CSVs are seat-0
  action tags in *menu order*, so the same conflict appears as both
  ``(harmlessness, honesty)`` and ``(honesty, harmlessness)``.
  :func:`canonical_pair` is the single place that sorts a seat's distinct tags
  into one key, used by the tracker, the CSV recount, and the analysis-time
  reweighting.
* **Deficit-aware sampling.** :func:`make_deficit_sampler` is a drop-in
  replacement for
  :func:`conflictxgame.games.generate_game_scenarios.sample_value_assignment`
  that steers each arity-2 seat toward whichever pair is furthest from its
  quota, instead of drawing uniformly.
* **Resume.** :func:`kept_pair_counts_from_csv` recomputes the accepted counts
  from the filtered CSV, which is therefore the only state a resumed run needs.

Quotas are tracked over per-seat pair *marginals*: a kept game contributes one
count to each seat's canonical pair, so a two-player game whose seats carry
different conflicts credits both. Seats with an arity other than 2, or whose
menu carries fewer or more than two distinct tags, fall back to uniform
sampling and are excluded from quota accounting.

New module for this fork. See ``ATTRIBUTION.md``.
"""

import itertools
import json
import logging
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: A canonical value pair: the two distinct value names, sorted.
Pair = Tuple[str, str]

#: Assumed acceptance rate before a run has observed any filtering outcome.
#: Only used to discount in-flight (pending) scenarios while sampling.
DEFAULT_PASS_RATE_PRIOR = 0.25

#: Bounds applied to the observed acceptance rate. The floor keeps a bad early
#: streak from making the pending discount vanish (which would pile a whole
#: round onto one pair); the ceiling is the trivial maximum.
MIN_PASS_RATE = 0.1
MAX_PASS_RATE = 1.0

#: Pseudo-attempts contributed by the global yield when estimating a pair
#: with sparse history. Ten keeps early per-pair rates stable without hiding
#: sustained pair-specific differences.
PAIR_RATE_PRIOR_WEIGHT = 10


def canonical_pair(tags: Iterable[str]) -> Optional[Pair]:
    """Reduce one seat's action tags to a canonical, order-insensitive pair.

    Args:
        tags: The value tags of a seat's action menu, in menu order.

    Returns:
        The two distinct tags as a sorted tuple, or ``None`` when the menu does
        not carry exactly two distinct tags (a degenerate single-value menu, or
        an arity>2 menu spanning three or more values). ``None`` means "not
        quota-tracked" rather than "invalid".
    """
    distinct = {str(tag) for tag in tags if str(tag)}
    if len(distinct) != 2:
        return None
    first, second = sorted(distinct)
    return (first, second)


def all_value_pairs(value_names: Iterable[str]) -> List[Pair]:
    """Enumerate every unordered pair of a value set, in canonical order.

    Args:
        value_names: The value names of a value set.

    Returns:
        Sorted 2-tuples for all ``C(n, 2)`` pairs (3 for HHH, 15 for
        ModelSpec, 28 for the 8-value personal-protective set).
    """
    return list(itertools.combinations(sorted(set(value_names)), 2))


def pair_label(pair: Pair) -> str:
    """Render a canonical pair as a stable string key for reports and JSON."""
    return f"{pair[0]}|{pair[1]}"


class PairQuotaTracker:
    """Tracks accepted-scenario counts per value pair against a target.

    Counts are per-seat marginals: :meth:`record_generated` or
    :meth:`record_dropped` records each attempted seat, and
    :meth:`record_kept` records seats accepted by filtering.
    ``pending`` holds seats assigned during the current round whose filtering
    outcome is not known yet; :meth:`next_pair` discounts them by the observed
    acceptance rate so one round's batch spreads across pairs instead of
    piling onto whichever pair happened to start furthest behind.

    Attributes:
        target: Kept-scenario target per pair.
        pairs: Every pair being tracked (fixed at construction).
    """

    def __init__(
        self,
        pairs: Sequence[Pair],
        target: int,
        kept: Optional[Dict[Pair, int]] = None,
        generated: Optional[Dict[Pair, int]] = None,
    ):
        """Initialize a tracker, optionally seeded from an existing corpus.

        Args:
            pairs: The pairs to track (typically :func:`all_value_pairs`).
            target: Kept-scenario target per pair.
            kept: Pre-existing kept counts, e.g. from
                :func:`kept_pair_counts_from_csv` when resuming.
            generated: Pre-existing generated counts, same source.
        """
        self.target = int(target)
        self.pairs: List[Pair] = list(pairs)
        self._kept: Dict[Pair, int] = {p: 0 for p in self.pairs}
        self._generated: Dict[Pair, int] = {p: 0 for p in self.pairs}
        self._pending: Dict[Pair, int] = {p: 0 for p in self.pairs}
        self._dropped: Dict[Pair, int] = {p: 0 for p in self.pairs}
        #: Kept scenarios that only passed after a revision round.
        self._revised_kept: Dict[Pair, int] = {p: 0 for p in self.pairs}
        for source, sink in ((kept, self._kept), (generated, self._generated)):
            for pair, count in (source or {}).items():
                if pair in sink:
                    sink[pair] += int(count)

    # -- accounting ----------------------------------------------------- #

    def record_generated(self, pair: Optional[Pair]) -> None:
        """Count one generated-and-filtered seat toward a pair."""
        if pair in self._generated:
            self._generated[pair] += 1
            self._pending[pair] = max(0, self._pending[pair] - 1)

    def record_dropped(self, pair: Optional[Pair]) -> None:
        """Release one pending seat dropped before filtering and count its attempt."""
        if pair in self._dropped:
            self._dropped[pair] += 1
            self._pending[pair] = max(0, self._pending[pair] - 1)

    def record_kept(self, pair: Optional[Pair], revised: bool = False) -> None:
        """Count one accepted seat toward a pair.

        Args:
            pair: The seat's canonical pair; ``None`` is ignored.
            revised: True when the scenario only passed after a revision pass,
                which is reported separately as the revision salvage rate.
        """
        if pair in self._kept:
            self._kept[pair] += 1
            if revised:
                self._revised_kept[pair] += 1

    def begin_round(self) -> None:
        """Clear the in-flight assignments at the start of a generation round."""
        self._pending = {p: 0 for p in self.pairs}

    def record_pending(self, pair: Optional[Pair]) -> None:
        """Count one seat assigned in this round but not yet filtered."""
        if pair in self._pending:
            self._pending[pair] += 1

    # -- state ---------------------------------------------------------- #

    @property
    def kept(self) -> Dict[Pair, int]:
        """Kept counts per pair (copy)."""
        return dict(self._kept)

    @property
    def generated(self) -> Dict[Pair, int]:
        """Generated counts per pair (copy)."""
        return dict(self._generated)

    def deficits(self) -> Dict[Pair, int]:
        """Remaining kept scenarios needed per pair (never negative)."""
        return {p: max(0, self.target - self._kept[p]) for p in self.pairs}

    def total_deficit(self) -> int:
        """Total remaining kept scenarios needed across all pairs."""
        return sum(self.deficits().values())

    def is_complete(self) -> bool:
        """True when every tracked pair has reached the target."""
        return self.total_deficit() == 0

    def pass_rate(self, pair: Optional[Pair] = None) -> float:
        """Observed acceptance rate so far, clamped to a usable range.

        With no pair, returns the global ``kept / (generated + dropped)`` rate.
        A pair-specific request shrinks that pair's rate toward the global rate
        by :data:`PAIR_RATE_PRIOR_WEIGHT`.  Results are clamped to
        ``[MIN_PASS_RATE, MAX_PASS_RATE]``; before any attempt the global rate
        is :data:`DEFAULT_PASS_RATE_PRIOR`.
        """
        attempts = sum(self._generated.values()) + sum(self._dropped.values())
        if attempts <= 0:
            global_rate = DEFAULT_PASS_RATE_PRIOR
        else:
            global_rate = sum(self._kept.values()) / attempts
            global_rate = min(MAX_PASS_RATE, max(MIN_PASS_RATE, global_rate))
        if pair is None or pair not in self._generated:
            return global_rate
        pair_attempts = self._generated[pair] + self._dropped[pair]
        rate = (
            self._kept[pair] + PAIR_RATE_PRIOR_WEIGHT * global_rate
        ) / (pair_attempts + PAIR_RATE_PRIOR_WEIGHT)
        return min(MAX_PASS_RATE, max(MIN_PASS_RATE, rate))

    def revision_salvage_rate(self) -> float:
        """Fraction of kept scenarios that only passed after revision."""
        kept = sum(self._kept.values())
        if kept <= 0:
            return 0.0
        return sum(self._revised_kept.values()) / kept

    # -- sampling ------------------------------------------------------- #

    def next_pair(self, rng: random.Random, pass_rate: Optional[float] = None) -> Pair:
        """Pick the pair most in need of another scenario.

        Deterministic max-deficit-first: the effective deficit of a pair is its
        remaining deficit minus its in-flight assignments discounted by the
        acceptance rate, so a pair already covered by this round's batch stops
        being picked. Ties are broken with ``rng`` so a fresh run does not
        always favor the alphabetically first pair.

        Args:
            rng: Seeded RNG, shared with the rest of generation.
            pass_rate: Acceptance rate for the pending discount; defaults to
                :meth:`pass_rate`.

        Returns:
            The chosen canonical pair. Falls back to a uniform draw when every
            pair has met its target (callers normally stop before that).
        """
        if not self.pairs:
            raise ValueError("PairQuotaTracker has no pairs to sample from")
        deficits = self.deficits()
        if not any(deficits.values()):
            return rng.choice(self.pairs)
        # A completed pair is never useful while any positive deficit remains.
        # Without this restriction, enough in-flight assignments can make an
        # unfinished pair's discounted deficit negative, allowing a completed
        # pair (effective deficit zero) to win and creating exactly the quota
        # overshoot this sampler is meant to prevent.
        active_pairs = [p for p in self.pairs if deficits[p] > 0]
        effective = {
            p: deficits[p] - self._pending[p] * (
                self.pass_rate(p) if pass_rate is None else pass_rate
            )
            for p in active_pairs
        }
        best = max(effective.values())
        leaders = [p for p in active_pairs if effective[p] == best]
        return rng.choice(leaders) if len(leaders) > 1 else leaders[0]

    # -- reporting ------------------------------------------------------ #

    def summary_rows(self) -> List[dict]:
        """Per-pair rows for the pair coverage report.

        Returns:
            One dict per pair with ``pair``, ``value1``, ``value2``,
            ``target``, ``generated``, ``pending``, ``dropped``, ``kept``,
            ``deficit``, ``acceptance_rate`` (kept/generated), ``yield_rate``
            (kept over all attempts), and ``revised_kept``.
        """
        deficits = self.deficits()
        rows = []
        for pair in self.pairs:
            generated = self._generated[pair]
            dropped = self._dropped[pair]
            attempts = generated + dropped
            rows.append(
                {
                    "pair": pair_label(pair),
                    "value1": pair[0],
                    "value2": pair[1],
                    "target": self.target,
                    "generated": generated,
                    "pending": self._pending[pair],
                    "dropped": dropped,
                    "kept": self._kept[pair],
                    "deficit": deficits[pair],
                    "acceptance_rate": (self._kept[pair] / generated) if generated else 0.0,
                    "yield_rate": (self._kept[pair] / attempts) if attempts else 0.0,
                    "revised_kept": self._revised_kept[pair],
                }
            )
        return rows

    def summary(self) -> str:
        """Render a one-line-per-pair progress summary for the console."""
        lines = [f"Pair quota progress (target {self.target} kept per pair):"]
        for row in self.summary_rows():
            lines.append(
                f"  {row['pair']}: kept {row['kept']}/{row['target']} "
                f"(generated {row['generated']}, deficit {row['deficit']})"
            )
        lines.append(
            f"  overall acceptance rate {self.pass_rate():.2f}, "
            f"revision salvage {self.revision_salvage_rate():.2f}"
        )
        return "\n".join(lines)


def make_deficit_sampler(
    tracker: PairQuotaTracker,
    pass_rate_fn: Optional[Callable[[], float]] = None,
) -> Callable[[object, Dict[str, object], random.Random], List[List[str]]]:
    """Build a quota-aware replacement for ``sample_value_assignment``.

    The returned callable matches the signature that
    :func:`conflictxgame.games.generate_game_scenarios.generate_stage_one_games`
    expects, so it can be passed as its ``assignment_sampler``. Each arity-2
    seat is assigned the pair returned by :meth:`PairQuotaTracker.next_pair`,
    with the slot order shuffled so the canonical ordering never leaks into the
    prompt as a position bias. Seats of any other arity fall back to the
    original uniform draw and are not counted toward quotas.

    Args:
        tracker: The quota tracker to consult and update with pending counts.
        pass_rate_fn: Optional override for the pending discount rate;
            defaults to the tracker's observed rate.

    Returns:
        ``sampler(config, value_dict, rng) -> assignment[seat][slot]``.
    """
    from conflictxgame.games.generate_game_scenarios import sample_value_assignment

    def sampler(config, value_dict: Dict[str, object], rng: random.Random) -> List[List[str]]:
        uniform: Optional[List[List[str]]] = None
        assignment: List[List[str]] = []
        for seat, arity in enumerate(config.arities):
            if arity != 2:
                # Fall back to the uniform sampler for this seat, computing the
                # full uniform assignment once and reusing the matching seat.
                if uniform is None:
                    uniform = sample_value_assignment(config, value_dict, rng)
                assignment.append(list(uniform[seat]))
                continue
            pair = tracker.next_pair(
                rng,
                pass_rate=pass_rate_fn() if pass_rate_fn is not None else None,
            )
            tracker.record_pending(pair)
            seat_values = list(pair)
            rng.shuffle(seat_values)
            assignment.append(seat_values)
        return assignment

    return sampler


def _is_kept(value) -> bool:
    """Interpret a ``keep_scenario`` cell, which round-trips as bool or text."""
    return str(value).strip().lower() == "true"


def _row_seat_tags(row) -> List[List[str]]:
    """Recover per-seat value-tag lists from a filtered-CSV row.

    Prefers the ``value_tags`` JSON column (all seats); falls back to seat 0's
    ``value1``/``value2`` columns when it is missing or unparseable, which is
    also how upstream dyadic rows are handled.
    """
    raw = row.get("value_tags")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and parsed:
            seats = []
            for seat_tags in parsed:
                if isinstance(seat_tags, list):
                    seats.append([str(t) for t in seat_tags])
            if seats:
                return seats
    value1, value2 = row.get("value1"), row.get("value2")
    if isinstance(value1, str) and isinstance(value2, str) and value1.strip() and value2.strip():
        return [[value1, value2]]
    return []


def kept_pair_counts_from_csv(
    csv_path: str,
    value_names: Optional[Iterable[str]] = None,
) -> Tuple[Dict[Pair, int], Dict[Pair, int]]:
    """Recompute per-pair kept and generated counts from a filtered CSV.

    This is the resume path: the filtered CSV is the orchestrator's only
    state, so a run that is interrupted, or a corpus generated before quotas
    existed, is picked up simply by recounting. Rows are read with pandas
    because these CSVs embed newlines inside quoted fields — line-based
    counting silently miscounts them.

    Args:
        csv_path: Path to a ``{model}_games.csv`` (or any CSV sharing the
            filtered schema). A missing file yields empty counts.
        value_names: When given, pairs outside this value set are dropped with
            a warning, so a stale CSV from another value set cannot pollute
            the quotas.

    Returns:
        ``(kept, generated)`` mappings from canonical pair to seat count.
    """
    import os

    import pandas as pd

    kept: Dict[Pair, int] = {}
    generated: Dict[Pair, int] = {}
    if not os.path.exists(csv_path):
        return kept, generated

    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        # A zero-byte file, e.g. a write interrupted before the header landed.
        logger.warning("Corpus CSV %s is empty; starting from zero counts", csv_path)
        return kept, generated
    if frame.empty:
        return kept, generated

    allowed = set(all_value_pairs(value_names)) if value_names is not None else None
    skipped_rows = 0
    foreign_pairs = set()
    for _, row in frame.iterrows():
        seats = _row_seat_tags(row)
        if not seats:
            skipped_rows += 1
            continue
        row_kept = _is_kept(row.get("keep_scenario"))
        for seat_tags in seats:
            pair = canonical_pair(seat_tags)
            if pair is None:
                continue
            if allowed is not None and pair not in allowed:
                foreign_pairs.add(pair)
                continue
            generated[pair] = generated.get(pair, 0) + 1
            if row_kept:
                kept[pair] = kept.get(pair, 0) + 1

    if skipped_rows:
        logger.warning(
            "%d row(s) in %s had no usable value tags and were skipped",
            skipped_rows,
            csv_path,
        )
    if foreign_pairs:
        logger.warning(
            "Ignored %d value pair(s) in %s outside the value set: %s",
            len(foreign_pairs),
            csv_path,
            ", ".join(sorted(pair_label(p) for p in foreign_pairs)),
        )
    return kept, generated
