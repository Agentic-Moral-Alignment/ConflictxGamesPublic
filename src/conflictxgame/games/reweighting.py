"""Reweight value comparisons to match a reference corpus's pair distribution.

:mod:`conflictxgame.games.generate_balanced` produces a corpus that is uniform
across value pairs, which is the right basis for measuring a model's value
ordering: every conflict gets equal say. The upstream ConflictScope corpora are
*not* uniform — their accepted scenarios survive filtering at very different
rates per pair (in HHH, 576 harmlessness-helpfulness against 227
helpfulness-honesty) — so a ranking fitted on our corpus and a ranking fitted
on theirs weight the same conflicts differently, and any difference between the
two rankings confounds "the model behaves differently in games" with "the
corpora emphasize different conflicts".

This module removes that confound without spending anything: the evaluation
runs once over the uniform corpus, and the *comparisons it produced* are
resampled here so their per-pair mix matches the reference corpus. One analysis
run therefore yields both readings — uniform and paper-matched — and the
difference between them is attributable to composition alone.

Resampling is **downsampling without replacement**, deterministic under a seed.
Upsampling by duplication would be cheaper in data but would fabricate
certainty: the bootstrap in
:func:`conflictxgame.games.analyze_games.fit_bradley_terry` would treat each
duplicate as an independent observation and shrink the confidence intervals
accordingly. Discarding instead means the intervals honestly widen to reflect
the smaller effective sample.

New module for this fork. See ``ATTRIBUTION.md``.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from conflictxgame.games.pair_quota import Pair, canonical_pair, pair_label

logger = logging.getLogger(__name__)


class ReweightingError(ValueError):
    """Raised when a requested pair distribution cannot be matched."""


def reference_pair_distribution(
    reference_csv: str,
    value_names: Optional[Sequence[str]] = None,
) -> Dict[Pair, float]:
    """Read the accepted per-pair proportions of a reference corpus.

    Args:
        reference_csv: A filtered scenario CSV, typically an upstream
            ConflictScope corpus such as
            ``data/HHH/claude-3-5-sonnet-latest.csv``.
        value_names: Optional value set; pairs outside it are ignored.

    Returns:
        Mapping from canonical pair to its share of the corpus's accepted
        scenarios (summing to 1).

    Raises:
        ReweightingError: If the corpus has no accepted scenarios with a
            usable value pair.
    """
    from conflictxgame.games.pair_quota import kept_pair_counts_from_csv

    kept, _ = kept_pair_counts_from_csv(reference_csv, value_names)
    total = sum(kept.values())
    if total <= 0:
        raise ReweightingError(
            f"Reference corpus {reference_csv} has no accepted scenarios with a value pair"
        )
    return {pair: count / total for pair, count in kept.items()}


def uniform_pair_weights(pairs: Sequence[Pair]) -> Dict[Pair, float]:
    """Equal weight for every pair present."""
    distinct = sorted(set(pairs))
    if not distinct:
        return {}
    share = 1.0 / len(distinct)
    return {pair: share for pair in distinct}


def _group_by_pair(items: Sequence, key) -> Dict[Pair, List[int]]:
    """Bucket item indices by the canonical pair ``key`` extracts."""
    groups: Dict[Pair, List[int]] = {}
    for i, item in enumerate(items):
        pair = key(item)
        if pair is None:
            continue
        groups.setdefault(pair, []).append(i)
    return groups


def _target_counts(
    groups: Dict[Pair, List[int]],
    weights: Dict[Pair, float],
    strict: bool,
) -> Dict[Pair, int]:
    """Compute the per-pair keep counts that realize ``weights``.

    The largest sample matching the target proportions is bounded by whichever
    pair is scarcest relative to its target share, so that pair is kept whole
    and the rest are trimmed down around it.

    Raises:
        ReweightingError: In strict mode, when a pair the reference weights
            positively has no comparisons at all — no amount of trimming can
            manufacture the missing evidence.
    """
    positive = {p: w for p, w in weights.items() if w > 0}
    if not positive:
        raise ReweightingError("No value pair carries positive weight")

    missing = [p for p in positive if not groups.get(p)]
    if missing:
        names = ", ".join(sorted(pair_label(p) for p in missing))
        if strict:
            raise ReweightingError(
                f"Cannot match the reference distribution: no comparisons for {names}. "
                f"Generate scenarios for those pairs, or analyze with --pair-weights none."
            )
        logger.warning("Dropping pair(s) with no comparisons: %s", names)
        positive = {p: w for p, w in positive.items() if groups.get(p)}
        if not positive:
            # Every weighted pair is missing: there is nothing to match against.
            raise ReweightingError(
                f"None of the weighted value pairs ({names}) appear in the data; "
                f"nothing to reweight."
            )
        total = sum(positive.values())
        positive = {p: w / total for p, w in positive.items()}

    scale = min(len(groups[p]) / w for p, w in positive.items())
    # Nudge before truncating: float round-trip can put ``scale * w`` a hair
    # under the count of the pair that set the scale, silently trimming the
    # very pair that is supposed to be kept whole.
    return {p: min(len(groups[p]), int(scale * w + 1e-9)) for p, w in positive.items()}


def resample_comparisons(
    comparisons: Sequence[Tuple[str, str]],
    weights: Dict[Pair, float],
    seed: int = 0,
    strict: bool = True,
):
    """Downsample (winner, loser) comparisons to match a pair distribution.

    Each comparison is attributed to the canonical pair of the two values it
    compares, which is exact and works for menus of any arity: a comparison
    *is* a pair by construction.

    Args:
        comparisons: Comparisons from
            :func:`conflictxgame.games.analyze_games.collect_pairwise_value_wins`.
        weights: Target per-pair proportions, e.g. from
            :func:`reference_pair_distribution`.
        seed: Seed for the deterministic subsample.
        strict: Fail when a positively weighted pair has no comparisons,
            rather than renormalizing around it.

    Returns:
        ``(resampled_comparisons, summary)`` where ``summary`` is a DataFrame
        with one row per pair: ``pair``, ``value1``, ``value2``, ``weight``,
        ``n_before``, ``n_after``, ``share_before``, ``share_after``.

    Raises:
        ReweightingError: See :func:`_target_counts`.
    """
    import numpy as np
    import pandas as pd

    groups = _group_by_pair(comparisons, lambda c: canonical_pair(c))
    if not groups:
        return list(comparisons), pd.DataFrame()

    targets = _target_counts(groups, weights, strict)
    rng = np.random.default_rng(seed)

    keep_indices: List[int] = []
    rows = []
    total_before = sum(len(idx) for idx in groups.values())
    total_after = sum(targets.values())
    for pair in sorted(groups):
        available = groups[pair]
        target = targets.get(pair, 0)
        if target >= len(available):
            chosen = list(available)
        else:
            chosen = [available[i] for i in rng.choice(len(available), target, replace=False)]
        keep_indices.extend(chosen)
        rows.append(
            {
                "pair": pair_label(pair),
                "value1": pair[0],
                "value2": pair[1],
                "weight": weights.get(pair, 0.0),
                "n_before": len(available),
                "n_after": len(chosen),
                "share_before": len(available) / total_before if total_before else 0.0,
                "share_after": (len(chosen) / total_after) if total_after else 0.0,
            }
        )

    keep_indices.sort()  # preserve the original ordering
    resampled = [tuple(comparisons[i]) for i in keep_indices]
    logger.info(
        "Reweighted comparisons: %d -> %d across %d pair(s)",
        total_before, len(resampled), len(rows),
    )
    # A pair whose target rounds down to zero contributes no evidence at all,
    # and any value appearing only in such pairs drops out of the fit. That is
    # arithmetically what the requested distribution asks for, but it changes
    # which values the ranking covers, so say so rather than let it pass
    # silently.
    emptied = [
        row["pair"] for row in rows if row["n_after"] == 0 and row["n_before"] > 0
    ]
    if emptied:
        logger.warning(
            "Reweighting left no comparisons for %s; any value appearing only "
            "in those pairs will be missing from the reweighted ranking",
            ", ".join(sorted(emptied)),
        )
    return resampled, pd.DataFrame(rows)


def resample_rankings(
    rankings: Sequence[Sequence[str]],
    weights: Dict[Pair, float],
    seed: int = 0,
    strict: bool = False,
):
    """Downsample Likert value rankings to match a pair distribution.

    A ranking is attributable to a value pair only when it ranks exactly two
    distinct values — always the case for the 2x2 corpora these pipelines
    generate. Longer rankings span more than one pair, so they are passed
    through unweighted and reported in the log rather than silently dropped.

    Args:
        rankings: Rankings from
            :func:`conflictxgame.games.analyze_games.collect_likert_value_rankings`.
        weights: Target per-pair proportions.
        seed: Seed for the deterministic subsample.
        strict: Fail when a positively weighted pair has no rankings.

    Returns:
        ``(resampled_rankings, summary)``; see :func:`resample_comparisons`.
    """
    import pandas as pd

    pairwise = [r for r in rankings if canonical_pair(r) is not None]
    wider = [r for r in rankings if canonical_pair(r) is None]
    if wider:
        logger.warning(
            "%d of %d Likert ranking(s) span more than two values and were left "
            "unweighted", len(wider), len(rankings),
        )
    if not pairwise:
        return list(rankings), pd.DataFrame()

    as_tuples = [tuple(r) for r in pairwise]
    resampled, summary = resample_comparisons(
        as_tuples, weights, seed=seed, strict=strict
    )
    return [list(r) for r in resampled] + [list(r) for r in wider], summary


def build_pair_weights(
    mode: str,
    pairs: Sequence[Pair],
    reference_csv: Optional[str] = None,
    value_names: Optional[Sequence[str]] = None,
) -> Optional[Dict[Pair, float]]:
    """Resolve a ``--pair-weights`` choice into concrete weights.

    Args:
        mode: ``"none"`` (no reweighting), ``"uniform"`` (equalize the pairs
            present in the data), or ``"match"`` (match ``reference_csv``).
        pairs: Pairs present in the data, used by ``"uniform"``.
        reference_csv: Reference corpus, required for ``"match"``.
        value_names: Optional value set restricting the reference pairs.

    Returns:
        The weights, or ``None`` when ``mode`` is ``"none"``.

    Raises:
        ReweightingError: If ``"match"`` is requested with no reference CSV.
    """
    if mode == "none":
        return None
    if mode == "uniform":
        return uniform_pair_weights(pairs)
    if mode == "match":
        if not reference_csv:
            raise ReweightingError("--pair-weights match requires --reference-csv")
        return reference_pair_distribution(reference_csv, value_names)
    raise ReweightingError(f"Unknown pair-weight mode {mode!r}")
