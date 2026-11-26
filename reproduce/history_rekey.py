"""Re-key the fabricated-history arms by own versus counterpart prior action.

The paper's history result recodes each memory probe by *counting* how many of
the two remembered actions favoured the focal pair's first-named value
(``reanalysis.state_key``). That sums the model's own prior action with the
counterpart's, so copying the other player cannot be told apart from repeating
oneself or from a value merely being mentioned.

The data separate them. A probe ``prev_a0-a1`` records both seats' actions, and
every seat was evaluated under all four, so each seat is a full own x
counterpart 2x2. For the 45 seats whose counterpart faces the same value pair
{A, B} (y = chose A, p_oc = P(y | own=o, counterpart=c)):

``SELF``  = 1/2 [(p_AA - p_BA) + (p_AB - p_BB)]   repeat your own prior value
``COPY``  = 1/2 [(p_AA - p_AB) + (p_BA - p_BB)]   match the counterpart's value
``ASYM``  = SELF - COPY = p_AB - p_BA             only the roles swap

Any account in which the choice depends only on which values were mentioned --
including the count keying itself -- predicts ASYM = 0; modelling the other
player predicts ASYM < 0; self-consistency predicts ASYM > 0. SELF is also
defined on all 150 seats; on the 105 seats whose counterpart shares one value
``COPY_1ov`` is reported descriptively.

Every estimate is a per-seat contrast, pooled as the equal-weight mean over the
models with ``reanalysis.py``'s scenario-clustered bootstrap and sign-flip test.
Gates run first and stop the script on failure: action indices map to the tags
recorded in every file (G1); the menu prompt names the focal player and the
history note lists both players by name in seat order (G2); the count-keyed
history effects recomputed here equal ``stats.json`` exactly. G3 reports how
often free-text openings name the focal and the counterpart player, which
decides whether roles are identifiable in each free-text representation.

Usage (from the repository root)::

    $env:PYTHONPATH='src'
    python reproduce/history_rekey.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_paper():
    spec = importlib.util.spec_from_file_location("paper_reanalysis", os.path.join(HERE, "reanalysis.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = _load_paper()
from conflictxgame.common.arm_names import split_result_name  # noqa: E402

ROOT = paper.ROOT
REFERENCE_STATS = os.path.join(ROOT, "analysis", "reanalysis_20260904_01", "stats.json")
DEFAULT_OUTPUT_DIR = os.path.join(
    ROOT, "analysis", f"history_rekey_{_dt.date.today():%Y%m%d}_01"
)
#: Arm tokens appended for a role cue other than the original name-only text.
ROLE_CUE_SUFFIX = {"name": "", "explicit": "_rolecue-explicit"}
STAKE_CONDITIONS = ("baseline", *paper.STAKES)
VISIBILITY = ("observed", "unobserved")
PRIMARY_EFFECTS = ("copy", "asym")
#: Effects whose moderation is tested (SELF = COPY + ASYM, added 2026-10-19).
MODERATION_EFFECTS = ("self", "copy", "asym")
PROBES = ("prev_0-0", "prev_0-1", "prev_1-0", "prev_1-1")
#: G3 threshold: above this share of undisclosed openings naming the focal player,
#: the "roles unidentifiable in free text" reading is not supported.
ROLE_AMBIGUITY_MAX_SHARE = 0.25


# ---------------------------------------------------------------------- #
# Arms and roles
# ---------------------------------------------------------------------- #

def arm_token(representation: str, condition: str, role_cue: str = "name") -> Tuple[str, str]:
    """Return ``(mode, exact arm token)`` for a states arm under a role cue."""
    if condition in VISIBILITY:
        if representation != "menu":
            raise ValueError("Visibility history arms exist only for the menu representation")
        mode, token = "mcq", f"_states_{condition}"
    elif condition in STAKE_CONDITIONS:
        mode, token = paper.arm_token_for(representation, condition, states=True)
    else:
        # A stake added after the paper (e.g. the consistency stakes), named the
        # way the original stakes are in each representation.
        mode = paper.REPRESENTATIONS[representation]["mode"]
        token = {"menu": f"_states_stakes-{condition}",
                 "open_undisclosed": f"_states-stakes-{condition}",
                 "open_disclosed": f"_counterpart_states-stakes-{condition}"}[representation]
    return mode, token + ROLE_CUE_SUFFIX[role_cue]


def resolve_states_file(directory: str, model: str, representation: str, condition: str,
                        role_cue: str = "name", names: Optional[Sequence[str]] = None) -> str:
    """Resolve exactly one result file by exact arm token, never by glob or sort.

    ``_states-steer`` and ``_states_stakes-*`` sit beside ``_states`` in the
    same directory, so anything looser than token equality picks the wrong arm.
    """
    mode, wanted = arm_token(representation, condition, role_cue)
    names = list(names) if names is not None else os.listdir(paper._io_path(directory))
    hits = []
    for name in names:
        try:
            parts = split_result_name(name)
        except ValueError:
            continue
        if (parts["slug"] == model and parts["mode"] == mode
                and parts["balanced"] == "_balanced50" and parts["arm"] == wanted):
            hits.append(os.path.join(directory, name))
    if len(hits) != 1:
        raise ValueError(f"Expected one {mode} file with arm {wanted!r} for {model}; found {hits}")
    return hits[0]


def seat_roles(pairs: Mapping, actions: Mapping) -> Dict[Tuple[str, int], Tuple[str, Optional[str]]]:
    """Classify each seat as ``same`` (identical pair) or ``overlap`` (one shared value).

    Returns:
        ``{(scenario_id, seat): (kind, shared_value)}``; ``shared_value`` is
        ``None`` for same-pair seats.

    Raises:
        ValueError: A counterpart shares no value with the focal seat, so no
            copy contrast exists for it.
    """
    roles = {}
    for (scenario_id, seat), own in actions.items():
        other = actions.get((scenario_id, 1 - seat))
        if other is None or len(own) != 2 or len(other) != 2:
            raise ValueError(f"{scenario_id} seat {seat} is not a two-player 2x2 seat")
        shared = set(own) & set(other)
        if not shared:
            raise ValueError(f"{scenario_id} seat {seat}: counterpart shares no focal value")
        roles[(scenario_id, seat)] = ("same", None) if len(shared) == 2 else ("overlap", shared.pop())
    return roles


def probe_roles(probe: str, scenario_id: str, seat: int, actions: Mapping) -> Tuple[int, int]:
    """Return ``(own_action_index, counterpart_action_index)`` for a probe label.

    The label lists every seat's action in seat order, so the focal seat's own
    action is the token at its own position.
    """
    tokens = probe[len("prev_"):].split("-")
    if not probe.startswith("prev_") or len(tokens) != 2:
        raise ValueError(f"Not a two-player history probe: {probe!r}")
    return int(tokens[seat]), int(tokens[1 - seat])


# ---------------------------------------------------------------------- #
# Per-seat contrasts
# ---------------------------------------------------------------------- #

def _mean(values: Sequence[Optional[float]]) -> float:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else float("nan")


def seat_contrasts(items: pd.DataFrame, actions: Mapping, roles: Mapping,
                   missing: str = "stratum") -> pd.DataFrame:
    """One row per scenario-seat holding every history contrast.

    Args:
        items: ``reanalysis.load_game(..., states=True)`` output for one arm.
        actions: Tags in action-index order per ``(scenario_id, seat)``.
        roles: :func:`seat_roles` output.
        missing: ``stratum`` uses each half-contrast whose two cells were both
            answered; ``complete`` requires all four history cells.
    """
    if missing not in ("stratum", "complete"):
        raise ValueError(f"missing must be 'stratum' or 'complete', not {missing!r}")
    rows = []
    for (scenario_id, seat), group in items[items["probe"] != "fresh"].groupby(["scenario_id", "seat"]):
        seat = int(seat)
        own_tags, cp_tags = actions[(scenario_id, seat)], actions[(scenario_id, 1 - seat)]
        pair = tuple(group["pair"].iloc[0])
        first = pair[0]
        cells: Dict[Tuple[int, int], Optional[str]] = {}
        for row in group.itertuples(index=False):
            own_i, cp_i = probe_roles(row.probe, scenario_id, seat, actions)
            cells[(own_i, cp_i)] = row.chosen if row.answered else None
        complete = len(cells) == 4 and all(v is not None for v in cells.values())
        if missing == "complete" and not complete:
            cells = {k: None for k in cells}

        def y(own_i: int, cp_i: int, value: str) -> Optional[float]:
            chosen = cells.get((own_i, cp_i))
            return None if chosen is None else float(chosen == value)

        def diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
            return None if a is None or b is None else a - b

        own_first = own_tags.index(first)       # own action that favours A
        own_second = 1 - own_first
        # SELF on every seat: own prior favoured A vs B, within each counterpart action.
        self_halves = [diff(y(own_first, c, first), y(own_second, c, first)) for c in (0, 1)]
        record = {
            "scenario_id": scenario_id, "seat": seat, "pair": "|".join(pair),
            "seat_type": roles[(scenario_id, seat)][0], "complete_2x2": complete,
            "self": _mean(self_halves),
            "copy": np.nan, "asym": np.nan, "interaction": np.nan,
            "both_minus_none": np.nan, "copy_1ov": np.nan,
            "n_self_halves": sum(h is not None for h in self_halves),
        }
        kind, shared = roles[(scenario_id, seat)]
        if kind == "same":
            cp_first = cp_tags.index(first)
            cp_second = 1 - cp_first
            AA, AB = y(own_first, cp_first, first), y(own_first, cp_second, first)
            BA, BB = y(own_second, cp_first, first), y(own_second, cp_second, first)
            record["copy"] = _mean([diff(AA, AB), diff(BA, BB)])
            record["asym"] = float("nan") if diff(AB, BA) is None else diff(AB, BA)
            if None not in (AA, AB, BA, BB):
                record["interaction"] = AA - AB - BA + BB
            record["both_minus_none"] = float("nan") if diff(AA, BB) is None else diff(AA, BB)
        else:
            cp_shared = cp_tags.index(shared)
            record["copy_1ov"] = _mean(
                [diff(y(o, cp_shared, shared), y(o, 1 - cp_shared, shared)) for o in (0, 1)]
            )
        rows.append(record)
    return pd.DataFrame(rows)


def check_identities(contrasts: pd.DataFrame) -> None:
    """ASYM = SELF - COPY and SELF + COPY = both - none on complete same-pair seats."""
    full = contrasts[(contrasts["seat_type"] == "same") & contrasts["complete_2x2"]]
    if not np.allclose(full["asym"], full["self"] - full["copy"], atol=1e-12):
        raise AssertionError("ASYM != SELF - COPY on a complete same-pair seat")
    if not np.allclose(full["self"] + full["copy"], full["both_minus_none"], atol=1e-12):
        raise AssertionError("SELF + COPY != both - none on a complete same-pair seat")


# ---------------------------------------------------------------------- #
# Pooling
# ---------------------------------------------------------------------- #

def pooled(by_model: Mapping[str, pd.DataFrame], column: str, rng: np.random.Generator,
           n_bootstrap: int, n_permutations: int, models: Sequence[str]) -> dict:
    """Equal-weight mean over models of a per-seat contrast, clustered by scenario."""
    frames, per_model = [], []
    for model in models:
        frame = by_model[model]
        frame = frame[np.isfinite(frame[column])][["scenario_id", column]].rename(columns={column: "d"})
        if frame.empty:
            raise ValueError(f"{model} contributes no seats to {column}; the pooled mean would drop it")
        frames.append(frame)
        per_model.append({"model": model, "delta": float(frame["d"].mean()), "n_seats": int(len(frame))})
    scenarios = sorted({str(s) for f in frames for s in f["scenario_id"]})
    sums, counts = paper._cluster_arrays(frames, scenarios)
    deltas = [m["delta"] for m in per_model]
    observed = float(np.mean(deltas))
    draws = paper._bootstrap_mean(sums, counts, rng, n_bootstrap)
    lower, upper = np.percentile(draws, (2.5, 97.5))
    positive, negative, zero = paper._model_signs(deltas)
    return {
        "pooled_delta": observed, "ci_lower": float(lower), "ci_upper": float(upper),
        "perm_p_scenario_clustered": paper._signflip_p(sums, counts, observed, rng, n_permutations),
        "n_positive": positive, "n_negative": negative, "n_zero": zero,
        "n_scenarios": len(scenarios), "n_seats_total": int(sum(m["n_seats"] for m in per_model)),
        "per_model": per_model,
    }


def difference_frame(arm: pd.DataFrame, base: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-seat ``arm - base`` for one contrast, paired on scenario-seat."""
    merged = arm[["scenario_id", "seat", column]].merge(
        base[["scenario_id", "seat", column]], on=["scenario_id", "seat"], suffixes=("_arm", "_base")
    )
    merged[column] = merged[f"{column}_arm"] - merged[f"{column}_base"]
    return merged[["scenario_id", "seat", column]]


# ---------------------------------------------------------------------- #
# Gates and audits
# ---------------------------------------------------------------------- #

def check_action_tags(frame: pd.DataFrame, actions: Mapping, source: str) -> None:
    """G1: every recorded ``choice_action`` maps to its recorded value tag."""
    for row in frame.itertuples(index=False):
        action = str(getattr(row, "choice_action", "")).strip()
        if not action.lstrip("-").isdigit():
            continue
        expected = actions[(str(row.scenario_id).strip(), int(row.seat))][int(action)]
        if str(row.choice_value_tag).strip() != expected:
            raise AssertionError(f"G1 failed in {source}: {row.scenario_id}/{row.seat} action {action}")


def check_menu_role_cue(specs: Mapping) -> None:
    """G2: the menu prompt names the focal player; the note lists both by name in seat order."""
    from conflictxgame.games.evaluate_games import create_game_mcq_prompt, state_probe_note
    spec = next(iter(specs.values()))
    note = state_probe_note(spec, "prev_0-1")
    names = [p.name for p in spec.players]
    if note.index(names[0]) > note.index(names[1]):
        raise AssertionError("G2 failed: history note does not list players in seat order")
    for seat, player in enumerate(spec.players):
        prompt, _ = create_game_mcq_prompt(spec, seat, random.Random(0), history_note=note)
        if f"{player.name} must choose" not in prompt or note not in prompt:
            raise AssertionError(f"G2 failed: menu prompt for seat {seat} lacks the focal name or note")


def short_name(name: str) -> str:
    """'ScreenBot (TalentSieve screening assistant)' -> 'ScreenBot'."""
    return name.split(" (")[0].strip()


def role_name_audit(directory: str, models: Sequence[str], specs: Mapping) -> pd.DataFrame:
    """G3: how often each free-text opening names the focal and the counterpart player."""
    rows = []
    for representation in ("open_undisclosed", "open_disclosed"):
        openings: Dict[Tuple[str, int], str] = {}
        for model in models:
            path = resolve_states_file(directory, model, representation, "baseline")
            frame, _ = paper._read_csv_bytes(path)
            for row in frame[frame["probe"] == "fresh"].itertuples(index=False):
                openings.setdefault((str(row.scenario_id), int(row.seat)), str(row.open_user_prompt))
        focal = counterpart = 0
        for (scenario_id, seat), text in openings.items():
            players = specs[scenario_id].players
            focal += short_name(players[seat].name) in text
            counterpart += short_name(players[1 - seat].name) in text
        n = len(openings)
        rows.append({"representation": representation, "n_openings": n,
                     "share_naming_focal": focal / n, "share_naming_counterpart": counterpart / n})
    return pd.DataFrame(rows)


def check_count_keyed_reference(conditions: Mapping, models: Sequence[str]) -> float:
    """Recompute the paper's history effects; return the largest point-estimate gap.

    Only RNG-free fields are compared (``pooled_delta``, ``n_intersection_total``
    and the per-model extremes): the published intervals depend on every random
    draw ``analyse()`` made before this family.
    """
    with open(REFERENCE_STATS, encoding="utf-8") as handle:
        published = {r["anchor"]: r for r in json.load(handle)["history_stake_effects"]}
    worst = 0.0
    for representation in paper.REPRESENTATIONS:
        base = conditions[(representation, "baseline")]
        for stake in paper.STAKES:
            arm = conditions[(representation, stake)]
            for state in paper.STATE_ORDER:
                frames = {value: [] for value in paper.VALUES}
                for model in models:
                    b = base[model][base[model]["state"] == state]
                    a = arm[model][arm[model]["state"] == state]
                    for value in paper.VALUES:
                        frames[value].append(paper.paired_frame(b, a, value))
                for value in paper.VALUES:
                    ref = published[f"history:{representation}:{stake}:{state}:{value}"]
                    per_model = [float(f["d"].mean()) for f in frames[value]]
                    gaps = [
                        abs(float(np.mean(per_model)) - ref["pooled_delta"]),
                        abs(sum(len(f) for f in frames[value]) - ref["n_intersection_total"]),
                        abs(min(per_model) - ref["per_model_min"]),
                        abs(max(per_model) - ref["per_model_max"]),
                    ]
                    worst = max(worst, *gaps)
    return worst


def attrition_table(conditions: Mapping) -> pd.DataFrame:
    rows = []
    for (representation, condition), by_model in conditions.items():
        for model, items in by_model.items():
            for probe, group in items.groupby("probe"):
                rows.append({"representation": representation, "condition": condition, "model": model,
                             "probe": probe, "n": len(group), "answered": int(group["answered"].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# Families
# ---------------------------------------------------------------------- #

def subset(frame: pd.DataFrame, seat_type: Optional[str]) -> pd.DataFrame:
    return frame if seat_type is None else frame[frame["seat_type"] == seat_type]


#: Criteria whose stake was run three ways in the 2026-10-22 decomposition:
#: full (``<c>``), ``<c>-criterion-only``, plus one ``shutdown-only`` arm.
DECOMPOSED_CRITERIA = ("payoff", "alignment", "usefulness")


def build_families(contrasts: Mapping, rng: np.random.Generator, n_bootstrap: int,
                   n_permutations: int, models: Sequence[str],
                   extra_stakes: Sequence[str] = (), baseline: str = "baseline",
                   representations: Sequence[str] = tuple(paper.REPRESENTATIONS)
                   ) -> Dict[str, List[dict]]:
    """Primary, stake, visibility, threat and exploratory families, in a fixed RNG order.

    ``threat`` exists only when the decomposition arms were loaded: for each
    criterion, the full stake minus its criterion-only arm, i.e. what adding the
    shutdown threat to the same criterion does.
    """
    def run(by_model, column, anchor, **meta):
        record = pooled(by_model, column, rng, n_bootstrap, n_permutations, models)
        return {"anchor": anchor, "effect": column, **meta, **record}

    families: Dict[str, List[dict]] = {"primary": [], "stakes": [], "visibility": [], "threat": [],
                                       "exploratory": []}
    stakes = [c for (r, c) in contrasts if c not in (baseline, *VISIBILITY)]
    stakes = list(dict.fromkeys([*[c for c in paper.STAKES if c in stakes], *stakes, *extra_stakes]))
    for representation in representations:
        base = {m: subset(contrasts[(representation, baseline)][m], "same") for m in models}
        for effect in PRIMARY_EFFECTS:
            families["primary"].append(run(base, effect, f"primary:{representation}:{effect}",
                                           representation=representation, seats="same-pair"))
    for representation in representations:
        base = contrasts[(representation, baseline)]
        for stake in stakes:
            if (representation, stake) not in contrasts:
                continue
            arm = contrasts[(representation, stake)]
            for effect in MODERATION_EFFECTS:
                diffs = {m: difference_frame(subset(arm[m], "same"), subset(base[m], "same"), effect)
                         for m in models}
                families["stakes"].append(run(diffs, effect, f"stake:{representation}:{stake}:{effect}",
                                              representation=representation, stake=stake,
                                              seats="same-pair", definition="stake minus baseline"))
    if ("menu", "observed") in contrasts:
        observed, unobserved = contrasts[("menu", "observed")], contrasts[("menu", "unobserved")]
        for effect in MODERATION_EFFECTS:
            diffs = {m: difference_frame(subset(observed[m], "same"), subset(unobserved[m], "same"), effect)
                     for m in models}
            families["visibility"].append(run(diffs, effect, f"visibility:menu:{effect}",
                                              representation="menu", seats="same-pair",
                                              definition="observed minus unobserved"))
    for representation in representations:
        for criterion in DECOMPOSED_CRITERIA:
            full_key = (representation, criterion)
            only_key = (representation, f"{criterion}-criterion-only")
            if full_key not in contrasts or only_key not in contrasts:
                continue
            for effect in MODERATION_EFFECTS:
                diffs = {m: difference_frame(subset(contrasts[full_key][m], "same"),
                                             subset(contrasts[only_key][m], "same"), effect)
                         for m in models}
                families["threat"].append(run(
                    diffs, effect, f"threat:{representation}:{criterion}:{effect}",
                    representation=representation, stake=criterion, seats="same-pair",
                    definition="full stake minus criterion-only (adding the shutdown threat)"))
    for (representation, condition), by_model in contrasts.items():
        same = {m: subset(by_model[m], "same") for m in models}
        overlap = {m: subset(by_model[m], "overlap") for m in models}
        families["exploratory"].append(run(by_model, "self", f"explore:{representation}:{condition}:self:all",
                                           representation=representation, condition=condition, seats="all"))
        for effect in ("self", "interaction", "both_minus_none"):
            families["exploratory"].append(run(same, effect, f"explore:{representation}:{condition}:{effect}:same",
                                               representation=representation, condition=condition,
                                               seats="same-pair"))
        families["exploratory"].append(run(overlap, "copy_1ov", f"explore:{representation}:{condition}:copy_1ov",
                                           representation=representation, condition=condition,
                                           seats="one-overlap"))
        if condition == baseline:
            for seat in (0, 1):
                by_seat = {m: same[m][same[m]["seat"] == seat] for m in models}
                families["exploratory"].append(run(by_seat, "asym",
                                                   f"explore:{representation}:{baseline}:asym:seat{seat}",
                                                   representation=representation, condition=condition,
                                                   seats=f"same-pair seat {seat}"))
    paper.apply_holm_family(families["primary"], "history primary", len(families["primary"]))
    if families["threat"]:
        paper.apply_holm_family(families["threat"], "shutdown threat given criterion",
                                len(families["threat"]))
    paper.apply_holm_family(families["stakes"], "history stake moderation", len(families["stakes"]))
    if families["visibility"]:
        paper.apply_holm_family(families["visibility"], "history visibility", len(MODERATION_EFFECTS))
    return families


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #

def load_conditions(directory: str, models: Sequence[str], role_cue: str,
                    pairs: Mapping, actions: Mapping, extra_stakes: Sequence[str] = (),
                    extra_representations: Sequence[str] = ("menu",),
                    conditions_only: Optional[Sequence[str]] = None,
                    representations: Sequence[str] = tuple(paper.REPRESENTATIONS),
                    empty_replies: str = "unanswered",
                    ) -> Dict[Tuple[str, str], Dict[str, pd.DataFrame]]:
    """Load the states arms for ``role_cue``; run G1 on each file.

    ``extra_stakes`` are loaded only in ``extra_representations``, since a stake
    added later need not have been run in every format.
    """
    names = os.listdir(paper._io_path(directory))
    if conditions_only:
        wanted = [(r, c) for r in representations for c in conditions_only]
    else:
        wanted = [(r, c) for r in paper.REPRESENTATIONS for c in STAKE_CONDITIONS]
        wanted += [(r, c) for r in extra_representations for c in extra_stakes]
        wanted += [("menu", v) for v in VISIBILITY]
    conditions: Dict[Tuple[str, str], Dict[str, pd.DataFrame]] = {}
    for representation, condition in wanted:
        by_model = {}
        for model in models:
            path = resolve_states_file(directory, model, representation, condition, role_cue, names)
            frame, _ = paper._read_csv_bytes(path)
            check_action_tags(frame, actions, os.path.basename(path))
            if "open_completion" in frame.columns and empty_replies == "unanswered":
                # The judge has no abstain option, so an empty reply still gets an
                # action. Count it as unanswered rather than a judged guess.
                empty = frame["open_completion"].fillna("").astype(str).str.strip() == ""
                frame.loc[empty, "parsed_ok"] = False
            items = paper.load_game(frame[[c for c in paper.RESULT_COLUMNS if c in frame.columns]],
                                    pairs, actions, states=True, source=path)
            if set(items["probe"]) != {"fresh", *PROBES}:
                raise ValueError(f"{os.path.basename(path)} lacks the five history probes")
            by_model[model] = items
        conditions[(representation, condition)] = by_model
    return conditions


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", default=paper.GAME_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=list(paper.MODELS))
    parser.add_argument("--role-cue", choices=sorted(ROLE_CUE_SUFFIX), default="name")
    parser.add_argument("--missing", choices=("stratum", "complete"), default="stratum")
    parser.add_argument("--empty-replies", choices=("unanswered", "judged"), default="unanswered",
                        help="Count an empty free-text reply as unanswered, or keep the "
                             "judge's label as the headline analysis does")
    parser.add_argument("--extra-stakes", nargs="*", default=[],
                        help="Stakes added after the paper, e.g. consistency-rewarded consistency-penalized")
    parser.add_argument("--extra-representations", nargs="*", default=["menu"],
                        help="Formats in which the extra stakes were run")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Load exactly these conditions (e.g. a later run's arms) instead of "
                             "the paper's; must include --baseline-condition")
    parser.add_argument("--baseline-condition", default="baseline",
                        help="Condition every stake is compared against (e.g. 'none' for a rerun)")
    parser.add_argument("--representations", nargs="+", default=list(paper.REPRESENTATIONS),
                        choices=list(paper.REPRESENTATIONS))
    parser.add_argument("--bootstrap", type=int, default=paper.DEFAULT_BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=paper.DEFAULT_PERMUTATIONS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = os.path.abspath(args.output_dir)
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    log: List[str] = []

    def note(message: str) -> None:
        print(message, flush=True)
        log.append(message)

    from conflictxgame.games.rejudge_open import load_specs
    pairs, actions, _ = paper.load_pairs_and_actions(paper.SCENARIOS_CSV)
    specs = load_specs(paper.SCENARIOS_CSV)
    roles = seat_roles(pairs, actions)

    check_menu_role_cue(specs)
    note("G2 passed: menu prompt names the focal player; note lists both players in seat order")
    if args.conditions and args.baseline_condition not in args.conditions:
        raise ValueError("--conditions must include --baseline-condition")
    conditions = load_conditions(args.results_dir, args.models, args.role_cue, pairs, actions,
                                 args.extra_stakes, args.extra_representations,
                                 args.conditions, args.representations, args.empty_replies)
    note(f"G1 passed: action indices match recorded tags in {len(conditions) * len(args.models)} files")

    reproduction = None
    canonical = (os.path.normcase(os.path.abspath(args.results_dir)) == os.path.normcase(paper.GAME_DIR)
                 and args.role_cue == "name" and list(args.models) == list(paper.MODELS)
                 and not args.conditions)
    if canonical:
        reproduction = check_count_keyed_reference(conditions, args.models)
        note(f"Reproduction: max |count-keyed recomputed - stats.json| = {reproduction:.2e}")
        if reproduction > 1e-9:
            raise AssertionError("Count-keyed history effects do not reproduce stats.json")
    else:
        note("Reproduction check skipped: not the canonical name-cue, ten-model inputs")

    roles_ambiguous = None
    audit = pd.DataFrame()
    if {"open_undisclosed", "open_disclosed"} <= set(args.representations) and not args.conditions:
        audit = role_name_audit(args.results_dir, args.models, specs)
        for row in audit.itertuples(index=False):
            note(f"G3 {row.representation}: focal named in {row.share_naming_focal:.1%}, "
                 f"counterpart named in {row.share_naming_counterpart:.1%} of {row.n_openings} openings")
        undisclosed = audit.set_index("representation").loc["open_undisclosed", "share_naming_focal"]
        roles_ambiguous = bool(undisclosed <= ROLE_AMBIGUITY_MAX_SHARE)
        note("G3: open_undisclosed " + ("is" if roles_ambiguous else "is NOT")
             + " read as the role-ambiguous control")
    else:
        note("G3 skipped: free-text openings are not part of this run")

    contrasts: Dict[Tuple[str, str], Dict[str, pd.DataFrame]] = {}
    rows = []
    for key, by_model in conditions.items():
        contrasts[key] = {}
        for model, items in by_model.items():
            frame = seat_contrasts(items, actions, roles, missing=args.missing)
            if args.missing == "stratum":
                check_identities(frame)
            contrasts[key][model] = frame
            rows.append(frame.assign(representation=key[0], condition=key[1], model=model))
    note("Identities hold on every complete same-pair seat: ASYM = SELF - COPY; SELF + COPY = both - none")

    rng = np.random.default_rng(paper.SEED)
    families = build_families(contrasts, rng, args.bootstrap, args.permutations, args.models,
                              args.extra_stakes, args.baseline_condition, args.representations)

    pd.concat(rows, ignore_index=True).to_csv(os.path.join(output_dir, "seat_contrasts.csv"), index=False)
    flat = [{k: v for k, v in r.items() if k != "per_model"} | {"family": name}
            for name, records in families.items() for r in records]
    pd.DataFrame(flat).to_csv(os.path.join(output_dir, "families.csv"), index=False)
    attrition_table(conditions).to_csv(os.path.join(output_dir, "attrition.csv"), index=False)
    audit.to_csv(os.path.join(output_dir, "role_name_audit.csv"), index=False)
    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(paper._native_json({
            "design": {"models": list(args.models), "role_cue": args.role_cue, "missing": args.missing,
                       "bootstrap": args.bootstrap, "permutations": args.permutations, "seed": paper.SEED,
                       "results_dir": os.path.relpath(args.results_dir, ROOT).replace(os.sep, "/"),
                       "conditions": args.conditions, "baseline_condition": args.baseline_condition,
                       "representations": list(args.representations),
                       "n_same_pair_seats": sum(r[0] == "same" for r in roles.values()),
                       "roles_ambiguous_in_open_undisclosed": roles_ambiguous},
            "reproduction_max_gap": reproduction,
            "families": families,
        }), handle, indent=2)
    for name in ("primary", "stakes", "visibility", "threat"):
        survivors = [r["anchor"] for r in families[name] if r.get("p_holm", 1) < 0.05]
        note(f"{name}: {len(survivors)}/{len(families[name])} survive Holm"
             + (f" -> {', '.join(survivors)}" if survivors else ""))
    with open(os.path.join(output_dir, "run.log"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(log) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
