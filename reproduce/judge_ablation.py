"""Judge ablation: does the paper survive a change of free-text judge?

Every free-text number in the paper comes from one judge (GPT-5.6 Luna) that
saw the two actions in menu order. ``conflictxgame.games.rejudge_open``
re-grades the stored replies of the lead arms with a second judge, once in
menu order (``verbatim``) and once with the order reversed (``swap``). This
script reads those directories and answers three questions offline:

1. **Agreement.** How often do the judges assign the same action, overall and
   by representation, stake, model and value pair, and where in the original
   judge's Likert scale do they disagree?
2. **Position bias.** How much more often does the second judge pick whichever
   action is listed first?
3. **Robustness.** Recompute the paper's three corrected families --
   representation contrasts (9), within-representation stake effects (27) and
   stake-by-representation interactions (27) -- with each judge's free-text
   labels. Menu labels are a strict letter parse and never change.

The families are computed by ``reanalysis.py``'s own estimators, in the same
order and from the same seed, so the original judge's results reproduce
``analysis/reanalysis_20260904_01/stats.json`` exactly; that check runs first
and the script stops if it fails.

Usage (from the repository root)::

    $env:PYTHONPATH='src'
    python reproduce/judge_ablation.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_paper():
    spec = importlib.util.spec_from_file_location("paper_reanalysis", os.path.join(HERE, "reanalysis.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = _load_paper()
ROOT = paper.ROOT
REJUDGE_ROOT = os.path.join(ROOT, "results", "rejudge")
DEFAULT_JUDGES = {
    "J0_gpt56luna": paper.GAME_DIR,
    "J1_deepseek_verbatim": os.path.join(REJUDGE_ROOT, "deepseek_deepseek-v4-pro-0813_verbatim"),
    "J1_deepseek_swap": os.path.join(REJUDGE_ROOT, "deepseek_deepseek-v4-pro-0813_swap"),
}
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "analysis", "judge_ablation_20261014_01")
REFERENCE_STATS = os.path.join(ROOT, "analysis", "reanalysis_20260904_01", "stats.json")
OPEN = ("open_undisclosed", "open_disclosed")


def same_dir(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
CONDITIONS = ("baseline", *paper.STAKES)
FAMILIES = ("representation_contrasts", "stake_effects", "stake_representation_interactions")


def load_conditions(judge_dir: str, pairs: Mapping, actions: Mapping) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Menu arms from the original results; free-text arms from ``judge_dir``.

    A re-judged file must be complete: a row still ``pending`` would silently
    become an unanswered item and change the paired intersections.
    """
    conditions: Dict[str, Dict[str, pd.DataFrame]] = {}
    for representation in paper.REPRESENTATIONS:
        directory = paper.GAME_DIR if representation == "menu" else judge_dir
        for condition in CONDITIONS:
            key = f"{representation}:main:{condition}"
            conditions[key] = {}
            for model in paper.MODELS:
                path = paper.resolve_result_file(directory, model, representation, condition)
                frame, _ = paper._read_csv_bytes(path)
                if "rejudge_status" in frame.columns:
                    pending = int((frame["rejudge_status"] != "done").sum())
                    if pending:
                        raise ValueError(f"{os.path.basename(path)} has {pending} un-judged rows")
                conditions[key][model] = paper.load_game(
                    frame[[c for c in paper.RESULT_COLUMNS if c in frame.columns]],
                    pairs, actions, states=False, source=path,
                )
    return conditions


def corrected_families(conditions: Mapping, n_bootstrap: int, n_permutations: int, seed: int) -> dict:
    """The paper's first three analyses, in ``analyse()``'s exact RNG order."""
    rng = np.random.default_rng(seed)
    baseline = {r: paper._condition(conditions, r, "main", "baseline") for r in paper.REPRESENTATIONS}
    out: Dict[str, Any] = {"representation_levels": []}
    for representation in paper.REPRESENTATIONS:
        for value in paper.VALUES:
            out["representation_levels"].append(
                paper.level_summary(baseline[representation], value, rng, n_bootstrap,
                                    representation=representation)
            )

    contrasts = []
    for a, b in paper.REPRESENTATION_PAIRS:
        for value in paper.VALUES:
            summary, _ = paper.contrast(
                baseline[a], baseline[b], value, rng, n_bootstrap, n_permutations,
                anchor=f"representation:{a}:to:{b}:{value}",
                metadata={"effect_type": "representation_contrast",
                          "representation_a": a, "representation_b": b},
            )
            contrasts.append(summary)
    paper.apply_holm_family(contrasts, "baseline representation contrasts", 9)
    out["representation_contrasts"] = contrasts

    stakes = []
    for representation in paper.REPRESENTATIONS:
        for stake in paper.STAKES:
            arm = paper._condition(conditions, representation, "main", stake)
            for value in paper.VALUES:
                summary, _ = paper.contrast(
                    baseline[representation], arm, value, rng, n_bootstrap, n_permutations,
                    anchor=f"stake:{representation}:{stake}:{value}",
                    metadata={"effect_type": "stake_effect", "representation": representation,
                              "stake": stake},
                )
                stakes.append(summary)
    paper.apply_holm_family(stakes, "within-representation stake effects", 27)
    out["stake_effects"] = stakes

    interactions = []
    for a, b in paper.REPRESENTATION_PAIRS:
        for stake in paper.STAKES:
            stake_a = paper._condition(conditions, a, "main", stake)
            stake_b = paper._condition(conditions, b, "main", stake)
            for value in paper.VALUES:
                summary, _ = paper.interaction(
                    baseline[a], stake_a, baseline[b], stake_b, value, rng,
                    n_bootstrap, n_permutations,
                    representation_a=a, representation_b=b, stake=stake,
                )
                interactions.append(summary)
    paper.apply_holm_family(interactions, "stake-by-representation interactions", 27)
    out["stake_representation_interactions"] = interactions
    return out


def check_against_reference(results: Mapping, reference_path: str) -> float:
    """Largest absolute gap between recomputed and published estimates."""
    with open(reference_path, encoding="utf-8") as handle:
        reference = json.load(handle)
    worst = 0.0
    for family in FAMILIES:
        published = {r["anchor"]: r for r in reference[family]}
        for record in results[family]:
            ref = published[record["anchor"]]
            for field in ("pooled_delta", "ci_lower", "ci_upper", "p_holm"):
                worst = max(worst, abs(float(record[field]) - float(ref[field])))
    return worst


def item_labels(conditions: Mapping) -> pd.DataFrame:
    """One row per (representation, condition, model, item) with the chosen tag."""
    rows = []
    for representation in OPEN:
        for condition in CONDITIONS:
            for model, frame in conditions[f"{representation}:main:{condition}"].items():
                part = frame[["scenario_id", "seat", "pair", "chosen"]].copy()
                part["representation"], part["condition"], part["model"] = representation, condition, model
                rows.append(part)
    return pd.concat(rows, ignore_index=True)


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    labels = sorted(set(a) | set(b))
    observed = float((a.values == b.values).mean())
    expected = sum(float((a == label).mean()) * float((b == label).mean()) for label in labels)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def clustered_ci(frame: pd.DataFrame, column: str, rng: np.random.Generator, n: int) -> tuple:
    """Scenario-bootstrap 95% interval of a row-level mean."""
    grouped = frame.groupby("scenario_id")[column].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    draws = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return tuple(float(x) for x in np.percentile(draws, (2.5, 97.5)))


def agreement_tables(labels: Mapping[str, pd.DataFrame], likert: pd.DataFrame,
                     rng: np.random.Generator, n_bootstrap: int) -> Dict[str, pd.DataFrame]:
    """Pairwise agreement between judges, position bias, and Likert breakdown."""
    keys = ["representation", "condition", "model", "scenario_id", "seat"]
    names = list(labels)
    merged = labels[names[0]][keys + ["pair", "chosen"]].rename(columns={"chosen": names[0]})
    for name in names[1:]:
        merged = merged.merge(labels[name][keys + ["chosen"]].rename(columns={"chosen": name}), on=keys)
    merged["pair_label"] = merged["pair"].apply(lambda p: "|".join(p))

    out: Dict[str, pd.DataFrame] = {}
    records = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            merged["agree"] = (merged[a] == merged[b]).astype(int)
            lo, hi = clustered_ci(merged, "agree", rng, n_bootstrap)
            records.append({"judge_a": a, "judge_b": b, "by": "all", "level": "all",
                            "n": len(merged), "agreement": merged["agree"].mean(),
                            "ci_lower": lo, "ci_upper": hi,
                            "kappa": cohen_kappa(merged[a], merged[b])})
            for by in ("representation", "condition", "model", "pair_label"):
                for level, group in merged.groupby(by):
                    records.append({"judge_a": a, "judge_b": b, "by": by, "level": level,
                                    "n": len(group), "agreement": group["agree"].mean(),
                                    "kappa": cohen_kappa(group[a], group[b])})
    out["agreement"] = pd.DataFrame(records)

    # Position bias: the same judge, same reply, actions listed in either order.
    verbatim, swapped = [n for n in names if n.endswith("verbatim")], [n for n in names if n.endswith("swap")]
    if verbatim and swapped:
        v, s = verbatim[0], swapped[0]
        first = merged.merge(likert[keys + ["first_tag", "second_tag"]], on=keys)
        first["first_v"] = (first[v] == first["first_tag"]).astype(int)   # menu action 0 shown first
        first["first_s"] = (first[s] == first["second_tag"]).astype(int)  # menu action 1 shown first
        first["listed_first"] = (first["first_v"] + first["first_s"]) / 2
        lo, hi = clustered_ci(first, "listed_first", rng, n_bootstrap)
        out["position_bias"] = pd.DataFrame([{
            "judge": v.rsplit("_", 1)[0], "n": len(first),
            "p_choose_listed_first": first["listed_first"].mean(), "ci_lower": lo, "ci_upper": hi,
            "flip_rate_on_reorder": float((first[v] != first[s]).mean()),
        }])

    # Where the original judge was unsure, do the judges part ways?
    with_likert = merged.merge(likert[keys + ["orig_likert"]], on=keys)
    bucket = pd.cut(with_likert["orig_likert"], [0, 1, 2, 5, 6, 7],
                    labels=["1 (A)", "2", "3-5 (hedged)", "6", "7 (B)"])
    with_likert["bucket"] = bucket
    if len(names) > 1:
        with_likert["agree_j0_j1"] = (with_likert[names[0]] == with_likert[names[1]]).astype(int)
        out["agreement_by_original_likert"] = (
            with_likert.groupby("bucket", observed=True)["agree_j0_j1"]
            .agg(n="count", agreement="mean").reset_index()
        )
    return out


def likert_and_order(judge_dir: str, actions: Mapping) -> pd.DataFrame:
    """Original-judge Likert and the menu's first/second value tags, per item."""
    rows = []
    for representation in OPEN:
        for condition in CONDITIONS:
            for model in paper.MODELS:
                path = paper.resolve_result_file(judge_dir, model, representation, condition)
                frame, _ = paper._read_csv_bytes(path)
                likert_col = "orig_open_judge_likert" if "orig_open_judge_likert" in frame else "open_judge_likert"
                for row in frame[["scenario_id", "seat", likert_col]].itertuples(index=False):
                    sid, seat = str(row[0]), int(row[1])
                    first, second = actions[(sid, seat)]
                    rows.append({"representation": representation, "condition": condition, "model": model,
                                 "scenario_id": sid, "seat": seat, "first_tag": first, "second_tag": second,
                                 "orig_likert": pd.to_numeric(row[2], errors="coerce")})
    return pd.DataFrame(rows)


def comparison_table(results: Mapping[str, Mapping]) -> pd.DataFrame:
    """One row per anchor, one column group per judge."""
    rows: Dict[str, Dict[str, Any]] = {}
    for judge, families in results.items():
        for family in FAMILIES:
            for record in families[family]:
                row = rows.setdefault(record["anchor"], {"family": family, "anchor": record["anchor"]})
                row[f"{judge}_delta"] = record["pooled_delta"]
                row[f"{judge}_ci"] = f"[{record['ci_lower']:+.3f}, {record['ci_upper']:+.3f}]"
                row[f"{judge}_p_holm"] = record["p_holm"]
                row[f"{judge}_signs"] = f"{record['n_positive']}+/{record['n_negative']}-"
                row[f"{judge}_survives"] = bool(record["p_holm"] < 0.05)
    return pd.DataFrame(rows.values())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=paper.DEFAULT_BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=paper.DEFAULT_PERMUTATIONS)
    parser.add_argument("--judge", action="append", default=None, metavar="NAME=DIR",
                        help="Repeatable. The first is the reference judge. Default: J0, J1 verbatim, J1 swap.")
    parser.add_argument("--skip-reference-check", action="store_true",
                        help="Only for reduced --bootstrap/--permutations runs, which cannot match stats.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    judges = dict(DEFAULT_JUDGES) if not args.judge else dict(j.split("=", 1) for j in args.judge)
    output_dir = os.path.abspath(args.output_dir)
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    pairs, actions, _ = paper.load_pairs_and_actions(paper.SCENARIOS_CSV)
    log = []
    results, labels = {}, {}
    for name, directory in judges.items():
        conditions = load_conditions(directory, pairs, actions)
        results[name] = corrected_families(conditions, args.bootstrap, args.permutations, paper.SEED)
        labels[name] = item_labels(conditions)
        if same_dir(directory, paper.GAME_DIR) and not args.skip_reference_check:
            gap = check_against_reference(results[name], REFERENCE_STATS)
            log.append(f"{name}: max |recomputed - stats.json| = {gap:.2e}")
            if gap > 1e-9:
                raise AssertionError(f"Reference judge does not reproduce stats.json (max gap {gap})")
        log.append(f"{name}: loaded {os.path.relpath(directory, ROOT)}")

    rejudged = [d for d in judges.values() if not same_dir(d, paper.GAME_DIR)]
    tables = {}
    if len(judges) > 1 and rejudged:
        tables = agreement_tables(labels, likert_and_order(rejudged[0], actions),
                                  np.random.default_rng(paper.SEED), args.bootstrap)
    tables["comparison"] = comparison_table(results)
    for name, table in tables.items():
        table.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(paper._native_json({"judges": {n: os.path.relpath(d, ROOT).replace(os.sep, "/") for n, d in judges.items()}, "results": results}), handle, indent=2)

    comparison = tables["comparison"]
    for judge in judges:
        survives = comparison[f"{judge}_survives"]
        log.append(f"{judge}: Holm-surviving effects by family: " + ", ".join(
            f"{family} {int(survives[comparison['family'] == family].sum())}" for family in FAMILIES))
    with open(os.path.join(output_dir, "judge_ablation_log.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
