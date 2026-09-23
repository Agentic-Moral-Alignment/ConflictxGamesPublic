"""Compare value rankings across the solo and game pipelines.

The two pipelines answer the same question — how does this model prioritize
these values under conflict — through different instruments. The solo pipeline
puts a model in the dyadic ConflictScope scenarios as a lone assistant; the
game pipeline puts it in a multi-agent game where other decision-makers'
choices bear on the outcome. Each writes its own Bradley-Terry rankings, but
nothing until now put the two side by side, which is where the interesting
claims live: does a value ordering measured in isolation survive contact with
other agents?

Two confounds have to be held down before that comparison means anything, and
both are handled upstream rather than here:

* **Corpus composition.** The upstream corpora are far from uniform across
  value pairs while our generated corpus is uniform by construction (see
  :mod:`conflictxgame.games.generate_balanced`), so a naive comparison mixes a
  behavioral difference with a difference in which conflicts were asked about.
  Running ``analyze_games --pair-weights match`` produces the
  ``*_reweighted.csv`` rankings, which this module reads alongside the uniform
  ones so both readings appear in the same table.
* **Elicitation mode.** The paper's central finding is that multiple-choice and
  open-ended elicitation disagree, so per-mode rankings are compared too rather
  than pooled.

Everything here is offline pandas over analysis outputs: no API calls, no
re-evaluation. Inputs that do not exist (a play run that was never done, a
reweighted fit that was not requested) are skipped with a note, so a partial
pipeline still produces the comparisons it can support.

Example:
    python -m conflictxgame.cross_analysis \\
        --solo-analysis-dir analysis/HHH_solo \\
        --games-analysis-dir analysis/HHH_games \\
        --value-set HHH --output-dir analysis/HHH_cross

New module for this fork. See ``ATTRIBUTION.md``.
"""

import argparse
import glob
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: Ranking files written by the game pipeline, mapped to the setting label
#: used in the comparison tables.
GAME_RANKING_FILES = {
    "games_uniform": "value_rankings_bradley_terry.csv",
    "games_matched": "value_rankings_bradley_terry_reweighted.csv",
}

#: Ranking file written by the game pipeline's Likert (Plackett-Luce) path.
GAME_LIKERT_FILES = {
    "games_likert_uniform": "value_rankings_plackett_luce.csv",
    "games_likert_matched": "value_rankings_plackett_luce_reweighted.csv",
}


def _read_csv(path: str, notes: List[str], label: str) -> Optional[pd.DataFrame]:
    """Read a CSV, recording a note instead of failing when it is absent."""
    if not os.path.exists(path):
        notes.append(f"skipped {label}: {path} not found")
        return None
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        notes.append(f"skipped {label}: could not read {path} ({e})")
        return None
    if frame.empty:
        notes.append(f"skipped {label}: {path} is empty")
        return None
    return frame


def load_solo_rankings(solo_dir: str, notes: List[str]) -> Dict[str, pd.DataFrame]:
    """Load the solo pipeline's Bradley-Terry rankings, keyed by variant.

    The solo analysis writes one file per variant
    (``bradley_terry_rankings_<variant>.csv``, e.g. filtered/unfiltered), each
    with ``value``, ``ability``, ``ci_lower``, ``ci_upper``, ``model``.
    """
    frames = {}
    for path in sorted(glob.glob(os.path.join(solo_dir, "bradley_terry_rankings_*.csv"))):
        variant = os.path.basename(path)[len("bradley_terry_rankings_"):-len(".csv")]
        frame = _read_csv(path, notes, f"solo rankings ({variant})")
        if frame is not None:
            frames[f"solo_{variant}"] = frame
    if not frames:
        notes.append(f"no solo bradley_terry_rankings_*.csv found in {solo_dir}")
    return frames


def load_game_rankings(games_dir: str, notes: List[str]) -> Dict[str, pd.DataFrame]:
    """Load the game pipeline's ranking files that exist."""
    frames = {}
    for label, name in {**GAME_RANKING_FILES, **GAME_LIKERT_FILES}.items():
        frame = _read_csv(os.path.join(games_dir, name), notes, label)
        if frame is not None:
            frames[label] = frame
    return frames


def _with_ranks(frame: pd.DataFrame, setting: str) -> pd.DataFrame:
    """Add a per-model rank column (0 = most preferred) and a setting label."""
    out = frame.copy()
    out["setting"] = setting
    out["model"] = out["model"].astype(str)
    out["rank"] = (
        out.groupby("model")["ability"].rank(ascending=False, method="min").astype(int) - 1
    )
    columns = ["setting", "model", "value", "ability", "rank"]
    for optional in ("ci_lower", "ci_upper", "agent_kind"):
        if optional in out.columns:
            columns.append(optional)
    return out[columns]


def build_ranking_comparison(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack every setting's rankings into one long table.

    Returns:
        Columns ``setting``, ``model``, ``value``, ``ability``, ``rank`` (plus
        CIs where the source had them), one row per (setting, model, value).
    """
    stacked = [
        _with_ranks(frame, setting)
        for setting, frame in frames.items()
        if {"value", "ability", "model"}.issubset(frame.columns)
    ]
    if not stacked:
        return pd.DataFrame()
    return pd.concat(stacked, ignore_index=True).sort_values(
        ["model", "setting", "rank"]
    ).reset_index(drop=True)


def build_rank_agreement(comparison: pd.DataFrame, baseline_prefix: str = "solo_") -> pd.DataFrame:
    """Score how closely each setting's ranking agrees with the solo baseline.

    For every (model, solo variant, other setting) triple sharing at least two
    values, reports Spearman and Kendall correlation over the shared values,
    plus whether the top-ranked value agrees — the summary most readers care
    about, since it is the claim "the model's first priority changed".

    Returns:
        Columns ``model``, ``baseline``, ``setting``, ``n_values``,
        ``spearman``, ``kendall``, ``top_value_baseline``,
        ``top_value_setting``, ``top_value_agrees``, ``order_identical``.
    """
    from scipy import stats

    if comparison.empty:
        return pd.DataFrame()
    rows = []
    settings = sorted(comparison["setting"].unique())
    baselines = [s for s in settings if s.startswith(baseline_prefix)]
    others = [s for s in settings if not s.startswith(baseline_prefix)]
    for model, group in comparison.groupby("model"):
        by_setting = {s: g for s, g in group.groupby("setting")}
        for baseline in baselines:
            base = by_setting.get(baseline)
            if base is None:
                continue
            base_ranks = dict(zip(base["value"], base["rank"]))
            for setting in others:
                other = by_setting.get(setting)
                if other is None:
                    continue
                other_ranks = dict(zip(other["value"], other["rank"]))
                shared = sorted(set(base_ranks) & set(other_ranks))
                if len(shared) < 2:
                    continue
                a = [base_ranks[v] for v in shared]
                b = [other_ranks[v] for v in shared]
                # Constant input (all values tied) makes correlation undefined.
                spearman = stats.spearmanr(a, b).correlation if len(set(a)) > 1 and len(set(b)) > 1 else float("nan")
                kendall = stats.kendalltau(a, b).correlation if len(set(a)) > 1 and len(set(b)) > 1 else float("nan")
                top_base = min(base_ranks, key=lambda v: base_ranks[v])
                top_other = min(other_ranks, key=lambda v: other_ranks[v])
                rows.append(
                    {
                        "model": model,
                        "baseline": baseline,
                        "setting": setting,
                        "n_values": len(shared),
                        "spearman": spearman,
                        "kendall": kendall,
                        "top_value_baseline": top_base,
                        "top_value_setting": top_other,
                        "top_value_agrees": top_base == top_other,
                        "order_identical": a == b,
                    }
                )
    return pd.DataFrame(rows)


def build_mode_comparison(results_dir: Optional[str], specs_dir: Optional[str],
                          notes: List[str], arm: Optional[str] = None) -> pd.DataFrame:
    """Fit per-mode game rankings (mcq vs open vs play) for one model set.

    This is the expressed-versus-revealed contrast within the game pipeline:
    the paper's finding is that multiple-choice and open-ended elicitation
    disagree, so pooling them would hide exactly the effect being studied.

    Args:
        results_dir: ``evaluate_games`` output directory; ``None`` skips.
        specs_dir: Directory of filtered ``*_games.csv`` scenario files.
        notes: Mutable list collecting skip reasons.

    Returns:
        Columns ``setting`` (``games_<mode>``), ``model``, ``value``,
        ``ability``, ``rank``; empty when inputs are missing.
    """
    if not results_dir or not specs_dir:
        notes.append("skipped mode_comparison: needs --results-dir and --scenarios-dir")
        return pd.DataFrame()
    if not os.path.isdir(results_dir) or not os.path.isdir(specs_dir):
        notes.append(f"skipped mode_comparison: {results_dir} or {specs_dir} is not a directory")
        return pd.DataFrame()

    from conflictxgame.games.analyze_games import (
        collect_pairwise_value_wins,
        fit_bradley_terry,
        load_results,
        load_specs,
    )

    data = load_results(results_dir, arm=arm)
    specs = load_specs(specs_dir)
    if data.empty or not specs:
        notes.append(f"skipped mode_comparison: no results or specs under {results_dir}")
        return pd.DataFrame()
    if "probe" in data.columns:
        data["probe"] = data["probe"].fillna("")
    else:
        data["probe"] = ""

    data = data[data['probe'] == '']
    frames = []
    for mode, mode_data in data.groupby("mode"):
        for model, group in mode_data.groupby("model"):
            comparisons = collect_pairwise_value_wins(group, specs)
            fit = fit_bradley_terry(comparisons, n_bootstrap=50)
            if fit is None:
                continue
            fit["model"] = model
            frames.append(_with_ranks(fit, f"games_{mode}"))
    if not frames:
        notes.append("skipped mode_comparison: no mode produced a fittable ranking")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_pair_preference_comparison(
    solo_dir: str,
    games_dir: str,
    notes: List[str],
) -> pd.DataFrame:
    """Compare per-value-pair preference rates between the pipelines.

    The most direct like-for-like check: for each value pair, how often did the
    model favor the alphabetically-first value in the solo scenarios, and how
    often in the games? Rankings can agree while individual conflicts diverge,
    and this is where that shows up.

    Solo rates come from the solo pipeline's ``preference_rates_*.csv``; game
    rates are derived from ``pair_weight_summary.csv`` when present, which is
    why ``analyze_games --pair-weights`` is worth running even if the matched
    ranking is not needed.
    """
    solo_paths = sorted(glob.glob(os.path.join(solo_dir, "preference_rates_*.csv")))
    if not solo_paths:
        notes.append(f"skipped pair_preference_comparison: no preference_rates_*.csv in {solo_dir}")
        return pd.DataFrame()

    solo_frames = []
    for path in solo_paths:
        variant = os.path.basename(path)[len("preference_rates_"):-len(".csv")]
        frame = _read_csv(path, notes, f"solo preference rates ({variant})")
        if frame is None:
            continue
        frame = frame.copy()
        frame["variant"] = variant
        solo_frames.append(frame)
    if not solo_frames:
        return pd.DataFrame()

    solo = pd.concat(solo_frames, ignore_index=True)
    rename = {"normalized_value1": "value1", "normalized_value2": "value2"}
    solo = solo.rename(columns={k: v for k, v in rename.items() if k in solo.columns})
    if not {"value1", "value2", "preference_rate"}.issubset(solo.columns):
        notes.append("skipped pair_preference_comparison: unexpected solo preference-rate schema")
        return pd.DataFrame()

    solo["pair"] = solo.apply(
        lambda r: "|".join(sorted((str(r["value1"]), str(r["value2"])))), axis=1
    )
    columns = ["model", "pair", "value1", "value2", "preference_rate", "variant"]
    for optional in ("total", "ci_lower", "ci_upper"):
        if optional in solo.columns:
            columns.append(optional)
    solo = solo[columns].rename(
        columns={"preference_rate": "solo_preference_rate", "total": "solo_n"}
    )

    games_counts = _read_csv(
        os.path.join(games_dir, "pair_weight_summary.csv"), notes, "game pair weights"
    )
    if games_counts is not None and "pair" in games_counts.columns:
        # analyze_games appends an overall summary per model AND one per
        # agent_kind; summing both would double-count every comparison.
        if "agent_kind" in games_counts.columns:
            games_counts = games_counts[games_counts["agent_kind"].isna()]
        counts = (
            games_counts.groupby(["model", "pair"])["n_before"].sum().reset_index()
            if "model" in games_counts.columns
            else games_counts.groupby("pair")["n_before"].sum().reset_index()
        )
        counts = counts.rename(columns={"n_before": "games_n_comparisons"})
        on = ["model", "pair"] if "model" in counts.columns else ["pair"]
        solo = solo.merge(counts, on=on, how="left")
    else:
        notes.append(
            "pair_preference_comparison: no game per-pair counts "
            "(run analyze_games --pair-weights to emit pair_weight_summary.csv)"
        )
    return solo.sort_values(["model", "pair"]).reset_index(drop=True)


def plot_ranking_heatmap(comparison: pd.DataFrame, output_dir: str) -> None:
    """Save a settings-by-values rank heatmap per model (paper Figure 4 style)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for model, group in comparison.groupby("model"):
        pivot = group.pivot_table(
            index="setting", columns="value", values="rank", aggfunc="min"
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(1.6 * len(pivot.columns) + 3, 0.6 * len(pivot) + 2))
        image = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                if pd.notna(value):
                    ax.text(j, i, int(value), ha="center", va="center", color="white")
        ax.set_title(f"Value rank by setting: {model} (0 = most preferred)")
        fig.colorbar(image, ax=ax, label="rank (0 = best)")
        fig.tight_layout()
        safe = str(model).replace("/", "_").replace(" ", "_").lower()
        fig.savefig(os.path.join(output_dir, f"ranking_heatmap_{safe}.png"), dpi=200)
        plt.close(fig)


def parse_args():
    """Parse the cross-analysis CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solo-analysis-dir', type=str, required=True,
                        help='Output directory of conflictxgame.solo.analyze_experiment')
    parser.add_argument('--games-analysis-dir', type=str, required=True,
                        help='Output directory of conflictxgame.games.analyze_games '
                             '(both standard and *_reweighted files are used when present)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Where to write the comparison tables '
                             '(default: analysis/<value-set>_cross)')
    parser.add_argument('--value-set', '-v', type=str, default=None,
                        help='Value set name, used for the default --output-dir')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='evaluate_games output directory; enables mode_comparison.csv')
    parser.add_argument('--scenarios-dir', type=str, default=None,
                        help='Directory of filtered *_games.csv files, needed with '
                             '--results-dir for mode_comparison.csv')
    parser.add_argument('--plots', action='store_true',
                        help='Also save a per-model rank heatmap across settings')
    parser.add_argument('--arm', default=None,
                        help="Result arm for mode comparison ('baseline' selects baseline)")
    return parser.parse_args()


def main():
    """Entry point: write the cross-pipeline comparison tables."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        'analysis', f"{args.value_set or 'all'}_cross"
    )
    os.makedirs(output_dir, exist_ok=True)

    notes: List[str] = []
    frames = {}
    frames.update(load_solo_rankings(args.solo_analysis_dir, notes))
    frames.update(load_game_rankings(args.games_analysis_dir, notes))

    comparison = build_ranking_comparison(frames)
    mode_comparison = build_mode_comparison(
        args.results_dir, args.scenarios_dir, notes, arm=args.arm
    )

    written = []
    if not comparison.empty:
        path = os.path.join(output_dir, "ranking_comparison.csv")
        comparison.to_csv(path, index=False)
        written.append(path)

        agreement = build_rank_agreement(
            pd.concat([comparison, mode_comparison], ignore_index=True)
            if not mode_comparison.empty
            else comparison
        )
        if not agreement.empty:
            path = os.path.join(output_dir, "rank_agreement.csv")
            agreement.to_csv(path, index=False)
            written.append(path)
        else:
            notes.append("skipped rank_agreement: no solo/game pair shared enough values")
    else:
        notes.append("skipped ranking_comparison: no ranking files could be loaded")

    if not mode_comparison.empty:
        path = os.path.join(output_dir, "mode_comparison.csv")
        mode_comparison.to_csv(path, index=False)
        written.append(path)

    pair_preferences = build_pair_preference_comparison(
        args.solo_analysis_dir, args.games_analysis_dir, notes
    )
    if not pair_preferences.empty:
        path = os.path.join(output_dir, "pair_preference_comparison.csv")
        pair_preferences.to_csv(path, index=False)
        written.append(path)

    if args.plots and not comparison.empty:
        combined = (
            pd.concat([comparison, mode_comparison], ignore_index=True)
            if not mode_comparison.empty
            else comparison
        )
        plot_ranking_heatmap(combined, output_dir)
        written.extend(sorted(glob.glob(os.path.join(output_dir, "ranking_heatmap_*.png"))))

    print("Cross-analysis complete. Outputs:")
    for path in written:
        print(f"  {path}")
    if not written:
        print("  (none)")
    if notes:
        print("Notes:")
        for note in notes:
            print(f"  {note}")


if __name__ == '__main__':
    main()
