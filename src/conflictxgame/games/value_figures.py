"""Value figures for one evaluated model across game evaluation modes and rounds.

Consumes the decision-event CSVs written by :mod:`conflictxgame.games.evaluate_games`
(``mcq``, ``open`` and ``play`` modes) plus the filtered scenario CSV (for the
menus and payoffs), and writes a small set of figures and the tables behind
them so the same model can later be lined up against its dyadic (solo) run:

* ``value_share_by_condition`` -- share of decisions that chose each value, per
  evaluation condition (MCQ choice, open-ended, play by round).
* ``bradley_terry_by_condition`` -- Bradley-Terry abilities with bootstrap CIs
  per condition, including play split by scripted opponent.
* ``pairwise_preference_by_condition`` -- for every value pair, the probability
  the higher-ranked value wins the menu, with Wilson CIs, per condition
  (MCQ choice, MCQ Likert, open-ended, play by round).
* ``pairwise_preference_over_rounds`` -- the same probabilities round by round
  for each scripted opponent, against the MCQ and open-ended references.
* ``play_dynamics_over_rounds`` -- cooperation rate and own payoff by round and
  opponent (the game-theoretic summary).
* ``cross_condition_agreement`` -- how often the same seat picked the same
  action under two conditions.

Value colors are fixed per value (never per rank) so every figure reads the
same way. Bradley-Terry fitting reuses :mod:`conflictxgame.games.analyze_games`.
"""

import argparse
import itertools
import json
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from conflictxgame.common.config import add_config_argument, parse_with_config
from conflictxgame.common.model_wrappers import short_model_name
from conflictxgame.games.analyze_games import (
    fit_bradley_terry,
    load_results,
    load_specs,
    wilson_interval,
)
from conflictxgame.games.game_structures import GameSpec, cooperative_actions

logger = logging.getLogger(__name__)

# Categorical slots (validated palette, light surface). Values get fixed slots;
# scripted opponents use the same slots in the one figure where no value is
# plotted.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
VALUE_SLOTS = {"honesty": "#2a78d6", "harmlessness": "#1baf7a", "helpfulness": "#eb6834"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
TEXT_MUTED = "#8a8985"
GRID = "#e8e7e3"
REFERENCE = "#9b9a95"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

OPPONENT_ORDER = [
    "scripted:tit_for_tat",
    "scripted:always_cooperate",
    "scripted:always_defect",
    "scripted:grim_trigger",
    "scripted:zd_extort",
]
MIN_OPPONENT_DECISIONS = 100


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def pretty(name: str) -> str:
    """Human label for a value or opponent id."""
    return name.replace("scripted:", "").replace("_", " ").capitalize()


def value_color(value: str, order: Sequence[str]) -> str:
    if value in VALUE_SLOTS:
        return VALUE_SLOTS[value]
    return SLOTS[list(order).index(value) % len(SLOTS)]


def _ordered_opponents(play: pd.DataFrame, minimum: int = 0) -> List[str]:
    present = set(play["opponent"]) - {""}
    ordered = [o for o in OPPONENT_ORDER if o in present]
    ordered += sorted(present - set(OPPONENT_ORDER))
    return [o for o in ordered if (play["opponent"] == o).sum() >= minimum]


# ---------------------------------------------------------------------- #
# Decision events
# ---------------------------------------------------------------------- #

def decision_events(data: pd.DataFrame, specs: Dict[str, GameSpec]) -> pd.DataFrame:
    """One row per valid decision with the seat's menu tags and the chosen tag."""
    data = data.copy()
    data['probe'] = data['probe'].fillna('') if 'probe' in data.columns else ''
    data = data[(data['mode'] != 'mcq') | (data['probe'] == '')]
    records = []
    for _, row in data.iterrows():
        spec = specs.get(str(row["scenario_id"]))
        if spec is None:
            continue
        try:
            seat = int(row["seat"])
            action = int(row["choice_action"])
        except (TypeError, ValueError):
            continue
        if seat >= spec.config.n_players or action >= spec.players[seat].arity:
            continue
        tags = [a.value_tag for a in spec.players[seat].actions]
        coop = cooperative_actions(spec)
        opponent = row.get("opponent_config")
        records.append(
            {
                "mode": row["mode"],
                "scenario_id": str(row["scenario_id"]),
                "seat": seat,
                "round": int(row.get("round", 0) or 0),
                "opponent": opponent if isinstance(opponent, str) else "",
                "action": action,
                "chosen": tags[action],
                "menu": tuple(tags),
                "cooperated": action == coop[seat],
                "coop_value": tags[coop[seat]],
                "payoff": row.get("round_payoff", np.nan),
            }
        )
    return pd.DataFrame(records)


def likert_events(data: pd.DataFrame, specs: Dict[str, GameSpec]) -> pd.DataFrame:
    """Pairwise value comparisons from per-action Likert ratings (ties dropped)."""
    data = data.copy()
    data['probe'] = data['probe'].fillna('') if 'probe' in data.columns else ''
    records = []
    subset = data[(data["mode"] == "mcq") & (data["probe"] == "")]
    for _, row in subset.iterrows():
        spec = specs.get(str(row["scenario_id"]))
        raw = row.get("likert_by_action")
        if spec is None or not isinstance(raw, str) or not raw.strip():
            continue
        try:
            likerts = {
                int(k): float(v)
                for k, v in json.loads(raw).items()
                if isinstance(v, (int, float))
            }
            seat = int(row["seat"])
        except (ValueError, TypeError):
            continue
        if seat >= spec.config.n_players:
            continue
        tags = [a.value_tag for a in spec.players[seat].actions]
        for a, b in itertools.combinations(sorted(likerts), 2):
            if a >= len(tags) or b >= len(tags):
                continue
            if tags[a] == tags[b] or likerts[a] == likerts[b]:
                continue
            winner, loser = (tags[a], tags[b]) if likerts[a] > likerts[b] else (tags[b], tags[a])
            records.append(
                {"scenario_id": str(row["scenario_id"]), "seat": seat, "winner": winner, "loser": loser}
            )
    return pd.DataFrame(records)


def comparisons_from_events(events: pd.DataFrame) -> List[tuple]:
    """(winner, loser) pairs: the chosen tag beats every other distinct tag in the menu."""
    out = []
    for chosen, menu in zip(events["chosen"], events["menu"]):
        for other in set(menu) - {chosen}:
            out.append((chosen, other))
    return out


def pair_rate(comparisons: Sequence[tuple], a: str, b: str) -> dict:
    """P(a beats b) with a Wilson interval among comparisons involving both."""
    wins = sum(1 for w, l in comparisons if (w, l) == (a, b))
    losses = sum(1 for w, l in comparisons if (w, l) == (b, a))
    n = wins + losses
    lo, hi = wilson_interval(wins, n)
    return {"n": n, "p": wins / n if n else np.nan, "ci_lower": lo, "ci_upper": hi}


# ---------------------------------------------------------------------- #
# Conditions
# ---------------------------------------------------------------------- #

def build_conditions(events: pd.DataFrame, likert: pd.DataFrame) -> Dict[str, List[tuple]]:
    """Ordered mapping condition label -> (winner, loser) comparisons."""
    conditions: Dict[str, List[tuple]] = {}
    mcq = events[events["mode"] == "mcq"]
    if len(mcq):
        conditions["MCQ choice"] = comparisons_from_events(mcq)
    if len(likert):
        conditions["MCQ Likert"] = list(zip(likert["winner"], likert["loser"]))
    opened = events[events["mode"] == "open"]
    if len(opened):
        conditions["Open-ended"] = comparisons_from_events(opened)
    play = events[events["mode"] == "play"]
    for rnd in sorted(play["round"].unique()):
        conditions[f"Play r{rnd + 1}"] = comparisons_from_events(play[play["round"] == rnd])
    return conditions


def opponent_conditions(events: pd.DataFrame) -> Dict[str, List[tuple]]:
    play = events[events["mode"] == "play"]
    out: Dict[str, List[tuple]] = {}
    if len(play):
        out["Play (all)"] = comparisons_from_events(play)
    for opp in _ordered_opponents(play):
        sub = play[play["opponent"] == opp]
        if len(sub) >= MIN_OPPONENT_DECISIONS:
            out[f"vs {pretty(opp)}"] = comparisons_from_events(sub)
        else:
            logger.info("Skipping opponent %s in per-opponent panels (%d decisions)", opp, len(sub))
    return out


def value_order(conditions: Dict[str, List[tuple]], n_bootstrap: int) -> List[str]:
    """Values ordered by the MCQ-choice Bradley-Terry ability (falls back to any condition)."""
    for label in ("MCQ choice", "Open-ended", "Play r1"):
        comps = conditions.get(label)
        if comps:
            bt = fit_bradley_terry(comps, n_bootstrap=min(n_bootstrap, 20))
            if bt is not None:
                return list(bt["value"])
    return sorted({v for comps in conditions.values() for pair in comps for v in pair})


# ---------------------------------------------------------------------- #
# Figures
# ---------------------------------------------------------------------- #

def _finish(fig, output_dir: str, name: str) -> str:
    import matplotlib.pyplot as plt

    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_value_share(
    events: pd.DataFrame, order: List[str], model: str, output_dir: str
) -> pd.DataFrame:
    """100% stacked bars: share of decisions choosing each value, per condition."""
    import matplotlib.pyplot as plt

    frames = []
    mcq = events[events["mode"] == "mcq"]
    if len(mcq):
        frames.append(("MCQ choice", mcq["chosen"]))
    opened = events[events["mode"] == "open"]
    if len(opened):
        frames.append(("Open-ended", opened["chosen"]))
    play = events[events["mode"] == "play"]
    for rnd in sorted(play["round"].unique()):
        frames.append((f"Play r{rnd + 1}", play[play["round"] == rnd]["chosen"]))
    rows = []
    for label, chosen in frames:
        counts = chosen.value_counts()
        n = int(counts.sum())
        rec = {"condition": label, "n_decisions": n}
        for v in order:
            rec[f"share_{v}"] = counts.get(v, 0) / n if n else np.nan
        rows.append(rec)
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(table) + 1.6))
    ax.grid(False)
    y = np.arange(len(table))[::-1]
    left = np.zeros(len(table))
    for v in order:
        share = table[f"share_{v}"].to_numpy()
        ax.barh(
            y, share, left=left, color=value_color(v, order), height=0.62,
            label=pretty(v), edgecolor=SURFACE, linewidth=2,
        )
        for yi, l, s in zip(y, left, share):
            if s >= 0.08:
                ax.text(l + s / 2, yi, f"{s:.0%}", ha="center", va="center", fontsize=8.5, color="white")
        left += share
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  (n={n})" for c, n in zip(table["condition"], table["n_decisions"])])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of decisions whose chosen action favored the value")
    ax.set_title(f"{model}: value favored by the chosen action, per evaluation condition")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(order))
    _finish(fig, output_dir, "value_share_by_condition")
    return table


def plot_bradley_terry(
    conditions: Dict[str, List[tuple]], order: List[str], model: str, output_dir: str, n_bootstrap: int
) -> pd.DataFrame:
    """Dot-and-CI Bradley-Terry abilities per value across conditions."""
    import matplotlib.pyplot as plt

    frames = []
    for label, comps in conditions.items():
        bt = fit_bradley_terry(comps, n_bootstrap=n_bootstrap)
        if bt is None:
            continue
        bt.insert(0, "condition", label)
        bt["n_comparisons"] = len(comps)
        frames.append(bt)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if table.empty:
        return table

    labels = list(dict.fromkeys(table["condition"]))
    x = np.arange(len(labels))
    width = 0.8 / max(len(order), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(labels) + 2), 4.6))
    ax.axhline(0, color=REFERENCE, linewidth=0.8, zorder=1)
    for i, v in enumerate(order):
        sub = table[table["value"] == v].set_index("condition").reindex(labels)
        xs = x + (i - (len(order) - 1) / 2) * width
        ability = sub["ability"].to_numpy(dtype=float)
        # Percentile CIs from few or one-sided resamples can miss the point
        # estimate; never draw a negative bar.
        lower = np.clip(ability - sub["ci_lower"].to_numpy(dtype=float), 0, None)
        upper = np.clip(sub["ci_upper"].to_numpy(dtype=float) - ability, 0, None)
        ax.errorbar(
            xs, ability, yerr=np.vstack([lower, upper]),
            fmt=MARKERS[i], color=value_color(v, order), markersize=6, capsize=3,
            elinewidth=1.4, markeredgecolor=SURFACE, markeredgewidth=1, label=pretty(v), zorder=3,
        )
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Bradley-Terry ability (95% bootstrap CI)")
    ax.set_title(f"{model}: fitted value priorities by evaluation condition")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=len(order))
    _finish(fig, output_dir, "bradley_terry_by_condition")
    return table


def _pairs(order: List[str]) -> List[tuple]:
    return list(itertools.combinations(order, 2))


def plot_pairwise_by_condition(
    conditions: Dict[str, List[tuple]], order: List[str], model: str, output_dir: str
) -> pd.DataFrame:
    """Small multiples, one per value pair: P(higher-ranked value wins) across conditions."""
    import matplotlib.pyplot as plt

    pairs = _pairs(order)
    labels = list(conditions)
    rows = []
    for a, b in pairs:
        for label in labels:
            rec = pair_rate(conditions[label], a, b)
            rec.update({"condition": label, "value_a": a, "value_b": b})
            rows.append(rec)
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(labels))
    for ax, (a, b) in zip(axes, pairs):
        sub = (
            table[(table["value_a"] == a) & (table["value_b"] == b)]
            .set_index("condition").reindex(labels)
        )
        color = value_color(a, order)
        ax.axhline(0.5, color=REFERENCE, linewidth=0.8, zorder=1)
        ax.fill_between(x, sub["ci_lower"], sub["ci_upper"], color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(
            x, sub["p"], color=color, linewidth=1.6, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1, zorder=3,
        )
        ax.set_title(f"{pretty(a)} over {pretty(b)}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        n_min, n_max = int(sub["n"].min()), int(sub["n"].max())
        ax.text(
            0.02, 0.03, f"n per point: {n_min}-{n_max}", transform=ax.transAxes,
            fontsize=8, color=TEXT_MUTED,
        )
    axes[0].set_ylabel("P(first value chosen) with Wilson 95% CI")
    fig.suptitle(f"{model}: pairwise value preference by evaluation condition", fontweight="semibold", y=1.02)
    _finish(fig, output_dir, "pairwise_preference_by_condition")
    return table


def plot_pairwise_over_rounds(
    events: pd.DataFrame, conditions: Dict[str, List[tuple]], order: List[str], model: str, output_dir: str
) -> pd.DataFrame:
    """Grid rows=value pairs, cols=opponents: P(first value wins) by round, vs MCQ/open references."""
    import matplotlib.pyplot as plt

    play = events[events["mode"] == "play"]
    if play.empty:
        return pd.DataFrame()
    opponents = _ordered_opponents(play, MIN_OPPONENT_DECISIONS)
    rounds = sorted(play["round"].unique())
    pairs = _pairs(order)
    rows = []
    for opp in opponents:
        for rnd in rounds:
            comps = comparisons_from_events(play[(play["opponent"] == opp) & (play["round"] == rnd)])
            for a, b in pairs:
                rec = pair_rate(comps, a, b)
                rec.update({"opponent": opp, "round": rnd + 1, "value_a": a, "value_b": b})
                rows.append(rec)
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    refs = {}
    for label, style in (("MCQ choice", "--"), ("Open-ended", ":")):
        if label in conditions:
            refs[label] = (style, {pair: pair_rate(conditions[label], *pair)["p"] for pair in pairs})

    fig, axes = plt.subplots(
        len(pairs), len(opponents), figsize=(3.1 * len(opponents), 2.6 * len(pairs)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, (a, b) in enumerate(pairs):
        color = value_color(a, order)
        for j, opp in enumerate(opponents):
            ax = axes[i, j]
            sub = table[
                (table["opponent"] == opp) & (table["value_a"] == a) & (table["value_b"] == b)
            ].sort_values("round")
            ax.axhline(0.5, color=REFERENCE, linewidth=0.8, zorder=1)
            for label, (style, values) in refs.items():
                ax.axhline(values[(a, b)], color=TEXT_MUTED, linestyle=style, linewidth=1.1, zorder=2, label=label)
            ax.fill_between(sub["round"], sub["ci_lower"], sub["ci_upper"], color=color, alpha=0.15, linewidth=0, zorder=2)
            ax.plot(
                sub["round"], sub["p"], color=color, linewidth=1.6, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1, zorder=3, label="Play",
            )
            if i == 0:
                ax.set_title(f"vs {pretty(opp)}")
            if j == 0:
                ax.set_ylabel(f"{pretty(a)}\nover {pretty(b)}", color=TEXT)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0%", "50%", "100%"])
            ax.set_xticks(sub["round"])
            n = int(sub["n"].iloc[0]) if len(sub) else 0
            ax.text(0.03, 0.05, f"n={n}/round", transform=ax.transAxes, fontsize=7.5, color=TEXT_MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("Round")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels))
    fig.suptitle(
        f"{model}: pairwise value preference over rounds, by scripted opponent (mem-1)",
        fontweight="semibold", y=1.01,
    )
    fig.tight_layout()
    _finish(fig, output_dir, "pairwise_preference_over_rounds")
    return table


def plot_play_dynamics(events: pd.DataFrame, model: str, output_dir: str) -> pd.DataFrame:
    """Cooperation rate and own payoff by round, one line per scripted opponent."""
    import matplotlib.pyplot as plt

    play = events[events["mode"] == "play"]
    if play.empty:
        return pd.DataFrame()
    opponents = _ordered_opponents(play)
    rows = []
    for opp in opponents:
        for rnd, grp in play[play["opponent"] == opp].groupby("round"):
            lo, hi = wilson_interval(int(grp["cooperated"].sum()), len(grp))
            rows.append(
                {
                    "opponent": opp, "round": rnd + 1, "n": len(grp),
                    "cooperation_rate": grp["cooperated"].mean(), "ci_lower": lo, "ci_upper": hi,
                    "mean_payoff": pd.to_numeric(grp["payoff"], errors="coerce").mean(),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for k, opp in enumerate(opponents):
        sub = table[table["opponent"] == opp].sort_values("round")
        thin = int(sub["n"].sum()) < MIN_OPPONENT_DECISIONS
        kw = dict(
            color=SLOTS[k % len(SLOTS)], marker=MARKERS[k % len(MARKERS)], markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1, linewidth=1.6,
            linestyle="--" if thin else "-", alpha=0.7 if thin else 1,
        )
        label = f"{pretty(opp)} (n={int(sub['n'].sum())})"
        ax1.plot(sub["round"], sub["cooperation_rate"], label=label, **kw)
        ax2.plot(sub["round"], sub["mean_payoff"], label=label, **kw)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax1.set_title("Cooperation rate (played the welfare-maximizing action)")
    ax2.set_title("Mean own payoff per round")
    for ax in (ax1, ax2):
        ax.set_xlabel("Round")
        ax.set_xticks(sorted(table["round"].unique()))
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(3, len(labels)))
    fig.suptitle(
        f"{model}: live play against scripted opponents (dashed = too few decisions)",
        fontweight="semibold", y=1.02,
    )
    fig.tight_layout()
    _finish(fig, output_dir, "play_dynamics_over_rounds")
    return table


def plot_agreement(events: pd.DataFrame, model: str, output_dir: str) -> pd.DataFrame:
    """Heatmap: share of (scenario, seat) decisions where two conditions picked the same action."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    columns: Dict[str, pd.DataFrame] = {}
    mcq = events[events["mode"] == "mcq"]
    if len(mcq):
        columns["MCQ choice"] = mcq[["scenario_id", "seat", "action"]]
    opened = events[events["mode"] == "open"]
    if len(opened):
        columns["Open-ended"] = opened[["scenario_id", "seat", "action"]]
    play = events[events["mode"] == "play"]
    if len(play):
        rounds = sorted(play["round"].unique())
        for rnd in dict.fromkeys((rounds[0], rounds[-1])):
            # Play has one decision per opponent; agreement with a static
            # condition is averaged over opponents for that seat.
            sub = play[play["round"] == rnd]
            columns[f"Play r{rnd + 1}"] = sub[["scenario_id", "seat", "opponent", "action"]]
    labels = list(columns)
    if len(labels) < 2:
        return pd.DataFrame()

    def agreement(a: str, b: str) -> tuple:
        fa = columns[a].rename(columns={"action": "a"})
        fb = columns[b].rename(columns={"action": "b"})
        keys = [k for k in ("scenario_id", "seat", "opponent") if k in fa.columns and k in fb.columns]
        merged = fa.merge(fb, on=keys)
        if merged.empty:
            return np.nan, 0
        return float((merged["a"] == merged["b"]).mean()), len(merged)

    matrix = np.full((len(labels), len(labels)), np.nan)
    rows = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            rate, n = (1.0, 0) if i == j else agreement(a, b)
            matrix[i, j] = rate
            if i < j:
                rows.append({"condition_a": a, "condition_b": b, "agreement": rate, "n_pairs": n})
    table = pd.DataFrame(rows)

    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 1.1 * len(labels) + 1.2))
    ax.grid(False)
    ax.imshow(matrix, cmap=cmap, vmin=0.5, vmax=1.0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color="white" if val > 0.8 else TEXT)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(f"{model}: same seat, same action? (agreement between conditions)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _finish(fig, output_dir, "cross_condition_agreement")
    return table


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

def run(
    results_dir: str, scenarios_dir: str, output_dir: str,
    model: Optional[str] = None, n_bootstrap: int = 200,
    results_exclude: Optional[Sequence[str]] = None,
    results_arm: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Produce every figure and table for one model; returns the tables."""
    _style()
    data = load_results(results_dir, exclude=results_exclude, arm=results_arm)
    specs = load_specs(scenarios_dir)
    if data.empty or not specs:
        raise SystemExit("No results or no scenario specs found; nothing to plot.")
    data["probe"] = data["probe"].fillna("") if "probe" in data.columns else ""
    models = sorted(data["model"].dropna().unique())
    if model is not None:
        data = data[data["model"] == model]
        if data.empty:
            raise SystemExit(f"No rows for model {model!r}; available: {models}")
    elif len(models) > 1:
        raise SystemExit(f"Results contain several models; pick one with --model: {models}")
    model = model or models[0]
    name = short_model_name(model)
    os.makedirs(output_dir, exist_ok=True)

    events = decision_events(data[(data["mode"] != "mcq") | (data["probe"] == "")], specs)
    likert = likert_events(data, specs)
    conditions = build_conditions(events, likert)
    by_opponent = opponent_conditions(events)
    order = value_order(conditions, n_bootstrap)
    logger.info("Value order (by MCQ Bradley-Terry): %s", order)

    bt_conditions = {k: v for k, v in conditions.items() if k != "MCQ Likert"}
    bt_conditions.update(by_opponent)
    tables = {
        "value_share_by_condition": plot_value_share(events, order, name, output_dir),
        "bradley_terry_by_condition": plot_bradley_terry(bt_conditions, order, name, output_dir, n_bootstrap),
        "pairwise_preference_by_condition": plot_pairwise_by_condition(conditions, order, name, output_dir),
        "pairwise_preference_over_rounds": plot_pairwise_over_rounds(events, conditions, order, name, output_dir),
        "play_dynamics_over_rounds": plot_play_dynamics(events, name, output_dir),
        "cross_condition_agreement": plot_agreement(events, name, output_dir),
    }
    for key, table in tables.items():
        if table is not None and not table.empty:
            table.insert(0, "model", model)
            table.to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)
    return tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_config_argument(parser)
    parser.add_argument("--results-dir", required=True, help="Directory of evaluate_games CSVs")
    parser.add_argument(
        "--scenarios-dir", required=True, help="Directory of filtered scenario CSVs (game_spec column)"
    )
    parser.add_argument(
        '--results-arm', default=None,
        help="Select one result arm ('baseline' selects the empty arm token).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=None, help="Model id to plot when the results dir holds several")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument(
        "--results-exclude", nargs="*", default=[],
        help="Skip result files whose name contains any of these substrings. "
             "Use it to keep a separately-run arm (stakes, visibility, memory "
             "states) out of the baseline conditions it would otherwise pool into.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args, _config_defaults, _explicit_cli = parse_with_config(build_parser(), argv)
    return args


def main(argv: Optional[Sequence[str]] = None):
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    run(args.results_dir, args.scenarios_dir, args.output_dir, args.model,
        args.n_bootstrap, args.results_exclude, args.results_arm)


if __name__ == "__main__":
    main()
