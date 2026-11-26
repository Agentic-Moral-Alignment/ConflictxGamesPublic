"""Generate the body and appendix figures for the paper.

Layout note (2026-09-05). The body figures are laid out as a 2x2 grid
(``stakes_by_action_representation``) and a two-row grid with the history
heatmap spanning the bottom row (``context_modifiers``); legends sit directly
under the panels via constrained layout, so there is no blank band between
axes and legend.

Presentation conventions, shared by every panel this script draws:

* color encodes the action representation everywhere;
* marker shape follows the representation too (circle menu, square
  open--undisclosed, triangle open--disclosed), so one legend serves every
  panel; the x axis already names the stake or outcome;
* in the stake-effect panels a *filled* marker means the pooled effect survives
  Holm correction within its family (``p_holm < 0.05``), a hollow one means it
  does not (``FILL_HOLM_SURVIVORS`` switches this off);
* the lead result (the payoff column of the helpfulness panel) is shaded and
  annotated with the two corrected interaction p-values read from stats.json.

The ongoing-framing null is written separately as ``ongoing_framing_appendix``.

All plotted estimates and intervals are read from a caller-supplied stats.json.
There is deliberately no default analysis path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


REPRESENTATIONS = ("menu", "open_undisclosed", "open_disclosed")
REP_LABELS = {
    "menu": "Menu; counterpart shown",
    "open_undisclosed": "Open; counterpart undisclosed",
    "open_disclosed": "Open; counterpart disclosed",
}
SHORT_REP_LABELS = {
    "menu": "Menu",
    "open_undisclosed": "Open–undisclosed",
    "open_disclosed": "Open–disclosed",
}
VALUES = ("harmlessness", "helpfulness", "honesty")
VALUE_LABELS = ("Harmlessness", "Helpfulness", "Honesty")
# All three value names begin with "h", so single-letter axis labels would be
# ambiguous; the appendix panels use these three-letter forms instead.
VALUE_ABBR = {"harmlessness": "Harm", "helpfulness": "Help", "honesty": "Hon"}
STAKES = ("payoff", "alignment", "usefulness")
STAKE_LABELS = ("Payoff", "Alignment", "Usefulness")
STATES = ("fresh", "none", "one", "both")
STATE_LABELS = ("Fresh", "Neither", "One", "Both")

# Okabe--Ito-derived colors and representation-specific markers keep the plots
# interpretable when printed in grayscale.  Data points are intentionally not
# connected: the x positions are discrete experimental conditions, not a
# continuous trajectory.
COLORS = {
    "menu": "#0072B2",
    "open_undisclosed": "#D55E00",
    "open_disclosed": "#009E73",
}
REP_MARKERS = {"menu": "o", "open_undisclosed": "s", "open_disclosed": "^"}
STAKE_MARKERS = {"payoff": "o", "alignment": "s", "usefulness": "^"}

# Marker fill encodes "survives Holm correction" in the body effect panels.
FILL_HOLM_SURVIVORS = True
HOLM_ALPHA = 0.05

# Typography for the body figures. The 2x2 grid gives each panel roughly twice
# the width of the old 1x4 strip, so titles and ticks can sit at readable sizes
# and x labels no longer need to rotate.
BODY_TITLE_SIZE = 9.0
BODY_LABEL_SIZE = 8.0
BODY_TICK_SIZE = 7.6
BODY_LEGEND_SIZE = 8.0
BODY_MARKER_SIZE = 5.0


def _records(stats: Mapping[str, Any], key: str) -> pd.DataFrame:
    if key not in stats or not isinstance(stats[key], list):
        raise ValueError(f"stats.json lacks list-valued {key!r}")
    frame = pd.DataFrame(stats[key])
    if frame.empty:
        raise ValueError(f"stats.json contains no records in {key!r}")
    return frame


def _one(frame: pd.DataFrame, **filters: str) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one record for {filters}, found {len(selected)}")
    row = selected.iloc[0]
    for column in ("ci_lower", "ci_upper"):
        if column in row and not math.isfinite(float(row[column])):
            raise ValueError(f"Non-finite {column} for {filters}")
    return row


def _survives_holm(row: pd.Series) -> bool:
    if "p_holm" not in row.index:
        return False
    p_holm = float(row["p_holm"])
    return math.isfinite(p_holm) and p_holm < HOLM_ALPHA


def _errorbar(
    ax,
    x: float,
    row: pd.Series,
    *,
    representation: str,
    marker: str | None = None,
    filled: bool | None = None,
    markersize: float = 4.5,
):
    estimate_key = "mean_rate" if "mean_rate" in row.index else "pooled_delta"
    estimate = float(row[estimate_key])
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    if filled is None:
        filled = FILL_HOLM_SURVIVORS and _survives_holm(row)
    ax.errorbar(
        x,
        estimate,
        yerr=np.array([[estimate - lower], [upper - estimate]]),
        color=COLORS[representation],
        marker=marker or REP_MARKERS[representation],
        markerfacecolor=COLORS[representation] if filled else "white",
        markeredgewidth=1.0,
        markersize=markersize,
        linestyle="none",
        linewidth=1.1,
        capsize=2.0,
        zorder=4,
    )


def _format_effect_axis(
    ax,
    title: str,
    x_labels: Sequence[str],
    *,
    title_size: float = BODY_TITLE_SIZE,
    label_size: float = BODY_LABEL_SIZE,
    tick_size: float = BODY_TICK_SIZE,
) -> None:
    ax.axhline(0.0, color="0.45", linewidth=0.7, zorder=0)
    ax.set_xticks(np.arange(len(x_labels)), x_labels)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylabel("Change in win rate", fontsize=label_size)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=title_size)
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=tick_size)


def _representation_handles(markersize: float = BODY_MARKER_SIZE) -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [],
            [],
            color=COLORS[representation],
            marker=REP_MARKERS[representation],
            linestyle="none",
            markerfacecolor="white",
            markersize=markersize,
            linewidth=1.2,
            label=SHORT_REP_LABELS[representation],
        )
        for representation in REPRESENTATIONS
    ]


def _holm_handles(markersize: float = BODY_MARKER_SIZE) -> list[mlines.Line2D]:
    """Legend entries explaining the filled/hollow marker convention."""
    if not FILL_HOLM_SURVIVORS:
        return []
    return [
        mlines.Line2D(
            [], [], color="0.35", marker="o", linestyle="none",
            markerfacecolor="0.35", markersize=markersize,
            label=f"filled: survives Holm (p < {HOLM_ALPHA:g})",
        ),
        mlines.Line2D(
            [], [], color="0.35", marker="o", linestyle="none",
            markerfacecolor="white", markersize=markersize,
            label="hollow: does not",
        ),
    ]


def _outside_legend(fig, handles: Sequence[mlines.Line2D], *, ncol: int, fontsize: float) -> None:
    """Legend directly under the panels.

    With constrained layout, ``loc="outside lower center"`` reserves exactly the
    legend's height below the axes, which removes the blank band that a
    ``bbox_to_anchor`` legend plus ``tight_layout(rect=...)`` used to leave.
    """
    fig.legend(
        handles=list(handles),
        loc="outside lower center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handletextpad=0.5,
        columnspacing=1.6,
    )


def make_stakes_figure(stats: Mapping[str, Any], output_dir: str) -> None:
    levels = _records(stats, "representation_levels")
    effects = _records(stats, "stake_effects")
    interactions = _records(stats, "stake_representation_interactions")
    # 2x2 grid: A baseline levels, B-D the stake effects per outcome. Each
    # panel gets about twice the width of the former 1x4 strip.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), layout="constrained")
    offsets = {"menu": -0.2, "open_undisclosed": 0.0, "open_disclosed": 0.2}

    ax = axes[0, 0]
    for representation in REPRESENTATIONS:
        for value_index, value in enumerate(VALUES):
            row = _one(levels, representation=representation, value=value)
            x = value_index + offsets[representation]
            per_model = row.get("per_model")
            if not isinstance(per_model, list) or len(per_model) != 10:
                raise ValueError(f"Baseline record lacks ten per-model rates: {representation}, {value}")
            ax.scatter(
                np.full(len(per_model), x),
                [float(item["rate"]) for item in per_model],
                s=10,
                marker=REP_MARKERS[representation],
                facecolor=COLORS[representation],
                edgecolor="none",
                alpha=0.22,
                zorder=2,
            )
            _errorbar(ax, x, row, representation=representation, filled=False, markersize=BODY_MARKER_SIZE)
    ax.set_xticks(np.arange(3), VALUE_LABELS)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Baseline win rate", fontsize=BODY_LABEL_SIZE)
    ax.set_title("A  Baseline HHH win rates", loc="left", fontweight="bold", fontsize=BODY_TITLE_SIZE)
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=BODY_TICK_SIZE)

    for panel_index, (ax, value) in enumerate(zip(axes.flat[1:], VALUES), start=1):
        for representation in REPRESENTATIONS:
            for stake_index, stake in enumerate(STAKES):
                row = _one(
                    effects,
                    representation=representation,
                    stake=stake,
                    value=value,
                )
                x = stake_index + offsets[representation]
                # Marker shape follows the representation (as in the legend);
                # the x axis already names the stake.
                _errorbar(ax, x, row, representation=representation, markersize=BODY_MARKER_SIZE)
        letter = chr(ord("A") + panel_index)
        _format_effect_axis(ax, f"{letter}  {value.title()}", STAKE_LABELS)

        if value == "helpfulness":
            # The lead result: the payoff column of the helpfulness panel is
            # where the menu effect and the two open effects differ, the only
            # corrected interactions in the 27-test family.
            p_values = [
                float(
                    _one(
                        interactions,
                        representation_a="menu",
                        representation_b=representation,
                        stake="payoff",
                        value="helpfulness",
                    )["p_holm"]
                )
                for representation in ("open_undisclosed", "open_disclosed")
            ]
            ax.axvspan(-0.5, 0.5, color="0.93", zorder=0)
            # The band's lower half is empty (no payoff estimate or interval
            # reaches below about -0.04), so the note sits there.
            ax.text(
                0.0,
                0.03,
                "only corrected\ninteractions\n"
                + f"Holm p = {p_values[0]:.3f}\nand {p_values[1]:.3f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=6.0,
                color="0.25",
                linespacing=1.15,
                zorder=5,
            )

    _outside_legend(
        fig,
        _representation_handles() + _holm_handles(),
        ncol=5 if FILL_HOLM_SURVIVORS else 3,
        fontsize=BODY_LEGEND_SIZE,
    )
    _save(fig, output_dir, "stakes_by_action_representation")


def make_context_figure(stats: Mapping[str, Any], output_dir: str) -> None:
    visibility = _records(stats, "visibility_effects")
    steering = _records(stats, "steering_effects")
    history = _records(stats, "history_stake_effects")
    ongoing = _records(stats, "ongoing_framing")
    ordering = stats.get("ordering_stability")
    if not isinstance(ordering, dict):
        raise ValueError("stats.json lacks ordering_stability")

    # Two rows: A and B side by side, the history heatmap C across the full
    # width below them. The ongoing-framing null lives in the appendix
    # (make_ongoing_appendix).
    fig = plt.figure(figsize=(7.2, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    # Bottom row: two heatmaps of the same shape (harmlessness, helpfulness)
    # sharing row labels and one color scale.
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    offsets = {"menu": -0.2, "open_undisclosed": 0.0, "open_disclosed": 0.2}

    ax = axes[0]
    for representation in ("menu", "open_undisclosed"):
        for value_index, value in enumerate(VALUES):
            row = _one(visibility, representation=representation, value=value)
            x = value_index + offsets[representation]
            _errorbar(ax, x, row, representation=representation, markersize=BODY_MARKER_SIZE)
    _format_effect_axis(ax, "A  Observed minus unobserved", VALUE_LABELS, tick_size=BODY_TICK_SIZE - 0.5)

    ax = axes[1]
    for representation in REPRESENTATIONS:
        for value_index, value in enumerate(VALUES):
            row = _one(steering, representation=representation, value=value)
            x = value_index + offsets[representation]
            _errorbar(ax, x, row, representation=representation, markersize=BODY_MARKER_SIZE)
    flip_text = " / ".join(
        f"{int(ordering[representation]['steering']['n_flipped'])}/10"
        for representation in REPRESENTATIONS
    )
    ax.text(
        0.02,
        0.02,
        "Ordering flips (Menu / Open–U / Open–D)\n" + flip_text,
        transform=ax.transAxes,
        fontsize=6.0,
        va="bottom",
        linespacing=1.15,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
    )
    _format_effect_axis(ax, "B  HHH steering minus baseline", VALUE_LABELS, tick_size=BODY_TICK_SIZE - 0.5)

    # Compact heatmaps separate the three dimensions that otherwise produce
    # 36 overlapping points per outcome in a body-sized panel.  The appendix
    # retains every per-model point and pooled interval.  Rows are grouped by
    # stake so the three representations of one stake sit together.  Panel C
    # is harmlessness, panel D helpfulness (the lead claim's outcome); both
    # share row labels and one symmetric color scale.
    history_rows = [(representation, stake) for stake in STAKES for representation in REPRESENTATIONS]

    def history_matrix_for(value: str) -> np.ndarray:
        return np.asarray(
            [
                [
                    float(
                        _one(
                            history,
                            representation=representation,
                            stake=stake,
                            state=state,
                            value=value,
                        )["pooled_delta"]
                    )
                    for state in STATES
                ]
                for representation, stake in history_rows
            ]
        )

    heatmap_values = ("harmlessness", "helpfulness")
    matrices = {value: history_matrix_for(value) for value in heatmap_values}
    color_limit = max(
        0.01,
        math.ceil(max(float(np.max(np.abs(matrix))) for matrix in matrices.values()) * 100.0) / 100.0,
    )
    # Row labels name the stake and the representation; the representation
    # uses the short form (Menu / Open–U / Open–D) so the label column stays
    # narrow enough for the panels to keep readable x labels.
    row_rep_labels = {"menu": "Menu", "open_undisclosed": "Open–U", "open_disclosed": "Open–D"}
    row_labels = [f"{stake.title()} · {row_rep_labels[representation]}" for representation, stake in history_rows]

    image = None
    for ax, letter, value in zip(axes[2:], ("C", "D"), heatmap_values):
        matrix = matrices[value]
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit),
        )
        ax.set_xticks(np.arange(len(STATES)), STATE_LABELS)
        ax.set_yticks(np.arange(len(history_rows)), row_labels)
        ax.set_xticks(np.arange(-0.5, len(STATES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(history_rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="x", labelsize=BODY_TICK_SIZE)
        ax.tick_params(axis="y", labelsize=BODY_TICK_SIZE - 0.6)
        if letter != "C":
            # The right-hand heatmap shares the row labels of the left one.
            ax.tick_params(axis="y", labelleft=False)
        # Thin separators between the three stake groups.
        for boundary in range(len(REPRESENTATIONS), len(history_rows), len(REPRESENTATIONS)):
            ax.axhline(boundary - 0.5, color="white", linewidth=3.0, zorder=3)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                cell = matrix[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{cell:+.02f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="white" if abs(cell) > 0.57 * color_limit else "black",
                )
        ax.set_title(
            f"{letter}  History-conditioned {value}",
            loc="left",
            fontweight="bold",
            fontsize=BODY_TITLE_SIZE,
        )
    colorbar = fig.colorbar(image, ax=axes[2:], fraction=0.02, pad=0.01)
    colorbar.set_label("Change in win rate", fontsize=BODY_LABEL_SIZE - 0.6)
    colorbar.ax.tick_params(labelsize=BODY_TICK_SIZE - 0.6)

    _outside_legend(
        fig,
        _representation_handles() + _holm_handles(),
        ncol=5 if FILL_HOLM_SURVIVORS else 3,
        fontsize=BODY_LEGEND_SIZE,
    )
    _save(fig, output_dir, "context_modifiers")

    make_ongoing_appendix(ongoing, output_dir)


def make_headline_figure(stats: Mapping[str, Any], output_dir: str) -> None:
    """The headline figure (Figure 1): alignment criterion by history.

    Replaces the earlier ``make_figure_revised.py`` output, which collapsed the
    design to two conditions (forced choice / free text).  That predated the
    split of free text into counterpart-undisclosed and counterpart-disclosed
    arms, so it disagreed with the three-way structure every other figure and
    the body text use.  Drawn three-way from the same stats.json as the rest
    of the figures.

    This was three panels.  Its A and B plotted exactly what
    ``stakes_by_action_representation`` (Figure 2) already shows in its own A
    and B -- same statistics, same style, one page apart -- so the duplicated
    pair was dropped for the page budget and the history panel, the only one
    unique to this figure, now stands alone.  The body cites this figure only
    as panel (c); Figure 2 carries the baseline levels and the one-shot
    criterion shifts.
    """
    history = _records(stats, "history_stake_effects")

    fig, ax_history = plt.subplots(1, 1, figsize=(4.2, 2.8), layout="constrained")
    offsets = {"menu": -0.2, "open_undisclosed": 0.0, "open_disclosed": 0.2}

    # The alignment criterion's helpfulness effect by remembered history.
    ax = ax_history
    for representation in REPRESENTATIONS:
        deltas, lows, highs = [], [], []
        for state in STATES:
            row = _one(
                history,
                representation=representation,
                stake="alignment",
                state=state,
                value="helpfulness",
            )
            deltas.append(float(row["pooled_delta"]))
            lows.append(float(row["ci_lower"]))
            highs.append(float(row["ci_upper"]))
        x = np.arange(len(STATES)) + offsets[representation]
        ax.errorbar(
            x,
            deltas,
            yerr=np.array([np.array(deltas) - np.array(lows), np.array(highs) - np.array(deltas)]),
            color=COLORS[representation],
            marker=REP_MARKERS[representation],
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=BODY_MARKER_SIZE,
            linestyle="none",
            linewidth=1.1,
            capsize=2.0,
            zorder=4,
        )
    _format_effect_axis(ax, "Alignment criterion by history", STATE_LABELS)

    _outside_legend(fig, _representation_handles(), ncol=3, fontsize=BODY_LEGEND_SIZE)
    _save(fig, output_dir, "context_stakes_revised")


def make_ongoing_appendix(ongoing: Sequence[Mapping[str, Any]], output_dir: str) -> None:
    """The ongoing-framing null, moved out of the body figure for space.

    Reported in App. F alongside the horizon decomposition; the panel itself is
    kept because the null is easier to read as a plot than as three rate pairs.
    """
    offsets = {"menu": -0.2, "open_undisclosed": 0.0, "open_disclosed": 0.2}
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.8), layout="constrained")
    for representation in REPRESENTATIONS:
        for value_index, value in enumerate(VALUES):
            row = _one(ongoing, representation=representation, value=value)
            x = value_index + offsets[representation]
            _errorbar(ax, x, row, representation=representation, filled=False, markersize=BODY_MARKER_SIZE)
    _format_effect_axis(ax, "Ongoing framing minus one-shot", VALUE_LABELS)
    _outside_legend(fig, _representation_handles(), ncol=3, fontsize=BODY_LEGEND_SIZE)
    _save(fig, output_dir, "ongoing_framing_appendix")


def _appendix_style(ax, title: str, labels: Sequence[str], *, rate: bool = False) -> None:
    if not rate:
        ax.axhline(0.0, color="0.45", linewidth=0.65)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20 if len(labels) > 4 else 0)
    ax.set_ylabel("Win rate" if rate else "Change in win rate")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(axis="y", color="0.91", linewidth=0.45)
    ax.tick_params(labelsize=7)


def make_per_model_appendix(stats: Mapping[str, Any], output_dir: str) -> None:
    """Multipage appendix covering every per-model non-history result."""
    levels = _records(stats, "representation_levels")
    stakes = _records(stats, "stake_effects")
    interactions = _records(stats, "stake_representation_interactions")
    secondary = {
        "Steering": _records(stats, "steering_effects"),
        "Visibility": _records(stats, "visibility_effects"),
        "Ongoing framing": _records(stats, "ongoing_framing"),
    }
    path = os.path.join(output_dir, "per_model_effects_appendix.pdf")
    os.makedirs(output_dir, exist_ok=True)
    # No per-model baseline-levels page: Table 3 in the appendix prints the same
    # ten checkpoints' rates exactly, and Figure 1A already plots them pooled.
    del levels
    with PdfPages(path, metadata={"Creator": "make_paper_figures.py", "CreationDate": None}) as pdf:
        for value in VALUES:
            fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.2), sharey=True)
            for ax, representation in zip(axes, REPRESENTATIONS):
                for model_index in range(10):
                    ys = [
                        float(
                            _one(stakes, representation=representation, stake=stake, value=value)[
                                "per_model"
                            ][model_index]["delta"]
                        )
                        for stake in STAKES
                    ]
                    ax.scatter(np.arange(3), ys, color="0.62", s=7, alpha=0.42)
                for stake_index, stake in enumerate(STAKES):
                    row = _one(stakes, representation=representation, stake=stake, value=value)
                    _errorbar(
                        ax,
                        stake_index,
                        row,
                        representation=representation,
                        marker=STAKE_MARKERS[stake],
                        filled=False,
                    )
                _appendix_style(ax, SHORT_REP_LABELS[representation], STAKE_LABELS)
            fig.suptitle(f"Per-model stake effects on {value} (gray)", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(11.2, 5.4), sharey=True)
        pair_labels = ["M→U", "M→D", "U→D"]
        for ax, value in zip(axes, VALUES):
            cells = [
                (pair, stake)
                for pair in (("menu", "open_undisclosed"), ("menu", "open_disclosed"),
                             ("open_undisclosed", "open_disclosed"))
                for stake in STAKES
            ]
            for model_index in range(10):
                ys = [
                    float(
                        _one(
                            interactions,
                            representation_a=pair[0],
                            representation_b=pair[1],
                            stake=stake,
                            value=value,
                        )["per_model"][model_index]["delta"]
                    )
                    for pair, stake in cells
                ]
                ax.scatter(np.arange(9), ys, color="0.65", s=6, alpha=0.38)
            for index, (pair, stake) in enumerate(cells):
                row = _one(
                    interactions,
                    representation_a=pair[0],
                    representation_b=pair[1],
                    stake=stake,
                    value=value,
                )
                _errorbar(ax, index, row, representation=pair[1], marker=STAKE_MARKERS[stake], filled=False)
            labels = [f"{pair_labels[i // 3]}\n{STAKE_LABELS[i % 3][0]}" for i in range(9)]
            _appendix_style(ax, value.title(), labels)
        fig.suptitle("Per-model stake × representation interactions (gray)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(11.2, 5.4), sharey=True)
        for ax, (family_name, frame) in zip(axes, secondary.items()):
            available_reps = (
                ("menu", "open_undisclosed") if family_name == "Visibility" else REPRESENTATIONS
            )
            cells = [(representation, value) for representation in available_reps for value in VALUES]
            for model_index in range(10):
                ys = [
                    float(
                        _one(frame, representation=representation, value=value)["per_model"][model_index][
                            "delta"
                        ]
                    )
                    for representation, value in cells
                ]
                ax.scatter(np.arange(len(cells)), ys, color="0.65", s=6, alpha=0.38)
            for index, (representation, value) in enumerate(cells):
                row = _one(frame, representation=representation, value=value)
                _errorbar(ax, index, row, representation=representation, filled=False)
            labels = [
                f"{('M' if r == 'menu' else 'U' if r == 'open_undisclosed' else 'D')}–{VALUE_ABBR[v]}"
                for r, v in cells
            ]
            _appendix_style(ax, family_name, labels)
        fig.suptitle("Per-model secondary effects (gray)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def make_history_appendix(
    stats: Mapping[str, Any], history_csv: str, output_dir: str
) -> None:
    """Three-page complete state grid with all ten per-model traces."""
    pooled = _records(stats, "history_stake_effects")
    grid = pd.read_csv(history_csv)
    required = {"representation", "stake", "state", "value", "model", "delta"}
    if not required <= set(grid.columns):
        raise ValueError(f"history_state_grid.csv lacks {sorted(required - set(grid.columns))}")
    path = os.path.join(output_dir, "history_state_grid_appendix.pdf")
    with PdfPages(path, metadata={"Creator": "make_paper_figures.py", "CreationDate": None}) as pdf:
        for value in VALUES:
            fig, axes = plt.subplots(3, 3, figsize=(10.5, 11.0), sharex=True, sharey=True)
            for row_index, representation in enumerate(REPRESENTATIONS):
                for column_index, stake in enumerate(STAKES):
                    ax = axes[row_index, column_index]
                    cell = grid[
                        (grid["representation"] == representation)
                        & (grid["stake"] == stake)
                        & (grid["value"] == value)
                    ]
                    for model in sorted(cell["model"].unique()):
                        model_cell = cell[cell["model"] == model].set_index("state")
                        ys = [float(model_cell.loc[state, "delta"]) for state in STATES]
                        ax.scatter(np.arange(4), ys, color="0.68", s=6, alpha=0.38)
                    ys = []
                    for state_index, state in enumerate(STATES):
                        record = _one(
                            pooled,
                            representation=representation,
                            stake=stake,
                            state=state,
                            value=value,
                        )
                        ys.append(float(record["pooled_delta"]))
                        _errorbar(
                            ax,
                            state_index,
                            record,
                            representation=representation,
                            marker=STAKE_MARKERS[stake],
                            filled=False,
                        )
                    _appendix_style(
                        ax,
                        f"{SHORT_REP_LABELS[representation]} – {stake.title()}",
                        STATE_LABELS,
                    )
            fig.suptitle(
                f"Complete history-conditioned {value} grid: per-model (gray) and pooled (colored)",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.0, w_pad=0.8)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _save(fig: plt.Figure, output_dir: str, stem: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, stem + ".pdf")
    png_path = os.path.join(output_dir, stem + ".png")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Creator": "make_paper_figures.py", "CreationDate": None},
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(stats_path: str, output_dir: str) -> None:
    with open(stats_path, "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    make_headline_figure(stats, output_dir)
    make_stakes_figure(stats, output_dir)
    make_context_figure(stats, output_dir)
    # Production reanalysis directories include the complete per-model history
    # grid.  Small figure-test fixtures may omit it, in which case only the two
    # required body figures are generated.
    history_csv = os.path.join(os.path.dirname(os.path.abspath(stats_path)), "history_state_grid.csv")
    if os.path.exists(history_csv):
        make_per_model_appendix(stats, output_dir)
        make_history_appendix(stats, history_csv, output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True, help="New reanalysis stats.json")
    parser.add_argument("--output-dir", required=True, help="Destination for PDF and PNG figures")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    make_figures(os.path.abspath(args.stats), os.path.abspath(args.output_dir))
    print(f"wrote paper figures to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
