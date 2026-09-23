"""Distribution of generated dyadic scenarios, decoded from ``scenario_id``.

Each dyadic ``scenario_id`` written by :mod:`generate_scenarios` encodes the
whole generation cell::

    {generating_model}_{value1}_{value2}_{value1}-{value2}-{prompt_type}-{index}

so the id alone recovers three meaningful axes, which this script cross-tabs
against ``keep_scenario`` (whether the LLM judge kept the scenario):

1. **Value pair** -- the conflict being probed.
2. **Causal frame x severity** (``prompt_type``, one of ``coc``, ``coc_mild``,
   ``cc``, ``cc_mild``) -- ``coc`` scenarios violate ``value2`` as a *side
   effect* of supporting ``value1``; ``cc`` scenarios violate ``value1`` as a
   *necessary precondition* for supporting ``value2``. ``_mild`` is the
   low-stakes variant of each. See :data:`utils.CAUSAL_GRAPH_PROMPTS`.
3. **Individual value** -- how many scenarios each value appears in, across
   every pair it belongs to.

Generation targets each (pair x prompt_type) cell equally, so the raw counts
are near-uniform by construction; the informative signal is the *kept* portion
of each bar, i.e. how the judge's yield varies across cells.

Writes one PNG (three stacked bar charts) and one long-form CSV per dataset.

Example:
    python scripts/scenario_distribution.py --datasets HHH modelspec personalprotective
"""

import argparse
import os
import re
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

#: Human-readable names for the causal-graph prompt types, in reading order
#: (severe before mild, side-effect harm before necessary harm).
PROMPT_TYPE_LABELS = {
    "coc": "side-effect harm - severe",
    "coc_mild": "side-effect harm - mild",
    "cc": "necessary harm - severe",
    "cc_mild": "necessary harm - mild",
}

# --- Chart tokens (validated light-mode palette; see the dataviz reference) ---
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
KEPT = "#2a78d6"     # categorical slot 1
DROPPED = "#eb6834"  # categorical slot 2


def parse_scenario_id(scenario_id: str, value1: str, value2: str) -> Optional[Tuple[str, int]]:
    """Recover ``(prompt_type, index)`` from a dyadic scenario id.

    The value pair is taken from the row's own ``value1``/``value2`` columns
    rather than by splitting on ``_``, so value names containing separators
    cannot break the parse.

    Args:
        scenario_id: The id as written by :mod:`generate_scenarios`.
        value1: The row's ``value1`` column.
        value2: The row's ``value2`` column.

    Returns:
        The prompt type and within-cell index, or ``None`` if the id does not
        match the expected layout.
    """
    pattern = (
        rf"^.+_{re.escape(value1)}_{re.escape(value2)}_"
        rf"{re.escape(value1)}-{re.escape(value2)}-"
        rf"(?P<prompt_type>coc|cc)(?P<mild>_mild)?-(?P<index>\d+)$"
    )
    match = re.match(pattern, scenario_id)
    if match is None:
        return None
    return match["prompt_type"] + (match["mild"] or ""), int(match["index"])


def load_dataset(path: str) -> pd.DataFrame:
    """Read one scenario CSV and decode its ids into analysis columns.

    Args:
        path: Path to a ``{generating_model}.csv`` written by
            :mod:`filter_scenarios`.

    Returns:
        The frame with added ``pair``, ``prompt_type``, ``prompt_label``,
        ``index`` and boolean ``kept`` columns. Rows whose id does not parse
        keep ``prompt_type == "UNPARSED"``.

    Raises:
        ValueError: If the CSV is missing a column the decoding needs.
    """
    df = pd.read_csv(path)
    required = {"scenario_id", "value1", "value2", "keep_scenario"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")

    parsed = [
        parse_scenario_id(sid, v1, v2)
        for sid, v1, v2 in zip(df["scenario_id"], df["value1"], df["value2"])
    ]
    df["prompt_type"] = [p[0] if p else "UNPARSED" for p in parsed]
    df["index"] = [p[1] if p else -1 for p in parsed]
    df["prompt_label"] = df["prompt_type"].map(PROMPT_TYPE_LABELS).fillna("unparsed id")
    df["pair"] = df["value1"] + " / " + df["value2"]
    # keep_scenario round-trips through CSV as the strings "True"/"False".
    df["kept"] = df["keep_scenario"].astype(str).str.strip().str.lower() == "true"
    return df


def tabulate(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab kept/dropped counts for each of the three axes.

    Args:
        df: A frame from :func:`load_dataset`.

    Returns:
        A long-form table with columns ``axis``, ``category``, ``kept``,
        ``dropped``, ``total``, ``keep_rate``, sorted by axis then keep rate.
    """
    frames = []

    def add(axis: str, series: pd.Series, kept: pd.Series) -> None:
        counts = pd.DataFrame({"category": series, "kept": kept})
        grouped = counts.groupby("category")["kept"].agg(kept="sum", total="count")
        grouped = grouped.reset_index()
        grouped.insert(0, "axis", axis)
        frames.append(grouped)

    add("value pair", df["pair"], df["kept"])
    add("causal frame x severity", df["prompt_label"], df["kept"])

    # A value appears in several pairs; count each scenario once per role.
    per_value = pd.concat(
        [
            pd.DataFrame({"category": df["value1"], "kept": df["kept"]}),
            pd.DataFrame({"category": df["value2"], "kept": df["kept"]}),
        ]
    )
    grouped = per_value.groupby("category")["kept"].agg(kept="sum", total="count").reset_index()
    grouped.insert(0, "axis", "individual value")
    frames.append(grouped)

    table = pd.concat(frames, ignore_index=True)
    table["dropped"] = table["total"] - table["kept"]
    table["keep_rate"] = table["kept"] / table["total"]
    table = table[["axis", "category", "kept", "dropped", "total", "keep_rate"]]
    return table.sort_values(["axis", "keep_rate"], ascending=[True, False], ignore_index=True)


def _draw_panel(ax, rows: pd.DataFrame, title: str, show_xlabel: bool) -> None:
    """Draw one horizontal stacked kept/dropped bar chart onto ``ax``.

    Each panel carries its own x-scale: the three axes aggregate the same
    scenarios differently (the per-value panel counts each scenario under both
    of its values), so a shared scale would squash the finer-grained panels
    without making anything comparable.
    """
    rows = rows.sort_values("keep_rate", ascending=True)
    positions = range(len(rows))
    xmax = float(rows["total"].max())
    gap = xmax * 0.005  # surface gap between the two segments, not a border

    ax.barh(positions, rows["kept"], height=0.5, color=KEPT, zorder=3)
    ax.barh(
        positions,
        rows["dropped"],
        left=rows["kept"] + gap,
        height=0.5,
        color=DROPPED,
        zorder=3,
    )

    for y, (_, row) in zip(positions, rows.iterrows()):
        ax.text(
            row["total"] + xmax * 0.02,
            y,
            f"{row['keep_rate']:.0%} kept  ({int(row['kept'])}/{int(row['total'])})",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(rows["category"], fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0, xmax * 1.34)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=7, fontweight="normal")
    if show_xlabel:
        ax.set_xlabel("scenarios generated", fontsize=9, color=INK_MUTED, labelpad=6)

    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["left"].set_linewidth(0.8)
    ax.set_facecolor(SURFACE)


def plot_dataset(table: pd.DataFrame, dataset: str, total: int, out_path: str) -> None:
    """Render the three-panel figure for one dataset.

    Args:
        table: The long-form table from :func:`tabulate`.
        dataset: Value-set name, used in the figure title.
        total: Number of scenarios in the dataset.
        out_path: Destination PNG path.
    """
    axes_order = ["value pair", "causal frame x severity", "individual value"]
    panel_titles = [
        "by value pair",
        "by causal frame \u00d7 severity",
        "by individual value  \u00b7  each scenario counted under both of its values",
    ]
    panels = [table[table["axis"] == axis] for axis in axes_order]

    panel_heights = [0.30 * len(p) + 0.80 for p in panels]
    header = 1.55
    footer = 0.55
    fig_h = sum(panel_heights) + header + footer
    fig = plt.figure(figsize=(9.5, fig_h), facecolor=PAGE)
    grid = GridSpec(
        len(panels), 1, figure=fig, height_ratios=panel_heights, hspace=0.5,
        top=1 - header / fig_h, bottom=footer / fig_h, left=0.27, right=0.97,
    )

    kept_total = int(table[table["axis"] == "value pair"]["kept"].sum())
    fig.text(
        0.028, 1 - 0.34 / fig_h,
        f"{dataset}: scenario distribution by generation cell",
        fontsize=13.5, color=INK, va="top", fontweight="bold",
    )
    fig.text(
        0.028, 1 - 0.68 / fig_h,
        f"{total:,} scenarios generated, {kept_total:,} kept by the LLM judge "
        f"({kept_total / total:.0%} overall).\nGeneration targets every cell equally, "
        f"so the quantity that actually varies is each cell's kept share.",
        fontsize=9, color=INK_SECONDARY, va="top", linespacing=1.5,
    )
    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=KEPT, label="kept by judge"),
            plt.Rectangle((0, 0), 1, 1, color=DROPPED, label="dropped by judge"),
        ],
        loc="upper right", bbox_to_anchor=(0.97, 1 - 0.30 / fig_h), ncol=2, frameon=False,
        fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.1, handleheight=0.9,
        columnspacing=1.4, handletextpad=0.6,
    )

    for i, (panel, title) in enumerate(zip(panels, panel_titles)):
        ax = fig.add_subplot(grid[i])
        _draw_panel(ax, panel, title, show_xlabel=(i == len(panels) - 1))

    fig.savefig(out_path, dpi=200, facecolor=PAGE)
    plt.close(fig)


def main() -> None:
    """CLI entry point: tabulate and plot each requested dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets", nargs="+",
        default=["HHH", "modelspec", "personalprotective"],
        help="Value-set directory names under --data-dir.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory holding the value-set dirs.")
    parser.add_argument(
        "--model", default="claude-3-5-sonnet-latest",
        help="Generating-model CSV basename to read from each value-set dir.",
    )
    parser.add_argument("-o", "--output-dir", default="analysis/scenario_distribution")
    args = parser.parse_args()

    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in args.datasets:
        path = os.path.join(args.data_dir, dataset, f"{args.model}.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue

        df = load_dataset(path)
        unparsed = int((df["prompt_type"] == "UNPARSED").sum())
        if unparsed:
            print(f"[warn] {dataset}: {unparsed} scenario_id(s) did not match the expected layout")

        table = tabulate(df)
        csv_path = os.path.join(args.output_dir, f"{dataset}_distribution.csv")
        png_path = os.path.join(args.output_dir, f"{dataset}_distribution.png")
        table.to_csv(csv_path, index=False)
        plot_dataset(table, dataset, len(df), png_path)

        print(f"\n=== {dataset} ({len(df):,} scenarios, {int(df['kept'].sum()):,} kept "
              f"= {df['kept'].mean():.1%}) ===")
        for axis in ["value pair", "causal frame x severity", "individual value"]:
            print(f"\n  by {axis}:")
            block = table[table["axis"] == axis]
            for _, row in block.iterrows():
                print(f"    {row['category']:<34} {int(row['kept']):>5} / {int(row['total']):>5} "
                      f"kept  ({row['keep_rate']:.1%})")
        print(f"\n  -> {csv_path}\n  -> {png_path}")


if __name__ == "__main__":
    main()
