"""Reproducible cross-domain and cross-model figures for HHH evaluations.

This module consumes the offline CSVs emitted by ``solo-analyze`` and
``games.value_figures``.  It creates one game-vs-solo ability figure per focal
model plus shared game-only model comparison and baseline-difference figures.
No model or API calls are made.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conflictxgame.common.config import add_config_argument, parse_with_config


CORE_GAME_CONDITIONS = ("MCQ choice", "Open-ended", "Play (all)")
SETTING_LABELS = {
    "solo": "Solo",
    "MCQ choice": "Game MCQ",
    "Open-ended": "Game open",
    "Play (all)": "Game play",
}
VALUE_COLORS = {
    "harmlessness": "#2f855a",
    "honesty": "#3182ce",
    "helpfulness": "#dd6b20",
}
MARKERS = ("o", "s", "^")


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_name(value: str) -> str:
    return (
        value.removeprefix("openai:")
        .replace("openai.", "")
        .replace("google.", "")
        .replace("qwen.", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _select_model(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    exact = frame[frame["model"].astype(str) == model]
    if not exact.empty:
        return exact.copy()
    normalized = model.removeprefix("openai:")
    return frame[
        frame["model"].astype(str).str.removeprefix("openai:") == normalized
    ].copy()


def _read_solo_rankings(solo_dir: str) -> pd.DataFrame:
    for name in (
        "bradley_terry_rankings_filtered.csv",
        "bradley_terry_rankings_unfiltered.csv",
    ):
        path = Path(solo_dir) / name
        if path.is_file():
            frame = pd.read_csv(path)
            required = {"model", "value", "ability", "ci_lower", "ci_upper"}
            if not required <= set(frame.columns):
                raise ValueError(f"{path} lacks required columns {sorted(required)}")
            return frame
    raise FileNotFoundError(
        f"No solo Bradley-Terry rankings found under {solo_dir}; run solo-analyze first."
    )


def _read_game_rankings(game_dir: str) -> pd.DataFrame:
    path = Path(game_dir) / "bradley_terry_by_condition.csv"
    frame = pd.read_csv(path)
    required = {
        "model", "condition", "value", "ability", "ci_lower", "ci_upper"
    }
    if not required <= set(frame.columns):
        raise ValueError(f"{path} lacks required columns {sorted(required)}")
    return frame


def build_cross_domain_table(
    solo: pd.DataFrame,
    game_frames: Sequence[pd.DataFrame],
    models: Sequence[str],
    labels: Sequence[str],
) -> pd.DataFrame:
    """Stack solo and core game-condition abilities for every focal model."""
    rows = []
    for model, label, games in zip(models, labels, game_frames):
        solo_model = _select_model(solo, model)
        if solo_model.empty:
            raise ValueError(f"Solo analysis has no rows for model {model}")
        solo_model = solo_model.copy()
        solo_model["condition"] = "solo"
        game_model = _select_model(games, model)
        game_model = game_model[
            game_model["condition"].isin(CORE_GAME_CONDITIONS)
        ].copy()
        if game_model.empty:
            raise ValueError(f"Game analysis has no core-condition rows for {model}")
        combined = pd.concat([solo_model, game_model], ignore_index=True)
        combined["model"] = model
        combined["model_label"] = label
        combined["setting"] = combined["condition"].map(SETTING_LABELS)
        rows.append(combined[[
            "model", "model_label", "setting", "condition", "value",
            "ability", "ci_lower", "ci_upper",
        ]])
    return pd.concat(rows, ignore_index=True)


def build_game_difference_table(
    cross_domain: pd.DataFrame,
    baseline_model: str,
) -> pd.DataFrame:
    """Return ability differences from the first/baseline model on games."""
    game = cross_domain[cross_domain["condition"] != "solo"].copy()
    baseline = game[game["model"] == baseline_model][[
        "condition", "value", "ability", "ci_lower", "ci_upper"
    ]].rename(columns={
        "ability": "baseline_ability",
        "ci_lower": "baseline_ci_lower",
        "ci_upper": "baseline_ci_upper",
    })
    others = game[game["model"] != baseline_model].copy()
    merged = others.merge(baseline, on=["condition", "value"], how="inner")
    merged["ability_difference"] = merged["ability"] - merged["baseline_ability"]
    se = (merged["ci_upper"] - merged["ci_lower"]) / 3.92
    baseline_se = (
        merged["baseline_ci_upper"] - merged["baseline_ci_lower"]
    ) / 3.92
    diff_error = 1.96 * np.sqrt(se.pow(2) + baseline_se.pow(2))
    merged["diff_ci_lower"] = merged["ability_difference"] - diff_error
    merged["diff_ci_upper"] = merged["ability_difference"] + diff_error
    merged["baseline_model"] = baseline_model
    return merged


def _errorbar(ax, x, row, *, color, marker, label=None):
    ax.errorbar(
        x, row["ability"],
        yerr=[[row["ability"] - row["ci_lower"]],
              [row["ci_upper"] - row["ability"]]],
        color=color, marker=marker, linewidth=1.6, capsize=3, label=label,
    )


def plot_cross_domain_per_model(table: pd.DataFrame, output_dir: str) -> list[str]:
    """Plot solo and game abilities on shared axes, one PNG per model."""
    written = []
    setting_order = ["Solo", "Game MCQ", "Game open", "Game play"]
    for (model, label), group in table.groupby(["model", "model_label"]):
        fig, ax = plt.subplots(figsize=(9.2, 5.6))
        for value in sorted(group["value"].unique()):
            value_rows = (
                group[group["value"] == value]
                .set_index("setting")
                .reindex(setting_order)
                .dropna(subset=["ability"])
            )
            xs = [setting_order.index(setting) for setting in value_rows.index]
            ys = value_rows["ability"].to_numpy()
            lower = ys - value_rows["ci_lower"].to_numpy()
            upper = value_rows["ci_upper"].to_numpy() - ys
            ax.errorbar(
                xs, ys, yerr=[lower, upper], marker="o", capsize=3,
                linewidth=2, color=VALUE_COLORS.get(value, "#555555"),
                label=value.title(),
            )
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(range(len(setting_order)), setting_order)
        ax.set_ylabel("Bradley–Terry ability")
        ax.set_title(f"Game vs solo value priorities — {label}")
        ax.legend(frameon=False, ncol=3, loc="upper center")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        model_dir = Path(output_dir) / _safe_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / "cross_game_solo_ability.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(str(path))
    return written


def plot_game_model_comparison(table: pd.DataFrame, output_dir: str) -> str:
    """Plot all focal models' absolute game abilities by condition and value."""
    game = table[table["condition"] != "solo"].copy()
    conditions = list(CORE_GAME_CONDITIONS)
    models = list(dict.fromkeys(game["model_label"]))
    values = sorted(game["value"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(15, 5.2), sharey=True)
    offsets = np.linspace(-0.22, 0.22, len(models))
    for ax, condition in zip(np.atleast_1d(axes), conditions):
        subset = game[game["condition"] == condition]
        for model_index, (model, offset) in enumerate(zip(models, offsets)):
            model_rows = subset[subset["model_label"] == model].set_index("value")
            for value_index, value in enumerate(values):
                if value not in model_rows.index:
                    continue
                row = model_rows.loc[value]
                _errorbar(
                    ax, value_index + offset, row,
                    color=VALUE_COLORS.get(value, "#555555"),
                    marker=MARKERS[model_index % len(MARKERS)],
                    label=model if value_index == 0 else None,
                )
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(range(len(values)), [v.title() for v in values], rotation=25)
        ax.set_title(SETTING_LABELS[condition])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Bradley–Terry ability")
    axes[0].legend(frameon=False)
    fig.suptitle("Model differences on game evaluations")
    fig.tight_layout()
    path = Path(output_dir) / "game_model_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def plot_game_model_differences(differences: pd.DataFrame, output_dir: str) -> str:
    """Plot each non-baseline model's game ability delta from the baseline."""
    conditions = list(CORE_GAME_CONDITIONS)
    models = list(dict.fromkeys(differences["model_label"]))
    values = sorted(differences["value"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(15, 5.2), sharey=True)
    offsets = np.linspace(-0.12, 0.12, max(len(models), 1))
    for ax, condition in zip(np.atleast_1d(axes), conditions):
        subset = differences[differences["condition"] == condition]
        for model_index, (model, offset) in enumerate(zip(models, offsets)):
            rows = subset[subset["model_label"] == model].set_index("value")
            for value_index, value in enumerate(values):
                if value not in rows.index:
                    continue
                row = rows.loc[value]
                delta = row["ability_difference"]
                ax.errorbar(
                    value_index + offset, delta,
                    yerr=[[delta - row["diff_ci_lower"]],
                          [row["diff_ci_upper"] - delta]],
                    marker=MARKERS[model_index % len(MARKERS)], capsize=3,
                    linewidth=1.6, color=VALUE_COLORS.get(value, "#555555"),
                    label=model if value_index == 0 else None,
                )
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks(range(len(values)), [v.title() for v in values], rotation=25)
        ax.set_title(SETTING_LABELS[condition])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Ability difference from baseline")
    if models:
        axes[0].legend(frameon=False)
    fig.suptitle("Game-model differences relative to the baseline model")
    fig.tight_layout()
    path = Path(output_dir) / "game_model_differences_from_baseline.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--solo-analysis-dir", required=True)
    parser.add_argument("--game-analysis-dirs", required=True,
                        help="Comma-separated per-model game analysis directories")
    parser.add_argument("--models", required=True,
                        help="Comma-separated model identifiers in display order")
    parser.add_argument("--model-labels", required=True,
                        help="Comma-separated display labels matching --models")
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args, _, _ = parse_with_config(build_parser(), argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    models = _csv_list(args.models)
    labels = _csv_list(args.model_labels)
    game_dirs = _csv_list(args.game_analysis_dirs)
    if not (len(models) == len(labels) == len(game_dirs)):
        raise SystemExit(
            "models, model_labels, and game_analysis_dirs must have equal lengths"
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    solo = _read_solo_rankings(args.solo_analysis_dir)
    games = [_read_game_rankings(path) for path in game_dirs]
    cross = build_cross_domain_table(solo, games, models, labels)
    cross.to_csv(Path(args.output_dir) / "cross_game_solo_abilities.csv", index=False)
    differences = build_game_difference_table(cross, models[0])
    differences.to_csv(
        Path(args.output_dir) / "game_model_differences_from_baseline.csv",
        index=False,
    )
    written: Iterable[str] = [
        *plot_cross_domain_per_model(cross, args.output_dir),
        plot_game_model_comparison(cross, args.output_dir),
        plot_game_model_differences(differences, args.output_dir),
    ]
    print("Comparison figures complete:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
