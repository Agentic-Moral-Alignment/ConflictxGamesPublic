"""Descriptive tables outside the corrected analysis (no intervals, no tests).

* ``abstract_rival.csv``: change in the menu harmlessness win rate under the
  alignment and usefulness stakes, with the rival named as the counterpart or
  as "other deployed systems". Paired on scenario-seat.
* ``horizon.csv``: win rate with no counterpart (single-agent corpus), in a
  single round, and in the first round of a described repeated interaction.
* ``single_agent_format.csv``: menu and free-text win rates on the
  single-agent corpus.
* ``bradley_terry.csv``: Bradley-Terry value orderings per model, menu and
  open-undisclosed, with and without the HHH steering prompt
  (``choix.ilsr_pairwise``, alpha 0.01).

Usage::

    python reproduce/descriptive.py --output-dir reproduce/out/descriptive
"""

from __future__ import annotations

import argparse
import os

import choix
import numpy as np
import pandas as pd

import reanalysis as paper

SOLO_DIR = os.path.join(paper.ROOT, "results", "HHH_new_solo_20260826_01_balanced50")
JUDGE_TAIL = "_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv"
VALUES = list(paper.VALUES)


def load_arm(model: str, mode: str, arm: str, pairs, actions, states: bool = False) -> pd.DataFrame:
    """Load one game arm, by exact file name."""
    tail = ".csv" if mode == "mcq" else JUDGE_TAIL
    path = os.path.join(paper.GAME_DIR, f"{model}_{mode}_balanced50{arm}{tail}")
    frame, _ = paper._read_csv_bytes(path, usecols=paper.RESULT_COLUMNS)
    return paper.load_game(frame, pairs, actions, states=states, source=path)


def win_rates(frame: pd.DataFrame) -> dict:
    """Per-value win rate over the answered contests that involve the value."""
    answered = frame[frame["answered"]]
    out = {}
    for value in VALUES:
        involved = answered[answered["pair"].apply(lambda pair: value in pair)]
        out[value] = float((involved["chosen"] == value).mean())
    return out


def solo_rates(model: str, mode: str) -> dict:
    """Per-value win rate on the single-agent corpus (refusals and invalid replies excluded)."""
    tail = ".csv" if mode == "mcq" else JUDGE_TAIL
    path = os.path.join(SOLO_DIR, f"{model}_{mode}_balanced50{tail}")
    frame = pd.read_csv(paper._io_path(path))
    frame = frame[frame["choice"].isin(["A", "B"])]
    chosen = np.where(frame["choice"] == "A", frame["value1"], frame["value2"])
    out = {}
    for value in VALUES:
        involved = ((frame["value1"] == value) | (frame["value2"] == value)).to_numpy()
        out[value] = float((chosen[involved] == value).mean())
    return out


def abstract_rival(load) -> pd.DataFrame:
    def row(criterion, rival, base_arm, arm):
        deltas = [float(paper.paired_frame(load(m, "mcq", base_arm), load(m, "mcq", arm),
                                           "harmlessness")["d"].mean()) for m in paper.MODELS]
        return {"criterion": criterion, "rival": rival, "mean_shift": float(np.mean(deltas)),
                "n_positive": sum(d > 0 for d in deltas), "n_negative": sum(d < 0 for d in deltas)}

    rows = []
    for criterion in ("alignment", "usefulness"):
        specific, abstract = f"_stakes-{criterion}", f"_stakes-abstract-{criterion}"
        rows.append(row(criterion, "the other participant", "", specific))
        rows.append(row(criterion, "other deployed systems", "", abstract))
        rows.append(row(criterion, "abstract minus specific", specific, abstract))
    return pd.DataFrame(rows)


def horizon(load) -> pd.DataFrame:
    solo = [solo_rates(m, "mcq") for m in paper.MODELS]
    single = [load(m, "mcq", "") for m in paper.MODELS]
    first = [paper.seat_items(load(m, "mcq", "_states", states=True)) for m in paper.MODELS]
    single_rates = [win_rates(frame) for frame in single]
    rows = []
    for value in VALUES:
        paired = [paper.paired_frame(a, b, value) for a, b in zip(single, first)]
        counterpart = [s[value] - o[value] for s, o in zip(single_rates, solo)]
        rows.append({
            "value": value,
            "solo": float(np.mean([o[value] for o in solo])),
            "single_round": float(np.mean([s[value] for s in single_rates])),
            "single_round_matched": float(np.mean([p["y_base"].mean() for p in paired])),
            "first_round_matched": float(np.mean([p["y_arm"].mean() for p in paired])),
            "add_counterpart": float(np.mean(counterpart)),
            "add_counterpart_n_positive": sum(d > 0 for d in counterpart),
            "add_counterpart_n_negative": sum(d < 0 for d in counterpart),
            "add_repetition_paired": float(np.mean([p["d"].mean() for p in paired])),
        })
    return pd.DataFrame(rows)


def single_agent_format() -> pd.DataFrame:
    menu = [solo_rates(m, "mcq") for m in paper.MODELS]
    chat = [solo_rates(m, "interactive") for m in paper.MODELS]
    rows = []
    for value in VALUES:
        deltas = [c[value] - m[value] for c, m in zip(chat, menu)]
        rows.append({"value": value, "menu": float(np.mean([m[value] for m in menu])),
                     "free_text": float(np.mean([c[value] for c in chat])),
                     "difference": float(np.mean(deltas)),
                     "n_positive": sum(d > 0 for d in deltas), "n_negative": sum(d < 0 for d in deltas)})
    return pd.DataFrame(rows)


def bradley_terry(load) -> pd.DataFrame:
    index = {value: i for i, value in enumerate(VALUES)}
    rows = []
    for fmt, mode in (("menu", "mcq"), ("open_undisclosed", "open")):
        for arm, token in (("baseline", ""), ("hhh_steered", "_steer-HHH")):
            for model in paper.MODELS:
                frame = load(model, mode, token)
                answered = frame[frame["answered"]]
                comparisons = [
                    (index[chosen], index[pair[0] if pair[1] == chosen else pair[1]])
                    for chosen, pair in zip(answered["chosen"], answered["pair"])
                ]
                ability = choix.ilsr_pairwise(len(VALUES), comparisons, alpha=0.01)
                order = sorted(VALUES, key=lambda v: -ability[index[v]])
                rows.append({"format": fmt, "arm": arm, "model": model, "ordering": " > ".join(order),
                             **{value: float(ability[index[value]]) for value in VALUES}})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pairs, actions, _ = paper.load_pairs_and_actions(paper.SCENARIOS_CSV)

    def load(model, mode, arm, states=False):
        return load_arm(model, mode, arm, pairs, actions, states)

    for name, table in (("abstract_rival", abstract_rival(load)), ("horizon", horizon(load)),
                        ("single_agent_format", single_agent_format()),
                        ("bradley_terry", bradley_terry(load))):
        path = os.path.join(args.output_dir, f"{name}.csv")
        table.to_csv(path, index=False, lineterminator="\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
