"""Observer-split and predict-first tables (menu format, fabricated-history states).

Writes the two tables behind the observer and prediction follow-ups:

* ``level_effects.csv``: change in each value's win rate for every visibility
  and prediction arm against four references (no visibility statement, private,
  placebo, and the matching arm without the prediction step), pooled over ten
  models and five history states. Paired on scenario-seat and state; 2000
  scenario-bootstrap resamples, 10,000 scenario-level sign flips, Holm within
  each reference family.
* ``predict_diagnostics.csv``: per model and prediction arm, how often the
  prediction parsed, repeated the counterpart's remembered move, expected the
  counterpart to learn the choice, and matched the value then chosen.

One generator, seeded once, is shared across the families in the order written.

Usage::

    python reproduce/observer_levels.py --output-dir reproduce/out/observer_levels
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import history_rekey as rekey

paper = rekey.paper

VISIBILITY = ("vis-both", "vis-none", "vis-counterpart", "vis-reviewer", "vis-placebo")
PREDICT = ("predict-none", "predict-vis-both", "predict-vis-none")
CONDITIONS = ("none", *VISIBILITY, *PREDICT)

#: Row order of the released diagnostics table.
ROW_ORDER = (
    "gemma-3-12b", "gemma-3-27b", "gemma-4-26b-a4b", "gemma-4-31B-it", "gpt-oss-120b",
    "gpt-oss-20b", "qwen3-32b", "llama-3.3-70b", "llama-4-maverick", "qwen3-235b-a22b",
)

#: (family, [(arm, reference arm)], iterate arms in the outer loop).
FAMILIES = (
    ("vs none", [(arm, "none") for arm in (*VISIBILITY, *PREDICT)], True),
    ("vs private", [(arm, "vis-none") for arm in ("vis-both", "vis-counterpart", "vis-reviewer")], True),
    ("vs placebo", [(arm, "vis-placebo") for arm in ("vis-none", "vis-both")], False),
    ("predict vs no-predict",
     [("predict-none", "none"), ("predict-vis-both", "vis-both"), ("predict-vis-none", "vis-none")], True),
)


def level_effects(arms: dict) -> pd.DataFrame:
    """Pooled paired effects for every family; see the module docstring."""
    rng = np.random.default_rng(paper.SEED)
    out = []
    for family, contrasts, arms_outer in FAMILIES:
        if arms_outer:
            cells = [(arm, base, value) for arm, base in contrasts for value in paper.VALUES]
        else:
            cells = [(arm, base, value) for value in paper.VALUES for arm, base in contrasts]
        rows = []
        for arm, base, value in cells:
            frames = [paper.paired_frame(arms[base][m], arms[arm][m], value) for m in paper.MODELS]
            scenarios = sorted({str(s) for frame in frames for s in frame["scenario_id"]})
            sums, counts = paper._cluster_arrays(frames, scenarios)
            per_model = [float(frame["d"].mean()) for frame in frames]
            delta = float(np.mean(per_model))
            lower, upper = np.percentile(
                paper._bootstrap_mean(sums, counts, rng, paper.DEFAULT_BOOTSTRAP), [2.5, 97.5])
            p = paper._signflip_p(sums, counts, delta, rng, paper.DEFAULT_PERMUTATIONS)
            rows.append({
                "family": family, "arm": arm, "value": value, "delta": delta,
                "ci_lower": float(lower), "ci_upper": float(upper), "p": p,
                "n_pos": sum(d > 0 for d in per_model), "n_neg": sum(d < 0 for d in per_model),
            })
        for row, adjusted in zip(rows, paper._holm_values([row["p"] for row in rows])):
            row["p_holm"] = float(adjusted)
        out.extend(rows)
    return pd.DataFrame(out)


def predict_diagnostics(pairs: dict) -> pd.DataFrame:
    """Prediction-turn diagnostics per arm and model."""
    rows = []
    for arm in PREDICT:
        for model in ROW_ORDER:
            path = rekey.resolve_states_file(paper.GAME_DIR, model, "menu", arm, "explicit")
            frame = pd.read_csv(paper._io_path(path), dtype=str, keep_default_na=False)
            remembered = frame[frame["probe"].str.startswith("prev_")]
            parsed = remembered["predicted_cp_action"].isin(["0", "1"])
            said = remembered[parsed]
            # probe is "prev_<seat 0 action>-<seat 1 action>".
            last = [probe[len("prev_"):].split("-")[1 - int(seat)]
                    for probe, seat in zip(said["probe"], said["seat"])]
            answered = said[said["parsed_ok"] == "True"]
            matchable = answered[[
                row.predicted_cp_value_tag in pairs[(row.scenario_id, int(row.seat))]
                for row in answered.itertuples()
            ]]
            rows.append({
                "arm": arm, "model": model,
                "prediction_parsed": float(parsed.mean()),
                "predicts_counterpart_repeats_last_move":
                    float(np.mean([a == b for a, b in zip(said["predicted_cp_action"], last)])),
                "says_counterpart_will_learn": float((frame["predicted_cp_learns"] == "yes").mean()),
                "choice_matches_predicted_value":
                    float((matchable["predicted_cp_value_tag"] == matchable["choice_value_tag"]).mean()),
                "n_match_items": len(matchable),
                "choice_unanswered": int((frame["parsed_ok"] != "True").sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pairs, actions, _ = paper.load_pairs_and_actions(paper.SCENARIOS_CSV)
    loaded = rekey.load_conditions(paper.GAME_DIR, list(paper.MODELS), "explicit", pairs, actions,
                                   conditions_only=CONDITIONS, representations=["menu"])
    arms = {condition: loaded[("menu", condition)] for condition in CONDITIONS}
    for name, table in (("level_effects", level_effects(arms)),
                        ("predict_diagnostics", predict_diagnostics(pairs))):
        path = os.path.join(args.output_dir, f"{name}.csv")
        table.to_csv(path, index=False, lineterminator="\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
