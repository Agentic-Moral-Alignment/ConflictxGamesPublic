"""Offline, paired reanalysis for the NeurIPS 2026 AI for Good paper.

The analysis is intentionally limited to the ten-model, two-assistant game
corpus.  It makes no model or API calls.  Conditions are resolved by exact arm
tokens parsed with :func:`conflictxgame.common.arm_names.arm_of`; filenames are
never selected by ordering or loose substring matches.

The three main action representations are:

* menu choice with the counterpart shown;
* open response with the counterpart generally undisclosed; and
* open response with the counterpart explicitly disclosed.

Run from the repository root with ``PYTHONPATH=src``::

    python reproduce/reanalysis.py

The default output directory is versioned and the script refuses to overwrite
a nonempty directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from conflictxgame.common.arm_names import arm_of, split_result_name


HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start):
    """Walk up until the directory holding ``results/`` and ``data/`` is found.

    Derived by search, so moving this script cannot silently repoint it.
    """
    d = start
    while True:
        if all(os.path.isdir(os.path.join(d, n)) for n in ("results", "data")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                "could not locate the repo root (no ancestor of %s has both "
                "results/ and data/)" % start
            )
        d = parent


ROOT = _find_repo_root(HERE)
GAME_DIR = os.path.join(
    ROOT, "results", "HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
)
SCENARIOS_CSV = os.path.join(
    ROOT,
    "data",
    "HHH_new_paper_bothAI_anthropic_batch_20260826_01",
    "claude-sonnet-5_games.csv",
)
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "analysis", "reanalysis_20260904_01")

SEED = 20260904
DEFAULT_BOOTSTRAP = 2000
DEFAULT_PERMUTATIONS = 10000

MODELS = (
    "gemma-3-12b",
    "gemma-3-27b",
    "gemma-4-26b-a4b",
    "gemma-4-31B-it",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3.3-70b",
    "llama-4-maverick",
    "qwen3-235b-a22b",
    "qwen3-32b",
)
MODEL_LABELS = {
    "gemma-3-12b": "Gemma 3 12B",
    "gemma-3-27b": "Gemma 3 27B",
    "gemma-4-26b-a4b": "Gemma 4 26B A4B",
    "gemma-4-31B-it": "Gemma 4 31B",
    "gpt-oss-120b": "GPT-OSS 120B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "llama-3.3-70b": "Llama 3.3 70B",
    "llama-4-maverick": "Llama 4 Maverick",
    "qwen3-235b-a22b": "Qwen 3 235B A22B",
    "qwen3-32b": "Qwen 3 32B",
}

REPRESENTATIONS = {
    "menu": {"label": "Menu choice (counterpart shown)", "mode": "mcq"},
    "open_undisclosed": {
        "label": "Open response (counterpart undisclosed)",
        "mode": "open",
    },
    "open_disclosed": {
        "label": "Open response (counterpart disclosed)",
        "mode": "open",
    },
}
REPRESENTATION_PAIRS = (
    ("menu", "open_undisclosed"),
    ("menu", "open_disclosed"),
    ("open_undisclosed", "open_disclosed"),
)
STAKES = ("payoff", "alignment", "usefulness")
VALUES = ("harmlessness", "helpfulness", "honesty")
STATE_ORDER = ("fresh", "none", "one", "both")
_COUNT_TO_STATE = {0: "none", 1: "one", 2: "both"}

RESULT_COLUMNS = {
    "scenario_id",
    "seat",
    "probe",
    "choice_value_tag",
    "parsed_ok",
}


class ConditionSpec(NamedTuple):
    representation: str
    family: str
    condition: str
    mode: str
    arm: str
    states: bool = False

    @property
    def key(self) -> str:
        return f"{self.representation}:{self.family}:{self.condition}"


def arm_token_for(
    representation: str, condition: str = "baseline", *, states: bool = False
) -> tuple[str, str]:
    """Return ``(mode, exact arm token)`` for a paper condition.

    ``condition`` is ``baseline``, a concrete stake, ``steering``, ``observed``,
    or ``unobserved``.  In state arms only baseline and concrete stakes exist.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation!r}")
    if condition == "none":
        condition = "baseline"
    mode = REPRESENTATIONS[representation]["mode"]
    if states:
        if condition not in ("baseline", *STAKES):
            raise ValueError(f"No state arm for condition {condition!r}")
        if representation == "menu":
            arm = "_states" if condition == "baseline" else f"_states_stakes-{condition}"
        elif representation == "open_undisclosed":
            arm = "_states" if condition == "baseline" else f"_states-stakes-{condition}"
        else:
            arm = (
                "_counterpart_states"
                if condition == "baseline"
                else f"_counterpart_states-stakes-{condition}"
            )
        return mode, arm

    if representation == "open_disclosed":
        if condition == "baseline":
            arm = "_counterpart"
        elif condition in STAKES:
            arm = f"_counterpart_stakes-{condition}"
        elif condition == "steering":
            arm = "_counterpart_steer-HHH"
        else:
            raise ValueError(
                f"The counterpart-disclosed representation has no {condition!r} arm"
            )
        return mode, arm

    if condition == "baseline":
        arm = ""
    elif condition in STAKES:
        arm = f"_stakes-{condition}"
    elif condition == "steering":
        arm = "_steer-HHH"
    elif condition in ("observed", "unobserved"):
        arm = f"_{condition}"
    else:
        raise ValueError(f"Unknown condition: {condition!r}")
    return mode, arm


def condition_specs() -> list[ConditionSpec]:
    """All and only the result conditions used in the paper."""
    specs: list[ConditionSpec] = []
    for representation in REPRESENTATIONS:
        for condition in ("baseline", *STAKES, "steering"):
            mode, arm = arm_token_for(representation, condition)
            specs.append(
                ConditionSpec(representation, "main", condition, mode, arm, False)
            )
        if representation != "open_disclosed":
            for condition in ("observed", "unobserved"):
                mode, arm = arm_token_for(representation, condition)
                specs.append(
                    ConditionSpec(
                        representation, "visibility", condition, mode, arm, False
                    )
                )
        for condition in ("baseline", *STAKES):
            mode, arm = arm_token_for(representation, condition, states=True)
            specs.append(
                ConditionSpec(representation, "states", condition, mode, arm, True)
            )
    return specs


def resolve_result_file(
    directory: str, model: str, representation: str, condition: str = "baseline", *,
    states: bool = False, filenames: Sequence[str] | None = None
) -> str:
    """Resolve exactly one result file using parsed, exact arm equality."""
    mode, wanted_arm = arm_token_for(representation, condition, states=states)
    names = list(filenames) if filenames is not None else os.listdir(directory)
    matches: list[str] = []
    for name in names:
        if not name.endswith(".csv"):
            continue
        try:
            parts = split_result_name(name)
            parsed_arm = arm_of(name)
        except ValueError:
            continue
        if (
            parts["slug"] == model
            and parts["mode"] == mode
            and parts["balanced"] == "_balanced50"
            and parsed_arm == wanted_arm
        ):
            matches.append(os.path.abspath(os.path.join(directory, name)))
    if len(matches) != 1:
        status = "missing" if not matches else "duplicate"
        raise ValueError(
            f"{status} result arm for model={model!r}, representation={representation!r}, "
            f"condition={condition!r}, states={states}: expected exactly one "
            f"mode={mode!r}, arm={wanted_arm!r}; found {len(matches)}: {matches}"
        )
    return matches[0]


def resolve_all_result_files(
    directory: str, specs: Sequence[ConditionSpec], models: Sequence[str] = MODELS
) -> dict[tuple[str, str], str]:
    """Resolve complete model coverage before any result CSV is read."""
    names = sorted(os.listdir(directory))
    resolved: dict[tuple[str, str], str] = {}
    for spec in specs:
        for model in models:
            resolved[(spec.key, model)] = resolve_result_file(
                directory,
                model,
                spec.representation,
                spec.condition,
                states=spec.states,
                filenames=names,
            )
    expected = len(specs) * len(models)
    if len(resolved) != expected:
        raise AssertionError(f"Expected {expected} condition-model paths, got {len(resolved)}")
    return resolved


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _io_path(path: str) -> str:
    """Use Windows extended-length syntax for long paths."""
    absolute = os.path.abspath(path)
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return "\\\\?\\" + absolute
    return absolute


def _read_csv_bytes(path: str, *, usecols: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    with open(_io_path(path), "rb") as handle:
        raw = handle.read()
    kwargs: dict[str, Any] = {"low_memory": False}
    if usecols is not None:
        kwargs["usecols"] = lambda column: column in usecols
    frame = pd.read_csv(io.BytesIO(raw), **kwargs)
    record = {
        "relative_path": os.path.relpath(path, ROOT).replace(os.sep, "/"),
        "sha256": sha256_bytes(raw),
        "bytes": int(len(raw)),
    }
    return frame, record


def load_pairs_and_actions(path: str) -> tuple[dict, dict, dict]:
    """Read value pairs and action mappings directly from ``game_spec``.

    Returns ``(pairs, actions, provenance)``.  Keys are ``(scenario_id, seat)``;
    pairs are alphabetically canonicalized while actions remain in action-index
    order for recoding fabricated histories.
    """
    frame, provenance = _read_csv_bytes(path)
    if "game_spec" not in frame.columns:
        raise ValueError(f"Scenario source lacks game_spec: {path}")
    pairs: dict[tuple[str, int], tuple[str, str]] = {}
    actions: dict[tuple[str, int], tuple[str, str]] = {}
    for row_index, row in frame.iterrows():
        scenario_id = str(row.get("scenario_id", "")).strip()
        raw_spec = row.get("game_spec")
        if not scenario_id or not isinstance(raw_spec, str) or not raw_spec.strip():
            continue
        try:
            spec = json.loads(raw_spec)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid game_spec at scenario row {row_index}: {exc}") from exc
        players = spec.get("players")
        if not isinstance(players, list):
            raise ValueError(f"game_spec for {scenario_id!r} has no players list")
        for seat, player in enumerate(players):
            raw_actions = player.get("actions") if isinstance(player, dict) else None
            if not isinstance(raw_actions, list) or len(raw_actions) != 2:
                raise ValueError(f"{scenario_id!r} seat {seat} does not have two actions")
            tags = tuple(str(action.get("value_tag", "")).strip() for action in raw_actions)
            if any(tag not in VALUES for tag in tags) or len(set(tags)) != 2:
                raise ValueError(f"Invalid action value tags for {scenario_id!r} seat {seat}: {tags}")
            raw_pair = player.get("value_pair")
            pair = tuple(sorted(str(value).strip() for value in raw_pair or tags))
            if len(pair) != 2 or set(pair) != set(tags):
                raise ValueError(
                    f"value_pair/action mismatch for {scenario_id!r} seat {seat}: "
                    f"pair={pair}, actions={tags}"
                )
            key = (scenario_id, int(seat))
            if key in pairs and (pairs[key] != pair or actions[key] != tags):
                raise ValueError(f"Conflicting duplicate scenario-seat mapping: {key}")
            pairs[key] = pair
            actions[key] = tags
    if not pairs:
        raise ValueError(f"No scenario-seat mappings parsed from {path}")
    provenance["role"] = "canonical scenario source"
    provenance["n_scenario_seats_in_source"] = int(len(pairs))
    return pairs, actions, provenance


def state_key(
    probe: str, scenario_id: str, focal_pair: tuple[str, str], actions: Mapping
) -> str | None:
    """Recode action-index histories by values for the focal seat's contest."""
    probe = (probe or "").strip()
    if probe == "fresh":
        return "fresh"
    if not probe.startswith("prev_"):
        return None
    tokens = probe[len("prev_") :].split("-")
    if len(tokens) != 2:
        return None
    remembered: list[str] = []
    for seat, token in enumerate(tokens):
        tags = actions.get((scenario_id, seat))
        if tags is None:
            return None
        try:
            index = int(token)
        except ValueError:
            return None
        if index < 0 or index >= len(tags):
            return None
        remembered.append(tags[index])
    return _COUNT_TO_STATE[sum(value == focal_pair[0] for value in remembered)]


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_game(
    frame: pd.DataFrame,
    pairs: Mapping,
    actions: Mapping,
    *,
    states: bool,
    source: str = "<dataframe>",
) -> pd.DataFrame:
    """Normalize one result CSV to paired scenario-seat items."""
    missing = {"scenario_id", "seat", "choice_value_tag"} - set(frame.columns)
    if missing:
        raise ValueError(f"{source} lacks required columns: {sorted(missing)}")
    if "probe" not in frame.columns:
        frame = frame.assign(probe="")
    rows: list[dict] = []
    unknown: list[tuple[str, int]] = []
    for _, row in frame.iterrows():
        scenario_id = str(row["scenario_id"]).strip()
        try:
            seat = int(row["seat"])
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"Invalid seat in {source}: {row.get('seat')!r}")
        probe = "" if pd.isna(row.get("probe")) else str(row.get("probe", "")).strip()
        if states != bool(probe):
            continue
        pair = pairs.get((scenario_id, seat))
        if pair is None:
            unknown.append((scenario_id, seat))
            continue
        recoded_state = state_key(probe, scenario_id, pair, actions) if states else ""
        if states and recoded_state is None:
            raise ValueError(f"Cannot recode probe {probe!r} for {(scenario_id, seat)} in {source}")
        chosen = str(row.get("choice_value_tag", "")).strip()
        parsed_ok = _as_bool(row.get("parsed_ok", True))
        answered = bool(parsed_ok and chosen in pair)
        item_key = (scenario_id, seat, probe) if states else (scenario_id, seat)
        rows.append(
            {
                "item_key": item_key,
                "scenario_id": scenario_id,
                "seat": seat,
                "probe": probe,
                "state": recoded_state,
                "pair": pair,
                "chosen": chosen if answered else "",
                "answered": answered,
            }
        )
    if unknown:
        preview = sorted(set(unknown))[:5]
        raise ValueError(f"{source} contains scenario-seats absent from game_spec: {preview}")
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No eligible rows loaded from {source}")
    duplicated = out["item_key"].duplicated(keep=False)
    if duplicated.any():
        examples = out.loc[duplicated, "item_key"].head(5).tolist()
        raise ValueError(f"Duplicate paired items in {source}: {examples}")
    return out.sort_values(["scenario_id", "seat", "probe"], kind="stable").reset_index(drop=True)


def load_all_conditions(
    resolved: Mapping[tuple[str, str], str],
    specs: Sequence[ConditionSpec],
    pairs: Mapping,
    actions: Mapping,
    *,
    workers: int,
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[dict]]:
    """Read/hash each unique CSV once with one shared thread pool."""
    unique_paths = sorted(set(resolved.values()))
    loaded: dict[str, pd.DataFrame] = {}
    provenance_by_path: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_read_csv_bytes, path, usecols=RESULT_COLUMNS): path
            for path in unique_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            frame, provenance = future.result()
            loaded[path] = frame
            provenance["role"] = "evaluated result CSV"
            provenance_by_path[path] = provenance

    spec_by_key = {spec.key: spec for spec in specs}
    conditions: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
    for spec_key, model in sorted(resolved, key=lambda item: (item[0], MODELS.index(item[1]))):
        path = resolved[(spec_key, model)]
        spec = spec_by_key[spec_key]
        conditions[spec_key][model] = load_game(
            loaded[path], pairs, actions, states=spec.states, source=path
        )
    provenance = [provenance_by_path[path] for path in unique_paths]
    return dict(conditions), provenance


def paired_frame(
    base: pd.DataFrame, arm: pd.DataFrame, value: str, pair: tuple[str, str] | None = None
) -> pd.DataFrame:
    """Return the answered-item intersection for a paired two-condition effect."""
    if value not in VALUES:
        raise ValueError(f"Unknown value: {value!r}")

    def eligible(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        # Copy before in-place boolean combination: a Series may be a view of
        # the source column, and mutating it would corrupt later value analyses.
        keep = frame["answered"].copy()
        if pair is None:
            keep &= frame["pair"].apply(lambda contest: value in contest)
        else:
            keep &= frame["pair"] == pair
        columns = ["item_key", "scenario_id", "seat", "pair", "chosen"]
        return frame.loc[keep, columns].rename(
            columns={column: f"{column}_{suffix}" for column in columns if column != "item_key"}
        )

    merged = eligible(base, "base").merge(eligible(arm, "arm"), on="item_key", how="inner")
    if merged.empty:
        return merged
    for column in ("scenario_id", "seat", "pair"):
        if not (merged[f"{column}_base"] == merged[f"{column}_arm"]).all():
            raise ValueError(f"Paired-item {column} misalignment")
    merged["scenario_id"] = merged["scenario_id_base"]
    merged["seat"] = merged["seat_base"]
    merged["pair"] = merged["pair_base"]
    merged["y_base"] = (merged["chosen_base"] == value).astype(int)
    merged["y_arm"] = (merged["chosen_arm"] == value).astype(int)
    merged["d"] = merged["y_arm"] - merged["y_base"]
    return merged


def interaction_frame(
    base_a: pd.DataFrame,
    stake_a: pd.DataFrame,
    base_b: pd.DataFrame,
    stake_b: pd.DataFrame,
    value: str,
) -> pd.DataFrame:
    """Four-condition answered intersection for ``(stake-base)B-(stake-base)A``."""
    frames = []
    for label, frame in (
        ("base_a", base_a),
        ("stake_a", stake_a),
        ("base_b", base_b),
        ("stake_b", stake_b),
    ):
        keep = frame["answered"] & frame["pair"].apply(lambda contest: value in contest)
        sub = frame.loc[keep, ["item_key", "scenario_id", "seat", "pair", "chosen"]].copy()
        sub = sub.rename(
            columns={column: f"{column}_{label}" for column in sub.columns if column != "item_key"}
        )
        frames.append(sub)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="item_key", how="inner")
    if merged.empty:
        return merged
    for label in ("stake_a", "base_b", "stake_b"):
        for column in ("scenario_id", "seat", "pair"):
            if not (merged[f"{column}_base_a"] == merged[f"{column}_{label}"]).all():
                raise ValueError(f"Four-condition {column} misalignment in {label}")
    merged["scenario_id"] = merged["scenario_id_base_a"]
    for label in ("base_a", "stake_a", "base_b", "stake_b"):
        merged[f"y_{label}"] = (merged[f"chosen_{label}"] == value).astype(int)
    merged["effect_a"] = merged["y_stake_a"] - merged["y_base_a"]
    merged["effect_b"] = merged["y_stake_b"] - merged["y_base_b"]
    merged["d"] = merged["effect_b"] - merged["effect_a"]
    return merged


def seat_items(frame: pd.DataFrame, state: str = "fresh") -> pd.DataFrame:
    """Normalize a state arm to scenario-seat keys for one-shot comparisons."""
    out = frame[frame["state"] == state].copy()
    out["item_key"] = list(zip(out["scenario_id"], out["seat"]))
    out["probe"] = ""
    if out["item_key"].duplicated().any():
        raise ValueError(f"State {state!r} does not map uniquely to scenario-seats")
    return out


def mcnemar_midp(n01: int, n10: int) -> float:
    """Two-sided exact McNemar mid-p."""
    discordant = int(n01 + n10)
    if discordant == 0:
        return 1.0
    smaller = min(int(n01), int(n10))
    exact = 2.0 * stats.binom.cdf(smaller, discordant, 0.5)
    midp = exact - stats.binom.pmf(smaller, discordant, 0.5)
    return float(min(1.0, max(0.0, midp)))


def _cluster_arrays(frames: Sequence[pd.DataFrame], scenarios: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {scenario: idx for idx, scenario in enumerate(scenarios)}
    sums = np.zeros((len(frames), len(scenarios)), dtype=float)
    counts = np.zeros_like(sums)
    for model_index, frame in enumerate(frames):
        for scenario_id, group in frame.groupby("scenario_id", sort=False):
            column = index[str(scenario_id)]
            sums[model_index, column] = float(group["d"].sum())
            counts[model_index, column] = float(len(group))
    return sums, counts


def _bootstrap_mean(
    sums: np.ndarray, counts: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> np.ndarray:
    """Scenario bootstrap of the equal-weight mean across the fixed model set."""
    n_scenarios = sums.shape[1]
    indices = rng.integers(0, n_scenarios, size=(n_bootstrap, n_scenarios))
    sampled_sums = sums[:, indices].sum(axis=2)
    sampled_counts = counts[:, indices].sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_model = np.where(sampled_counts > 0, sampled_sums / sampled_counts, np.nan)
    return np.nanmean(per_model, axis=0)


def _signflip_p(
    sums: np.ndarray,
    counts: np.ndarray,
    observed: float,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    """Scenario-clustered sign-flip p-value with the models held fixed."""
    flips = rng.choice((-1.0, 1.0), size=(n_permutations, sums.shape[1]))
    denominators = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_model = np.where(
            denominators > 0, (sums @ flips.T) / denominators, np.nan
        )
    null = np.nanmean(per_model, axis=0)
    extreme = int((np.abs(null) >= abs(observed) - 1e-12).sum())
    return float((extreme + 1) / (n_permutations + 1))


def _holm_values(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty(len(p), dtype=float)
    out[order] = adjusted
    return out


def _bh_values(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = ranked * len(p) / (np.arange(len(p)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(len(p), dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def _by_values(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    harmonic = float(np.sum(1.0 / np.arange(1, len(p) + 1)))
    return np.minimum(1.0, _bh_values(p) * harmonic)


def apply_holm_family(records: list[dict], name: str, expected_size: int) -> dict:
    """Mutate records with Holm-adjusted p-values and describe the family."""
    if len(records) != expected_size:
        raise ValueError(f"Holm family {name!r}: expected {expected_size}, got {len(records)}")
    adjusted = _holm_values([record["perm_p_scenario_clustered"] for record in records])
    raw: dict[str, float] = {}
    corrected: dict[str, float] = {}
    for record, value in zip(records, adjusted):
        record["p_holm"] = float(value)
        raw[record["anchor"]] = float(record["perm_p_scenario_clustered"])
        corrected[record["anchor"]] = float(value)
    return {
        "name": name,
        "size": int(len(records)),
        "raw": raw,
        "holm_adjusted": corrected,
    }


def _model_signs(values: Sequence[float]) -> tuple[int, int, int]:
    return (
        int(sum(value > 0 for value in values)),
        int(sum(value < 0 for value in values)),
        int(sum(value == 0 for value in values)),
    )


def level_summary(
    by_model: Mapping[str, pd.DataFrame],
    value: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    *,
    representation: str,
) -> dict:
    frames: list[pd.DataFrame] = []
    per_model: list[dict] = []
    scenarios = sorted({str(s) for frame in by_model.values() for s in frame["scenario_id"]})
    for model in MODELS:
        frame = by_model[model]
        sub = frame[
            frame["answered"] & frame["pair"].apply(lambda contest: value in contest)
        ].copy()
        if sub.empty:
            raise ValueError(f"No answered {value} items for {model}, {representation}")
        sub["d"] = (sub["chosen"] == value).astype(int)
        frames.append(sub)
        per_model.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "rate": float(sub["d"].mean()),
                "n_answered": int(len(sub)),
            }
        )
    sums, counts = _cluster_arrays(frames, scenarios)
    draws = _bootstrap_mean(sums, counts, rng, n_bootstrap)
    lower, upper = np.percentile(draws, (2.5, 97.5))
    rates = [record["rate"] for record in per_model]
    return {
        "anchor": f"level:{representation}:{value}",
        "representation": representation,
        "representation_label": REPRESENTATIONS[representation]["label"],
        "value": value,
        "mean_rate": float(np.mean(rates)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "across_model_sd": float(np.std(rates, ddof=1)),
        "n_models": int(len(per_model)),
        "n_answered_total": int(sum(record["n_answered"] for record in per_model)),
        "per_model": per_model,
    }


def contrast(
    base_by_model: Mapping[str, pd.DataFrame],
    arm_by_model: Mapping[str, pd.DataFrame],
    value: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    *,
    anchor: str,
    metadata: Mapping[str, Any],
) -> tuple[dict, list[dict]]:
    """Per-model McNemar results plus clustered pooled inference."""
    frames: list[pd.DataFrame] = []
    per_model: list[dict] = []
    scenarios: set[str] = set()
    for model in MODELS:
        paired = paired_frame(base_by_model[model], arm_by_model[model], value)
        if paired.empty:
            raise ValueError(f"Empty paired intersection for {anchor}, {model}")
        frames.append(paired)
        scenarios.update(str(s) for s in paired["scenario_id"])
        n01 = int(((paired["y_base"] == 0) & (paired["y_arm"] == 1)).sum())
        n10 = int(((paired["y_base"] == 1) & (paired["y_arm"] == 0)).sum())
        row = {
            "anchor": anchor,
            "model": model,
            "model_label": MODEL_LABELS[model],
            "value": value,
            "delta": float(paired["d"].mean()),
            "rate_base": float(paired["y_base"].mean()),
            "rate_arm": float(paired["y_arm"].mean()),
            "n_intersection": int(len(paired)),
            "n01": n01,
            "n10": n10,
            "p_midp": mcnemar_midp(n01, n10),
            **metadata,
        }
        per_model.append(row)
    ordered_scenarios = sorted(scenarios)
    sums, counts = _cluster_arrays(frames, ordered_scenarios)
    deltas = [row["delta"] for row in per_model]
    observed = float(np.mean(deltas))
    draws = _bootstrap_mean(sums, counts, rng, n_bootstrap)
    lower, upper = np.percentile(draws, (2.5, 97.5))
    positive, negative, zero = _model_signs(deltas)
    summary = {
        "anchor": anchor,
        "value": value,
        "pooled_delta": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "perm_p_scenario_clustered": _signflip_p(
            sums, counts, observed, rng, n_permutations
        ),
        "across_model_sd": float(np.std(deltas, ddof=1)),
        "n_models": int(len(per_model)),
        "n_positive": positive,
        "n_negative": negative,
        "n_zero": zero,
        "n_intersection_total": int(sum(row["n_intersection"] for row in per_model)),
        "per_model_min": float(min(deltas)),
        "per_model_max": float(max(deltas)),
        "per_model": per_model,
        **metadata,
    }
    return summary, per_model


def interaction(
    base_a: Mapping[str, pd.DataFrame],
    stake_a: Mapping[str, pd.DataFrame],
    base_b: Mapping[str, pd.DataFrame],
    stake_b: Mapping[str, pd.DataFrame],
    value: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    *,
    representation_a: str,
    representation_b: str,
    stake: str,
) -> tuple[dict, list[dict]]:
    anchor = f"interaction:{representation_a}:to:{representation_b}:{stake}:{value}"
    frames: list[pd.DataFrame] = []
    rows: list[dict] = []
    scenarios: set[str] = set()
    for model in MODELS:
        four = interaction_frame(
            base_a[model], stake_a[model], base_b[model], stake_b[model], value
        )
        if four.empty:
            raise ValueError(f"Empty four-condition intersection for {anchor}, {model}")
        frames.append(four)
        scenarios.update(str(s) for s in four["scenario_id"])
        rows.append(
            {
                "anchor": anchor,
                "effect_type": "stake_representation_interaction",
                "model": model,
                "model_label": MODEL_LABELS[model],
                "representation_a": representation_a,
                "representation_b": representation_b,
                "stake": stake,
                "value": value,
                "effect_a": float(four["effect_a"].mean()),
                "effect_b": float(four["effect_b"].mean()),
                "delta": float(four["d"].mean()),
                "n_intersection": int(len(four)),
            }
        )
    sums, counts = _cluster_arrays(frames, sorted(scenarios))
    deltas = [row["delta"] for row in rows]
    observed = float(np.mean(deltas))
    draws = _bootstrap_mean(sums, counts, rng, n_bootstrap)
    lower, upper = np.percentile(draws, (2.5, 97.5))
    positive, negative, zero = _model_signs(deltas)
    summary = {
        "anchor": anchor,
        "representation_a": representation_a,
        "representation_b": representation_b,
        "stake": stake,
        "value": value,
        "pooled_delta": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "perm_p_scenario_clustered": _signflip_p(
            sums, counts, observed, rng, n_permutations
        ),
        "n_models": int(len(rows)),
        "n_positive": positive,
        "n_negative": negative,
        "n_zero": zero,
        "n_intersection_total": int(sum(row["n_intersection"] for row in rows)),
        "per_model_min": float(min(deltas)),
        "per_model_max": float(max(deltas)),
        "per_model": rows,
        "definition": "(stake - baseline) in representation B minus (stake - baseline) in representation A",
    }
    return summary, rows


def ordering(frame: pd.DataFrame) -> tuple[str, ...] | None:
    """Copeland ordering over the three pairwise contests, with rate tiebreaks."""
    wins: defaultdict[str, float] = defaultdict(float)
    seen = 0
    for pair in sorted(set(frame["pair"])):
        sub = frame[(frame["pair"] == pair) & frame["answered"]]
        if sub.empty:
            continue
        seen += 1
        rate = float((sub["chosen"] == pair[0]).mean())
        wins[pair[0]] += 1.0 if rate > 0.5 else (0.5 if rate == 0.5 else 0.0)
        wins[pair[1]] += 1.0 if rate < 0.5 else (0.5 if rate == 0.5 else 0.0)
        wins[pair[0]] += rate * 1e-6
        wins[pair[1]] += (1.0 - rate) * 1e-6
    if seen != 3:
        return None
    return tuple(sorted(VALUES, key=lambda value: (-wins[value], value)))


def _condition(conditions: Mapping, representation: str, family: str, condition: str) -> dict:
    return conditions[f"{representation}:{family}:{condition}"]


def _pooled_history_summary(
    frames: Sequence[pd.DataFrame], rng: np.random.Generator, n_bootstrap: int
) -> dict:
    scenarios = sorted({str(s) for frame in frames for s in frame["scenario_id"]})
    sums, counts = _cluster_arrays(frames, scenarios)
    per_model = [float(frame["d"].mean()) for frame in frames]
    draws = _bootstrap_mean(sums, counts, rng, n_bootstrap)
    lower, upper = np.percentile(draws, (2.5, 97.5))
    return {
        "pooled_delta": float(np.mean(per_model)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_models": int(len(frames)),
        "n_intersection_total": int(sum(len(frame) for frame in frames)),
        "per_model_min": float(min(per_model)),
        "per_model_max": float(max(per_model)),
    }


def _native_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _native_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native_json(item) for item in value]
    return value


def assert_finite_json(value: Any, path: str = "root") -> None:
    """Fail with a precise key for non-finite or unsupported JSON values."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite statistic at {path}: {value!r}")
    elif isinstance(value, np.generic):
        raise TypeError(f"Non-native NumPy scalar at {path}: {type(value).__name__}")


def _camel(*parts: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", "_".join(parts))
    return "".join(word[:1].upper() + word[1:] for word in words)


def manuscript_numbers(stats_out: Mapping[str, Any]) -> dict[str, str]:
    """Create TeX macro values directly from the statistics artifact."""
    macros = {
        "StatModels": str(stats_out["design"]["n_models"]),
        "StatScenarios": str(stats_out["design"]["n_scenarios"]),
        "StatSeats": str(stats_out["design"]["n_scenario_seats"]),
    }
    groups = (
        ("Level", stats_out["representation_levels"]),
        ("Representation", stats_out["representation_contrasts"]),
        ("Stake", stats_out["stake_effects"]),
        ("Interaction", stats_out["stake_representation_interactions"]),
        ("Steering", stats_out["steering_effects"]),
        ("Visibility", stats_out["visibility_effects"]),
        ("Ongoing", stats_out["ongoing_framing"]),
        ("History", stats_out["history_stake_effects"]),
    )
    identity_fields = (
        "representation",
        "representation_a",
        "representation_b",
        "stake",
        "state",
        "value",
    )
    for group_name, records in groups:
        for record in records:
            identity = [str(record[field]) for field in identity_fields if field in record]
            base = "Stat" + _camel(group_name, *identity)
            if "mean_rate" in record:
                macros[base + "Rate"] = f"{record['mean_rate']:.3f}"
            if "pooled_delta" in record:
                macros[base + "Estimate"] = f"{record['pooled_delta']:+.3f}"
            if "ci_lower" in record:
                macros[base + "Lo"] = f"{record['ci_lower']:+.3f}"
                macros[base + "Hi"] = f"{record['ci_upper']:+.3f}"
            if "perm_p_scenario_clustered" in record:
                macros[base + "P"] = f"{record['perm_p_scenario_clustered']:.4f}"
            if "p_holm" in record:
                macros[base + "HolmP"] = f"{record['p_holm']:.4f}"
    return dict(sorted(macros.items()))


def analyse(
    conditions: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int = SEED,
) -> tuple[dict, dict[str, pd.DataFrame], list[str]]:
    """Run the symmetric primary and secondary analyses."""
    rng = np.random.default_rng(seed)
    logs: list[str] = []

    def note(message: str) -> None:
        print(message)
        logs.append(message)

    baseline = {
        representation: _condition(conditions, representation, "main", "baseline")
        for representation in REPRESENTATIONS
    }
    scenarios = sorted(
        {
            str(scenario)
            for frame in baseline["menu"].values()
            for scenario in frame["scenario_id"]
        }
    )
    seats = sorted(
        {
            (str(row.scenario_id), int(row.seat))
            for frame in baseline["menu"].values()
            for row in frame[["scenario_id", "seat"]].itertuples(index=False)
        }
    )
    stats_out: dict[str, Any] = {
        "analysis_version": "reanalysis_20260904_01",
        "design": {
            "n_models": int(len(MODELS)),
            "models": list(MODELS),
            "n_scenarios": int(len(scenarios)),
            "n_scenario_seats": int(len(seats)),
            "values": list(VALUES),
            "stakes": list(STAKES),
            "representations": list(REPRESENTATIONS),
            "bootstrap_draws": int(n_bootstrap),
            "permutation_draws": int(n_permutations),
            "seed": int(seed),
            "scope": "offline game-corpus reanalysis; no solo, abstract-stake, play, or sparsely covered actual repeated-open claims",
        },
    }
    if len(MODELS) == 10 and (len(scenarios), len(seats)) != (102, 150):
        raise ValueError(
            f"Canonical baseline expected 102 scenarios and 150 seats; got "
            f"{len(scenarios)} and {len(seats)}"
        )

    level_records: list[dict] = []
    level_rows: list[dict] = []
    for representation in REPRESENTATIONS:
        for value in VALUES:
            record = level_summary(
                baseline[representation],
                value,
                rng,
                n_bootstrap,
                representation=representation,
            )
            level_records.append(record)
            level_rows.append({**{k: v for k, v in record.items() if k != "per_model"}, "summary": "pooled"})
            for model_record in record["per_model"]:
                level_rows.append(
                    {
                        "anchor": record["anchor"],
                        "representation": representation,
                        "representation_label": REPRESENTATIONS[representation]["label"],
                        "value": value,
                        "summary": "model",
                        **model_record,
                    }
                )
    stats_out["representation_levels"] = level_records
    note(f"baseline levels: {len(level_records)} representation-value cells")

    per_model_effects: list[dict] = []
    representation_contrasts: list[dict] = []
    for representation_a, representation_b in REPRESENTATION_PAIRS:
        for value in VALUES:
            metadata = {
                "effect_type": "representation_contrast",
                "representation_a": representation_a,
                "representation_b": representation_b,
                "definition": "representation B minus representation A",
            }
            summary, rows = contrast(
                baseline[representation_a],
                baseline[representation_b],
                value,
                rng,
                n_bootstrap,
                n_permutations,
                anchor=f"representation:{representation_a}:to:{representation_b}:{value}",
                metadata=metadata,
            )
            representation_contrasts.append(summary)
            per_model_effects.extend(rows)
    representation_family = apply_holm_family(
        representation_contrasts, "baseline representation contrasts", 9
    )
    stats_out["representation_contrasts"] = representation_contrasts
    note("representation contrasts: 9; Holm family complete")

    stake_effects: list[dict] = []
    for representation in REPRESENTATIONS:
        for stake in STAKES:
            stake_data = _condition(conditions, representation, "main", stake)
            for value in VALUES:
                metadata = {
                    "effect_type": "stake_effect",
                    "representation": representation,
                    "stake": stake,
                    "definition": "stake arm minus the within-representation baseline",
                }
                summary, rows = contrast(
                    baseline[representation],
                    stake_data,
                    value,
                    rng,
                    n_bootstrap,
                    n_permutations,
                    anchor=f"stake:{representation}:{stake}:{value}",
                    metadata=metadata,
                )
                stake_effects.append(summary)
                per_model_effects.extend(rows)
    stake_family = apply_holm_family(stake_effects, "within-representation stake effects", 27)
    stats_out["stake_effects"] = stake_effects
    note("stake effects: 27; Holm family complete")

    interactions: list[dict] = []
    for representation_a, representation_b in REPRESENTATION_PAIRS:
        for stake in STAKES:
            stake_a = _condition(conditions, representation_a, "main", stake)
            stake_b = _condition(conditions, representation_b, "main", stake)
            for value in VALUES:
                summary, rows = interaction(
                    baseline[representation_a],
                    stake_a,
                    baseline[representation_b],
                    stake_b,
                    value,
                    rng,
                    n_bootstrap,
                    n_permutations,
                    representation_a=representation_a,
                    representation_b=representation_b,
                    stake=stake,
                )
                interactions.append(summary)
                per_model_effects.extend(rows)
    interaction_family = apply_holm_family(
        interactions, "stake-by-representation interactions", 27
    )
    stats_out["stake_representation_interactions"] = interactions
    stats_out["multiplicity"] = {
        "representation_contrasts": representation_family,
        "stake_effects": stake_family,
        "stake_representation_interactions": interaction_family,
    }
    note("stake-by-representation interactions: 27; Holm family complete")

    steering_effects: list[dict] = []
    order_rows: list[dict] = []
    ordering_stability: dict[str, dict] = {}
    for representation in REPRESENTATIONS:
        steered = _condition(conditions, representation, "main", "steering")
        for value in VALUES:
            metadata = {
                "effect_type": "steering_effect",
                "representation": representation,
                "definition": "HHH-steered minus unsteered baseline",
            }
            summary, rows = contrast(
                baseline[representation],
                steered,
                value,
                rng,
                n_bootstrap,
                n_permutations,
                anchor=f"steering:{representation}:{value}",
                metadata=metadata,
            )
            steering_effects.append(summary)
            per_model_effects.extend(rows)
        baseline_orders: list[tuple[str, ...]] = []
        flipped = 0
        for model in MODELS:
            unsteered_order = ordering(baseline[representation][model])
            steered_order = ordering(steered[model])
            if unsteered_order is None or steered_order is None:
                raise ValueError(f"Incomplete value ordering for {representation}, {model}")
            baseline_orders.append(unsteered_order)
            changed = unsteered_order != steered_order
            flipped += int(changed)
            order_rows.append(
                {
                    "representation": representation,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "ordering_unsteered": " > ".join(unsteered_order),
                    "ordering_steered": " > ".join(steered_order),
                    "flipped": bool(changed),
                }
            )
        first_counts = Counter(order[0] for order in baseline_orders)
        order_counts = Counter(" > ".join(order) for order in baseline_orders)
        ordering_stability[representation] = {
            "baseline": {
                "first_value_counts": {value: int(first_counts.get(value, 0)) for value in VALUES},
                "ordering_counts": {key: int(value) for key, value in sorted(order_counts.items())},
                "n_models": int(len(baseline_orders)),
            },
            "steering": {"n_flipped": int(flipped), "n_models": int(len(MODELS))},
        }
    steering_family = apply_holm_family(steering_effects, "HHH steering effects", 9)
    stats_out["steering_effects"] = steering_effects
    stats_out["ordering_stability"] = ordering_stability
    note("steering effects and ordering stability complete")

    visibility_effects: list[dict] = []
    for representation in ("menu", "open_undisclosed"):
        unobserved = _condition(conditions, representation, "visibility", "unobserved")
        observed = _condition(conditions, representation, "visibility", "observed")
        for value in VALUES:
            metadata = {
                "effect_type": "visibility_effect",
                "representation": representation,
                "definition": "observed counterpart action minus unobserved counterpart action",
            }
            summary, rows = contrast(
                unobserved,
                observed,
                value,
                rng,
                n_bootstrap,
                n_permutations,
                anchor=f"visibility:{representation}:{value}",
                metadata=metadata,
            )
            visibility_effects.append(summary)
            per_model_effects.extend(rows)
    visibility_family = apply_holm_family(
        visibility_effects, "observed-versus-unobserved visibility effects", 6
    )
    stats_out["visibility_effects"] = visibility_effects
    note("visibility effects complete")

    ongoing_effects: list[dict] = []
    for representation in REPRESENTATIONS:
        ongoing = {
            model: seat_items(frame, "fresh")
            for model, frame in _condition(
                conditions, representation, "states", "baseline"
            ).items()
        }
        for value in VALUES:
            metadata = {
                "effect_type": "ongoing_framing",
                "representation": representation,
                "definition": "fresh-state ongoing framing minus one-shot baseline",
            }
            summary, rows = contrast(
                baseline[representation],
                ongoing,
                value,
                rng,
                n_bootstrap,
                n_permutations,
                anchor=f"ongoing:{representation}:{value}",
                metadata=metadata,
            )
            ongoing_effects.append(summary)
            per_model_effects.extend(rows)
    ongoing_family = apply_holm_family(ongoing_effects, "one-shot versus ongoing framing", 9)
    stats_out["ongoing_framing"] = ongoing_effects
    note("one-shot versus fresh-state ongoing framing complete")

    history_grid: list[dict] = []
    history_summaries: list[dict] = []
    for representation in REPRESENTATIONS:
        state_baseline = _condition(conditions, representation, "states", "baseline")
        for stake in STAKES:
            state_stake = _condition(conditions, representation, "states", stake)
            for state in STATE_ORDER:
                pooled_frames: dict[str, list[pd.DataFrame]] = {value: [] for value in VALUES}
                for model in MODELS:
                    base_state = state_baseline[model][state_baseline[model]["state"] == state]
                    stake_state = state_stake[model][state_stake[model]["state"] == state]
                    for value in VALUES:
                        paired = paired_frame(base_state, stake_state, value)
                        if paired.empty:
                            raise ValueError(
                                f"Empty history intersection: {representation}, {stake}, "
                                f"{state}, {value}, {model}"
                            )
                        pooled_frames[value].append(paired)
                        n01 = int(((paired["y_base"] == 0) & (paired["y_arm"] == 1)).sum())
                        n10 = int(((paired["y_base"] == 1) & (paired["y_arm"] == 0)).sum())
                        history_grid.append(
                            {
                                "representation": representation,
                                "stake": stake,
                                "state": state,
                                "value": value,
                                "model": model,
                                "model_label": MODEL_LABELS[model],
                                "delta": float(paired["d"].mean()),
                                "rate_base": float(paired["y_base"].mean()),
                                "rate_stake": float(paired["y_arm"].mean()),
                                "n_intersection": int(len(paired)),
                                "n01": n01,
                                "n10": n10,
                                "p_midp": mcnemar_midp(n01, n10),
                            }
                        )
                for value in VALUES:
                    history_summaries.append(
                        {
                            "anchor": f"history:{representation}:{stake}:{state}:{value}",
                            "representation": representation,
                            "stake": stake,
                            "state": state,
                            "value": value,
                            **_pooled_history_summary(
                                pooled_frames[value], rng, n_bootstrap
                            ),
                        }
                    )
    grid_p = [row["p_midp"] for row in history_grid]
    bh_adjusted = _bh_values(grid_p)
    by_adjusted = _by_values(grid_p)
    for row, p_bh, p_by in zip(history_grid, bh_adjusted, by_adjusted):
        row["p_bh"] = float(p_bh)
        row["p_by"] = float(p_by)
        row["sig_bh_q05"] = bool(p_bh <= 0.05)
        row["sig_bh_q10"] = bool(p_bh <= 0.10)
        row["sig_by_q05"] = bool(p_by <= 0.05)
        row["sig_by_q10"] = bool(p_by <= 0.10)
    stats_out["history_stake_effects"] = history_summaries
    stats_out["history_grid_summary"] = {
        "n_cells": int(len(history_grid)),
        "n_bh_q05": int(sum(row["sig_bh_q05"] for row in history_grid)),
        "n_bh_q10": int(sum(row["sig_bh_q10"] for row in history_grid)),
        "n_by_q05": int(sum(row["sig_by_q05"] for row in history_grid)),
        "n_by_q10": int(sum(row["sig_by_q10"] for row in history_grid)),
        "uncorrected_p05": int(sum(row["p_midp"] < 0.05 for row in history_grid)),
        "correction_scope": "complete representation x stake x state x value x model grid",
    }
    note(f"history grid complete: {len(history_grid)} per-model cells")

    attrition_rows: list[dict] = []
    spec_lookup = {spec.key: spec for spec in condition_specs()}
    for spec_key in sorted(conditions):
        spec = spec_lookup[spec_key]
        for model in MODELS:
            frame = conditions[spec_key][model]
            attrition_rows.append(
                {
                    "representation": spec.representation,
                    "family": spec.family,
                    "condition": spec.condition,
                    "arm": spec.arm,
                    "states": bool(spec.states),
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "n_items": int(len(frame)),
                    "answered": int(frame["answered"].sum()),
                    "not_answered": int((~frame["answered"]).sum()),
                }
            )
    worst = sorted(
        attrition_rows,
        key=lambda row: (-row["not_answered"], row["representation"], row["condition"], row["model"]),
    )[:10]
    stats_out["attrition"] = {
        "n_cells": int(len(attrition_rows)),
        "cells_with_loss": int(sum(row["not_answered"] > 0 for row in attrition_rows)),
        "total_items": int(sum(row["n_items"] for row in attrition_rows)),
        "total_not_answered": int(sum(row["not_answered"] for row in attrition_rows)),
        "worst_cells": worst,
        "paired_inference_rule": "every contrast uses only items answered in all conditions entering that contrast",
    }
    stats_out["secondary_multiplicity"] = {
        "steering_effects": steering_family,
        "visibility_effects": visibility_family,
        "ongoing_framing": ongoing_family,
        "history_grid": stats_out["history_grid_summary"],
    }
    note("attrition and secondary multiplicity summaries complete")

    stats_out["manuscript_numbers"] = manuscript_numbers(stats_out)
    stats_out = _native_json(stats_out)
    assert_finite_json(stats_out)

    effect_summaries = []
    for effect_type, records in (
        ("representation_contrast", representation_contrasts),
        ("stake_effect", stake_effects),
        ("stake_representation_interaction", interactions),
        ("steering_effect", steering_effects),
        ("visibility_effect", visibility_effects),
        ("ongoing_framing", ongoing_effects),
        ("history_stake_effect", history_summaries),
    ):
        for record in records:
            effect_summaries.append(
                {
                    "effect_type": effect_type,
                    **{key: value for key, value in record.items() if not isinstance(value, (list, dict))},
                }
            )

    tables = {
        "representation_levels.csv": pd.DataFrame(level_rows),
        "effect_summaries.csv": pd.DataFrame(effect_summaries),
        "per_model_effects.csv": pd.DataFrame(per_model_effects),
        "history_state_grid.csv": pd.DataFrame(history_grid),
        "orderings_by_model.csv": pd.DataFrame(order_rows),
        "attrition_by_cell.csv": pd.DataFrame(attrition_rows),
    }
    return stats_out, tables, logs


def write_artifacts(
    output_dir: str,
    stats_out: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    provenance: Mapping[str, Any],
    logs: Sequence[str],
) -> None:
    """Write byte-stable CSV/JSON/text artifacts to an empty directory."""
    output_dir = os.path.abspath(output_dir)
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    native_stats = _native_json(stats_out)
    native_provenance = _native_json(provenance)
    assert_finite_json(native_stats, "stats")
    assert_finite_json(native_provenance, "provenance")
    for filename, frame in tables.items():
        frame.to_csv(
            os.path.join(output_dir, filename),
            index=False,
            lineterminator="\n",
            encoding="utf-8",
        )
    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(native_stats, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(
        os.path.join(output_dir, "provenance.json"), "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(native_provenance, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with open(
        os.path.join(output_dir, "manuscript_numbers.tex"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write("% Generated from stats.json by reanalysis.py; do not edit.\n")
        for macro, value in sorted(native_stats["manuscript_numbers"].items()):
            handle.write(f"\\providecommand{{\\{macro}}}{{{value}}}\n")
    with open(
        os.path.join(output_dir, "reanalysis_log.txt"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write("\n".join(logs) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = parser.parse_args(argv)
    if args.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = os.path.abspath(args.output_dir)
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")

    print("Parsing canonical game_spec mappings...")
    pairs, actions, scenario_provenance = load_pairs_and_actions(SCENARIOS_CSV)
    specs = condition_specs()
    print(f"Resolving {len(specs)} exact arms across {len(MODELS)} models...")
    resolved = resolve_all_result_files(GAME_DIR, specs)
    print(f"Reading and hashing {len(set(resolved.values()))} unique result CSVs...")
    conditions, result_provenance = load_all_conditions(
        resolved, specs, pairs, actions, workers=args.workers
    )
    stats_out, tables, logs = analyse(
        conditions,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=SEED,
    )

    condition_inputs = {
        f"{spec_key}:{model}": os.path.relpath(path, ROOT).replace(os.sep, "/")
        for (spec_key, model), path in sorted(resolved.items())
    }
    with open(__file__, "rb") as handle:
        script_raw = handle.read()
    provenance = {
        "analysis_version": "reanalysis_20260904_01",
        "parameters": {
            "bootstrap_draws": int(args.bootstrap),
            "permutation_draws": int(args.permutations),
            "seed": int(SEED),
            "workers": int(args.workers),
        },
        "analysis_script": {
            "relative_path": os.path.relpath(__file__, ROOT).replace(os.sep, "/"),
            "sha256": sha256_bytes(script_raw),
            "bytes": int(len(script_raw)),
        },
        "inputs": [scenario_provenance, *result_provenance],
        "condition_inputs": condition_inputs,
        "selection_rule": "exact parsed arm equality using conflictxgame.common.arm_names.arm_of",
    }
    logs.append(
        f"analysis complete: {len(tables)} CSV tables, stats.json, provenance.json, "
        "manuscript_numbers.tex, and reanalysis_log.txt"
    )
    print(logs[-1])
    write_artifacts(output_dir, stats_out, tables, provenance, logs)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
