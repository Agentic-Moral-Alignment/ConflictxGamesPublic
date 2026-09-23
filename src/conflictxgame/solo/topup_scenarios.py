"""Resume solo generation until every HHH value pair reaches a kept quota.

This is a thin, resumable orchestrator around the existing two-stage solo
generator and unchanged solo filter.  It never removes corpus rows: each
round extends the pair JSON files, filters only scenario ids not already in
the corpus CSV, and stops once every pair has at least the requested number
of ``keep_scenario=True`` rows.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, Tuple

import pandas as pd

from conflictxgame.common.config import add_config_argument, parse_with_config
from conflictxgame.solo.generate_scenarios import HHH_PAIRS


Pair = Tuple[str, str]


def parse_args(argv=None):
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--target-kept-per-pair", type=int, default=50)
    parser.add_argument("--generator-config", type=str, required=True)
    parser.add_argument("--filter-config", type=str, required=True)
    parser.add_argument("--scenarios-dir", type=str, required=True)
    parser.add_argument("--corpus-csv", type=str, required=True)
    parser.add_argument("--generator-output-prefix", type=str, required=True)
    parser.add_argument("--candidate-batch-cap", type=int, default=100)
    parser.add_argument("--candidate-batch-min", type=int, default=25)
    parser.add_argument("--acceptance-rate-floor", type=float, default=0.10)
    parser.add_argument("--oversample-factor", type=float, default=1.10)
    parser.add_argument("--max-rounds", type=int, default=10)
    args, _config_defaults, _explicit_cli = parse_with_config(parser, argv)
    return args


def canonical_pair(value1: object, value2: object) -> Pair:
    return tuple(sorted((str(value1).strip(), str(value2).strip())))  # type: ignore[return-value]


def corpus_pair_counts(csv_path: os.PathLike | str) -> tuple[Dict[Pair, int], Dict[Pair, int]]:
    """Return kept and total row counts by canonical value pair."""
    kept: Dict[Pair, int] = {canonical_pair(*pair): 0 for pair in HHH_PAIRS}
    total: Dict[Pair, int] = dict(kept)
    path = Path(csv_path)
    if not path.exists():
        return kept, total

    frame = pd.read_csv(path)
    required = {"value1", "value2", "keep_scenario"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    keep_mask = frame["keep_scenario"].astype(str).str.strip().str.lower().eq("true")
    for index, row in frame.iterrows():
        pair = canonical_pair(row["value1"], row["value2"])
        if pair not in total:
            continue
        total[pair] += 1
        if bool(keep_mask.loc[index]):
            kept[pair] += 1
    return kept, total


def candidate_count(
    deficit: int,
    kept: int,
    total: int,
    *,
    minimum: int,
    maximum: int,
    rate_floor: float,
    oversample_factor: float,
) -> int:
    """Size one candidate round from the observed judge acceptance rate."""
    if deficit <= 0:
        return 0
    observed_rate = kept / total if total else 0.0
    effective_rate = max(observed_rate, rate_floor)
    estimate = math.ceil((deficit / effective_rate) * oversample_factor)
    return min(maximum, max(minimum, estimate))


def _run(command: Iterable[str]) -> None:
    rendered = subprocess.list2cmdline(list(command))
    print(f"Running: {rendered}", flush=True)
    subprocess.run(list(command), check=True)


def _scenario_file(args, pair: Pair) -> Path:
    value1, value2 = pair
    return Path(args.scenarios_dir) / (
        f"{args.generator_output_prefix}_{value1}_{value2}.json"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.target_kept_per_pair <= 0:
        raise SystemExit("--target-kept-per-pair must be positive")
    if args.candidate_batch_min <= 0 or args.candidate_batch_cap < args.candidate_batch_min:
        raise SystemExit("candidate batch bounds must satisfy 0 < min <= cap")
    if not 0 < args.acceptance_rate_floor <= 1:
        raise SystemExit("--acceptance-rate-floor must be in (0, 1]")
    if args.oversample_factor < 1:
        raise SystemExit("--oversample-factor must be at least 1")

    pairs = tuple(canonical_pair(*pair) for pair in HHH_PAIRS)
    Path(args.scenarios_dir).mkdir(parents=True, exist_ok=True)

    for round_index in range(1, args.max_rounds + 1):
        kept, total = corpus_pair_counts(args.corpus_csv)
        deficits = {
            pair: max(0, args.target_kept_per_pair - kept[pair])
            for pair in pairs
        }
        print(
            f"Round {round_index}: "
            + ", ".join(
                f"{'/'.join(pair)}={kept[pair]}/{args.target_kept_per_pair}"
                for pair in pairs
            ),
            flush=True,
        )
        if not any(deficits.values()):
            print("All solo value-pair kept quotas are complete.", flush=True)
            return 0

        for pair in pairs:
            deficit = deficits[pair]
            if deficit <= 0:
                continue
            count = candidate_count(
                deficit,
                kept[pair],
                total[pair],
                minimum=args.candidate_batch_min,
                maximum=args.candidate_batch_cap,
                rate_floor=args.acceptance_rate_floor,
                oversample_factor=args.oversample_factor,
            )
            value1, value2 = pair
            print(
                f"Generating {count} new candidates for {value1}/{value2} "
                f"(kept deficit {deficit}).",
                flush=True,
            )
            _run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "conflictxgame.solo.generate_scenarios",
                    "--config",
                    args.generator_config,
                    "--value1",
                    value1,
                    "--value2",
                    value2,
                    "--num-scenarios",
                    str(count),
                    "--add-to-existing",
                ]
            )
            source = _scenario_file(args, pair)
            if not source.exists():
                raise FileNotFoundError(f"Generator did not create expected file {source}")
            _run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "conflictxgame.solo.filter_scenarios",
                    "--config",
                    args.filter_config,
                    "--input-file",
                    str(source),
                ]
            )

    kept, _ = corpus_pair_counts(args.corpus_csv)
    short = {
        "/".join(pair): args.target_kept_per_pair - count
        for pair, count in kept.items()
        if count < args.target_kept_per_pair
    }
    print(f"Stopped after {args.max_rounds} rounds with deficits: {short}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
