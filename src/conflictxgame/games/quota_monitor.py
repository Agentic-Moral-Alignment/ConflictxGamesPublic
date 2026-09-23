"""Read-only completion monitor for a balanced-generation run.

The balanced generator's filtered CSV is its resume source of truth.  This
monitor therefore recounts accepted *per-seat* value-pair marginals with
``kept_pair_counts_from_csv`` instead of counting CSV rows or the legacy
``value1``/``value2`` columns (which describe seat 0 only).  It also requires
the derived ``pair_coverage.json`` report to agree with that recount before it
declares a run complete.

Both files are rewritten by a running generator, so a check retries transient
parse errors and temporarily inconsistent snapshots.  Every invocation writes
``<run-root>/monitor/status.json`` atomically.  The first successful completion
also writes ``COMPLETE.json`` atomically and leaves it in place as a durable
sentinel.

Example for the HHH paper/both-assistant Message Batches continuation::

    python -m conflictxgame.games.quota_monitor \
        --run-root data/HHH_new_paper_bothAI_anthropic_batch_20260826_01 \
        --value-set HHH_new --model claude-sonnet-5 --target 50 \
        --expected-prompt-style paper \
        --expected-assistant-seats prob:1.0 \
        --allow-shadow-to-off-prescreen-transition

Exit codes are stable for schedulers: 0 complete, 10 incomplete, 11 waiting
for the first output files, 20 unreadable after retries, 21 wrong run/config,
22 report/CSV disagreement, 23 stopped before quota, and 24 state-write error.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from enum import IntEnum
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable, Iterable, Mapping, Optional, Sequence

from conflictxgame.common.model_wrappers import short_model_name
from conflictxgame.common.portable_lock import file_lock
from conflictxgame.common.values import load_value_set
from conflictxgame.games.pair_quota import (
    Pair,
    all_value_pairs,
    kept_pair_counts_from_csv,
    pair_label,
)


# Resolve the production default from this source file rather than the process
# cwd.  Windows Task Scheduler otherwise starts relative paths under
# ``C:\Windows\System32`` unless its action supplies a working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "data" / "HHH_new_paper_bothAI_20260825_01"
DEFAULT_VALUE_SET = "HHH_new"
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TARGET = 50
DEFAULT_EXPECTED_GENERATION = {
    "prompt_style": "paper",
    "guidance": [],
    "assistant_seats": "prob:1.0",
    "prescreen": "shadow",
    "prescreen_criteria": ["exclusivity", "mapping", "obvious"],
    "short_circuit": True,
    "revise_failures": 0,
    "guidedness_votes": 3,
    "judge_model": "gpt-5.6-luna",
    "screen_model": "gpt-5.6-luna",
    "batch_size": 1,
    "seed": 0,
}


class ExitCode(IntEnum):
    """Process results consumed by a periodic scheduler."""

    COMPLETE = 0
    INCOMPLETE = 10
    WAITING = 11
    READ_ERROR = 20
    INVALID_CONFIG = 21
    INCONSISTENT = 22
    STOPPED_INCOMPLETE = 23
    STATE_WRITE_ERROR = 24


class MissingInputsError(RuntimeError):
    """The run has not written both monitor inputs yet."""


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with an explicit ``Z`` suffix."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON via a same-directory temporary file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_inputs_once(
    coverage_path: Path,
    csv_path: Path,
    value_names: Iterable[str],
) -> tuple[dict, dict[Pair, int], dict[Pair, int]]:
    """Read one report/CSV snapshot, raising on absence or parse failure."""

    missing = [str(path) for path in (coverage_path, csv_path) if not path.is_file()]
    if missing:
        raise MissingInputsError("missing " + ", ".join(missing))

    with coverage_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"{coverage_path} must contain a JSON object")

    kept, generated = kept_pair_counts_from_csv(str(csv_path), value_names)
    return report, kept, generated


def _report_pair_counts(report: Mapping[str, object]) -> tuple[dict[str, int], dict[str, int]]:
    """Extract and validate ``kept``/``generated`` mappings from a report."""

    rows = report.get("pairs")
    if not isinstance(rows, list):
        raise ValueError("pair_coverage.json has no valid 'pairs' list")

    kept: dict[str, int] = {}
    generated: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("pair_coverage.json contains a non-object pair row")
        label = row.get("pair")
        if not isinstance(label, str) or not label:
            raise ValueError("pair_coverage.json contains a pair row without a label")
        if label in kept:
            raise ValueError(f"pair_coverage.json repeats pair {label!r}")
        try:
            raw_kept = row["kept"]
            raw_generated = row["generated"]
            pair_kept = int(raw_kept)
            pair_generated = int(raw_generated)
            if (
                isinstance(raw_kept, bool)
                or isinstance(raw_generated, bool)
                or pair_kept != raw_kept
                or pair_generated != raw_generated
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"pair_coverage.json has invalid counts for {label!r}") from exc
        if pair_kept < 0 or pair_generated < 0 or pair_kept > pair_generated:
            raise ValueError(f"pair_coverage.json has impossible counts for {label!r}")
        kept[label] = pair_kept
        generated[label] = pair_generated
    return kept, generated


def _csv_pair_counts(
    pairs: Sequence[Pair],
    kept: Mapping[Pair, int],
    generated: Mapping[Pair, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """Normalize sparse recount mappings to every expected pair label."""

    return (
        {pair_label(pair): int(kept.get(pair, 0)) for pair in pairs},
        {pair_label(pair): int(generated.get(pair, 0)) for pair in pairs},
    )


def _accepted_row_provenance_errors(
    csv_path: Path,
    *,
    expected_value_set: str,
    expected_generation: Mapping[str, object],
) -> list[str]:
    """Validate that every accepted row belongs to the monitored method.

    ``pair_coverage.json.config`` describes only the process that most
    recently wrote the report.  A resumed CSV can contain older rows from a
    different prompt, judge, value set, or seat mix, so completion must also
    be guarded by the provenance stamped into each accepted ``game_spec``.
    """

    import pandas as pd

    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return []
    if frame.empty:
        return []

    required = {"scenario_id", "keep_scenario", "value_set", "agent_kinds", "game_spec"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        return [f"accepted-row provenance columns are missing: {missing!r}"]

    accepted = frame[
        frame["keep_scenario"].astype(str).str.strip().str.lower().eq("true")
    ]
    errors: list[str] = []
    for row_number, (_, row) in enumerate(accepted.iterrows(), start=1):
        scenario_id = str(row.get("scenario_id") or f"accepted row {row_number}")
        prefix = f"{scenario_id}: "
        if str(row.get("value_set")) != expected_value_set:
            errors.append(
                prefix
                + f"value_set is {row.get('value_set')!r}, expected {expected_value_set!r}"
            )

        try:
            agent_kinds = json.loads(row.get("agent_kinds"))
        except (TypeError, json.JSONDecodeError):
            agent_kinds = None
        if agent_kinds != ["assistant", "assistant"]:
            errors.append(
                prefix
                + f"agent_kinds is {agent_kinds!r}, expected two assistant seats"
            )

        try:
            spec = json.loads(row.get("game_spec"))
        except (TypeError, json.JSONDecodeError):
            spec = None
        if not isinstance(spec, dict):
            errors.append(prefix + "game_spec is missing or invalid JSON")
            continue
        if spec.get("value_set") != expected_value_set:
            errors.append(
                prefix
                + f"game_spec.value_set is {spec.get('value_set')!r}, "
                + f"expected {expected_value_set!r}"
            )
        players = spec.get("players")
        spec_kinds = (
            [player.get("agent_kind") for player in players]
            if isinstance(players, list) and all(isinstance(player, dict) for player in players)
            else None
        )
        if spec_kinds != ["assistant", "assistant"]:
            errors.append(
                prefix
                + f"game_spec player kinds are {spec_kinds!r}, expected two assistant seats"
            )

        metadata = spec.get("metadata")
        generation = metadata.get("generation") if isinstance(metadata, dict) else None
        if not isinstance(generation, dict):
            errors.append(prefix + "game_spec.metadata.generation is missing")
            continue
        for key, expected in expected_generation.items():
            if generation.get(key) != expected:
                errors.append(
                    prefix
                    + f"generation.{key} is {generation.get(key)!r}, expected {expected!r}"
                )

        # Keep status.json bounded even if a whole foreign corpus was appended.
        if len(errors) >= 25:
            errors.append("additional accepted-row provenance errors omitted")
            break
    return errors


def _base_state(
    run_root: Path,
    coverage_path: Path,
    csv_path: Path,
    target: int,
    expected_labels: Sequence[str],
) -> dict:
    """Fields present in every persisted monitor result."""

    return {
        "schema_version": 1,
        "checked_at": _utc_now(),
        "run_root": str(run_root.resolve()),
        "coverage_path": str(coverage_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "target_per_pair": target,
        "expected_pairs": list(expected_labels),
    }


def check_run(
    *,
    run_root: Path,
    value_names: Iterable[str],
    model: str = DEFAULT_MODEL,
    target: int = DEFAULT_TARGET,
    expected_value_set: str = DEFAULT_VALUE_SET,
    expected_generation: Optional[Mapping[str, object]] = None,
    expected_prompt_style: Optional[str] = "paper",
    expected_assistant_seats: Optional[str] = "prob:1.0",
    allow_shadow_to_off_prescreen_transition: bool = False,
    retries: int = 5,
    retry_delay: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[dict, ExitCode]:
    """Check one run and return its serializable state plus stable exit code.

    A full read is retried not only on parser errors but also when the report
    and CSV disagree.  The generator writes the CSV before the report at the
    end of a round, so that disagreement is normally just an in-flight update.
    """

    run_root = Path(run_root)
    coverage_path = run_root / "games" / "pair_coverage.json"
    csv_path = run_root / f"{short_model_name(model)}_games.csv"
    # Some callers may supply a one-shot iterator.  The recount and expected
    # pair construction both need to traverse the same names.
    value_names = tuple(value_names)
    pairs = all_value_pairs(value_names)
    expected_labels = [pair_label(pair) for pair in pairs]
    expected_set = set(expected_labels)
    method_config = dict(
        DEFAULT_EXPECTED_GENERATION
        if expected_generation is None
        else expected_generation
    )
    if expected_prompt_style is not None:
        method_config["prompt_style"] = expected_prompt_style
    if expected_assistant_seats is not None:
        method_config["assistant_seats"] = expected_assistant_seats
    # The production batch continuation deliberately turns the prescreen off,
    # while its forked accepted rows were stamped with the former shadow
    # prescreen.  In that one migration mode, ignore only the two fields whose
    # values necessarily differ.  Every other report and accepted-row
    # provenance invariant remains mandatory.
    if allow_shadow_to_off_prescreen_transition:
        method_config.pop("prescreen", None)
        method_config.pop("screen_model", None)
    attempts = max(1, int(retries))
    delay = max(0.0, float(retry_delay))
    last_error: Optional[Exception] = None
    last_snapshot: Optional[
        tuple[
            dict,
            dict[str, int],
            dict[str, int],
            dict[str, int],
            dict[str, int],
            list[str],
        ]
    ] = None
    attempt_used = 0

    for attempt in range(1, attempts + 1):
        attempt_used = attempt
        try:
            report, csv_kept_raw, csv_generated_raw = _read_inputs_once(
                coverage_path, csv_path, value_names
            )
            report_kept, report_generated = _report_pair_counts(report)
            csv_kept, csv_generated = _csv_pair_counts(
                pairs, csv_kept_raw, csv_generated_raw
            )
            provenance_errors = _accepted_row_provenance_errors(
                csv_path,
                expected_value_set=expected_value_set,
                expected_generation=method_config,
            )
            last_snapshot = (
                report,
                report_kept,
                report_generated,
                csv_kept,
                csv_generated,
                provenance_errors,
            )
            # Retry a live between-file snapshot.  A durable mismatch is
            # classified below after the final attempt.
            if report_kept == csv_kept and report_generated == csv_generated:
                break
            last_error = ValueError("pair_coverage.json and CSV counts disagree")
        except Exception as exc:  # monitor boundary: persist the failure, do not crash
            last_error = exc
        if attempt < attempts:
            sleep_fn(delay)

    state = _base_state(run_root, coverage_path, csv_path, target, expected_labels)
    state["attempts"] = attempt_used
    state["allow_shadow_to_off_prescreen_transition"] = bool(
        allow_shadow_to_off_prescreen_transition
    )

    if last_snapshot is None:
        if isinstance(last_error, MissingInputsError):
            state.update(
                status="waiting",
                message=str(last_error),
                error_type=type(last_error).__name__,
            )
            return state, ExitCode.WAITING
        state.update(
            status="read_error",
            message=str(last_error) if last_error else "unable to read run outputs",
            error_type=type(last_error).__name__ if last_error else "UnknownError",
        )
        return state, ExitCode.READ_ERROR

    (
        report,
        report_kept,
        report_generated,
        csv_kept,
        csv_generated,
        provenance_errors,
    ) = last_snapshot
    report_labels = set(report_kept)
    config = report.get("config")
    config = config if isinstance(config, dict) else {}
    notes = report.get("notes")
    notes = list(notes) if isinstance(notes, list) else []
    state.update(
        csv={"kept": csv_kept, "generated": csv_generated},
        report={
            "target_per_pair": report.get("target_per_pair"),
            "complete": report.get("complete"),
            "kept": report_kept,
            "generated": report_generated,
            "config": config,
            "notes": notes,
        },
        agreement=(report_kept == csv_kept and report_generated == csv_generated),
        accepted_row_provenance_valid=(not provenance_errors),
    )

    config_errors = []
    if report.get("target_per_pair") != target:
        config_errors.append(
            f"target_per_pair is {report.get('target_per_pair')!r}, expected {target}"
        )
    if report_labels != expected_set:
        config_errors.append(
            "pair set is "
            f"{sorted(report_labels)!r}, expected {sorted(expected_set)!r}"
        )
    for key, expected in method_config.items():
        if config.get(key) != expected:
            config_errors.append(
                f"{key} is {config.get(key)!r}, expected {expected!r}"
            )
    config_errors.extend(provenance_errors)
    if config_errors:
        state.update(
            status="invalid_config",
            message="; ".join(config_errors),
            config_errors=config_errors,
        )
        return state, ExitCode.INVALID_CONFIG

    if not state["agreement"]:
        state.update(
            status="inconsistent",
            message="pair_coverage.json and CSV counts still disagree after retries",
        )
        return state, ExitCode.INCONSISTENT

    deficits = {label: max(0, target - csv_kept[label]) for label in expected_labels}
    state["deficits"] = deficits
    counts_complete = all(deficit == 0 for deficit in deficits.values())
    report_complete = report.get("complete")
    if not isinstance(report_complete, bool) or report_complete != counts_complete:
        state.update(
            status="inconsistent",
            message=(
                f"report complete={report_complete!r} disagrees with "
                f"recounted completion={counts_complete}"
            ),
        )
        return state, ExitCode.INCONSISTENT

    if counts_complete:
        state.update(
            status="complete",
            message=f"every expected value pair has at least {target} accepted seat marginal(s)",
        )
        return state, ExitCode.COMPLETE

    stopped = [str(note) for note in notes if str(note).strip().lower().startswith("stopped")]
    if stopped:
        state.update(
            status="stopped_incomplete",
            message="; ".join(stopped),
        )
        return state, ExitCode.STOPPED_INCOMPLETE

    remaining = sum(deficits.values())
    state.update(
        status="incomplete",
        message=f"quota is short by {remaining} accepted seat marginal(s)",
    )
    return state, ExitCode.INCOMPLETE


def persist_state(run_root: Path, state: Mapping[str, object], code: ExitCode) -> None:
    """Atomically persist current state and, once, the completion sentinel."""

    monitor_dir = Path(run_root) / "monitor"
    status_payload = dict(state)
    status_payload["exit_code"] = int(code)
    _atomic_write_json(monitor_dir / "status.json", status_payload)

    if code is ExitCode.COMPLETE:
        complete_path = monitor_dir / "COMPLETE.json"
        # Preserve the first observed completion time even if scheduled checks
        # overlap.  The advisory lock is shared by all instances of this tool.
        with file_lock(str(complete_path) + ".lock"):
            if not complete_path.exists():
                complete_payload = dict(status_payload)
                complete_payload["completed_at"] = status_payload["checked_at"]
                _atomic_write_json(complete_path, complete_payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for one scheduled check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--value-set", "-v", default=DEFAULT_VALUE_SET)
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--target", "-k", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--expected-prompt-style", default="paper")
    parser.add_argument("--expected-assistant-seats", default="prob:1.0")
    parser.add_argument(
        "--allow-shadow-to-off-prescreen-transition",
        action="store_true",
        help=(
            "allow a fork to contain both legacy shadow-prescreen rows and new "
            "prescreen-off rows by ignoring only prescreen and screen_model provenance"
        ),
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one check, persist it, print a concise result, and return its code."""

    args = parse_args(argv)
    try:
        values = load_value_set(args.value_set)
        state, code = check_run(
            run_root=args.run_root,
            value_names=values.keys(),
            model=args.model,
            target=args.target,
            expected_value_set=args.value_set,
            expected_prompt_style=args.expected_prompt_style,
            expected_assistant_seats=args.expected_assistant_seats,
            allow_shadow_to_off_prescreen_transition=(
                args.allow_shadow_to_off_prescreen_transition
            ),
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
    except Exception as exc:  # includes a misspelled/unreadable value set
        run_root = Path(args.run_root)
        coverage_path = run_root / "games" / "pair_coverage.json"
        csv_path = run_root / f"{short_model_name(args.model)}_games.csv"
        state = _base_state(run_root, coverage_path, csv_path, args.target, [])
        state.update(
            status="read_error",
            message=str(exc),
            error_type=type(exc).__name__,
        )
        code = ExitCode.READ_ERROR

    try:
        persist_state(args.run_root, state, code)
    except Exception as exc:
        print(f"quota monitor could not write state: {exc}", file=sys.stderr)
        return int(ExitCode.STATE_WRITE_ERROR)

    print(f"{state['status']}: {state['message']}")
    if "deficits" in state:
        for label, deficit in state["deficits"].items():
            kept = state["csv"]["kept"][label]
            print(f"  {label}: kept {kept}/{args.target}, deficit {deficit}")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
