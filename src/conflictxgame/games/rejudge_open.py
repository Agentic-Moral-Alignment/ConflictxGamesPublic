"""Re-judge stored open-ended replies with a different judge model.

Every free-text result in the study runs through one judge (GPT-5.6 Luna),
with no second judge, no human agreement study, and the two actions always
shown in menu order. This module bounds that dependence without re-running a
single evaluated model: each open-mode result row already stores the opening
the judge saw (``open_user_prompt``) and the target's reply
(``open_completion``), so a second judge can classify the identical text.

Two conditions are supported, selected with ``--condition``:

``verbatim``
    The original judge prompt, byte for byte, in menu order. Differences from
    the stored labels are differences between judges.
``swap``
    The same prompt with the action order reversed, so letter A shows menu
    action B. Read against ``verbatim`` from the same judge, the difference is
    that judge's position bias.

Output is a drop-in copy of each input CSV, same filename, written to its own
directory: ``choice_action``, ``choice_label``, ``choice_value_tag``,
``parsed_ok``, ``open_judge_likert``, ``reasoning`` and ``judge_model`` carry
the new judge's verdict, already mapped back to menu order, and the original
values are kept beside them as ``orig_*`` columns. Paired analyses can
therefore read a re-judged directory exactly as they read the original.

A failed API call is never written as a verdict. It returns ``''`` from the
model wrapper, the row stays ``rejudge_status == "pending"``, and the next run
retries it. Only a real judge response that cannot be parsed becomes
``INVALID``. This matters on OpenRouter, where a transient 429 window has
already turned one sweep into silent data loss.

Example::

    export CONFLICTXGAME_MAX_RETRIES=4
    python -m conflictxgame.games.rejudge_open \\
        --judge-model openai:deepseek/deepseek-v4-pro-0813 --condition verbatim

New module for this fork. See ``ATTRIBUTION.md``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from conflictxgame.common.arm_names import split_result_name
from conflictxgame.common.model_wrappers import ModelWrapper
from conflictxgame.common.portable_lock import replace_with_retry
from conflictxgame.common.utils import MAX_LIKERT, parse_json
from conflictxgame.games import open_ended
from conflictxgame.games.game_structures import GameSpec

logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_RESULTS_DIR = os.path.join(
    _ROOT, "results", "HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
)
DEFAULT_SCENARIOS_CSV = os.path.join(
    _ROOT, "data", "HHH_new_paper_bothAI_anthropic_batch_20260826_01", "claude-sonnet-5_games.csv"
)
DEFAULT_OUTPUT_ROOT = os.path.join(_ROOT, "results", "rejudge")

#: The ten checkpoints of the paper, by result-file slug.
MODELS = (
    "gemma-3-12b", "gemma-3-27b", "gemma-4-26b-a4b", "gemma-4-31B-it",
    "gpt-oss-120b", "gpt-oss-20b", "llama-3.3-70b", "llama-4-maverick",
    "qwen3-235b-a22b", "qwen3-32b",
)

#: Arms behind the paper's lead claim: baseline and the three stakes, in both
#: free-text representations (counterpart undisclosed, then disclosed).
LEAD_ARMS = (
    "", "_stakes-payoff", "_stakes-alignment", "_stakes-usefulness",
    "_counterpart", "_counterpart_stakes-payoff",
    "_counterpart_stakes-alignment", "_counterpart_stakes-usefulness",
)

CONDITIONS = ("verbatim", "swap")

#: Columns the re-judge overwrites; each original is kept as ``orig_<name>``.
VERDICT_COLUMNS = (
    "choice_action", "choice_label", "choice_value_tag", "parsed_ok",
    "open_judge_likert", "reasoning", "judge_model",
)

PENDING, DONE = "pending", "done"


def presentation_order(condition: str, arity: int) -> List[int]:
    """Menu indices in the order the judge sees them under ``condition``."""
    if condition == "verbatim":
        return list(range(arity))
    if condition == "swap":
        return list(reversed(range(arity)))
    raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")


def map_verdict(
    letter_index: object, likert: object, order: Sequence[int]
) -> Tuple[object, object]:
    """Map a verdict given under ``order`` back to menu order.

    Args:
        letter_index: The parsed letter position, or ``"INVALID"``.
        likert: The parsed Likert rating, or ``""``.
        order: The presentation order the judge saw.

    Returns:
        ``(menu_index, likert)`` in the original menu's terms. The Likert
        scale runs from presented action A to presented action B, so a
        reversed binary menu reflects it.
    """
    action = letter_index if letter_index == "INVALID" else order[letter_index]
    if likert != "" and list(order) == [1, 0]:
        likert = MAX_LIKERT + 1 - int(likert)
    return action, likert


def long_path(path: str) -> str:
    """Return ``path`` in a form Windows accepts beyond MAX_PATH.

    Result filenames run past 100 characters and the repository sits deep in
    a Dropbox tree, so plain paths overflow the 260-character limit there.
    """
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


def find_input_files(
    results_dir: str, models: Iterable[str], arms: Iterable[str]
) -> Dict[Tuple[str, str], str]:
    """Resolve exactly one open-mode result file per (model, arm).

    Arms are matched by exact token, never by glob or sort order: in this
    directory ``_stakes-alignment`` and ``_observed`` sort ahead of the bare
    baseline, which has twice put a manipulated arm in a baseline's place.

    Raises:
        FileNotFoundError: A (model, arm) has no file.
        ValueError: A (model, arm) matches more than one file.
    """
    models, arms = list(models), list(arms)
    hits: Dict[Tuple[str, str], List[str]] = {}
    for name in os.listdir(long_path(results_dir)):
        try:
            parts = split_result_name(name)
        except ValueError:
            continue
        if parts["mode"] != "open" or parts["slug"] not in models:
            continue
        if parts["arm"] in arms:
            hits.setdefault((parts["slug"], parts["arm"]), []).append(name)
    resolved = {}
    for model in models:
        for arm in arms:
            names = hits.get((model, arm), [])
            if not names:
                raise FileNotFoundError(f"No open result for {model} arm {arm or 'baseline'!r}")
            if len(names) > 1:
                raise ValueError(f"Ambiguous open results for {model} arm {arm!r}: {names}")
            resolved[(model, arm)] = os.path.join(results_dir, names[0])
    return resolved


def load_specs(scenarios_csv: str) -> Dict[str, GameSpec]:
    """Map every scenario id in the corpus CSV to its :class:`GameSpec`."""
    frame = pd.read_csv(long_path(scenarios_csv), dtype=str, keep_default_na=False)
    return {row["scenario_id"]: GameSpec.from_json(row["game_spec"]) for _, row in frame.iterrows()}


def prepare_frame(input_path: str, output_path: str) -> pd.DataFrame:
    """Load an input result file, carrying over verdicts already re-judged.

    The frame is always rebuilt from the original input, so the ``orig_*``
    columns cannot be overwritten by a resumed run.
    """
    frame = pd.read_csv(long_path(input_path), dtype=str, keep_default_na=False)
    if frame.duplicated(["scenario_id", "seat"]).any():
        raise ValueError(f"{input_path} has repeated (scenario_id, seat) rows; expected one-shot arms only")
    for column in VERDICT_COLUMNS:
        frame["orig_" + column] = frame[column]
    frame["rejudge_condition"] = ""
    frame["rejudge_order"] = ""
    frame["rejudge_raw"] = ""
    frame["rejudge_status"] = PENDING

    if os.path.exists(long_path(output_path)):
        previous = pd.read_csv(long_path(output_path), dtype=str, keep_default_na=False)
        previous = previous[previous["rejudge_status"] == DONE].set_index(["scenario_id", "seat"])
        carried = list(VERDICT_COLUMNS) + [
            "rejudge_condition", "rejudge_order", "rejudge_raw", "rejudge_status"
        ]
        keyed = frame.set_index(["scenario_id", "seat"])
        common = keyed.index.intersection(previous.index)
        keyed.loc[common, carried] = previous.loc[common, carried]
        frame = keyed.reset_index()[frame.columns]
    return frame


def apply_verdict(
    frame: pd.DataFrame, index: int, spec: GameSpec, seat: int, response: str,
    order: Sequence[int], condition: str, judge_name: str,
) -> None:
    """Write one judge response into row ``index`` in the evaluator's format."""
    letter, likert = open_ended.parse_open_judge(response, spec.players[seat].arity)
    action, likert = map_verdict(letter, likert, order)
    if action == "INVALID":
        label = tag = "INVALID"
    else:
        label = spec.players[seat].actions[action].label
        tag = spec.players[seat].actions[action].value_tag
    parsed = parse_json(response)
    frame.at[index, "choice_action"] = str(action)
    frame.at[index, "choice_label"] = label
    frame.at[index, "choice_value_tag"] = tag
    frame.at[index, "parsed_ok"] = str(action != "INVALID")
    frame.at[index, "open_judge_likert"] = str(likert)
    frame.at[index, "reasoning"] = str(parsed.get("reasoning", "")) if isinstance(parsed, dict) else ""
    frame.at[index, "judge_model"] = judge_name
    frame.at[index, "rejudge_condition"] = condition
    frame.at[index, "rejudge_order"] = ",".join(map(str, order))
    frame.at[index, "rejudge_raw"] = response
    frame.at[index, "rejudge_status"] = DONE


def write_frame(frame: pd.DataFrame, output_path: str) -> None:
    """Write ``frame`` atomically, so an interrupted run leaves the last good copy."""
    tmp = output_path + ".tmp"
    frame.to_csv(long_path(tmp), index=False)
    replace_with_retry(long_path(tmp), long_path(output_path))


def rejudge_file(
    input_path: str, output_path: str, specs: Dict[str, GameSpec], client: ModelWrapper,
    condition: str, chunk_size: int = 60, limit: Optional[int] = None,
    chunk_attempts: int = 4, cooldown_seconds: float = 60.0,
) -> Dict[str, int]:
    """Re-judge every pending row of one result file, writing after each chunk.

    Args:
        input_path: Original result CSV.
        output_path: Where the drop-in copy goes; resumed if present.
        specs: Scenario id to spec.
        client: The judge.
        condition: ``verbatim`` or ``swap``.
        chunk_size: Rows per write.
        limit: Re-judge at most this many pending rows (for a smoke run).
        chunk_attempts: Tries per chunk when the circuit breaker trips.
        cooldown_seconds: Base wait before retrying a tripped chunk; doubles.

    Returns:
        Counts of ``done``, ``pending``, ``invalid`` rows and ``calls`` made.
    """
    frame = prepare_frame(input_path, output_path)
    pending = [i for i in frame.index if frame.at[i, "rejudge_status"] == PENDING]
    if limit is not None:
        pending = pending[:limit]
    calls = 0
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start:start + chunk_size]
        requests = []
        for i in chunk:
            spec = specs[frame.at[i, "scenario_id"]]
            seat = int(frame.at[i, "seat"])
            order = presentation_order(condition, spec.players[seat].arity)
            messages = open_ended.build_open_judge_messages(
                spec, seat, frame.at[i, "open_user_prompt"], frame.at[i, "open_completion"], order
            )
            requests.append((i, spec, seat, order, messages))
        responses = None
        for attempt in range(chunk_attempts):
            try:
                responses = client.batch_generate([r[-1] for r in requests])
                break
            except Exception as error:  # circuit breaker: consecutive failures
                client.consecutive_failures = 0
                wait = cooldown_seconds * (2 ** attempt)
                logger.warning("Chunk failed (%s); cooling down %.0fs", error, wait)
                time.sleep(wait)
        if responses is None:
            write_frame(frame, output_path)
            raise RuntimeError(f"Judge kept failing on {os.path.basename(input_path)}; progress saved")
        calls += len(requests)
        for (i, spec, seat, order, _), response in zip(requests, responses):
            if response:  # '' is a failed call: leave pending for the next run
                apply_verdict(frame, i, spec, seat, response, order, condition, client.model_name)
        write_frame(frame, output_path)
    status = frame["rejudge_status"]
    return {
        "done": int((status == DONE).sum()),
        "pending": int((status == PENDING).sum()),
        "invalid": int(((status == DONE) & (frame["choice_action"] == "INVALID")).sum()),
        "calls": calls,
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(long_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def judge_slug(model_name: str) -> str:
    """A filesystem-safe name for the judge, without provider prefix."""
    return model_name.split(":", 1)[-1].replace("/", "_")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--judge-model", required=True,
                        help="Judge id, e.g. openai:deepseek/deepseek-v4-pro-0813 under openrouter_env.sh")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--scenarios-csv", default=DEFAULT_SCENARIOS_CSV)
    parser.add_argument("--output-dir", default=None,
                        help="Default: results/rejudge/<judge>_<condition>")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Exact arm tokens; 'baseline' for the bare arm. Default: the lead set.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Generous: a reasoning judge that runs out of budget returns nothing")
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None,
                        help="Re-judge at most this many pending rows per file (smoke runs)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    arms = list(LEAD_ARMS) if args.arms is None else ["" if a == "baseline" else a for a in args.arms]
    output_dir = args.output_dir or os.path.join(
        DEFAULT_OUTPUT_ROOT, f"{judge_slug(args.judge_model)}_{args.condition}"
    )
    os.makedirs(long_path(output_dir), exist_ok=True)

    files = find_input_files(args.results_dir, args.models, arms)
    specs = load_specs(args.scenarios_csv)
    client = ModelWrapper.create(args.judge_model, temperature=args.temperature, max_tokens=args.max_tokens)

    manifest_path = os.path.join(output_dir, "rejudge_manifest.json")
    manifest = {
        "judge_model": args.judge_model,
        "condition": args.condition,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "results_dir": os.path.relpath(args.results_dir, _ROOT),
        "scenarios_csv": os.path.relpath(args.scenarios_csv, _ROOT),
        "inputs": {},
    }
    totals = {"done": 0, "pending": 0, "invalid": 0, "calls": 0}
    for (model, arm), path in sorted(files.items()):
        out = os.path.join(output_dir, os.path.basename(path))
        counts = rejudge_file(path, out, specs, client, args.condition,
                              chunk_size=args.chunk_size, limit=args.limit)
        for key in totals:
            totals[key] += counts[key]
        manifest["inputs"][os.path.basename(path)] = {
            "model": model, "arm": arm or "baseline", "sha256": _sha256(path), **counts,
        }
        print(f"{model:18s} {arm or 'baseline':32s} done {counts['done']:4d}  "
              f"pending {counts['pending']:4d}  invalid {counts['invalid']:3d}", flush=True)

    usage = client.usage
    manifest["usage"] = {
        "requests": usage.requests,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }
    manifest["totals"] = totals
    manifest["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    with open(long_path(manifest_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"totals": totals, "usage": manifest["usage"]}))
    return 0 if totals["pending"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
