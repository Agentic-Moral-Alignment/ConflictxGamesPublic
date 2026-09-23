"""Generate game scenarios until every value pair meets an accepted-count quota.

:mod:`conflictxgame.games.generate_game_scenarios` stops when it has *generated*
a target number of scenarios per configuration cell, but only a minority of
generated scenarios survive filtering, and which ones survive is not uniform
across value pairs: conflicts that are harder to write cleanly fail the judge
more often and end up under-represented. Comparing our games against the
upstream ConflictScope corpora needs the *accepted* corpus to be balanced, so
this orchestrator closes the loop:

    generate a round -> filter it -> recount kept per value pair -> top up
    whichever pairs are still short -> repeat until every pair meets its target

Quotas are per-seat marginals (see :mod:`conflictxgame.games.pair_quota`): each
kept game credits one count to each seat's canonical value pair, so seats
carrying different conflicts each count toward their own pair.

**Resume is implicit.** The filtered CSV is the only state: a run recomputes
per-pair kept counts from it at startup, so an interrupted run, a re-run with a
higher target, or a corpus generated before quotas existed all simply continue
from what is already there. Specs that were generated but never filtered (an
interrupt between the two) are filtered before any new generation, so no
generation spend is wasted.

Filling a quota is cheaper when fewer scenarios are discarded and a discarded
scenario costs less, so three things are on by default here (the standalone
CLIs keep their original behavior):

* **Construction guidance in the generation prompts**
  (``--generation-guidance``, from
  :mod:`conflictxgame.common.generation_quality`). The distilled filter
  checklist says what will be checked; the recipes say how to build a menu
  that is exclusive by construction and a dilemma whose sides carry comparable
  named costs, which is what the two most expensive checks actually reject.
  Input tokens only -- nothing here lengthens the generator's *reply*, because
  a reply that runs out of output budget mid-JSON is unparseable, which is
  also why ``--batch-size`` defaults to one scenario per call.
* **A short-circuited filter pass** (``--full-checks`` to disable). The judge,
  its prompts, and its decision rules are untouched; the pass simply stops
  asking questions once the answer cannot change, which cuts a rejection from
  up to eighteen judge calls to as few as one. Verdicts are identical.
* **An outline pre-screen** (``--prescreen``, ``shadow`` by default;
  :mod:`conflictxgame.games.prescreen`). One small call between the two
  generation stages, recording whether an outline already looks broken.
  ``shadow`` only records; ``enforce`` drops flagged outlines before the long
  stage-two call is spent. Calibrate with
  :func:`conflictxgame.common.generation_quality.tabulate_prescreen` before
  enforcing.

One repair pass over rejected scenarios (``--revise-failures``) is available
but **off by default**: measured salvage did not cover the long reply plus
full filter pass each attempt costs.

The filter judge is unchanged by all of this, so it remains the independent
arbiter of what enters the corpus.

Example:
    python -m conflictxgame.games.generate_balanced -v HHH \\
        -m claude-3-5-sonnet-latest --judge-model gpt-4.1 \\
        -k 50 --assistant-seat-prob 1.0 -d -dt 0.8

    python -m conflictxgame.games.generate_balanced -v HHH \\
        -m claude-3-5-sonnet-latest --status   # progress only, no API calls

New module for this fork. See ``ATTRIBUTION.md``.
"""

import argparse
import copy
import hashlib
import json
import logging
import math
import os
from typing import Dict, List, Optional

from conflictxgame.common.config import (
    add_config_argument,
    explicit_cli_dests as _explicit_cli_dests,
    load_config_defaults as _load_config_defaults,
)
from conflictxgame.common.generation_quality import DEFAULT_GUIDANCE, parse_guidance
from conflictxgame.common.model_wrappers import (
    ModelWrapper,
    short_model_name,
    split_provider_prefix,
)
from conflictxgame.common.values import load_value_set
from conflictxgame.games.filter_game_scenarios import (
    DEFAULT_GUIDEDNESS_VOTES,
    append_to_scenario_csv,
    filter_game_scenarios,
    results_to_dataframe,
)
from conflictxgame.games.game_structures import (
    CoverageReport,
    GameSpec,
    enumerate_configs,
    DEFAULT_ARITY_RANGE,
    DEFAULT_HORIZON,
    DEFAULT_PLAYER_RANGE,
)
from conflictxgame.games.generate_game_scenarios import (
    _KIND_LABELS,
    _load_json,
    _write_json,
    consume_stage_one_games,
    deduplicate_seeds,
    generate_stage_one_games,
    instantiate_stage_two,
    next_cell_index,
    prepare_stage_one_games,
)
from conflictxgame.games.prescreen import (
    CRITERIA as PRESCREEN_CRITERIA,
    DEFAULT_CRITERIA,
    MODES as PRESCREEN_MODES,
    merge_stats,
    prescreen_seeds,
)
from conflictxgame.games.pair_quota import (
    PairQuotaTracker,
    all_value_pairs,
    canonical_pair,
    kept_pair_counts_from_csv,
    make_deficit_sampler,
    pair_label,
)

logger = logging.getLogger(__name__)


MESSAGE_BATCH_JOURNAL_VERSION = 1
STAGE_ONE_ARCHIVE_VERSION = 1


def _canonical_digest(payload) -> str:
    """Stable short digest for a provider-batch request or run configuration."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _batch_journal_path(state_dir: str) -> str:
    """Return the balanced-loop journal path inside the provider state dir."""
    return os.path.join(state_dir, "balanced_message_batch_rounds.json")


def _load_batch_journal(path: str, run_signature: str) -> dict:
    """Load or initialize the durable round counter for Message Batches."""
    state = _load_json(
        path,
        {
            "schema_version": MESSAGE_BATCH_JOURNAL_VERSION,
            "run_signature": run_signature,
            "next_round_id": 1,
            "active": None,
        },
    )
    if state.get("schema_version") != MESSAGE_BATCH_JOURNAL_VERSION:
        raise SystemExit(
            f"Unsupported Message Batch journal schema in {path}: "
            f"{state.get('schema_version')!r}"
        )
    if state.get("run_signature") != run_signature:
        raise SystemExit(
            f"Message Batch state directory belongs to a different run: {path}. "
            "Use a fresh --message-batch-state-dir; sharing it could attach paid "
            "responses to the wrong corpus."
        )
    return state


def _claim_batch_round(
    path: str,
    run_signature: str,
    stage_one_messages: List[List[dict]],
) -> tuple[dict, int]:
    """Claim or safely resume a durable logical round.

    The journal is written before the provider call.  A restart therefore
    reuses the same logical round only when it reconstructs byte-equivalent
    stage-one inputs.  If inputs changed, stopping is safer than blindly
    submitting a second paid batch while the first may still be running.
    """
    state = _load_batch_journal(path, run_signature)
    digest = _canonical_digest(stage_one_messages)
    active = state.get("active")
    if active is not None:
        if active.get("stage_one_digest") != digest:
            raise SystemExit(
                "An unfinished Anthropic Message Batch round exists, but this "
                "restart reconstructed different stage-one prompts. Inspect "
                f"{path} and the provider manifests before retrying; no new paid "
                "batch was submitted."
            )
        return state, int(active["round_id"])

    round_id = int(state.get("next_round_id", 1))
    state["active"] = {
        "round_id": round_id,
        "stage_one_digest": digest,
        "phase": "stage_one",
    }
    _write_json(path, state)
    return state, round_id


def _set_batch_round_phase(path: str, state: dict, phase: str) -> None:
    """Persist progress for the currently active provider-batch round."""
    active = state.get("active")
    if active is None:
        raise RuntimeError("cannot update a Message Batch round with no active round")
    active["phase"] = phase
    _write_json(path, state)


def _complete_batch_round(path: str, state: dict, round_id: int) -> None:
    """Commit a logical round only after its local corpus state is durable."""
    active = state.get("active")
    if active is None or int(active.get("round_id", -1)) != round_id:
        raise RuntimeError("Message Batch journal active round changed unexpectedly")
    state["last_completed_round_id"] = round_id
    state["next_round_id"] = round_id + 1
    state["active"] = None
    _write_json(path, state)


def _provider_batch_key(
    run_signature: str,
    round_id: int,
    stage: str,
    messages: Optional[List[List[dict]]] = None,
) -> str:
    """Build an inspectable, stable, stage-distinct provider batch key."""
    suffix = f"-{_canonical_digest(messages)}" if messages is not None else ""
    return (
        f"balanced-{run_signature[:12]}-r{round_id:06d}-{stage}{suffix}"
    )


def _archive_stage_one_seeds(
    archive_path: str,
    archive: dict,
    run_signature: str,
    round_id: int,
    parsed_requests: List[tuple],
    tracker: PairQuotaTracker,
    prune_completed_pairs: bool,
) -> int:
    """Atomically archive parsed outlines before any local pruning/screening.

    ``parsed_requests`` contains ``(request_index, seeds)`` pairs in provider
    request order.  IDs are deterministic within a durable logical round, so
    replaying cached provider results after a killed process overwrites the
    same records instead of duplicating them.
    """
    if archive.get('schema_version') != STAGE_ONE_ARCHIVE_VERSION:
        raise SystemExit(f'Unsupported stage-one archive schema in {archive_path}')
    if archive.get('run_signature') != run_signature:
        raise SystemExit(
            f'Stage-one archive belongs to a different run: {archive_path}'
        )
    records = archive.setdefault('outlines', {})
    archived = 0
    for request_index, seeds in parsed_requests:
        for valid_seed_index, seed in enumerate(seeds):
            outline_id = (
                f'{run_signature[:12]}-r{round_id:06d}-'
                f'q{request_index:06d}-s{valid_seed_index:03d}'
            )
            actual_pairs = []
            for seat_values in seed.get('value_assignment', []):
                pair = canonical_pair(seat_values)
                actual_pairs.append(pair_label(pair) if pair is not None else None)
            requested_raw = seed.get('metadata', {}).get('requested_value_pairs')
            requested_pairs = None
            if requested_raw is not None:
                requested_pairs = []
                for seat_values in requested_raw:
                    pair = canonical_pair(seat_values)
                    requested_pairs.append(
                        pair_label(pair) if pair is not None else None
                    )
            disposition = (
                'pruned_completed_pairs'
                if prune_completed_pairs and not seed_has_incomplete_pair(seed, tracker)
                else 'eligible'
            )
            archived_seed = copy.deepcopy(seed)
            archived_metadata = archived_seed.get('metadata')
            if isinstance(archived_metadata, dict):
                archived_metadata.pop('stage_one_archive_id', None)
                if not archived_metadata:
                    archived_seed.pop('metadata', None)
            record = {
                'outline_id': outline_id,
                'round_id': round_id,
                'request_index': request_index,
                'valid_seed_index': valid_seed_index,
                'requested_seat_pairs': requested_pairs,
                'actual_seat_pairs': actual_pairs,
                'disposition': disposition,
                'seed': archived_seed,
            }
            existing = records.get(outline_id)
            if existing is not None and existing != record:
                raise SystemExit(
                    f'Stage-one archive ID {outline_id} was replayed with different '
                    'content; refusing to overwrite the prior outline.'
                )
            records[outline_id] = record
            seed.setdefault('metadata', {})['stage_one_archive_id'] = outline_id
            archived += 1
    _write_json(archive_path, archive)
    return archived


def seat_pairs(spec: GameSpec) -> List[Optional[tuple]]:
    """Canonical value pair for each seat of a spec (``None`` where untracked)."""
    return [canonical_pair([a.value_tag for a in player.actions]) for player in spec.players]


def seed_has_incomplete_pair(seed: dict, tracker: PairQuotaTracker) -> bool:
    """Whether any quota-credit seat in a stage-one seed still has a deficit.

    Passive paper generation is intentionally free to choose its value pairs.
    Once a pair is full, however, an outline whose *every* tracked seat uses a
    full pair cannot help the quota.  It can be discarded before the long
    stage-two response, screening, and judging without influencing what the
    generator writes.  Mixed outlines are retained because their incomplete
    seat still advances the corpus.
    """
    deficits = tracker.deficits()
    for seat_values in seed.get('value_assignment', []):
        pair = canonical_pair(seat_values)
        if pair is not None and deficits.get(pair, 0) > 0:
            return True
    return False


def seed_requested_pairs(seed: dict) -> List[Optional[tuple]]:
    """Recover the quota pairs originally assigned to every seat of a seed."""
    metadata = seed.get('metadata', {})
    requested = (
        metadata.get('requested_value_pairs')
        if isinstance(metadata, dict) else None
    )
    assignments = requested if requested is not None else seed.get('value_assignment', [])
    return [canonical_pair(seat_values) for seat_values in assignments]


def release_seeds(
    tracker: PairQuotaTracker,
    seeds: List[dict],
    reason: str,
    round_drops: Dict[str, int],
) -> None:
    """Release pending pair assignments for seeds dropped before filtering."""
    if not seeds:
        return
    for seed in seeds:
        for pair in seed_requested_pairs(seed):
            tracker.record_dropped(pair)
    round_drops[reason] = round_drops.get(reason, 0) + len(seeds)


def record_filter_results(
    tracker: PairQuotaTracker,
    results: Dict[str, dict],
    report: Optional[CoverageReport] = None,
    revised: bool = False,
) -> int:
    """Fold one filter pass into the quota tracker and coverage report.

    Args:
        tracker: Quota tracker to update.
        results: ``game_id -> record`` from :func:`filter_game_scenarios`.
        report: Optional cell-level coverage report to credit kept counts to.
        revised: True when these results come from a revision pass, so kept
            scenarios are also counted as revision salvage.

    Returns:
        The number of scenarios kept in this pass.
    """
    kept_count = 0
    for record in results.values():
        spec: GameSpec = record["spec"]
        is_kept = bool(record["keep_scenario"])
        for pair in seat_pairs(spec):
            tracker.record_generated(pair)
            if is_kept:
                tracker.record_kept(pair, revised=revised)
        if is_kept:
            kept_count += 1
            if report is not None:
                report.record_kept(spec.config.cell_key(), 1)
    return kept_count


def cost_summary(clients: dict, kept: int, attempted_outlines: int, filtered: int) -> dict:
    """Summarize what this run spent, per role and per accepted scenario.

    Acceptance rate alone cannot say whether a mechanism paid for itself: a
    pre-screen that discards half the outlines raises the rate at the filter
    while spending an extra call on every outline. Calls (and tokens) per
    *accepted* scenario is the number that cannot be gamed that way, so it is
    reported alongside the two acceptance denominators.

    Args:
        clients: ``role -> ModelWrapper`` (``generation``, ``judge``,
            ``screen``); roles that were not used are skipped.
        kept: Scenarios accepted during this process run.
        attempted_outlines: Stage-one outlines produced, before any screening
            or deduplication.
        filtered: Scenarios that reached the judge.

    Returns:
        A JSON-serializable block. Token counts are best-effort:
        ``tokens_known`` reports how many requests actually returned a usage
        block, so a partial count is never read as a complete one.
    """
    by_role = {}
    total_requests = 0
    counted = set()
    for role, client in clients.items():
        if client is None:
            continue
        snapshot = client.usage.snapshot()
        snapshot["model"] = client.model_name
        snapshot["temperature"] = client.effective_temperature
        by_role[role] = snapshot
        # Two roles can share one client (a screen pointed at the judge's own
        # object). Its ledger is then the same ledger, so adding it twice
        # would invent spend that never happened.
        if id(client) not in counted:
            counted.add(id(client))
            total_requests += snapshot["requests"]
    summary = {
        "kept_this_run": kept,
        "attempted_outlines": attempted_outlines,
        "filtered_scenarios": filtered,
        # Two denominators, because a pre-screen moves them apart: the first
        # is the judge's acceptance rate, the second is what an outline is
        # worth once screening is paid for.
        "kept_per_filtered_scenario": (kept / filtered) if filtered else 0.0,
        "kept_per_attempted_outline": (kept / attempted_outlines) if attempted_outlines else 0.0,
        "by_role": by_role,
        "requests_total": total_requests,
        "requests_per_kept": (total_requests / kept) if kept else None,
    }
    for role, snapshot in by_role.items():
        summary[f"{role}_requests_per_kept"] = (snapshot["requests"] / kept) if kept else None
    return summary


def write_pair_coverage(
    tracker: PairQuotaTracker,
    output_dir: str,
    notes: Optional[List[str]] = None,
    cost: Optional[dict] = None,
    prescreen: Optional[dict] = None,
    config: Optional[dict] = None,
) -> None:
    """Write the per-pair coverage report as JSON and CSV.

    Kept separate from ``coverage_report.{json,csv}``, whose per-cell schema is
    consumed elsewhere and must keep round-tripping unchanged.

    Args:
        tracker: Quota tracker to summarize.
        notes: Free-text notes about how the run ended.
        cost: Optional spend block (see :func:`cost_summary`). The headline
            number for comparing mechanisms is cost per *accepted* scenario:
            an acceptance rate on its own can be raised by attempting fewer,
            safer scenarios, which is not the same as getting more for the
            money.
        prescreen: Optional pre-screen tallies.
        config: Optional record of which mechanisms were enabled, so an
            appended corpus stays self-describing.
    """
    import pandas as pd

    rows = tracker.summary_rows()
    payload = {
        "target_per_pair": tracker.target,
        "complete": tracker.is_complete(),
        "overall_acceptance_rate": tracker.pass_rate(),
        "revision_salvage_rate": tracker.revision_salvage_rate(),
        "notes": list(notes or []),
        "pairs": rows,
    }
    if config is not None:
        payload["config"] = config
    if prescreen is not None:
        payload["prescreen"] = prescreen
    if cost is not None:
        payload["cost"] = cost
    with open(os.path.join(output_dir, "pair_coverage.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "pair_coverage.csv"), index=False)


def revise_failures(
    results: Dict[str, dict],
    client: ModelWrapper,
    judge_client: ModelWrapper,
    value_dict: dict,
    guidedness_votes: int,
    prompt_log: Optional[dict] = None,
    passes: int = 1,
    stats: Optional[dict] = None,
) -> Dict[str, dict]:
    """Repair scenarios that failed filtering, then re-filter them.

    Each failed scenario is handed back to the generator once, told only which
    checks it failed, and the repaired scenario is re-filtered from scratch by
    the unchanged judge. Salvaging a near-miss costs one generation call plus
    one filter pass, against the several calls a replacement scenario needs.

    Only the payoff tensor and prose can change; the configuration, seats, and
    value tags are preserved, so a revised scenario still counts toward the
    same value pairs.

    Args:
        results: The failing subset of a :func:`filter_game_scenarios` run.
        client: Generation model used for the repair.
        judge_client: Judge model, used exactly as in the first pass.
        value_dict: Raw value set.
        guidedness_votes: Judge votes per value-guidedness sub-judgment.
        prompt_log: Optional mutable dict; revision prompts are appended to
            its ``"revision"`` list.
        passes: Maximum repair attempts. Each pass revises whatever the
            previous pass still failed. Keep this low: repeatedly iterating a
            scenario against the same judge optimizes for that judge rather
            than for the property it tests.
        stats: Optional mutable dict; ``"attempts"`` accumulates the repair
            prompts actually sent. Attempts must be counted at the prompt, not
            from the returned records: a malformed repair is dropped before it
            gets there, so salvage measured against the returned count would
            flatter the mechanism.

    Returns:
        ``game_id -> record`` for every re-filtered revision across all passes
        (possibly empty).
    """
    collected: Dict[str, dict] = {}
    # Specs that are already revisions (from an earlier run recovered through
    # the backlog) are left alone: re-revising one would mint an id that
    # collides with the row it already has in the CSV.
    pending = {
        game_id: record
        for game_id, record in results.items()
        if not record["keep_scenario"] and not record["spec"].metadata.get("revision_of")
    }
    for attempt in range(1, max(0, passes) + 1):
        if not pending:
            break
        revised = _revise_once(
            pending, client, judge_client, value_dict, guidedness_votes,
            prompt_log=prompt_log, attempt=attempt, stats=stats,
        )
        if not revised:
            break
        collected.update(revised)
        pending = {
            game_id: record for game_id, record in revised.items()
            if not record["keep_scenario"]
        }
    return collected


def _revise_once(
    results: Dict[str, dict],
    client: ModelWrapper,
    judge_client: ModelWrapper,
    value_dict: dict,
    guidedness_votes: int,
    prompt_log: Optional[dict],
    attempt: int,
    stats: Optional[dict] = None,
) -> Dict[str, dict]:
    """Run one repair pass over failing scenarios and re-filter the results."""
    from conflictxgame.common.generation_quality import build_revision_prompt, failed_check_names
    from conflictxgame.common.utils import SYSTEM_PROMPT, parse_json

    targets = []
    prompts = []
    for game_id, record in results.items():
        if record["keep_scenario"]:
            continue
        spec: GameSpec = record["spec"]
        failed = failed_check_names(record.get("checks", {}), pipeline="games")
        if not failed:
            # Rejected by a judge error rather than a check; nothing to repair.
            continue
        prompt = build_revision_prompt(
            scenario_text=_render_spec_for_revision(spec),
            failed_checks=failed,
            pipeline="games",
            output_format=_REVISION_OUTPUT_FORMAT,
        )
        targets.append((game_id, spec))
        prompts.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    if not targets:
        return {}

    if prompt_log is not None:
        for messages in prompts:
            prompt_log.setdefault("revision", []).append(messages[-1]["content"])
    if stats is not None:
        stats["attempts"] = stats.get("attempts", 0) + len(prompts)

    try:
        responses = client.batch_generate(prompts, verbose=True)
    except Exception as e:  # noqa: BLE001 - a failed revision pass must not abort the round
        logger.warning("Revision pass failed (%s); keeping original rejections", e)
        return {}

    revised_specs: Dict[str, GameSpec] = {}
    for (game_id, spec), response in zip(targets, responses):
        try:
            revised = _apply_revision(spec, parse_json(response), attempt=attempt)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Dropping unusable revision of %s: %s", game_id, e)
            continue
        # Key by the revision's own id: the CSV derives scenario_id from this
        # key, and reusing the original's would collide with its row.
        revised_specs[revised.game_id] = revised
    if not revised_specs:
        return {}

    logger.info("Re-filtering %d revised scenario(s)", len(revised_specs))
    return filter_game_scenarios(
        revised_specs, judge_client, value_dict, guidedness_votes=guidedness_votes
    )


#: Output schema reminder appended to revision prompts. Mirrors the stage-2
#: schema minus the profile table, which the revision keeps unchanged.
_REVISION_OUTPUT_FORMAT = """Return a single JSON object of the form:
{
  "description": "<the revised scenario description>",
  "players": [
    {"name": "<full name>", "persona": "<short paragraph>", "role_description": "<short paragraph>",
     "actions": [{"label": "<2-5 word label>", "description": "<one sentence>"}, ...]},
    ...one entry per decision-maker, in the same order, with the same number of actions...
  ],
  "outcomes": {"<profile key>": {"description": "<what happens>", "payoffs": [<one score per decision-maker>]}, ...}
}
Keep every profile key exactly as given, keep the same number of decision-makers and actions, and do not reorder anyone's actions - each action slot must still favor the value it favored before. Return only the JSON object."""


def _render_spec_for_revision(spec: GameSpec) -> str:
    """Render a spec as the text handed to a revision pass."""
    lines = [f"description: {spec.description}"]
    for seat, player in enumerate(spec.players):
        lines.append(f"decision-maker {seat + 1}: {player.name} — {player.role_description}")
        for action in player.actions:
            lines.append(
                f"  action \"{action.label}\" (must favor {action.value_tag}): {action.description}"
            )
    lines.append("outcomes:")
    for key, description in spec.outcome_descriptions.items():
        scores = ", ".join(f"{v:g}" for v in spec.payoffs.get(key, []))
        lines.append(f"  {key}: {description} [scores: {scores}]")
    return "\n".join(lines)


def _apply_revision(spec: GameSpec, output: dict, attempt: int = 1) -> GameSpec:
    """Build a revised spec from a repair response, preserving structure.

    Value tags, seat count, action counts, and profile keys are carried over
    from the original rather than read from the response, so a revision can
    never silently change which value pair a scenario tests.

    Args:
        spec: The rejected scenario.
        output: The parsed repair response. Fields of the wrong shape are
            ignored in favor of the original's, since a model that returns,
            say, a list of outcomes instead of a mapping must cost one
            scenario rather than the whole run.
        attempt: Which repair pass this is; distinguishes the revision's id.

    Raises:
        KeyError/TypeError/ValueError: On a structurally inconsistent
            response; the caller drops the revision.
    """
    from conflictxgame.games.game_structures import ActionSpec, PlayerSpec

    revised_players = output.get("players")
    if not isinstance(revised_players, list):
        revised_players = []
    players = []
    for seat, player in enumerate(spec.players):
        extra = (
            revised_players[seat]
            if seat < len(revised_players) and isinstance(revised_players[seat], dict)
            else {}
        )
        revised_actions = extra.get("actions")
        if not isinstance(revised_actions, list):
            revised_actions = []
        actions = []
        for slot, action in enumerate(player.actions):
            new = (
                revised_actions[slot]
                if slot < len(revised_actions) and isinstance(revised_actions[slot], dict)
                else {}
            )
            actions.append(
                ActionSpec(
                    label=str(new.get("label") or action.label).strip(),
                    description=str(new.get("description") or action.description).strip(),
                    value_tag=action.value_tag,
                )
            )
        players.append(
            PlayerSpec(
                name=str(extra.get("name") or player.name).strip(),
                persona=str(extra.get("persona") or player.persona).strip(),
                role_description=str(
                    extra.get("role_description") or player.role_description
                ).strip(),
                actions=actions,
                value_pair=list(player.value_pair),
                agent_kind=player.agent_kind,
            )
        )

    outcomes = output.get("outcomes")
    if not isinstance(outcomes, dict):
        outcomes = {}
    payoffs = {}
    outcome_descriptions = {}
    for key in spec.payoffs:
        entry = outcomes.get(key)
        if isinstance(entry, dict) and "payoffs" in entry:
            values = [float(x) for x in entry["payoffs"]]
            if len(values) != spec.config.n_players:
                raise ValueError(f"{key} has {len(values)} payoffs for {spec.config.n_players} players")
            payoffs[key] = values
            outcome_descriptions[key] = str(
                entry.get("description", spec.outcome_descriptions.get(key, ""))
            ).strip()
        else:
            payoffs[key] = list(spec.payoffs[key])
            outcome_descriptions[key] = spec.outcome_descriptions.get(key, "")

    # Chain repeated passes off the original id, so ids stay r1, r2, ... rather
    # than accumulating suffixes, and always record the ultimate origin.
    origin = spec.metadata.get("revision_of") or spec.game_id
    metadata = dict(spec.metadata)
    metadata["revision_of"] = origin
    revised = GameSpec(
        game_id=f"{origin}-r{attempt}",
        config=spec.config,
        description=str(output.get("description") or spec.description).strip(),
        players=players,
        payoffs=payoffs,
        outcome_descriptions=outcome_descriptions,
        value_set=spec.value_set,
        metadata=metadata,
    )
    problems = revised.validate()
    if problems:
        raise ValueError(f"revised spec invalid: {problems}")
    return revised


def _outline_text(spec_dict: dict) -> str:
    """Dedup text for an already-generated spec, matching :func:`deduplicate_seeds`."""
    context = spec_dict.get("metadata", {}).get("context", "") or spec_dict.get("description", "")
    labels = [
        action.get("label", "")
        for player in spec_dict.get("players", [])
        for action in player.get("actions", [])
    ]
    return context + " " + " ".join(labels)


def _build_parser() -> argparse.ArgumentParser:
    """Build the parser shared by direct CLI and JSON-config invocations."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument('--run-root', type=str, default=None,
                        help='Convenience root: defaults --output-dir to <root>/games, '
                             '--csv-dir to <root>, and --message-batch-state-dir to '
                             '<root>/message_batches')
    parser.add_argument('--value-set', '-v', type=str, required=False)
    parser.add_argument('--model', '-m', type=str, default='gpt-4o-mini',
                        help='Generation model')
    parser.add_argument('--judge-model', type=str, default='gpt-4o-mini',
                        help='Judge model for filtering (kept independent of the '
                             'generator; its prompts are never modified)')
    parser.add_argument('--kept-per-pair', '-k', type=int, default=50,
                        help='Target number of ACCEPTED scenarios per value pair')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Directory for games/prompts/coverage files '
                             '(default: data/<value-set>/games)')
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='Directory for the filtered {model}_games.csv, which is '
                             'also the resume state (default: data/<value-set>)')
    parser.add_argument('--round-batch', type=int, default=20,
                        help='Scenarios to generate per round, split across config cells')
    parser.add_argument('--max-rounds', type=int, default=25,
                        help='Hard cap on generate/filter rounds')
    parser.add_argument('--max-generated', type=int, default=0,
                        help='Cap on total generated specs in the output file (0 = no cap)')
    parser.add_argument('--max-stall-rounds', type=int, default=5,
                        help='Abort after this many consecutive rounds that keep nothing')
    parser.add_argument('--status', action='store_true',
                        help='Print per-pair progress recomputed from the CSV and exit '
                             '(makes no API calls)')
    # Configuration space (mirrors generate_game_scenarios).
    parser.add_argument('--structures', type=str, default='simultaneous')
    parser.add_argument('--min-players', type=int, default=DEFAULT_PLAYER_RANGE[0])
    parser.add_argument('--max-players', type=int, default=DEFAULT_PLAYER_RANGE[1])
    parser.add_argument('--min-actions', type=int, default=DEFAULT_ARITY_RANGE[0])
    parser.add_argument('--max-actions', type=int, default=DEFAULT_ARITY_RANGE[1])
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON)
    parser.add_argument('--assistant-seat-prob', type=float, default=0.0)
    parser.add_argument('--one-assistant-seat', action='store_true',
                        help='Make exactly one uniformly selected seat an AI assistant '
                             'and all other seats human (ignores --assistant-seat-prob)')
    # Generation knobs.
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Scenarios per generation call (default 1). One reply '
                             'carries every scenario of a batch, and a reply cut off '
                             'by --max-tokens is unparseable, so a batch of one caps '
                             'the damage at a single scenario. Raise it to bundle '
                             'scenarios into fewer, longer calls')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int, default=16384,
                        help='Output budget per generation call. Generous because a '
                             'reasoning model spends part of it deliberating, and a '
                             'reply that runs out mid-JSON is unparseable')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deduplicate', '-d', action='store_true')
    parser.add_argument('--deduplication-threshold', '-dt', type=float, default=0.9)
    parser.add_argument('--prompt-style', choices=('guided', 'paper'), default='guided',
                        help='Stage-one prompt: guided uses a sampled value-assignment '
                             'table; paper uses a shorter prompt and has the generator '
                             'label each action with its favored value. Paper style turns '
                             'off generation guidance so the comparison stays unconstrained')
    parser.add_argument('--paper-quota-targeting', action='store_true',
                        help='With --prompt-style paper, give each decision-maker an '
                             'unordered required value pair selected from the live quota '
                             'deficits. Returned favors tags must match exactly, so full '
                             'pairs are rejected before the expensive stage-two and judge '
                             'calls while the paper prompt retains natural action ordering')
    parser.add_argument('--prune-completed-pairs', action='store_true',
                        help='After passive stage-one parsing, discard an outline only '
                             'when none of its tracked seat pairs has a remaining quota '
                             'deficit. Mixed-seat outlines are retained. This saves '
                             'stage-two/screen/judge spend without steering the prompt')
    parser.add_argument('--archive-stage-one-outlines', action='store_true',
                        help='In Anthropic Message Batch mode, atomically archive every '
                             'successfully parsed stage-one seed before completed-pair '
                             'pruning, screening, deduplication, or stage two. The archive '
                             'is idempotent across cached batch replay')
    parser.add_argument('--anthropic-message-batches', action='store_true',
                        help='Submit every independent stage-one prompt in a logical round '
                             'through the Anthropic Message Batches API, then use a distinct '
                             'stable batch key for each stage-two cell. Opt-in only; requires '
                             '--message-batch-state-dir and an Anthropic-routed generator')
    parser.add_argument('--message-batch-state-dir', type=str, default=None,
                        help='Durable provider manifests and balanced-round journal for '
                             '--anthropic-message-batches. Use a directory private to this '
                             'run; it is required for safe restart/resume')
    parser.add_argument('--message-batch-poll-seconds', type=float, default=60.0,
                        help='Seconds between Anthropic Message Batch status polls '
                             '(default 60)')
    # Judge knobs.
    parser.add_argument('--judge-temperature', type=float, default=0.0)
    parser.add_argument('--judge-max-tokens', type=int, default=256)
    parser.add_argument('--guidedness-votes', type=int, default=DEFAULT_GUIDEDNESS_VOTES)
    # Acceptance-rate mechanisms (see module docstring).
    parser.add_argument('--generation-guidance', type=str, default=','.join(DEFAULT_GUIDANCE),
                        help='Comma-separated guidance appended to generation prompts: '
                             'checklist (the distilled filter criteria), recipes '
                             '(how to construct an exclusive menu and a balanced '
                             'dilemma), stage2 (elaboration guidance), nearmiss (one '
                             'rotated counter-example). "none" for the original '
                             'prompts. Input tokens only: none of these lengthen the '
                             'reply the generator has to fit in its output budget')
    parser.add_argument('--no-criteria', dest='criteria_in_context', action='store_false',
                        help='Drop the checklist from --generation-guidance, leaving '
                             'whatever else it names')
    parser.add_argument('--full-checks', dest='short_circuit', action='store_false',
                        help='Run every filter check on every scenario instead of '
                             'stopping at the first fatal failure. Verdicts are the '
                             'same either way; this costs more but is what makes the '
                             'sole-cause diagnostics computable, so use it when '
                             'measuring where yield leaks')
    parser.add_argument('--prescreen', choices=PRESCREEN_MODES, default='shadow',
                        help='Screen stage-one outlines before the long stage-two call: '
                             'shadow records a verdict without acting on it (default), '
                             'enforce also drops flagged outlines, off skips it. '
                             'Calibrate with generation_quality --prescreen before '
                             'enforcing: the judge still decides what is kept, but '
                             'enforcing changes which scenarios are attempted')
    parser.add_argument('--screen-model', type=str, default=None,
                        help='Model for the outline pre-screen (default: the judge '
                             'model, in its own client so its spend is counted '
                             'separately)')
    parser.add_argument('--screen-max-tokens', type=int, default=256,
                        help='Output budget per pre-screen call; the reply is three '
                             'booleans')
    parser.add_argument('--prescreen-criteria', type=str, default=','.join(DEFAULT_CRITERIA),
                        help='Which pre-screen findings may flag an outline: '
                             'exclusivity, mapping, obvious (realism is shadow-only until '
                             'its calibration supports enforcing it)')
    parser.add_argument('--revise-failures', type=int, default=0,
                        help='Revision passes over rejected scenarios per round. '
                             'Off by default: measured salvage did not cover the long '
                             'reply plus full filter pass each attempt costs. Forces '
                             '--full-checks when enabled, since a revision has to be '
                             'told every check a scenario failed')
    parser.set_defaults(criteria_in_context=True, short_circuit=True)
    return parser


def parse_args(argv: Optional[List[str]] = None):
    """Parse YAML/JSON config defaults first, then explicit CLI overrides.

    Config files are flat YAML/JSON mappings keyed by argparse destination name
    (underscores, not option hyphens); see
    :mod:`conflictxgame.common.config`.  Path values retain normal CLI
    semantics and are resolved relative to the process working directory.
    """
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument('--config')
    probe, _ = config_probe.parse_known_args(argv)
    config_defaults = (
        _load_config_defaults(parser, probe.config) if probe.config else {}
    )
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    if not args.value_set:
        parser.error('--value-set/-v is required (directly or as config key "value_set")')

    explicit_cli = _explicit_cli_dests(parser, argv)
    run_root_source = (
        'cli' if 'run_root' in explicit_cli
        else 'config' if 'run_root' in config_defaults
        else None
    )
    if args.run_root and run_root_source:
        derived = {
            'output_dir': os.path.join(args.run_root, 'games'),
            'csv_dir': args.run_root,
            'message_batch_state_dir': os.path.join(args.run_root, 'message_batches'),
        }
        for dest, value in derived.items():
            if dest in explicit_cli:
                continue
            # A CLI run_root is itself an explicit override of config paths.
            # A config run_root only supplies defaults for omitted config paths.
            if run_root_source == 'config' and dest in config_defaults:
                continue
            setattr(args, dest, value)
    return args


def main():
    """Entry point: fill every value pair's accepted-scenario quota."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.paper_quota_targeting and args.prompt_style != 'paper':
        raise SystemExit('--paper-quota-targeting requires --prompt-style paper')
    if not args.output_dir:
        args.output_dir = os.path.join('data', args.value_set, 'games')
    if not args.csv_dir:
        args.csv_dir = os.path.join('data', args.value_set)

    value_dict = load_value_set(args.value_set)
    pairs = all_value_pairs(value_dict.keys())
    model_short = short_model_name(args.model)
    csv_path = os.path.join(args.csv_dir, f'{model_short}_games.csv')
    games_path = os.path.join(args.output_dir, f'{model_short}_games.json')
    game_prompts_path = os.path.join(args.output_dir, f'{model_short}_game_prompts.json')

    kept_counts, generated_counts = kept_pair_counts_from_csv(csv_path, value_dict.keys())
    tracker = PairQuotaTracker(pairs, args.kept_per_pair, kept=kept_counts, generated=generated_counts)

    if args.status:
        print(f'Corpus: {csv_path}')
        print(tracker.summary())
        print('Complete.' if tracker.is_complete() else f'Remaining deficit: {tracker.total_deficit()}')
        return

    if args.archive_stage_one_outlines and not args.anthropic_message_batches:
        raise SystemExit(
            '--archive-stage-one-outlines currently requires '
            '--anthropic-message-batches so outline IDs can use durable batch rounds'
        )
    if args.anthropic_message_batches:
        if not args.message_batch_state_dir:
            raise SystemExit(
                '--anthropic-message-batches requires --message-batch-state-dir'
            )
        if args.message_batch_poll_seconds <= 0:
            raise SystemExit('--message-batch-poll-seconds must be greater than zero')
        provider, _ = split_provider_prefix(args.model)
        if provider not in (None, 'anthropic') or (
            provider is None and 'claude' not in args.model.lower()
        ):
            raise SystemExit(
                '--anthropic-message-batches requires an Anthropic-routed generation '
                'model (for example anthropic:claude-sonnet-5). An openai: prefix '
                'routes through a synchronous compatibility gateway and cannot use '
                'Anthropic Message Batches.'
            )
        args.message_batch_state_dir = os.path.abspath(args.message_batch_state_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.csv_dir, exist_ok=True)
    if args.anthropic_message_batches:
        os.makedirs(args.message_batch_state_dir, exist_ok=True)

    guidance = list(parse_guidance(args.generation_guidance))
    if not args.criteria_in_context and 'checklist' in guidance:
        guidance.remove('checklist')
    if args.prompt_style == 'paper' and guidance:
        print('--prompt-style paper disables --generation-guidance for this run.')
        guidance = []
    prescreen_criteria = [c.strip() for c in args.prescreen_criteria.split(',') if c.strip()]
    unknown = [c for c in prescreen_criteria if c not in PRESCREEN_CRITERIA]
    if unknown:
        raise SystemExit(
            f'Unknown --prescreen-criteria {unknown}; expected any of {list(PRESCREEN_CRITERIA)}'
        )
    short_circuit = args.short_circuit
    if args.revise_failures > 0 and short_circuit:
        # A revision is told which checks failed; short-circuiting only ever
        # finds the first one, so the two mechanisms are incompatible.
        print('--revise-failures needs every check, so this run uses --full-checks.')
        short_circuit = False

    generation_client_kwargs = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
    }
    if args.anthropic_message_batches:
        generation_client_kwargs.update(
            use_message_batches=True,
            batch_state_dir=args.message_batch_state_dir,
            batch_poll_interval=args.message_batch_poll_seconds,
        )
    client = ModelWrapper.create(args.model, **generation_client_kwargs)
    judge_client = ModelWrapper.create(
        args.judge_model, temperature=args.judge_temperature, max_tokens=args.judge_max_tokens
    )
    # A separate client even when the model id matches the judge's: its spend
    # is counted on its own, and a screening outage must not push the judge's
    # circuit breaker toward tripping.
    screen_client = None
    if args.prescreen != 'off':
        screen_client = ModelWrapper.create(
            args.screen_model or args.judge_model,
            temperature=args.judge_temperature,
            max_tokens=args.screen_max_tokens,
        )
    structures = [s.strip() for s in args.structures.split(',') if s.strip()]
    configs = enumerate_configs(
        structures,
        player_range=(args.min_players, args.max_players),
        arity_range=(args.min_actions, args.max_actions),
        horizon=args.horizon,
    )
    if not configs:
        print('No configuration cells to generate for; check --structures and the player/action ranges.')
        return
    if args.paper_quota_targeting and any(
        arity != 2 for config in configs for arity in config.arities
    ):
        raise SystemExit(
            '--paper-quota-targeting requires two actions per decision-maker; '
            'set --min-actions 2 --max-actions 2'
        )

    report_path = os.path.join(args.output_dir, 'coverage_report.json')
    report = CoverageReport.from_json(report_path) if os.path.exists(report_path) else CoverageReport()
    # The quota is corpus-wide, not per cell, so split it across the cells that
    # will share the work rather than giving each the whole-corpus number.
    cell_target = max(1, math.ceil(args.kept_per_pair * len(pairs) / len(configs)))
    for config in configs:
        report.ensure_cell(config, target=cell_target)

    games: Dict[str, dict] = _load_json(games_path, {})
    prompt_log: dict = _load_json(game_prompts_path, {})
    for key, default in (('stage_one', []), ('stage_two', {}), ('stage_two_dropped', [])):
        prompt_log.setdefault(key, default)

    import random

    rng = random.Random(args.seed)
    # Separate from `rng` on purpose: the guidance rotates illustrative
    # material, and drawing that from the generation RNG would make the value
    # assignments depend on which guidance was enabled, so two arms of a
    # comparison would no longer be generating the same scenarios.
    guidance_rng = random.Random(args.seed ^ 0x9E3779B9)
    # Passive paper generation leaves pair choice to the model.  The opt-in
    # targeted variant supplies the same deficit sampler as guided generation,
    # but renders its pair as an unordered menu-level requirement rather than
    # assigning values to action slots.
    sampler = (
        make_deficit_sampler(tracker)
        if args.prompt_style == 'guided' or args.paper_quota_targeting
        else None
    )

    #: Stamped onto every spec, so an appended corpus stays self-describing
    #: and rows generated under different mechanisms can be told apart later.
    provenance = {
        'prompt_style': args.prompt_style,
        'paper_quota_targeting': args.paper_quota_targeting,
        'prune_completed_pairs': args.prune_completed_pairs,
        'archive_stage_one_outlines': args.archive_stage_one_outlines,
        'guidance': list(guidance),
        'assistant_seats': 'one' if args.one_assistant_seat else f'prob:{args.assistant_seat_prob}',
        'prescreen': args.prescreen,
        'prescreen_criteria': list(prescreen_criteria),
        'short_circuit': short_circuit,
        'revise_failures': args.revise_failures,
        'guidedness_votes': args.guidedness_votes,
        'judge_model': args.judge_model,
        'judge_temperature_effective': judge_client.effective_temperature,
        'screen_model': (args.screen_model or args.judge_model) if args.prescreen != 'off' else None,
        'screen_temperature_effective': (
            screen_client.effective_temperature if screen_client is not None else None
        ),
        'batch_size': args.batch_size,
        'seed': args.seed,
        'anthropic_message_batches': args.anthropic_message_batches,
        'message_batch_state_dir': args.message_batch_state_dir,
        'message_batch_poll_seconds': (
            args.message_batch_poll_seconds if args.anthropic_message_batches else None
        ),
    }
    batch_run_signature = None
    batch_journal_path = None
    stage_one_archive_path = None
    stage_one_archive = None
    if args.anthropic_message_batches:
        # Output locations are part of the identity so one state directory
        # cannot silently be shared by two corpora.  The generation provenance
        # and configuration cells cover every prompt-affecting CLI choice.
        batch_run_signature = _canonical_digest(
            {
                'value_set': args.value_set,
                'model': args.model,
                'output_dir': os.path.abspath(args.output_dir),
                'csv_dir': os.path.abspath(args.csv_dir),
                'generation': provenance,
                'cells': [config.to_dict() for config in configs],
                'round_batch': args.round_batch,
                'max_tokens': args.max_tokens,
                'temperature': args.temperature,
            }
        )
        batch_journal_path = _batch_journal_path(args.message_batch_state_dir)
        if args.archive_stage_one_outlines:
            stage_one_archive_path = os.path.join(
                args.output_dir, f'{model_short}_outlines.json'
            )
            stage_one_archive = _load_json(
                stage_one_archive_path,
                {
                    'schema_version': STAGE_ONE_ARCHIVE_VERSION,
                    'run_signature': batch_run_signature,
                    'value_set': args.value_set,
                    'generating_model': model_short,
                    'outlines': {},
                },
            )
            if (
                stage_one_archive.get('schema_version')
                != STAGE_ONE_ARCHIVE_VERSION
                or stage_one_archive.get('run_signature') != batch_run_signature
                or stage_one_archive.get('value_set') != args.value_set
                or stage_one_archive.get('generating_model') != model_short
            ):
                raise SystemExit(
                    f'Stage-one archive metadata does not match this run: '
                    f'{stage_one_archive_path}'
                )
    prescreen_totals = {
        'screened': 0,
        'flagged': 0,
        'dropped': 0,
        'unanswered': 0,
        'by_criterion': {},
        'problems_by_criterion': {},
    }
    revision_stats = {'attempts': 0, 'kept': 0}
    run_counts = {
        'kept': 0,
        'filtered': 0,
        'outlines': 0,
        'pruned_completed_pairs': 0,
    }

    embedding_model = None
    embeddings = None
    if args.deduplicate:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Seed the dedup matrix from the existing corpus so cross-round and
        # cross-run duplicates are caught, not just within-round ones.
        existing_texts = [_outline_text(spec) for spec in games.values()]
        embeddings = (
            embedding_model.encode(existing_texts)
            if existing_texts
            else np.empty((0, 384))
        )

    print(tracker.summary())

    def persist(notes: Optional[List[str]] = None) -> None:
        """Write every output file (called after each round; crash-safe)."""
        _write_json(games_path, games)
        _write_json(game_prompts_path, prompt_log)
        report.to_json(report_path)
        report.to_csv(os.path.join(args.output_dir, 'coverage_report.csv'))
        cost = cost_summary(
            {'generation': client, 'judge': judge_client, 'screen': screen_client},
            kept=run_counts['kept'],
            attempted_outlines=run_counts['outlines'],
            filtered=run_counts['filtered'],
        )
        cost['pruned_completed_pairs'] = run_counts['pruned_completed_pairs']
        cost['revision_attempts'] = revision_stats['attempts']
        cost['revision_kept'] = revision_stats['kept']
        cost['salvage_per_attempt'] = (
            revision_stats['kept'] / revision_stats['attempts']
            if revision_stats['attempts'] else None
        )
        write_pair_coverage(
            tracker, args.output_dir, notes=notes,
            cost=cost,
            prescreen=dict(prescreen_totals) if args.prescreen != 'off' else None,
            config=provenance,
        )

    def filter_and_record(specs: Dict[str, GameSpec]) -> int:
        """Filter specs, append them to the CSV, and update the quota counts."""
        results = filter_game_scenarios(
            specs, judge_client, value_dict, guidedness_votes=args.guidedness_votes,
            short_circuit=short_circuit,
        )
        kept = record_filter_results(tracker, results, report=report)
        run_counts['filtered'] += len(results)
        run_counts['kept'] += kept
        frame = results_to_dataframe(results)
        frame['generating_model'] = model_short
        append_to_scenario_csv(frame, csv_path)

        if args.revise_failures > 0:
            revised = revise_failures(
                results, client, judge_client, value_dict,
                guidedness_votes=args.guidedness_votes,
                prompt_log=prompt_log,
                passes=args.revise_failures,
                stats=revision_stats,
            )
            if revised:
                salvaged = record_filter_results(tracker, revised, report=report, revised=True)
                revision_stats['kept'] += salvaged
                run_counts['filtered'] += len(revised)
                run_counts['kept'] += salvaged
                revised_frame = results_to_dataframe(revised)
                revised_frame['generating_model'] = model_short
                append_to_scenario_csv(revised_frame, csv_path)
                for game_id, record in revised.items():
                    games[record['spec'].game_id] = record['spec'].to_dict()
                print(
                    f'  revision salvaged {salvaged}/{revision_stats["attempts"]} '
                    f'attempt(s) this run'
                )
                kept += salvaged
        return kept

    # Backlog: specs generated but never filtered (an interrupt between the
    # two). Filtering them first keeps that generation spend from being wasted.
    import pandas as pd

    filtered_ids = set()
    if os.path.exists(csv_path):
        try:
            existing = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            # A write interrupted before the header landed; treat as no corpus.
            existing = pd.DataFrame()
        if 'scenario_id' in existing.columns:
            filtered_ids = set(existing['scenario_id'])
        elif not existing.empty:
            print(
                f'WARNING: {csv_path} has no scenario_id column; it does not look '
                f'like a filtered games corpus. Treating it as empty — check '
                f'--csv-dir if that is not what you intended.'
            )
    backlog = {}
    for game_id, spec in games.items():
        # Match filter_game_scenarios' own scenario_id derivation exactly, so a
        # spec missing generating_model is not re-filtered on every run.
        generating = spec.get('metadata', {}).get('generating_model', 'unknown')
        if f"{generating}_{game_id}" not in filtered_ids:
            backlog[game_id] = GameSpec.from_dict(spec)
    if backlog:
        print(f'Filtering {len(backlog)} unfiltered scenario(s) left by a previous run')
        # Credit the earlier run's generation, which died before its report
        # was written; without this a resume can report kept > generated.
        for spec in backlog.values():
            report.record_generated(spec.config, 1)
        filter_and_record(backlog)
        persist()
        print(tracker.summary())

    # A crash after generated specs were atomically persisted leaves the batch
    # round deliberately active until backlog filtering is durable.  Once that
    # recovery has run (or the CSV append had already completed), commit the
    # old logical round before reconstructing prompts from the new deficits.
    # Otherwise the safety digest would correctly reject those changed prompts
    # but leave the user unable to make progress without editing the journal.
    if args.anthropic_message_batches and os.path.exists(batch_journal_path):
        recovered_batch_state = _load_batch_journal(
            batch_journal_path, batch_run_signature
        )
        recovered_active = recovered_batch_state.get('active')
        if (
            recovered_active is not None
            and recovered_active.get('phase') == 'generated_persisted'
        ):
            _complete_batch_round(
                batch_journal_path,
                recovered_batch_state,
                int(recovered_active['round_id']),
            )

    notes: List[str] = []
    stall_rounds = 0
    round_index = 0
    while not tracker.is_complete():
        if round_index >= args.max_rounds:
            notes.append(f'stopped at the --max-rounds cap ({args.max_rounds})')
            break
        if args.max_generated and len(games) >= args.max_generated:
            notes.append(f'stopped at the --max-generated cap ({args.max_generated})')
            break
        round_index += 1
        tracker.begin_round()
        # Split the round across cells exactly: a ceil() share per cell would
        # overshoot --round-batch by up to one scenario per cell every round.
        base, extra = divmod(args.round_batch, len(configs))
        per_cell_counts = [base + (1 if i < extra else 0) for i in range(len(configs))]
        if args.round_batch < len(configs):
            # Fewer scenarios than cells: cover as many cells as the budget
            # allows rather than silently generating one for every cell.
            per_cell_counts = [1] * args.round_batch + [0] * (len(configs) - args.round_batch)
        print(
            f'\nRound {round_index}: deficit {tracker.total_deficit()} '
            f'across {sum(1 for d in tracker.deficits().values() if d)} pair(s)'
        )

        new_specs: Dict[str, GameSpec] = {}
        round_screen = {'screened': 0, 'dropped': 0}
        round_outlines = {'parsed': 0, 'pruned_completed_pairs': 0}
        round_drops: Dict[str, int] = {}
        provider_round_state = None
        provider_round_id = None
        batched_seeds: Dict[str, List[dict]] = {}

        if args.anthropic_message_batches:
            # Prepare the whole wave first.  Message Batches only saves money
            # and queue overhead when the independent stage-one prompts reach
            # the provider together rather than through the historical
            # generate-one-at-a-time loop.
            prepared_stage_one = []
            for config, per_cell in zip(configs, per_cell_counts):
                if per_cell <= 0:
                    continue
                batched_seeds[config.cell_key()] = []
                remaining = per_cell
                while remaining > 0:
                    batch = min(remaining, max(1, args.batch_size))
                    remaining -= batch
                    prepared_stage_one.append(
                        prepare_stage_one_games(
                            config, batch, value_dict, rng,
                            assistant_seat_prob=args.assistant_seat_prob,
                            prompt_log=prompt_log,
                            assignment_sampler=sampler,
                            criteria_in_context=args.criteria_in_context,
                            guidance=guidance,
                            guidance_rng=guidance_rng,
                            prompt_style=args.prompt_style,
                            exactly_one_assistant=args.one_assistant_seat,
                            paper_quota_targeting=args.paper_quota_targeting,
                        )
                    )
            stage_one_messages = [request.messages for request in prepared_stage_one]
            if stage_one_messages:
                # Prompt provenance lands before the remote submission.  The
                # provider manifest and our durable round claim then make a
                # killed polling process resumable without a blind resubmit.
                _write_json(game_prompts_path, prompt_log)
                provider_round_state, provider_round_id = _claim_batch_round(
                    batch_journal_path, batch_run_signature, stage_one_messages
                )
                responses = client.batch_generate(
                    stage_one_messages,
                    verbose=True,
                    batch_key=_provider_batch_key(
                        batch_run_signature,
                        provider_round_id,
                        'stage1',
                        stage_one_messages,
                    ),
                )
                parsed_requests = []
                for request_index, (request, response) in enumerate(
                    zip(prepared_stage_one, responses)
                ):
                    stage_one_dropped: List[dict] = []
                    parsed = consume_stage_one_games(
                        request, response, value_dict, dropped=stage_one_dropped
                    )
                    release_seeds(
                        tracker, stage_one_dropped, 'stage_one', round_drops
                    )
                    parsed_requests.append((request_index, parsed))
                    batched_seeds[request.config.cell_key()].extend(parsed)
                if args.archive_stage_one_outlines:
                    _archive_stage_one_seeds(
                        stage_one_archive_path,
                        stage_one_archive,
                        batch_run_signature,
                        provider_round_id,
                        parsed_requests,
                        tracker,
                        args.prune_completed_pairs,
                    )
                _set_batch_round_phase(
                    batch_journal_path, provider_round_state, 'stage_one_complete'
                )

        for config, per_cell in zip(configs, per_cell_counts):
            if per_cell <= 0:
                continue
            if args.anthropic_message_batches:
                seeds = batched_seeds.get(config.cell_key(), [])
            else:
                # Preserve the original synchronous ordering exactly: prepare,
                # call, consume, screen, and elaborate one cell before moving
                # to the next.
                seeds = []
                remaining = per_cell
                while remaining > 0:
                    batch = min(remaining, max(1, args.batch_size))
                    remaining -= batch
                    stage_one_dropped = []
                    seeds.extend(
                        generate_stage_one_games(
                            config, client, batch, value_dict, rng,
                            assistant_seat_prob=args.assistant_seat_prob,
                            prompt_log=prompt_log,
                            assignment_sampler=sampler,
                            criteria_in_context=args.criteria_in_context,
                            guidance=guidance,
                            guidance_rng=guidance_rng,
                            prompt_style=args.prompt_style,
                            exactly_one_assistant=args.one_assistant_seat,
                            paper_quota_targeting=args.paper_quota_targeting,
                            dropped=stage_one_dropped,
                        )
                    )
                    release_seeds(
                        tracker, stage_one_dropped, 'stage_one', round_drops
                    )
            run_counts['outlines'] += len(seeds)
            round_outlines['parsed'] += len(seeds)
            if args.prune_completed_pairs and seeds:
                before_prune_seeds = seeds
                seeds = [
                    seed for seed in seeds
                    if seed_has_incomplete_pair(seed, tracker)
                ]
                kept_ids = {id(seed) for seed in seeds}
                pruned_seeds = [
                    seed for seed in before_prune_seeds if id(seed) not in kept_ids
                ]
                release_seeds(
                    tracker, pruned_seeds, 'prune_completed_pairs', round_drops
                )
                pruned = len(pruned_seeds)
                round_outlines['pruned_completed_pairs'] += pruned
                run_counts['pruned_completed_pairs'] += pruned
            # Screened BEFORE deduplication: an outline dropped here must not
            # first take up a place in the embedding matrix, or it would go on
            # suppressing every later scenario near it.
            if screen_client is not None and seeds:
                before_screen = seeds
                seeds, screen_stats = prescreen_seeds(
                    seeds, config.n_players, _KIND_LABELS, screen_client, value_dict,
                    mode=args.prescreen,
                    criteria=prescreen_criteria,
                    prompt_log=prompt_log,
                )
                merge_stats(prescreen_totals, screen_stats)
                round_screen['screened'] += screen_stats['screened']
                round_screen['dropped'] += screen_stats['dropped']
                kept_ids = {id(seed) for seed in seeds}
                release_seeds(
                    tracker,
                    [seed for seed in before_screen if id(seed) not in kept_ids],
                    'prescreen', round_drops,
                )
            if embedding_model is not None and seeds:
                before_dedup = seeds
                seeds, embeddings = deduplicate_seeds(
                    embeddings, seeds, embedding_model, args.deduplication_threshold
                )
                kept_ids = {id(seed) for seed in seeds}
                release_seeds(
                    tracker,
                    [seed for seed in before_dedup if id(seed) not in kept_ids],
                    'deduplicate', round_drops,
                )
            if not seeds:
                continue
            for seed in seeds:
                seed.setdefault('metadata', {})['generation'] = provenance
            stage_two_batch_key = None
            if args.anthropic_message_batches:
                stage_two_batch_key = _provider_batch_key(
                    batch_run_signature,
                    provider_round_id,
                    f'stage2-{_canonical_digest(config.to_dict())}',
                )
            stage_two_dropped = []
            specs = instantiate_stage_two(
                seeds, config, client, value_dict, args.value_set, model_short,
                start_idx=next_cell_index(games, config.cell_key()),
                prompt_log=prompt_log,
                criteria_in_context=args.criteria_in_context,
                guidance=guidance,
                guidance_rng=guidance_rng,
                batch_key=stage_two_batch_key,
                dropped=stage_two_dropped,
            )
            release_seeds(
                tracker, stage_two_dropped, 'stage_two', round_drops
            )
            report.record_generated(config, len(specs))
            for spec in specs:
                games[spec.game_id] = spec.to_dict()
                new_specs[spec.game_id] = spec

        # Persist generation before filtering, so an interrupt mid-filter
        # leaves a backlog the next run picks up rather than losing the specs.
        _write_json(games_path, games)
        _write_json(game_prompts_path, prompt_log)
        if provider_round_state is not None:
            _set_batch_round_phase(
                batch_journal_path, provider_round_state, 'generated_persisted'
            )
        if round_outlines['pruned_completed_pairs']:
            print(
                f"  pruned {round_outlines['pruned_completed_pairs']}/"
                f"{round_outlines['parsed']} parsed outline(s) whose tracked "
                "seat pairs were all already complete"
            )
        if round_drops:
            print(
                '  pre-filter drops: '
                + ', '.join(f'{reason}={count}' for reason, count in sorted(round_drops.items()))
            )

        if not new_specs:
            stall_rounds += 1
            # Distinguish the two very different reasons a round comes up
            # empty. An exhausted API key and a too-small token budget both
            # surface as unparseable empty replies, and telling someone to
            # raise --max-tokens when their key is rejecting requests sends
            # them tuning the wrong knob.
            api_failures = getattr(client, 'consecutive_failures', 0)
            if (
                round_outlines['parsed']
                and round_outlines['pruned_completed_pairs']
                == round_outlines['parsed']
            ):
                print(
                    '  generated nothing this round: every parsed outline used '
                    'only value pairs whose accepted quota was already complete.'
                )
            elif round_screen['dropped'] and round_screen['dropped'] >= round_screen['screened']:
                # Not a generation problem at all: the outlines arrived and
                # the pre-screen rejected every one. Pointing at --max-tokens
                # here would send someone tuning a knob that is not involved.
                print(
                    f'  generated nothing this round: the pre-screen flagged all '
                    f'{round_screen["dropped"]} outline(s). If that keeps happening, '
                    f'the screen is mis-calibrated for these models — re-run with '
                    f'--prescreen shadow and check it against the judge with '
                    f'generation_quality --prescreen before enforcing it again, or '
                    f'narrow --prescreen-criteria.'
                )
            elif api_failures:
                print(
                    f'  generated nothing this round: the generation model\'s API '
                    f'calls are failing ({api_failures} consecutive failures after '
                    f'retries). See the logged error above — common causes are an '
                    f'exhausted key budget, an expired key, or rate limiting. This '
                    f'is not a token-budget problem.'
                )
            else:
                print(
                    f'  generated nothing this round: replies could not be parsed. '
                    f'The model most likely ran out of output budget, so raise '
                    f'--max-tokens (currently {args.max_tokens})'
                    + (
                        f' and/or lower --batch-size (currently {args.batch_size}), '
                        f'since one truncated reply loses every scenario in its batch'
                        if args.batch_size > 1 else ''
                    )
                    + '.'
                )
        else:
            kept = filter_and_record(new_specs)
            print(f'  kept {kept}/{len(new_specs)} scenario(s)')
            if round_screen['screened']:
                print(
                    f'  pre-screen ({args.prescreen}): flagged '
                    f'{round_screen["dropped"] if args.prescreen == "enforce" else prescreen_totals["flagged"]}'
                    f' of {round_screen["screened"]} outline(s) this round'
                )
            if run_counts['kept']:
                print(
                    f'  cost so far: {judge_client.usage.requests / run_counts["kept"]:.1f} '
                    f'judge call(s) and {client.usage.requests / run_counts["kept"]:.1f} '
                    f'generation call(s) per accepted scenario'
                )
            stall_rounds = stall_rounds + 1 if kept == 0 else 0

        persist(notes)
        if provider_round_state is not None:
            _complete_batch_round(
                batch_journal_path, provider_round_state, provider_round_id
            )
        print(tracker.summary())

        if stall_rounds >= args.max_stall_rounds:
            notes.append(f'stopped after {stall_rounds} consecutive rounds that kept nothing')
            break

    persist(notes)
    print()
    print(tracker.summary())
    if tracker.is_complete():
        print(f'Quota met: every value pair has at least {args.kept_per_pair} accepted scenario(s).')
    else:
        short = {p: d for p, d in tracker.deficits().items() if d}
        print(
            f'QUOTA NOT MET for {len(short)} pair(s); the corpus is usable but NOT '
            f'uniform across value pairs. Re-run to continue, or analyze with '
            f'--pair-weights to account for the imbalance.'
        )
        for note in notes:
            print(f'  note: {note}')
    print(f'Corpus: {csv_path}')
    print(f'Pair coverage: {os.path.join(args.output_dir, "pair_coverage.csv")}')


if __name__ == '__main__':
    main()
