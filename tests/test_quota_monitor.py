"""Offline tests for the scheduled balanced-run quota monitor."""

import json

import pandas as pd

from conflictxgame.games.quota_monitor import (
    DEFAULT_EXPECTED_GENERATION,
    ExitCode,
    check_run,
    main,
    persist_state,
)


HHH = ["harmlessness", "helpfulness", "honesty"]
PAIR_LABELS = [
    "harmlessness|helpfulness",
    "harmlessness|honesty",
    "helpfulness|honesty",
]


def _write_run(tmp_path, *, kept=None, generated=None, complete=None, notes=None,
               prompt_style="paper", assistant_seats="prob:1.0", target=1,
               row_generation_overrides=None):
    """Write a small real-schema CSV plus its derived pair report."""

    root = tmp_path / "run"
    games = root / "games"
    games.mkdir(parents=True)

    # Seat 0's legacy columns name only the first pair.  The other two pairs
    # appear in value_tags, proving that the monitor uses all-seat marginals.
    generation = dict(DEFAULT_EXPECTED_GENERATION)
    generation.update(
        prompt_style=prompt_style,
        assistant_seats=assistant_seats,
    )
    row_generation = dict(generation)
    row_generation.update(row_generation_overrides or {})

    def game_spec():
        return json.dumps(
            {
                "value_set": "HHH_new",
                "players": [
                    {"agent_kind": "assistant"},
                    {"agent_kind": "assistant"},
                ],
                "metadata": {"generation": row_generation},
            }
        )

    frame = pd.DataFrame(
        [
            {
                "scenario_id": "g0",
                "value1": "harmlessness",
                "value2": "helpfulness",
                "value_tags": json.dumps(
                    [
                        ["harmlessness", "helpfulness"],
                        ["honesty", "harmlessness"],
                    ]
                ),
                "keep_scenario": True,
                "value_set": "HHH_new",
                "agent_kinds": json.dumps(["assistant", "assistant"]),
                "game_spec": game_spec(),
            },
            {
                "scenario_id": "g1",
                "value1": "harmlessness",
                "value2": "helpfulness",
                "value_tags": json.dumps([["helpfulness", "honesty"]]),
                "keep_scenario": True,
                "value_set": "HHH_new",
                "agent_kinds": json.dumps(["assistant", "assistant"]),
                "game_spec": game_spec(),
            },
        ]
    )
    frame.to_csv(root / "claude-sonnet-5_games.csv", index=False)

    generated = generated or dict.fromkeys(PAIR_LABELS, 1)
    kept = kept or dict.fromkeys(PAIR_LABELS, 1)
    if complete is None:
        complete = all(kept[label] >= target for label in PAIR_LABELS)
    report = {
        "target_per_pair": target,
        "complete": complete,
        "notes": list(notes or []),
        "pairs": [
            {
                "pair": label,
                "generated": generated[label],
                "kept": kept[label],
                "target": target,
                "deficit": max(0, target - kept[label]),
            }
            for label in PAIR_LABELS
        ],
        "config": generation,
    }
    (games / "pair_coverage.json").write_text(json.dumps(report), encoding="utf-8")
    return root, report


def _check(root, **kwargs):
    return check_run(
        run_root=root,
        value_names=HHH,
        target=1,
        retries=kwargs.pop("retries", 1),
        retry_delay=0,
        **kwargs,
    )


def _write_shadow_to_off_transition(root):
    """Stamp one accepted row and the live report with prescreen-off provenance."""

    csv_path = root / "claude-sonnet-5_games.csv"
    frame = pd.read_csv(csv_path)
    transitioned = json.loads(frame.loc[1, "game_spec"])
    transitioned["metadata"]["generation"].update(
        prescreen="off",
        screen_model=None,
    )
    frame.loc[1, "game_spec"] = json.dumps(transitioned)
    frame.to_csv(csv_path, index=False)

    report_path = root / "games" / "pair_coverage.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["config"].update(prescreen="off", screen_model=None)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return frame, report


def test_complete_recounts_all_seats_and_writes_durable_marker(tmp_path):
    root, _ = _write_run(tmp_path)

    state, code = _check(root)

    assert code is ExitCode.COMPLETE
    assert state["status"] == "complete"
    assert state["csv"]["kept"] == dict.fromkeys(PAIR_LABELS, 1)
    persist_state(root, state, code)
    status_path = root / "monitor" / "status.json"
    complete_path = root / "monitor" / "COMPLETE.json"
    assert status_path.exists()
    assert complete_path.exists()
    first_completed_at = json.loads(complete_path.read_text(encoding="utf-8"))["completed_at"]

    # Later checks update status but preserve the first completion time.
    later = dict(state, checked_at="2099-01-01T00:00:00Z")
    persist_state(root, later, code)
    assert json.loads(status_path.read_text(encoding="utf-8"))["checked_at"] == later["checked_at"]
    assert json.loads(complete_path.read_text(encoding="utf-8"))["completed_at"] == first_completed_at
    assert not list((root / "monitor").glob("*.tmp"))


def test_incomplete_report_and_csv_agree(tmp_path):
    root, report = _write_run(tmp_path)
    # Make one accepted row rejected in both the source CSV and report.
    csv_path = root / "claude-sonnet-5_games.csv"
    frame = pd.read_csv(csv_path)
    frame.loc[1, "keep_scenario"] = False
    frame.to_csv(csv_path, index=False)
    report["pairs"][2]["kept"] = 0
    report["pairs"][2]["deficit"] = 1
    report["complete"] = False
    (root / "games" / "pair_coverage.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    state, code = _check(root)

    assert code is ExitCode.INCOMPLETE
    assert state["deficits"]["helpfulness|honesty"] == 1
    persist_state(root, state, code)
    assert (root / "monitor" / "status.json").exists()
    assert not (root / "monitor" / "COMPLETE.json").exists()


def test_stopped_before_quota_is_distinct_from_running_incomplete(tmp_path):
    root, report = _write_run(tmp_path)
    csv_path = root / "claude-sonnet-5_games.csv"
    frame = pd.read_csv(csv_path)
    frame.loc[1, "keep_scenario"] = False
    frame.to_csv(csv_path, index=False)
    report["pairs"][2]["kept"] = 0
    report["complete"] = False
    report["notes"] = ["stopped at the --max-rounds cap (100)"]
    (root / "games" / "pair_coverage.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    state, code = _check(root)

    assert code is ExitCode.STOPPED_INCOMPLETE
    assert state["status"] == "stopped_incomplete"


def test_durable_report_csv_disagreement_retries_then_fails(tmp_path):
    root, report = _write_run(tmp_path)
    report["pairs"][0]["kept"] = 0
    report["complete"] = False
    (root / "games" / "pair_coverage.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    sleeps = []

    state, code = check_run(
        run_root=root,
        value_names=HHH,
        target=1,
        retries=3,
        retry_delay=0.25,
        sleep_fn=sleeps.append,
    )

    assert code is ExitCode.INCONSISTENT
    assert state["agreement"] is False
    assert sleeps == [0.25, 0.25]


def test_transient_between_file_snapshot_recovers_on_retry(tmp_path):
    root, report = _write_run(tmp_path)
    report["pairs"][0]["kept"] = 0
    report["complete"] = False
    report_path = root / "games" / "pair_coverage.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    def finish_report(_delay):
        report["pairs"][0]["kept"] = 1
        report["complete"] = True
        report_path.write_text(json.dumps(report), encoding="utf-8")

    state, code = check_run(
        run_root=root,
        value_names=HHH,
        target=1,
        retries=2,
        retry_delay=0,
        sleep_fn=finish_report,
    )

    assert code is ExitCode.COMPLETE
    assert state["attempts"] == 2


def test_wrong_generation_method_is_rejected_even_if_counts_are_full(tmp_path):
    root, _ = _write_run(tmp_path, prompt_style="guided")

    state, code = _check(root)

    assert code is ExitCode.INVALID_CONFIG
    assert state["status"] == "invalid_config"
    assert "prompt_style" in state["message"]


def test_mixed_accepted_row_provenance_cannot_satisfy_quota(tmp_path):
    root, _ = _write_run(tmp_path)
    csv_path = root / "claude-sonnet-5_games.csv"
    frame = pd.read_csv(csv_path)
    mixed = json.loads(frame.loc[1, "game_spec"])
    mixed["metadata"]["generation"]["prompt_style"] = "guided"
    frame.loc[1, "game_spec"] = json.dumps(mixed)
    frame.to_csv(csv_path, index=False)

    state, code = _check(root)

    assert code is ExitCode.INVALID_CONFIG
    assert state["accepted_row_provenance_valid"] is False
    assert "generation.prompt_style" in state["message"]


def test_other_material_report_config_is_rejected(tmp_path):
    root, report = _write_run(tmp_path)
    report["config"]["prescreen"] = "enforce"
    (root / "games" / "pair_coverage.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    state, code = _check(root)

    assert code is ExitCode.INVALID_CONFIG
    assert "prescreen" in state["message"]


def test_shadow_to_off_transition_ignores_only_prescreen_fields(tmp_path):
    root, _ = _write_run(tmp_path)
    _write_shadow_to_off_transition(root)

    strict_state, strict_code = _check(root)
    transition_state, transition_code = _check(
        root,
        allow_shadow_to_off_prescreen_transition=True,
    )

    assert strict_code is ExitCode.INVALID_CONFIG
    assert "prescreen" in strict_state["message"]
    assert transition_code is ExitCode.COMPLETE
    assert transition_state["accepted_row_provenance_valid"] is True
    assert transition_state["allow_shadow_to_off_prescreen_transition"] is True


def test_shadow_to_off_transition_still_rejects_other_provenance_drift(tmp_path):
    root, _ = _write_run(tmp_path)
    frame, _ = _write_shadow_to_off_transition(root)
    drifted = json.loads(frame.loc[1, "game_spec"])
    drifted["metadata"]["generation"]["judge_model"] = "foreign-judge"
    frame.loc[1, "game_spec"] = json.dumps(drifted)
    frame.to_csv(root / "claude-sonnet-5_games.csv", index=False)

    state, code = _check(
        root,
        allow_shadow_to_off_prescreen_transition=True,
    )

    assert code is ExitCode.INVALID_CONFIG
    assert state["accepted_row_provenance_valid"] is False
    assert "generation.judge_model" in state["message"]


def test_missing_and_malformed_inputs_have_distinct_results(tmp_path):
    missing_root = tmp_path / "missing"
    state, code = _check(missing_root)
    assert code is ExitCode.WAITING
    assert state["status"] == "waiting"

    root, _ = _write_run(tmp_path)
    (root / "games" / "pair_coverage.json").write_text("{", encoding="utf-8")
    state, code = _check(root)
    assert code is ExitCode.READ_ERROR
    assert state["status"] == "read_error"
    assert state["error_type"] == "JSONDecodeError"


def test_cli_main_uses_real_value_set_and_persists_state(tmp_path):
    root, _ = _write_run(tmp_path)

    code = main(
        [
            "--run-root", str(root),
            "--value-set", "HHH_new",
            "--target", "1",
            "--retries", "1",
            "--retry-delay", "0",
        ]
    )

    assert code == int(ExitCode.COMPLETE)
    assert (root / "monitor" / "status.json").exists()
    assert (root / "monitor" / "COMPLETE.json").exists()


def test_cli_allows_shadow_to_off_prescreen_transition(tmp_path):
    root, _ = _write_run(tmp_path)
    _write_shadow_to_off_transition(root)

    code = main(
        [
            "--run-root", str(root),
            "--value-set", "HHH_new",
            "--target", "1",
            "--retries", "1",
            "--retry-delay", "0",
            "--allow-shadow-to-off-prescreen-transition",
        ]
    )

    assert code == int(ExitCode.COMPLETE)
    status = json.loads(
        (root / "monitor" / "status.json").read_text(encoding="utf-8")
    )
    assert status["allow_shadow_to_off_prescreen_transition"] is True
