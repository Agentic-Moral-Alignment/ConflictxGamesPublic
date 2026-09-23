"""Crash-safe, idempotent appends to the filtered-scenario corpus."""

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from conflictxgame.games.filter_game_scenarios import append_to_scenario_csv


def test_new_rows_replace_ids_and_column_union_keeps_existing_order(tmp_path):
    output = tmp_path / "scenarios.csv"
    append_to_scenario_csv(
        pd.DataFrame(
            [
                {"scenario_id": "a", "verdict": "old-a", "legacy": "A"},
                {"scenario_id": "b", "verdict": "old-b", "legacy": "B"},
            ]
        ),
        output,
    )

    new = pd.DataFrame(
        [
            {"scenario_id": "b", "verdict": "new-b", "batch": 1},
            {"scenario_id": "c", "verdict": "new-c", "batch": 1},
        ]
    )
    combined = append_to_scenario_csv(new, output)

    assert list(combined.columns) == ["scenario_id", "verdict", "legacy", "batch"]
    assert combined["scenario_id"].tolist() == ["a", "b", "c"]
    assert combined.set_index("scenario_id").loc["b", "verdict"] == "new-b"
    assert pd.isna(combined.set_index("scenario_id").loc["b", "legacy"])

    persisted = pd.read_csv(output)
    pdt.assert_frame_equal(persisted, combined, check_dtype=False)


def test_replaying_the_same_batch_is_idempotent(tmp_path):
    output = tmp_path / "scenarios.csv"
    batch = pd.DataFrame(
        [
            {"scenario_id": "one", "accepted": True},
            {"scenario_id": "two", "accepted": False},
        ]
    )

    first = append_to_scenario_csv(batch, output)
    second = append_to_scenario_csv(batch, output)

    assert second["scenario_id"].tolist() == ["one", "two"]
    assert len(second) == len(first) == 2
    pdt.assert_frame_equal(pd.read_csv(output), second, check_dtype=False)


def test_last_duplicate_wins_across_existing_and_new_rows(tmp_path):
    output = tmp_path / "scenarios.csv"
    # Simulate duplicate rows left by the former non-idempotent writer.
    pd.DataFrame(
        [
            {"scenario_id": "a", "version": "old-a-1"},
            {"scenario_id": "a", "version": "old-a-2"},
            {"scenario_id": "b", "version": "old-b"},
        ]
    ).to_csv(output, index=False)
    new = pd.DataFrame(
        [
            {"scenario_id": "b", "version": "new-b-1"},
            {"scenario_id": "b", "version": "new-b-2"},
            {"scenario_id": "c", "version": "new-c"},
        ]
    )

    combined = append_to_scenario_csv(new, output)

    assert combined.to_dict("records") == [
        {"scenario_id": "a", "version": "old-a-2"},
        {"scenario_id": "b", "version": "new-b-2"},
        {"scenario_id": "c", "version": "new-c"},
    ]


def test_rows_without_usable_ids_keep_append_semantics(tmp_path):
    output = tmp_path / "scenarios.csv"
    append_to_scenario_csv(
        pd.DataFrame(
            [
                {"scenario_id": None, "value": "missing-1"},
                {"scenario_id": "", "value": "blank-1"},
            ]
        ),
        output,
    )
    combined = append_to_scenario_csv(
        pd.DataFrame(
            [
                {"scenario_id": None, "value": "missing-2"},
                {"scenario_id": "", "value": "blank-2"},
            ]
        ),
        output,
    )

    assert combined["value"].tolist() == [
        "missing-1", "blank-1", "missing-2", "blank-2"
    ]


def test_header_only_existing_file_contributes_its_columns(tmp_path):
    output = tmp_path / "scenarios.csv"
    output.write_text("scenario_id,legacy\n", encoding="utf-8")

    combined = append_to_scenario_csv(
        pd.DataFrame([{"scenario_id": "new", "batch": 4}]), output
    )

    assert list(combined.columns) == ["scenario_id", "legacy", "batch"]
    assert combined.loc[0, "scenario_id"] == "new"


def test_serialization_failure_leaves_previous_csv_intact(
    monkeypatch, tmp_path
):
    output = tmp_path / "scenarios.csv"
    append_to_scenario_csv(
        pd.DataFrame([{"scenario_id": "safe", "value": "original"}]), output
    )
    original = output.read_bytes()

    def fail_after_partial_write(self, destination, *args, **kwargs):
        destination.write("partial,output\n")
        destination.flush()
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated"):
        append_to_scenario_csv(
            pd.DataFrame([{"scenario_id": "unsafe", "value": "new"}]), output
        )

    assert output.read_bytes() == original
    assert not list(Path(tmp_path).glob(".scenarios.csv.*.tmp"))


def test_without_scenario_id_the_function_remains_a_plain_append(tmp_path):
    output = tmp_path / "legacy.csv"
    append_to_scenario_csv(pd.DataFrame([{"value": 1}]), output)
    combined = append_to_scenario_csv(pd.DataFrame([{"value": 1}]), output)

    assert combined["value"].tolist() == [1, 1]

