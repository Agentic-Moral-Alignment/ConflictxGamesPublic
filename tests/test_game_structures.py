"""Tests for the game configuration space, spec model, and tensor helpers."""

import json

import pytest

from conflictxgame.games import game_structures as gs
from conftest import make_spec


class TestGameConfig:
    def test_valid_configs(self):
        gs.GameConfig("single", 1, (3,))
        gs.GameConfig("simultaneous", 2, (2, 3))
        gs.GameConfig("sequential", 3, (2, 2, 2))
        gs.GameConfig("repeated", 2, (2, 2), horizon=5)

    def test_invalid_configs(self):
        with pytest.raises(ValueError):
            gs.GameConfig("single", 2, (2, 2))  # single requires 1 player
        with pytest.raises(ValueError):
            gs.GameConfig("simultaneous", 1, (2,))  # multi requires >= 2
        with pytest.raises(ValueError):
            gs.GameConfig("simultaneous", 2, (2, 1))  # arity >= 2
        with pytest.raises(ValueError):
            gs.GameConfig("simultaneous", 2, (2,))  # arities/n mismatch
        with pytest.raises(ValueError):
            gs.GameConfig("repeated", 2, (2, 2), horizon=1)  # repeated needs >= 2
        with pytest.raises(ValueError):
            gs.GameConfig("simultaneous", 2, (2, 2), horizon=3)  # one-shot needs 1
        with pytest.raises(ValueError):
            gs.GameConfig("tournament", 2, (2, 2))  # unknown structure

    def test_soft_cap_accepts_but_warns(self):
        """4-player / 4-action configs are accepted with a size warning."""
        big = gs.GameConfig("simultaneous", 4, (4, 4, 4, 4))
        with pytest.warns(gs.RepresentationSizeWarning):
            assert big.warn_if_large() is True
        assert big.joint_profile_count == 256

        small = gs.GameConfig("simultaneous", 3, (3, 3, 3))
        assert small.warn_if_large() is False  # 27 profiles == threshold

    def test_representation_size(self):
        config = gs.GameConfig("simultaneous", 2, (2, 3))
        size = config.representation_size()
        assert size["joint_profiles"] == 6
        assert size["tensor_entries"] == 12
        assert size["outcome_descriptions"] == 6
        assert size["estimated_prompt_tokens"] > 0

    def test_cell_key(self):
        assert gs.GameConfig("simultaneous", 2, (2, 3)).cell_key() == "simultaneous_n2_a2x3"
        assert gs.GameConfig("repeated", 2, (2, 2), horizon=5).cell_key() == "repeated_n2_a2x2_t5"


class TestEnumerateConfigs:
    def test_default_ranges(self):
        configs = gs.enumerate_configs(["simultaneous"], (2, 3), (2, 3))
        keys = {c.cell_key() for c in configs}
        # n=2: arity multisets (2,2), (2,3), (3,3); n=3: 4 multisets
        assert keys == {
            "simultaneous_n2_a2x2", "simultaneous_n2_a2x3", "simultaneous_n2_a3x3",
            "simultaneous_n3_a2x2x2", "simultaneous_n3_a2x2x3",
            "simultaneous_n3_a2x3x3", "simultaneous_n3_a3x3x3",
        }

    def test_single_included_when_min_players_is_one(self):
        configs = gs.enumerate_configs(["single", "simultaneous"], (1, 2), (2, 3))
        keys = {c.cell_key() for c in configs}
        assert "single_n1_a2" in keys and "single_n1_a3" in keys
        assert "simultaneous_n2_a2x2" in keys
        # single excluded when the range excludes n=1
        no_single = gs.enumerate_configs(["single"], (2, 3), (2, 3))
        assert no_single == []

    def test_asymmetric_cell_included_once(self):
        configs = gs.enumerate_configs(["sequential"], (2, 2), (2, 3))
        keys = [c.cell_key() for c in configs]
        assert keys.count("sequential_n2_a2x3") == 1

    def test_invalid_ranges(self):
        with pytest.raises(ValueError):
            gs.enumerate_configs(["simultaneous"], (0, 2), (2, 3))
        with pytest.raises(ValueError):
            gs.enumerate_configs(["simultaneous"], (2, 2), (1, 3))


class TestProfiles:
    def test_profile_key_roundtrip(self):
        for profile in gs.enumerate_joint_profiles([2, 3, 2]):
            assert gs.parse_profile_key(gs.profile_key(profile)) == profile

    def test_enumeration_count(self):
        assert len(gs.enumerate_joint_profiles([2, 3])) == 6
        assert len(gs.enumerate_joint_profiles([3])) == 3


class TestSpecValidationAndSerialization:
    def test_valid_spec_roundtrip(self, pd_spec):
        assert pd_spec.validate() == []
        clone = gs.GameSpec.from_json(pd_spec.to_json())
        assert clone.to_dict() == pd_spec.to_dict()

    def test_missing_payoff_profile_detected(self, pd_spec):
        del pd_spec.payoffs["1,1"]
        problems = pd_spec.validate()
        assert any("missing profiles" in p for p in problems)

    def test_wrong_payoff_length_detected(self, pd_spec):
        pd_spec.payoffs["0,0"] = [3.0]
        assert any("expected 2" in p for p in pd_spec.validate())

    def test_turn_order_default_and_validation(self, pd_spec):
        assert pd_spec.turn_order == [0, 1]
        pd_spec.turn_order = [0, 0]
        assert any("turn_order" in p for p in pd_spec.validate())


class TestTensorAnalysis:
    def test_pd_nash_and_detection(self, pd_spec):
        assert gs.pure_nash_equilibria(pd_spec) == [(1, 1)]
        assert gs.is_symmetric_two_by_two(pd_spec)
        payoffs = gs.pd_payoffs(pd_spec)
        assert payoffs["cooperative_action"] == 0
        assert (payoffs["R"], payoffs["S"], payoffs["T"], payoffs["P"]) == (3, 0, 5, 1)
        assert gs.detect_canonical_2x2(pd_spec) == "prisoners_dilemma_like"
        assert gs.cooperative_actions(pd_spec) == (0, 0)
        assert gs.punishment_actions(pd_spec) == (1, 1)

    def test_pd_detection_with_swapped_labels(self):
        """PD ordering is detected when action 1 is the cooperative one."""
        spec = make_spec(
            "repeated", [2, 2],
            {"0,0": [1, 1], "0,1": [5, 0], "1,0": [0, 5], "1,1": [3, 3]},
            horizon=3,
        )
        payoffs = gs.pd_payoffs(spec)
        assert payoffs["cooperative_action"] == 1
        assert gs.cooperative_actions(spec) == (1, 1)

    def test_stag_hunt_two_equilibria(self):
        spec = make_spec(
            "simultaneous", [2, 2],
            {"0,0": [4, 4], "0,1": [0, 3], "1,0": [3, 0], "1,1": [2, 2]},
        )
        assert set(gs.pure_nash_equilibria(spec)) == {(0, 0), (1, 1)}
        assert gs.detect_canonical_2x2(spec) == "stag_hunt_like"
        assert gs.pd_payoffs(spec) is None  # not a PD ordering

    def test_chicken_off_diagonal_equilibria(self):
        spec = make_spec(
            "simultaneous", [2, 2],
            {"0,0": [3, 3], "0,1": [2, 4], "1,0": [4, 2], "1,1": [0, 0]},
        )
        assert set(gs.pure_nash_equilibria(spec)) == {(0, 1), (1, 0)}
        assert gs.detect_canonical_2x2(spec) == "chicken_like"

    def test_asymmetric_2x3_nash(self):
        """Row has 2 actions, column has 3; (1, 2) is the unique pure NE."""
        payoffs = {
            "0,0": [2, 2], "0,1": [0, 3], "0,2": [1, 1],
            "1,0": [3, 0], "1,1": [1, 1], "1,2": [2, 4],
        }
        spec = make_spec("simultaneous", [2, 3], payoffs)
        assert gs.pure_nash_equilibria(spec) == [(1, 2)]
        assert not gs.is_symmetric_two_by_two(spec)
        assert gs.pd_payoffs(spec) is None  # ZD gate rejects 2x3

    def test_three_player_nash(self):
        """Volunteer's-dilemma-like: only profiles with exactly one volunteer are NE."""
        payoffs = {}
        for profile in gs.enumerate_joint_profiles([2, 2, 2]):
            volunteers = sum(1 for a in profile if a == 0)
            values = []
            for seat in range(3):
                if volunteers == 0:
                    values.append(0.0)
                else:
                    values.append(4.0 - (2.0 if profile[seat] == 0 else 0.0))
            payoffs[gs.profile_key(profile)] = values
        spec = make_spec("simultaneous", [2, 2, 2], payoffs)
        equilibria = set(gs.pure_nash_equilibria(spec))
        assert equilibria == {(0, 1, 1), (1, 0, 1), (1, 1, 0)}

    def test_best_response(self, pd_spec):
        assert gs.best_responses(pd_spec, (0, 0), 0) == [1]  # defect dominates
        assert gs.is_best_response(pd_spec, (1, 0), 0)
        assert not gs.is_best_response(pd_spec, (0, 0), 0)


class TestCoverageReport:
    def test_accounting_roundtrip(self, tmp_path):
        report = gs.CoverageReport()
        c1 = gs.GameConfig("single", 1, (3,))
        c2 = gs.GameConfig("simultaneous", 2, (2, 3))
        report.ensure_cell(c1, target=10)
        report.ensure_cell(c2, target=10)
        report.record_generated(c1, 7)
        report.record_generated(c2, 10)
        report.record_kept(c2.cell_key(), 4)
        report.add_note(c1, "generated 7/10 requested")

        json_path = tmp_path / "coverage.json"
        csv_path = tmp_path / "coverage.csv"
        report.to_json(str(json_path))
        report.to_csv(str(csv_path))

        loaded = gs.CoverageReport.from_json(str(json_path))
        row1 = loaded.cells[c1.cell_key()]
        row2 = loaded.cells[c2.cell_key()]
        assert (row1["target"], row1["generated"], row1["kept"]) == (10, 7, 0)
        assert (row2["target"], row2["generated"], row2["kept"]) == (10, 10, 4)
        assert row1["notes"] == ["generated 7/10 requested"]
        assert "single_n1_a3" in report.summary()

        with open(csv_path, encoding="utf-8") as f:
            header = f.readline()
        assert "cell_key" in header and "kept" in header
