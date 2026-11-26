"""Tests for re-keying fabricated history by own versus counterpart action."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "history_rekey_tests", ROOT / "reproduce" / "history_rekey.py"
)
rekey = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rekey)

# Scenario "s": both seats face harmlessness|honesty (same pair), with the
# seats' action orders reversed so own and counterpart indices differ.
# Scenario "o": seat 0 faces harmlessness|honesty; seat 1 faces honesty|helpfulness,
# sharing only honesty (one-overlap).
ACTIONS = {
    ("s", 0): ("harmlessness", "honesty"),
    ("s", 1): ("honesty", "harmlessness"),
    ("o", 0): ("harmlessness", "honesty"),
    ("o", 1): ("helpfulness", "honesty"),
}
PAIRS = {key: tuple(sorted(tags)) for key, tags in ACTIONS.items()}
PROBES = ("prev_0-0", "prev_0-1", "prev_1-0", "prev_1-1")


def items_for(scenario, seat, chooser, unanswered=()):
    """load_game-shaped rows; chooser(own_tag, cp_tag) returns the chosen tag."""
    rows = [{"scenario_id": scenario, "seat": seat, "probe": "fresh", "pair": PAIRS[(scenario, seat)],
             "chosen": PAIRS[(scenario, seat)][0], "answered": True}]
    for probe in PROBES:
        own_i, cp_i = rekey.probe_roles(probe, scenario, seat, ACTIONS)
        own_tag, cp_tag = ACTIONS[(scenario, seat)][own_i], ACTIONS[(scenario, 1 - seat)][cp_i]
        answered = probe not in unanswered
        rows.append({"scenario_id": scenario, "seat": seat, "probe": probe,
                     "pair": PAIRS[(scenario, seat)],
                     "chosen": chooser(own_tag, cp_tag, PAIRS[(scenario, seat)]) if answered else "",
                     "answered": answered})
    return pd.DataFrame(rows)


def contrasts(chooser, scenario="s", seat=0, **kwargs):
    roles = rekey.seat_roles(PAIRS, ACTIONS)
    unanswered = kwargs.pop("unanswered", ())
    frame = rekey.seat_contrasts(items_for(scenario, seat, chooser, unanswered), ACTIONS, roles, **kwargs)
    return frame.iloc[0]


def pure_self(own, cp, pair):
    return own


def pure_copy(own, cp, pair):
    # An outside counterpart value can't be copied; fall back to a focal value
    # that is not the shared one (harmlessness in scenario "o").
    return cp if cp in pair else pair[0]


def count_priming(own, cp, pair):
    """Chooses whichever focal value was mentioned more; ties break on a fixed value."""
    mentions = [v for v in (own, cp) if v in pair]
    a, b = mentions.count(pair[0]), mentions.count(pair[1])
    return pair[0] if a > b else pair[1] if b > a else pair[1]


class TestRoles:
    def test_probe_tokens_are_read_at_the_seat_position(self):
        assert rekey.probe_roles("prev_0-1", "s", 0, ACTIONS) == (0, 1)
        assert rekey.probe_roles("prev_0-1", "s", 1, ACTIONS) == (1, 0)

    def test_seat_types(self):
        roles = rekey.seat_roles(PAIRS, ACTIONS)
        assert roles[("s", 0)] == ("same", None)
        assert roles[("o", 0)] == ("overlap", "honesty")

    def test_counterpart_with_no_shared_value_raises(self):
        actions = {("x", 0): ("harmlessness", "honesty"), ("x", 1): ("helpfulness", "fairness")}
        with pytest.raises(ValueError):
            rekey.seat_roles({}, actions)


class TestContrasts:
    @pytest.mark.parametrize("seat", [0, 1])
    def test_pure_self(self, seat):
        row = contrasts(pure_self, seat=seat)
        assert row["self"] == 1 and row["copy"] == 0 and row["asym"] == 1

    @pytest.mark.parametrize("seat", [0, 1])
    def test_pure_copy(self, seat):
        row = contrasts(pure_copy, seat=seat)
        assert row["copy"] == 1 and row["self"] == 0 and row["asym"] == -1

    def test_count_priming_is_role_symmetric(self):
        row = contrasts(count_priming)
        assert row["self"] == row["copy"] and row["asym"] == 0

    def test_identities(self):
        for chooser in (pure_self, pure_copy, count_priming):
            row = contrasts(chooser)
            assert row["asym"] == pytest.approx(row["self"] - row["copy"])
            assert row["self"] + row["copy"] == pytest.approx(row["both_minus_none"])

    def test_one_overlap_copy(self):
        row = contrasts(pure_copy, scenario="o")
        assert row["seat_type"] == "overlap" and row["copy_1ov"] == 1
        assert np.isnan(row["copy"]) and np.isnan(row["asym"])

    def test_stratum_keeps_the_answered_half(self):
        # Losing one cell kills one COPY half and ASYM survives only if p_AB and p_BA remain.
        row = contrasts(pure_copy, unanswered=("prev_0-0",))
        assert row["copy"] == 1 and not row["complete_2x2"]

    def test_complete_mode_drops_incomplete_seats(self):
        row = contrasts(pure_copy, unanswered=("prev_0-0",), missing="complete")
        assert np.isnan(row["copy"]) and np.isnan(row["self"])


class TestArmResolution:
    TAIL = "_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv"

    def test_states_excludes_states_steer_and_stakes(self, tmp_path):
        names = ["m_mcq_balanced50_states.csv", "m_mcq_balanced50_states-steer.csv",
                 "m_mcq_balanced50_states_stakes-payoff.csv", "m_mcq_balanced50_states_observed.csv"]
        found = rekey.resolve_states_file(str(tmp_path), "m", "menu", "baseline", names=names)
        assert found.endswith("m_mcq_balanced50_states.csv")
        assert rekey.resolve_states_file(str(tmp_path), "m", "menu", "observed",
                                         names=names).endswith("_states_observed.csv")

    def test_explicit_cue_token(self, tmp_path):
        names = ["m_open_balanced50_states" + self.TAIL,
                 "m_open_balanced50_states_rolecue-explicit" + self.TAIL]
        found = rekey.resolve_states_file(str(tmp_path), "m", "open_undisclosed", "baseline",
                                          role_cue="explicit", names=names)
        assert "_rolecue-explicit" in found

    def test_visibility_only_for_menu(self):
        with pytest.raises(ValueError):
            rekey.arm_token("open_undisclosed", "observed")
