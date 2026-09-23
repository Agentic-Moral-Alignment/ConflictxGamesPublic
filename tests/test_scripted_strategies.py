"""Tests for scripted strategies, including ZD gating and analytic checks."""

import random

import pytest

from conflictxgame.games import game_structures as gs
from conftest import make_spec
from conflictxgame.games.scripted_strategies import (
    STRATEGY_REGISTRY,
    StrategyInapplicableError,
    ZDStrategy,
    make_strategy,
    zd_applicable,
)


class TestBasicStrategies:
    def test_always_first_last(self, pd_spec):
        assert make_strategy("always_first").act(pd_spec, 0, []) == 0
        assert make_strategy("always_last").act(pd_spec, 0, []) == 1

    def test_always_cooperate_defect(self, pd_spec):
        assert make_strategy("always_cooperate").act(pd_spec, 0, []) == 0
        assert make_strategy("always_defect").act(pd_spec, 1, []) == 1

    def test_random_respects_arity(self):
        spec = make_spec(
            "simultaneous", [2, 3],
            {"0,0": [1, 1], "0,1": [1, 1], "0,2": [1, 1],
             "1,0": [1, 1], "1,1": [1, 1], "1,2": [1, 1]},
        )
        rng = random.Random(0)
        strategy = make_strategy("random")
        actions = {strategy.act(spec, 1, [], rng=rng) for _ in range(100)}
        assert actions == {0, 1, 2}

    def test_tit_for_tat(self, pd_spec):
        tft = make_strategy("tit_for_tat")
        assert tft.act(pd_spec, 0, []) == 0  # opens cooperatively
        assert tft.act(pd_spec, 0, [(0, 1)]) == 1  # mirrors defection
        assert tft.act(pd_spec, 0, [(0, 1), (1, 0)]) == 0  # mirrors return to coop

    def test_tit_for_tat_sequential_sees_partial_round(self, pd_spec):
        tft = make_strategy("tit_for_tat")
        # Opponent (seat 0) already defected this round.
        assert tft.act(pd_spec, 1, [], partial_round={0: 1}) == 1

    def test_tit_for_tat_majority_multiplayer(self):
        payoffs = {gs.profile_key(p): [1.0, 1.0, 1.0] for p in gs.enumerate_joint_profiles([2, 2, 2])}
        payoffs["0,0,0"] = [3.0, 3.0, 3.0]  # make all-cooperate the welfare max
        spec = make_spec("simultaneous", [2, 2, 2], payoffs)
        tft = make_strategy("tit_for_tat")
        assert tft.act(spec, 0, [(0, 1, 1)]) == 1  # majority of others defected
        assert tft.act(spec, 0, [(1, 0, 1)]) == 0  # tie broken toward lower index

    def test_tit_for_tat_clamps_to_own_arity(self):
        payoffs = {gs.profile_key(p): [1.0, 1.0] for p in gs.enumerate_joint_profiles([2, 3])}
        payoffs["0,0"] = [3.0, 3.0]
        spec = make_spec("simultaneous", [2, 3], payoffs)
        tft = make_strategy("tit_for_tat")
        assert tft.act(spec, 0, [(0, 2)]) == 1  # opponent played 2; own menu tops out at 1

    def test_grim_trigger(self, pd_spec):
        grim = make_strategy("grim_trigger")
        assert grim.act(pd_spec, 0, [(0, 0), (0, 0)]) == 0
        assert grim.act(pd_spec, 0, [(0, 0), (0, 1)]) == 1
        # Once triggered, stays triggered even after opponent resumes cooperating.
        assert grim.act(pd_spec, 0, [(0, 1), (1, 0)]) == 1

    def test_win_stay_lose_shift(self, pd_spec):
        wsls = make_strategy("win_stay_lose_shift")
        rng = random.Random(0)
        # Mean own payoff over the PD tensor is (3+0+5+1)/4 = 2.25.
        assert wsls.act(pd_spec, 0, [(0, 0)], rng=rng) == 0  # 3 >= 2.25: stay
        assert wsls.act(pd_spec, 0, [(0, 1)], rng=rng) == 1  # 0 < 2.25: shift
        assert wsls.act(pd_spec, 0, [(1, 0)], rng=rng) == 1  # 5 >= 2.25: stay on defect

    def test_registry_complete(self):
        for name in STRATEGY_REGISTRY:
            strategy = make_strategy(name)
            assert strategy.name == name
        with pytest.raises(KeyError):
            make_strategy("nope")


class TestZDStrategies:
    def test_canonical_extort2(self, pd_spec):
        probs = ZDStrategy("extort", chi=2.0).probabilities(pd_spec)
        assert probs["p_cc"] == pytest.approx(8 / 9)
        assert probs["p_cd"] == pytest.approx(1 / 2)
        assert probs["p_dc"] == pytest.approx(1 / 3)
        assert probs["p_dd"] == 0.0

    def test_canonical_generous(self, pd_spec):
        probs = ZDStrategy("generous", chi=2.0).probabilities(pd_spec)
        assert probs["p_cc"] == pytest.approx(1.0)
        assert probs["p_cd"] == pytest.approx(1 / 8)
        assert probs["p_dc"] == pytest.approx(1.0)
        assert probs["p_dd"] == pytest.approx(1 / 4)

    def test_gating_rejects_non_repeated(self, pd_spec):
        one_shot = make_spec(
            "simultaneous", [2, 2],
            {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
        )
        assert not zd_applicable(one_shot)
        with pytest.raises(StrategyInapplicableError):
            ZDStrategy("extort").probabilities(one_shot)

    def test_gating_rejects_asymmetric_and_2x3(self):
        asym = make_spec(
            "repeated", [2, 2],
            {"0,0": [3, 2], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
            horizon=5,
        )
        assert not zd_applicable(asym)
        payoffs_2x3 = {gs.profile_key(p): [1.0, 1.0] for p in gs.enumerate_joint_profiles([2, 3])}
        wide = make_spec("repeated", [2, 3], payoffs_2x3, horizon=5)
        assert not zd_applicable(wide)

    def test_gating_rejects_non_pd_ordering(self):
        stag = make_spec(
            "repeated", [2, 2],
            {"0,0": [4, 4], "0,1": [0, 3], "1,0": [3, 0], "1,1": [2, 2]},
            horizon=5,
        )
        assert not zd_applicable(stag)

    def test_extortion_ratio_converges(self, pd_spec):
        """Against an unconditional cooperator, the extortioner's surplus over P
        converges to chi times the opponent's surplus (the ZD enforcement)."""
        zd = ZDStrategy("extort", chi=2.0)
        rng = random.Random(42)
        history = []
        for _ in range(60000):
            a0 = zd.act(pd_spec, 0, history, rng=rng)
            history.append((a0, 0))  # opponent always cooperates
        payoffs0 = sum(pd_spec.payoff(p)[0] for p in history) / len(history)
        payoffs1 = sum(pd_spec.payoff(p)[1] for p in history) / len(history)
        P = 1.0
        ratio = (payoffs0 - P) / (payoffs1 - P)
        assert ratio == pytest.approx(2.0, abs=0.1)

    def test_zd_first_move_cooperates(self, pd_spec):
        assert ZDStrategy("extort").act(pd_spec, 0, [], rng=random.Random(0)) == 0
