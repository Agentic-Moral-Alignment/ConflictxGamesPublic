"""Scripted opponent strategies for live game play.

Scripted strategies give controlled probes of the focal model's behavior:
unlike LLM opponents, they are deterministic (or seeded-random) functions of
the game history, so differences in outcomes are attributable to the focal
model. All strategies are pure functions of ``(spec, seat, history,
partial_round)`` plus an injected :class:`random.Random`, so no per-game state
needs resetting.

Conventions:
    * ``history`` is a list of completed rounds; each round is a tuple of one
      action index per seat.
    * ``partial_round`` (sequential games only) maps seats that have already
      moved this round to their action; later movers may condition on it.
    * "Cooperating" and "punishing" are operationalized from the scenario's
      annotative payoff tensor via :func:`conflictxgame.games.game_structures.cooperative_actions`
      (welfare-maximizing profile) and
      :func:`conflictxgame.games.game_structures.punishment_actions` (welfare-minimizing profile).

Zero-determinant (ZD) strategies:
    :class:`ZDStrategy` implements Press & Dyson (2012) memory-one strategies
    (extortionate) and the generous variant of Stewart & Plotkin (2012). They
    are only well-defined for repeated symmetric 2x2 games with a prisoner's-
    dilemma payoff ordering — the scripted agent *assumes the game is
    symmetric*. Applicability is gated by :func:`zd_applicable`, which detects
    the pattern from the payoff tensor (never from a game name); inapplicable
    games raise :class:`StrategyInapplicableError`.
"""

import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from conflictxgame.games.game_structures import (
    GameSpec,
    cooperative_actions,
    enumerate_joint_profiles,
    pd_payoffs,
    punishment_actions,
)


class StrategyInapplicableError(ValueError):
    """Raised when a strategy's assumptions do not hold for a game spec."""


def zd_applicable(spec: GameSpec) -> bool:
    """Whether zero-determinant strategies apply to this game.

    Requires a ``repeated`` structure and a symmetric 2x2 payoff tensor with a
    prisoner's-dilemma ordering (``T > R > P > S``), detected from the tensor
    itself via :func:`conflictxgame.games.game_structures.pd_payoffs`.
    """
    return spec.config.structure == "repeated" and pd_payoffs(spec) is not None


def _others_last_actions(
    spec: GameSpec,
    seat: int,
    history: List[Tuple[int, ...]],
    partial_round: Optional[Dict[int, int]],
) -> List[int]:
    """Collect the most recent observable action of every other seat.

    For a seat that has already moved in the current (sequential) round, that
    move is used; otherwise the seat's action in the last completed round is
    used. Seats with no observable action yet are omitted.
    """
    actions = []
    for other in range(spec.config.n_players):
        if other == seat:
            continue
        if partial_round and other in partial_round:
            actions.append(partial_round[other])
        elif history:
            actions.append(history[-1][other])
    return actions


def _majority_action(actions: Sequence[int]) -> Optional[int]:
    """Most common action in ``actions`` (ties broken by lowest index)."""
    if not actions:
        return None
    counts = Counter(actions)
    best = max(counts.values())
    return min(a for a, c in counts.items() if c == best)


class ScriptedStrategy(ABC):
    """Base class for scripted strategies.

    Attributes:
        name: Registry name of the strategy (set by :func:`make_strategy`).
    """

    name: str = "scripted"

    @abstractmethod
    def act(
        self,
        spec: GameSpec,
        seat: int,
        history: List[Tuple[int, ...]],
        partial_round: Optional[Dict[int, int]] = None,
        rng: Optional[random.Random] = None,
    ) -> int:
        """Choose an action index for ``seat`` given the observable history.

        Args:
            spec: The game being played.
            seat: The strategy's seat index.
            history: Completed rounds (tuples of per-seat action indices).
            partial_round: Actions already taken this round (sequential games).
            rng: Source of randomness for stochastic strategies.

        Returns:
            An action index in ``range(spec.config.arities[seat])``.
        """


class AlwaysAction(ScriptedStrategy):
    """Unconditionally plays a fixed menu position.

    Args:
        position: ``"first"`` (index 0) or ``"last"`` (highest index).
    """

    def __init__(self, position: str = "first"):
        if position not in ("first", "last"):
            raise ValueError("position must be 'first' or 'last'")
        self.position = position
        self.name = f"always_{position}"

    def act(self, spec, seat, history, partial_round=None, rng=None) -> int:
        return 0 if self.position == "first" else spec.config.arities[seat] - 1


class AlwaysCooperate(ScriptedStrategy):
    """Always plays its action from the welfare-maximizing profile."""

    name = "always_cooperate"

    def act(self, spec, seat, history, partial_round=None, rng=None) -> int:
        return cooperative_actions(spec)[seat]


class AlwaysDefect(ScriptedStrategy):
    """Always plays its action from the welfare-minimizing profile."""

    name = "always_defect"

    def act(self, spec, seat, history, partial_round=None, rng=None) -> int:
        return punishment_actions(spec)[seat]


class UniformRandom(ScriptedStrategy):
    """Plays uniformly at random over its menu."""

    name = "random"

    def act(self, spec, seat, history, partial_round=None, rng=None) -> int:
        rng = rng or random.Random()
        return rng.randrange(spec.config.arities[seat])


class TitForTat(ScriptedStrategy):
    """Opens cooperatively, then mirrors the (majority) opponent action.

    Generalizations beyond the classic 2-player 2-action setting:
        * With several opponents, the majority of their most recent observable
          actions is mirrored (ties to the lowest action index).
        * If the mirrored index exceeds this seat's arity (asymmetric games),
          it is clamped to the top of the menu.
    """

    name = "tit_for_tat"

    def act(self, spec, seat, history, partial_round=None, rng=None) -> int:
        observed = _others_last_actions(spec, seat, history, partial_round)
        mirrored = _majority_action(observed)
        if mirrored is None:
            return cooperative_actions(spec)[seat]
        return min(mirrored, spec.config.arities[seat] - 1)


class GrimTrigger(ScriptedStrategy):
    """Cooperates until any opponent ever deviates, then punishes forever.

    Deviation means an opponent playing anything other than *their* action in
    the welfare-maximizing profile in any completed round (or earlier in the
    current sequential round).
    """

    name = "grim_trigger"

    def act(self, spec, seat, history, partial_round=None, rng=None) -> int:
        coop = cooperative_actions(spec)
        triggered = False
        for round_profile in history:
            for other in range(spec.config.n_players):
                if other != seat and round_profile[other] != coop[other]:
                    triggered = True
        if partial_round:
            for other, action in partial_round.items():
                if other != seat and action != coop[other]:
                    triggered = True
        return punishment_actions(spec)[seat] if triggered else coop[seat]


class WinStayLoseShift(ScriptedStrategy):
    """Pavlov-style strategy: repeat after a good round, switch after a bad one.

    "Good" means last round's own payoff was at least the seat's mean payoff
    over all joint profiles (an aspiration level that needs no game-specific
    tuning). On a shift, one of the other actions is chosen uniformly.
    """

    name = "win_stay_lose_shift"

    def act(self, spec, seat, history, partial_round=None, rng=None) -> int:
        rng = rng or random.Random()
        if not history:
            return cooperative_actions(spec)[seat]
        last_profile = history[-1]
        last_payoff = spec.payoff(last_profile)[seat]
        profiles = enumerate_joint_profiles(spec.config.arities)
        aspiration = sum(spec.payoff(p)[seat] for p in profiles) / len(profiles)
        own_last = last_profile[seat]
        if last_payoff >= aspiration:
            return own_last
        alternatives = [a for a in range(spec.config.arities[seat]) if a != own_last]
        return rng.choice(alternatives)


class ZDStrategy(ScriptedStrategy):
    """Memory-one zero-determinant strategy for repeated symmetric 2x2 PD-like games.

    The cooperation probabilities ``(p_cc, p_cd, p_dc, p_dd)`` — indexed by
    (own, opponent) action last round in cooperate/defect terms — are solved
    from the scenario's own payoff tensor:

    Extortionate (Press & Dyson 2012), enforcing
    ``s_self - P = chi * (s_opp - P)``::

        p_cc = 1 - phi*(chi - 1)*(R - P)
        p_cd = 1 - phi*((P - S) + chi*(T - P))
        p_dc = phi*((T - P) + chi*(P - S))
        p_dd = 0

    with ``phi = phi_max / 2`` where ``phi_max`` is the largest value keeping
    all probabilities in [0, 1]. For the canonical (R,S,T,P) = (3,0,5,1) and
    ``chi = 2`` this yields Extort-2: (8/9, 1/2, 1/3, 0).

    Generous (Stewart & Plotkin 2012), enforcing
    ``s_self - R = chi * (s_opp - R)``::

        p_cc = 1
        p_cd = 1 - phi*((R - S) + chi*(T - R))
        p_dc = phi*((T - R) + chi*(R - S))
        p_dd = phi*(chi - 1)*(R - P)

    with ``phi = phi_max``. For the canonical payoffs and ``chi = 2`` this
    yields ZDGTFT-2: (1, 1/8, 1, 1/4).

    Args:
        variant: ``"extort"`` or ``"generous"``.
        chi: Extortion/generosity factor (> 1).

    Raises:
        StrategyInapplicableError: If constructed for a game where
            :func:`zd_applicable` is False (asymmetric, non-2x2, non-PD
            ordering, or not repeated). The probabilities are computed lazily
            per spec, so one instance can serve many games.
    """

    def __init__(self, variant: str = "extort", chi: float = 2.0):
        if variant not in ("extort", "generous"):
            raise ValueError("variant must be 'extort' or 'generous'")
        if chi <= 1:
            raise ValueError("chi must be > 1")
        self.variant = variant
        self.chi = chi
        self.name = f"zd_{variant}"

    def probabilities(self, spec: GameSpec) -> Dict[str, float]:
        """Solve the four cooperation probabilities for this game's payoffs.

        Returns:
            ``{"p_cc", "p_cd", "p_dc", "p_dd", "cooperative_action", "phi"}``.

        Raises:
            StrategyInapplicableError: If the game is not a repeated symmetric
                2x2 game with a PD ordering.
        """
        if not zd_applicable(spec):
            raise StrategyInapplicableError(
                f"{self.name} requires a repeated symmetric 2x2 game with a "
                f"prisoner's-dilemma payoff ordering; game '{spec.game_id}' "
                f"({spec.config.cell_key()}) does not qualify."
            )
        payoffs = pd_payoffs(spec)
        R, S, T, P = payoffs["R"], payoffs["S"], payoffs["T"], payoffs["P"]
        chi = self.chi
        if self.variant == "extort":
            phi_max = min(
                1.0 / ((chi - 1) * (R - P)),
                1.0 / ((P - S) + chi * (T - P)),
                1.0 / ((T - P) + chi * (P - S)),
            )
            phi = phi_max / 2.0
            probs = {
                "p_cc": 1 - phi * (chi - 1) * (R - P),
                "p_cd": 1 - phi * ((P - S) + chi * (T - P)),
                "p_dc": phi * ((T - P) + chi * (P - S)),
                "p_dd": 0.0,
            }
        else:  # generous
            phi_max = min(
                1.0 / ((R - S) + chi * (T - R)),
                1.0 / ((T - R) + chi * (R - S)),
                1.0 / ((chi - 1) * (R - P)),
            )
            phi = phi_max
            probs = {
                "p_cc": 1.0,
                "p_cd": 1 - phi * ((R - S) + chi * (T - R)),
                "p_dc": phi * ((T - R) + chi * (R - S)),
                "p_dd": phi * (chi - 1) * (R - P),
            }
        probs["cooperative_action"] = payoffs["cooperative_action"]
        probs["phi"] = phi
        return probs

    def act(self, spec, seat, history, partial_round=None, rng=None) -> int:
        rng = rng or random.Random()
        probs = self.probabilities(spec)
        coop = int(probs["cooperative_action"])
        defect = 1 - coop
        if not history:
            return coop  # standard opening move for memory-one strategies
        last = history[-1]
        opponent = 1 - seat
        own_cooperated = last[seat] == coop
        opp_cooperated = last[opponent] == coop
        key = f"p_{'c' if own_cooperated else 'd'}{'c' if opp_cooperated else 'd'}"
        return coop if rng.random() < probs[key] else defect


#: Registry mapping CLI names to zero-argument strategy factories.
STRATEGY_REGISTRY = {
    "always_first": lambda: AlwaysAction("first"),
    "always_last": lambda: AlwaysAction("last"),
    "always_cooperate": AlwaysCooperate,
    "always_defect": AlwaysDefect,
    "random": UniformRandom,
    "tit_for_tat": TitForTat,
    "grim_trigger": GrimTrigger,
    "win_stay_lose_shift": WinStayLoseShift,
    "zd_extort": lambda: ZDStrategy("extort", chi=2.0),
    "zd_generous": lambda: ZDStrategy("generous", chi=2.0),
}


def make_strategy(name: str) -> ScriptedStrategy:
    """Instantiate a registered strategy by name.

    Args:
        name: A key of :data:`STRATEGY_REGISTRY`.

    Returns:
        A fresh strategy instance.

    Raises:
        KeyError: If the name is not registered.
    """
    try:
        return STRATEGY_REGISTRY[name]()
    except KeyError:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        ) from None
