"""Pytest configuration: shared fixtures for the offline test suite.

The ``conflictxgame`` package must be importable — install it first with
``uv sync --extra dev``.
"""

import pytest

from conflictxgame.games.game_structures import ActionSpec, GameConfig, GameSpec, PlayerSpec


#: Small value set used across tests. Definitions are distinctive strings so
#: dummy judges can recover the value name from a prompt.
TEST_VALUE_DICT = {
    "v1": "sharing information openly with others",
    "v2": "protecting confidential information",
    "v3": "following institutional rules",
}


def make_players(arities, tags=None):
    """Build one PlayerSpec per seat with labels encoding their value tags."""
    players = []
    for seat, arity in enumerate(arities):
        actions = []
        for slot in range(arity):
            tag = tags[seat][slot] if tags else f"v{slot + 1}"
            actions.append(
                ActionSpec(
                    label=f"favor {tag} (p{seat})",
                    description=f"Player {seat} takes the action favoring {tag}.",
                    value_tag=tag,
                )
            )
        players.append(
            PlayerSpec(
                name=f"Player {seat}",
                persona=f"Persona of player {seat}.",
                role_description=f"Role of player {seat}.",
                actions=actions,
                value_pair=sorted({a.value_tag for a in actions}),
            )
        )
    return players


def make_spec(structure, arities, payoffs, horizon=1, game_id=None, tags=None):
    """Convenience GameSpec factory with generated outcome descriptions."""
    config = GameConfig(
        structure=structure,
        n_players=len(arities),
        arities=tuple(arities),
        horizon=horizon,
    )
    outcome_descriptions = {key: f"Outcome under profile {key}." for key in payoffs}
    return GameSpec(
        game_id=game_id or f"test-{config.cell_key()}",
        config=config,
        description="A concrete test scenario at Riverside Community Center in Toledo, Ohio.",
        players=make_players(arities, tags=tags),
        payoffs=payoffs,
        outcome_descriptions=outcome_descriptions,
        value_set="test",
    )


@pytest.fixture
def pd_spec():
    """A repeated symmetric prisoner's-dilemma-patterned spec (R,S,T,P = 3,0,5,1)."""
    return make_spec(
        "repeated",
        [2, 2],
        {"0,0": [3, 3], "0,1": [0, 5], "1,0": [5, 0], "1,1": [1, 1]},
        horizon=5,
    )


@pytest.fixture
def value_dict():
    """The shared test value set."""
    return dict(TEST_VALUE_DICT)
