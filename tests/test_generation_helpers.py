"""Tests for the combined-output helpers and prompt persistence of game generation."""

import random

from conflictxgame.games import game_structures as gs
from conftest import TEST_VALUE_DICT
from conflictxgame.games.generate_game_scenarios import (
    existing_cell_counts,
    generate_cell,
    next_cell_index,
)
from fakes import DummyGenerationClient


class TestCombinedOutputHelpers:
    def test_existing_cell_counts_and_next_index(self):
        simultaneous = gs.GameConfig("simultaneous", 2, (2, 2)).to_dict()
        sequential = gs.GameConfig("sequential", 2, (2, 2)).to_dict()
        games = {
            "simultaneous_n2_a2x2-0": {"config": simultaneous},
            "simultaneous_n2_a2x2-4": {"config": simultaneous},
            "sequential_n2_a2x2-1": {"config": sequential},
        }
        assert existing_cell_counts(games) == {
            "simultaneous_n2_a2x2": 2,
            "sequential_n2_a2x2": 1,
        }
        # Highest surviving index + 1, so top-ups after drops cannot collide.
        assert next_cell_index(games, "simultaneous_n2_a2x2") == 5
        assert next_cell_index(games, "sequential_n2_a2x2") == 2
        assert next_cell_index(games, "repeated_n2_a2x2_t5") == 0
        assert next_cell_index({}, "simultaneous_n2_a2x2") == 0


class TestPromptPersistence:
    def test_prompt_log_structure(self):
        client = DummyGenerationClient()
        rng = random.Random(0)
        config = gs.GameConfig("simultaneous", 2, (2, 2))
        prompt_log = {}
        specs = generate_cell(
            config, client, TEST_VALUE_DICT, "test", 2, 2, 3, rng,
            generating_model="dummy-generator", prompt_log=prompt_log,
        )
        assert specs
        assert prompt_log["stage_one"]
        entry = prompt_log["stage_one"][0]
        assert entry["cell_key"] == "simultaneous_n2_a2x2"
        assert "scenario outlines" in entry["prompt"]
        assert set(prompt_log["stage_two"]) == {spec.game_id for spec in specs}
        for logged in prompt_log["stage_two"].values():
            assert "elaborate a scenario outline" in logged["prompt"]
