import pandas as pd

from conflictxgame.figures.comparison_figures import (
    build_cross_domain_table,
    build_game_difference_table,
)


def _solo(models):
    return pd.DataFrame([
        {
            "model": model,
            "value": value,
            "ability": ability,
            "ci_lower": ability - 0.1,
            "ci_upper": ability + 0.1,
        }
        for model in models
        for value, ability in (("harmlessness", 1.0), ("honesty", 0.0), ("helpfulness", -1.0))
    ])


def _games(model, shift):
    return pd.DataFrame([
        {
            "model": model,
            "condition": condition,
            "value": value,
            "ability": ability + shift,
            "ci_lower": ability + shift - 0.2,
            "ci_upper": ability + shift + 0.2,
        }
        for condition in ("MCQ choice", "Open-ended", "Play (all)")
        for value, ability in (("harmlessness", 1.0), ("honesty", 0.0), ("helpfulness", -1.0))
    ])


def test_cross_domain_table_and_game_differences_use_the_requested_models():
    models = ["openai:model-a", "openai:model-b"]
    cross = build_cross_domain_table(
        _solo(models), [_games(models[0], 0.0), _games(models[1], 0.5)],
        models, ["A", "B"],
    )
    differences = build_game_difference_table(cross, models[0])

    assert set(cross["setting"]) == {"Solo", "Game MCQ", "Game open", "Game play"}
    assert set(differences["model"]) == {models[1]}
    assert set(differences["ability_difference"].round(6)) == {0.5}
