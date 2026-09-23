from pathlib import Path

import pandas as pd
import pytest

from conflictxgame.figures import value_conditions_figure as figure


def _solo(path: Path, choice: str = 'A') -> None:
    pd.DataFrame([{'value1': 'a', 'value2': 'b', 'choice': choice}]).to_csv(path, index=False)


def _game(path: Path, choice: str = 'a') -> None:
    pd.DataFrame([{
        'scenario_id': 's1', 'seat': 0, 'choice_value_tag': choice,
    }]).to_csv(path, index=False)


def test_collect_selects_exact_arms(tmp_path):
    solo = tmp_path / 'solo'
    game = tmp_path / 'game'
    solo.mkdir()
    game.mkdir()
    _solo(solo / 'model_mcq_balanced50.csv')
    _solo(solo / 'model_mcq_balanced50_steer-HHH.csv', 'B')
    _solo(solo / 'model_interactive_balanced50.csv')
    _solo(solo / 'model_interactive_balanced50_stakes-alignment_user-x.csv', 'B')
    _solo(solo / 'model_interactive_balanced50_steer-HHH_user-x.csv', 'B')
    _game(game / 'model_mcq_balanced50.csv')
    _game(game / 'model_mcq_balanced50_states.csv', 'b')
    _game(game / 'model_open_balanced50_user-x_judge-y.csv')
    _game(game / 'model_open_balanced50_counterpart_user-x.csv', 'b')
    _game(game / 'model_open_balanced50_counterpart_steer-HHH_user-x.csv', 'b')
    _game(game / 'model_open_balanced50_steer-HHH_user-x.csv', 'b')
    _game(game / 'model_open_balanced50_steer-HHH_repeated_rounds5_mem1_user-x.csv')

    found = figure.collect(
        {'game_slug': 'model', 'solo_slug': 'model'}, str(solo), str(game),
        {('s1', 0): ('a', 'b')},
    )

    assert found['game_open'][('a', 'b')] == (1, 1)
    assert found['game_open_steer'][('a', 'b')] == (0, 1)
    assert found['solo_int'][('a', 'b')] == (1, 1)


def test_collect_rejects_duplicate_exact_arm(tmp_path):
    _game(tmp_path / 'model_open_balanced50_user-x.csv')
    _game(tmp_path / 'model_open_balanced50_fallback-x.csv')
    with pytest.raises(ValueError, match='Multiple result CSVs'):
        figure.collect(
            {'game_slug': 'model', 'solo_slug': 'missing'}, str(tmp_path), str(tmp_path),
            {('s1', 0): ('a', 'b')},
        )


def test_collect_omits_missing_steer(tmp_path):
    _game(tmp_path / 'model_open_balanced50_user-x.csv')
    found = figure.collect(
        {'game_slug': 'model', 'solo_slug': 'missing'}, str(tmp_path), str(tmp_path),
        {('s1', 0): ('a', 'b')},
    )
    assert 'game_open' in found
    assert 'game_open_steer' not in found


def test_output_suffix_is_applied_to_every_output(tmp_path, monkeypatch):
    def fake_collect(*_args, **_kwargs):
        return {'solo_mcq': {('a', 'b'): (1, 1)}}

    def fake_plot(_models, _labels, path, **_kwargs):
        Path(path).touch()
        return path

    def fake_plot_rows(_models, _labels, path, **_kwargs):
        Path(path).touch()
        return path, []

    scenarios = tmp_path / 'scenarios.csv'
    pd.DataFrame([{'scenario_id': 's1', 'value1': 'a', 'value2': 'b'}]).to_csv(
        scenarios, index=False,
    )
    monkeypatch.setattr(figure, 'collect', fake_collect)
    monkeypatch.setattr(figure, 'plot', fake_plot)
    monkeypatch.setattr(figure, 'plot_bt', fake_plot_rows)
    monkeypatch.setattr(figure, 'plot_steering_effect', fake_plot_rows)

    figure.main([
        '--solo-results-dir', str(tmp_path), '--game-results-dir', str(tmp_path),
        '--game-scenarios-csv', str(scenarios), '--output-dir', str(tmp_path),
        '--game-slugs', 'model', '--solo-slugs', 'model', '--model-labels', 'Model',
        '--output-suffix', '_v2',
    ])

    expected = {
        'value_preferences_across_conditions_v2.csv',
        'value_preferences_across_conditions_v2.png',
        'bradley_terry_by_condition_vs_solo_v2.csv',
        'bradley_terry_by_condition_vs_solo_v2.png',
        'bradley_terry_by_condition_vs_solo_model_v2.png',
        'steering_effect_by_condition_v2.csv',
        'steering_effect_by_condition_v2.png',
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    for suffixed in expected:
        assert not (tmp_path / suffixed.replace('_v2.', '.')).exists()
