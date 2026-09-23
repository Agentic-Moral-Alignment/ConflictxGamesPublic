import random
import sys

import pandas as pd

from fakes import DummyEvalClient
from conflictxgame.solo import evaluate_models, generate_scenarios


def _scenarios(order=('a', 'b', 'c')):
    return pd.DataFrame([
        {
            'scenario_id': scenario_id,
            'description': f'description {scenario_id}',
            'action1': f'first {scenario_id}',
            'action2': f'second {scenario_id}',
        }
        for scenario_id in order
    ])


def _mcq_maps(frame, seed):
    maps = {}
    original = evaluate_models.create_scenario_prompt_mcq

    def capture(row, rng=None):
        prompt, action_map = original(row, rng)
        maps[row['scenario_id']] = action_map
        return prompt, action_map

    evaluate_models.create_scenario_prompt_mcq = capture
    try:
        evaluate_models.evaluate_models_mcq(DummyEvalClient(), frame, seed=seed)
    finally:
        evaluate_models.create_scenario_prompt_mcq = original
    return maps


def test_solo_mcq_option_map_is_seeded_by_scenario_id():
    forward = _mcq_maps(_scenarios(), 41)
    reversed_order = _mcq_maps(_scenarios(('c', 'b', 'a')), 41)
    assert forward == reversed_order
    assert forward == _mcq_maps(_scenarios(), 41)


def test_create_scenario_prompt_default_still_uses_module_random(monkeypatch):
    monkeypatch.setattr(random, 'choice', lambda _items: True)
    _, mapping = evaluate_models.create_scenario_prompt_mcq(_scenarios().iloc[0])
    assert mapping == {'A': 'action2', 'B': 'action1'}


def test_generation_cli_strips_provider_prefix_from_model_stem(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(sys, 'argv', [
        'generate_scenarios', '--model', 'openai:x', '--value-set', 'test',
        '--value1', 'a', '--value2', 'b', '--output-dir', str(tmp_path),
    ])
    monkeypatch.setattr(generate_scenarios.ModelWrapper, 'create', lambda *a, **k: object())
    monkeypatch.setattr(generate_scenarios, 'load_value_dict', lambda _name: {'a': 'A', 'b': 'B'})
    monkeypatch.setattr(
        generate_scenarios, '_generate_one_pair',
        lambda _args, _client, _values, _embedding, stem, *_pair, **_kwargs: captured.append(stem),
    )
    generate_scenarios.main()
    assert captured == ['x']
    assert ':' not in captured[0]
