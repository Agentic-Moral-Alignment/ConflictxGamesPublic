import json
import random

from conftest import make_spec
from fakes import DummyEvalClient
from conflictxgame.games.evaluate_games import create_game_mcq_prompt, evaluate_mcq


PAYOFFS = {'0,0': [3, 3], '0,1': [0, 5], '1,0': [5, 0], '1,1': [1, 1]}


def _spec(game_id):
    return make_spec('simultaneous', [2, 2], PAYOFFS, game_id=game_id)


def _orders(rows):
    return {
        (str(row['scenario_id']), row['seat']): json.loads(row['option_order'])
        for row in rows if row['probe'] in ('', 'fresh')
    }


def test_all_state_probes_of_a_seat_share_one_option_order():
    rows = evaluate_mcq(
        DummyEvalClient(), [('s', _spec('s'))], random.Random(999),
        state_probes=True, seed=17,
    )
    for seat in range(2):
        assert len({row['option_order'] for row in rows if row['seat'] == seat}) == 1


def test_option_order_is_stable_across_probe_setting_and_scenario_subsets():
    specs = [('a', _spec('a')), ('b', _spec('b'))]
    plain = evaluate_mcq(DummyEvalClient(), specs, random.Random(1), seed=23)
    probed = evaluate_mcq(
        DummyEvalClient(), specs, random.Random(2), state_probes=True, seed=23,
    )
    subset = evaluate_mcq(DummyEvalClient(), specs[1:], random.Random(3), seed=23)
    assert _orders(plain) == _orders(probed)
    assert _orders(subset) == {key: value for key, value in _orders(plain).items() if key[0] == 'b'}


def test_option_order_changes_with_seed():
    specs = [(str(i), _spec(str(i))) for i in range(6)]
    first = evaluate_mcq(DummyEvalClient(), specs, random.Random(1), seed=1)
    second = evaluate_mcq(DummyEvalClient(), specs, random.Random(1), seed=2)
    assert _orders(first) != _orders(second)


def test_create_prompt_honours_explicit_order():
    prompt, inverse = create_game_mcq_prompt(
        _spec('s'), 0, random.Random(0), order=[1, 0],
    )
    assert inverse == {'A': 1, 'B': 0}
    assert prompt.index('favor v2') < prompt.index('favor v1')
