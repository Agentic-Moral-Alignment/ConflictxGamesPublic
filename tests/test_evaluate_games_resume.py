import sys

import pandas as pd

from conftest import make_spec
from fakes import DummyEvalClient
from conflictxgame.games import evaluate_games


PAYOFFS = {'0,0': [3, 3], '0,1': [0, 5], '1,0': [5, 0], '1,1': [1, 1]}


def _write_scenarios(path):
    rows = []
    for index in range(3):
        spec = make_spec('simultaneous', [2, 2], PAYOFFS, game_id=f's{index}')
        rows.append({'scenario_id': f's{index}', 'game_spec': spec.to_json()})
    pd.DataFrame(rows).to_csv(path / 'gen_games.csv', index=False)


def _run(monkeypatch, root, client, pair_limit=None):
    args = [
        'evaluate_games', '--model', 'dummy', '--mode', 'mcq',
        '--scenarios-dir', str(root), '--output-dir', str(root),
        '--output-name', 'eval.csv', '--state-probes', '--seed', '9',
    ]
    if pair_limit is not None:
        args.extend(['--pair-limit', str(pair_limit)])
    monkeypatch.setattr(sys, 'argv', args)
    monkeypatch.setattr(
        evaluate_games.ModelClientManager, 'get_client',
        lambda self, *a, **k: client,
    )
    evaluate_games.main()
    return pd.read_csv(root / 'eval.csv')


def _key_counts(frame):
    return frame.groupby(['scenario_id', 'seat', 'probe']).size()


def test_completed_seats_require_the_full_probe_set():
    frame = pd.DataFrame([
        {'scenario_id': 'a', 'seat': 0, 'mode': 'mcq', 'probe': probe}
        for probe in ['', 'coop_history', 'defect_history']
    ] + [
        {'scenario_id': 'a', 'seat': 1, 'mode': 'mcq', 'probe': probe}
        for probe in ['', 'coop_history']
    ])
    expected = {'', 'coop_history', 'defect_history'}
    assert evaluate_games._completed_seats(frame, 'mcq', expected) == {('a', 0)}
    assert evaluate_games._completed_seats(frame.drop(columns='probe'), 'mcq', {''}) == {
        ('a', 0), ('a', 1),
    }


def test_pair_limited_partial_seat_is_rerun_in_full(monkeypatch, tmp_path):
    _write_scenarios(tmp_path)
    initial = _run(monkeypatch, tmp_path, DummyEvalClient(), pair_limit=2)
    first_key = initial[['scenario_id', 'seat']].drop_duplicates().iloc[0]
    is_first = initial['scenario_id'].eq(first_key.scenario_id) & initial['seat'].eq(first_key.seat)
    truncated = pd.concat([initial.loc[~is_first], initial.loc[is_first].iloc[:1]])
    truncated.to_csv(tmp_path / 'eval.csv', index=False)

    resumed = _run(monkeypatch, tmp_path, DummyEvalClient(), pair_limit=2)
    assert len(resumed) == 10
    assert _key_counts(resumed).eq(1).all()


def test_unlimited_resume_preserves_complete_seats(monkeypatch, tmp_path):
    _write_scenarios(tmp_path)
    initial = _run(monkeypatch, tmp_path, DummyEvalClient())
    first_key = initial[['scenario_id', 'seat']].drop_duplicates().iloc[0]
    is_first = initial['scenario_id'].eq(first_key.scenario_id) & initial['seat'].eq(first_key.seat)
    complete_before = initial.loc[~is_first].copy().reset_index(drop=True)
    pd.concat([complete_before, initial.loc[is_first].iloc[:2]]).to_csv(
        tmp_path / 'eval.csv', index=False,
    )

    resumed = _run(monkeypatch, tmp_path, DummyEvalClient())
    assert len(resumed) == 30
    assert _key_counts(resumed).eq(1).all()
    complete_after = resumed.merge(
        complete_before[['scenario_id', 'seat']].drop_duplicates(),
        on=['scenario_id', 'seat'], how='inner',
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(complete_after, complete_before, check_dtype=False)


def test_complete_resume_makes_no_model_calls(monkeypatch, tmp_path):
    _write_scenarios(tmp_path)
    _run(monkeypatch, tmp_path, DummyEvalClient())
    unused = DummyEvalClient()
    result = _run(monkeypatch, tmp_path, unused)
    assert len(result) == 30
    assert unused.usage.snapshot()['requests'] == 0
