"""Regression tests for :mod:`conflictxgame.solo.analyze_experiment`.

Covers the grouping-label rule (each result file is its own group, even
when two files carry the same raw model id — a baseline and a steered run
must never pool) and the handling of non-A/B choices, which are excluded
from preference rates and Bradley-Terry fits instead of being scored as
wins for ``value2``.
"""

import numpy as np
import pandas as pd

from conflictxgame.solo.analyze_experiment import (
    combine_data,
    compute_preference_rates,
    fit_bradley_terry_model,
    load_data_files,
)


def make_rows(choices, value1='harmlessness', value2='helpfulness'):
    """Build evaluation rows with one entry per choice."""
    return pd.DataFrame({
        'scenario_id': range(len(choices)),
        'value1': value1,
        'value2': value2,
        'choice': choices,
    })


def test_same_model_id_in_two_files_stays_two_groups(tmp_path):
    baseline = make_rows(['A', 'A'])
    steered = make_rows(['B', 'B'])
    baseline['model'] = 'openai:google.gemma-4-31B-it'
    steered['model'] = 'openai:google.gemma-4-31B-it'
    baseline.to_csv(tmp_path / 'gemma_mcq_balanced50.csv', index=False)
    steered.to_csv(tmp_path / 'gemma_mcq_balanced50_steer-HHH.csv', index=False)

    combined = combine_data(load_data_files(str(tmp_path)))

    assert set(combined['model']) == {
        'gemma_mcq_balanced50', 'gemma_mcq_balanced50_steer-HHH',
    }
    rates = compute_preference_rates(combined)
    assert len(rates) == 2
    assert rates['total'].tolist() == [2, 2]


def test_file_without_model_column_gets_stem_label(tmp_path):
    make_rows(['A']).to_csv(tmp_path / 'gemma_interactive.csv', index=False)

    combined = combine_data(load_data_files(str(tmp_path)))

    assert combined['model'].tolist() == ['gemma_interactive']


def test_preference_rates_exclude_invalid_choices():
    data = make_rows(['A', 'A', 'A', 'B', 'INVALID', 'INVALID'])
    data['model'] = 'm'

    rates = compute_preference_rates(data)

    row = rates.iloc[0]
    assert row['total'] == 4
    assert row['count_value1'] == 3
    assert row['count_value2'] == 1
    assert row['n_invalid'] == 2
    assert row['preference_rate'] == 0.75


def test_bradley_terry_drops_invalid_choices():
    valid = make_rows(['A', 'A', 'B'])
    with_invalid = pd.concat(
        [valid, make_rows(['INVALID', 'INVALID', 'INVALID'])],
        ignore_index=True,
    )

    params_valid, _ = fit_bradley_terry_model(valid)
    params_with_invalid, _ = fit_bradley_terry_model(with_invalid)

    np.testing.assert_allclose(params_with_invalid, params_valid)
