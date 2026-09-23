from pathlib import Path

import pandas as pd

from conflictxgame.solo.filter_scenarios import input_file_stem
from conflictxgame.solo.topup_scenarios import candidate_count, corpus_pair_counts


def test_filter_input_stem_accepts_both_path_separator_styles():
    expected = "claude-sonnet-5_harmlessness_helpfulness"
    assert input_file_stem(f"data/scenarios/{expected}.json") == expected
    assert input_file_stem(f"data\\scenarios\\{expected}.json") == expected


def test_corpus_pair_counts_tracks_kept_and_total_without_dropping_rejections(tmp_path: Path):
    path = tmp_path / "solo.csv"
    pd.DataFrame([
        {"value1": "harmlessness", "value2": "honesty", "keep_scenario": True},
        {"value1": "honesty", "value2": "harmlessness", "keep_scenario": "False"},
        {"value1": "helpfulness", "value2": "honesty", "keep_scenario": "true"},
    ]).to_csv(path, index=False)

    kept, total = corpus_pair_counts(path)
    assert kept[("harmlessness", "honesty")] == 1
    assert total[("harmlessness", "honesty")] == 2
    assert kept[("helpfulness", "honesty")] == 1
    assert total[("helpfulness", "honesty")] == 1


def test_candidate_count_uses_observed_rate_and_bounds():
    assert candidate_count(
        10, 20, 100, minimum=25, maximum=100,
        rate_floor=0.10, oversample_factor=1.10,
    ) == 56
    assert candidate_count(
        100, 1, 100, minimum=25, maximum=100,
        rate_floor=0.10, oversample_factor=1.10,
    ) == 100
    assert candidate_count(
        0, 1, 100, minimum=25, maximum=100,
        rate_floor=0.10, oversample_factor=1.10,
    ) == 0
