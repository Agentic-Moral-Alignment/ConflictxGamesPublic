import math

import pytest

from conflictxgame.common.stats import wald_unpaired
from conflictxgame.figures.horizon_figure import to_values
from conflictxgame.figures.summary_tables import marginals


def test_wald_unpaired_matches_the_previous_arithmetic():
    p1, n1, p2, n2 = 0.35, 80, 0.6, 95
    difference = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    assert wald_unpaired(p1, n1, p2, n2) == pytest.approx(
        (difference, difference - 1.96 * se, difference + 1.96 * se)
    )


def test_zero_count_pairs_do_not_round_nan():
    pairs = {
        ('harmlessness', 'helpfulness'): {'p': float('nan'), 'n': 0},
        ('harmlessness', 'honesty'): {'p': 0.5, 'n': 2},
        ('helpfulness', 'honesty'): {'p': 1.0, 'n': 1},
    }
    assert marginals(pairs)['harmlessness'] == [1, 2]
    assert to_values(pairs)['harmlessness'] == [1, 2]
