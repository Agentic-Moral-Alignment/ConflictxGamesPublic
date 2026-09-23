"""Shared statistical helpers with explicit estimator assumptions."""
from __future__ import annotations

import math


UNPAIRED_WALD_CSV_NOTE = '# NOTE: unpaired Wald CIs — descriptive only'


def wald_unpaired(p1: float, n1: int, p2: float, n2: int) -> tuple:
    """Difference ``p2 - p1`` with an unpaired two-sample Wald interval.

    This descriptive interval treats the two samples as independent. For the
    experiment's paired observations, it is superseded by the McNemar and
    scenario-clustered analyses used for the reported results.
    """
    difference = p2 - p1
    standard_error = (
        math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        if n1 and n2 else float('nan')
    )
    return (
        difference,
        difference - 1.96 * standard_error,
        difference + 1.96 * standard_error,
    )
