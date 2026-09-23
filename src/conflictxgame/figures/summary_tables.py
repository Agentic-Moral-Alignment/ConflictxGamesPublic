"""Cross-model summary tables for the headline cards.

The two tables the watch page leads with -- the format effect and the stake
matrix -- were hand-written from one model's numbers. That made them wrong in
two ways: they named a single model above the tabs, where the reader should meet
the question before the model, and being hardcoded they did not move when new
results landed.

This module recomputes both from the result CSVs, one column per model, so the
page states them across models and stays current.

Start: bash scripts/HHH_new_pipeline.sh summary-tables
"""
from __future__ import annotations

import argparse
import os
from typing import Sequence

import pandas as pd

from conflictxgame.common.config import add_config_argument, parse_with_config
from conflictxgame.common.stats import UNPAIRED_WALD_CSV_NOTE, wald_unpaired
from conflictxgame.games.analyze_games import load_specs
from conflictxgame.games.value_figures import (
    decision_events, comparisons_from_events, pair_rate)

PAIRS = [('harmlessness', 'helpfulness'),
         ('harmlessness', 'honesty'),
         ('helpfulness', 'honesty')]
STATES = ['fresh', 'prev_0-0', 'prev_0-1', 'prev_1-0', 'prev_1-1']
VALUES = ('harmlessness', 'helpfulness', 'honesty')
ARMS = [('payoff', 'Payoff'), ('alignment', 'Alignment'), ('usefulness', 'Usefulness')]


def rates(path: str, specs: dict, states: bool) -> dict | None:
    """Pairwise preference rates from one game CSV."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['probe'] = df['probe'].fillna('') if 'probe' in df.columns else ''
    sub = df[df['probe'].isin(STATES)] if states else df[df['probe'] == '']
    if sub.empty:
        return None
    comps = comparisons_from_events(decision_events(sub, specs))
    return {p: pair_rate(comps, *p) for p in PAIRS}


def marginals(src: dict) -> dict:
    """Per-value win rate, pooled over the two pairs each value appears in."""
    tot = {v: [0, 0] for v in VALUES}
    for p in PAIRS:
        n = src[p]['n']
        if n == 0:
            continue
        hits = round(src[p]['p'] * n)
        tot[p[0]][0] += hits
        tot[p[0]][1] += n
        tot[p[1]][0] += n - hits
        tot[p[1]][1] += n
    return tot


def format_rows(game_dir: str, slug: str, label: str, judge_tail: str, specs: dict) -> list:
    """Open minus mcq on the same states, per value pair."""
    mcq = rates(os.path.join(game_dir, f'{slug}_mcq_balanced50_states.csv'), specs, True)
    opn = rates(os.path.join(game_dir, f'{slug}_open_balanced50_states{judge_tail}'), specs, True)
    if not mcq or not opn:
        return []
    out = []
    for p in PAIRS:
        d, lo, hi = wald_unpaired(mcq[p]['p'], mcq[p]['n'], opn[p]['p'], opn[p]['n'])
        out.append({'model': label, 'value_first': p[0], 'value_second': p[1],
                    'mcq': mcq[p]['p'], 'open': opn[p]['p'], 'shift': d,
                    'ci_lower': lo, 'ci_upper': hi,
                    'significant': not (lo <= 0 <= hi)})
    return out


def stake_rows(game_dir: str, slug: str, label: str, judge_tail: str, specs: dict) -> list:
    """Marginal shift per value under each stake arm, in both formats."""
    out = []
    for mode, tail, sep in (('mcq', '.csv', '_states_'), ('open', judge_tail, '_states-')):
        base = rates(os.path.join(game_dir, f'{slug}_{mode}_balanced50_states{tail}'), specs, True)
        if not base:
            continue
        mb = marginals(base)
        for arm, arm_label in ARMS:
            path = os.path.join(game_dir, f'{slug}_{mode}_balanced50{sep}stakes-{arm}{tail}')
            got = rates(path, specs, True)
            if not got:
                continue
            ma = marginals(got)
            for v in VALUES:
                bh, bn = mb[v]
                ah, an = ma[v]
                d, lo, hi = wald_unpaired(bh / bn, bn, ah / an, an)
                out.append({'model': label, 'format': mode, 'arm': arm_label, 'value': v,
                            'baseline': bh / bn, 'stake': ah / an, 'shift': d,
                            'ci_lower': lo, 'ci_upper': hi,
                            'significant': not (lo <= 0 <= hi)})
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_config_argument(parser)
    parser.add_argument('--game-results-dir', required=True)
    parser.add_argument('--scenarios-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-slugs', required=True)
    parser.add_argument('--model-labels', required=True)
    parser.add_argument('--judge-tail',
                        default='_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv')
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args, _d, _e = parse_with_config(build_parser(), argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    slugs = [x.strip() for x in args.model_slugs.split(',') if x.strip()]
    labels = [x.strip() for x in args.model_labels.split(',') if x.strip()]
    if len(slugs) != len(labels):
        raise ValueError('model-slugs and model-labels must be the same length.')
    specs = load_specs(args.scenarios_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    fmt, stake = [], []
    for slug, label in zip(slugs, labels):
        fmt.extend(format_rows(args.game_results_dir, slug, label, args.judge_tail, specs))
        stake.extend(stake_rows(args.game_results_dir, slug, label, args.judge_tail, specs))
    for name, rows in (('format_effect_by_model', fmt), ('stake_matrix_by_model', stake)):
        path = os.path.join(args.output_dir, name + '.csv')
        with open(path, 'w', encoding='utf-8', newline='') as handle:
            handle.write(UNPAIRED_WALD_CSV_NOTE + '\n')
            pd.DataFrame(rows).to_csv(handle, index=False)
        print(f'wrote {path} ({len(rows)} rows)')
        if rows:
            got = sorted({r['model'] for r in rows})
            print(f'  models covered: {got}')


if __name__ == '__main__':
    main()
