"""Raw value-preference rates, per model, domain and elicitation format.

The format sections previously reported differences, which answer "did it move"
but hide where each model actually sits. A reader comparing models wants the
levels: a shift of +0.36 from 0.38 and one of +0.69 from 0.13 are the same
arrow and very different behaviour.

Emits one row per (model, domain, format, value) with the share of contests
that value won, so the report can lay them out as four tables -- solo and
multi-agent, each split by forced choice and free text -- with models down the
left and values across.

Rows the model did not answer are excluded from the rates and counted
separately, because a model that declines the hardest scenarios would otherwise
look decisive on an easier subset than its peers are judged on.

Start: bash scripts/HHH_new_pipeline.sh raw-rates
"""
from __future__ import annotations

import argparse
import os
from typing import Sequence

import pandas as pd

from conflictxgame.common.config import add_config_argument, parse_with_config
from conflictxgame.games.analyze_games import load_specs
from conflictxgame.games.value_figures import decision_events, comparisons_from_events
from conflictxgame.figures.value_conditions_figure import load_pairs

VALUES = ('harmlessness', 'helpfulness', 'honesty')

#: (domain, format key, label) -> how to find the file for a model slug.
#: Solo and game name their two formats differently; the pairing here is what
#: makes them comparable -- forced choice against forced choice, free text
#: against free text.
LAYOUT = [
    ('solo', 'forced', 'forced choice'),
    ('solo', 'free', 'chat'),
    ('game', 'forced', 'forced choice'),
    ('game', 'free', 'open-ended'),
]


def solo_rates(path: str) -> tuple | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'choice' not in df.columns:
        return None
    counts = df['choice'].value_counts()
    skipped = int(counts.get('REFUSAL', 0)) + int(counts.get('INVALID', 0))
    kept = df[df['choice'].isin(['A', 'B'])]
    if kept.empty:
        return None
    tot = {v: [0, 0] for v in VALUES}
    for _, r in kept.iterrows():
        first, second = str(r['value1']), str(r['value2'])
        chosen = first if r['choice'] == 'A' else second
        for v in (first, second):
            if v in tot:
                tot[v][1] += 1
        if chosen in tot:
            tot[chosen][0] += 1
    return tot, skipped


def game_rates(path: str, specs: dict) -> tuple | None:
    """Single-state game rows only, so the level is not mixed with memory."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['probe'] = df['probe'].fillna('') if 'probe' in df.columns else ''
    sub = df[df['probe'] == '']
    if sub.empty:
        return None
    comps = comparisons_from_events(decision_events(sub, specs))
    if not comps:
        return None
    tot = {v: [0, 0] for v in VALUES}
    for winner, loser in comps:
        for v in (winner, loser):
            if v in tot:
                tot[v][1] += 1
        if winner in tot:
            tot[winner][0] += 1
    skipped = int((~sub.get('parsed_ok', pd.Series(True, index=sub.index)).fillna(True)).sum())
    return tot, skipped


def paths_for(slug: str, domain: str, fmt: str, arm: str,
              solo_dir: str, game_dir: str, judge_tail: str) -> str:
    if domain == 'solo':
        mode = 'mcq' if fmt == 'forced' else 'interactive'
        tail = '.csv' if mode == 'mcq' else judge_tail
        return os.path.join(solo_dir, f'{slug}_{mode}_balanced50{arm}{tail}')
    mode = 'mcq' if fmt == 'forced' else 'open'
    tail = '.csv' if mode == 'mcq' else judge_tail
    return os.path.join(game_dir, f'{slug}_{mode}_balanced50{arm}{tail}')


def collect(slug: str, label: str, solo_dir: str, game_dir: str,
            judge_tail: str, specs: dict) -> list:
    rows = []
    for arm, arm_label in (('', 'baseline'), ('_steer-HHH', 'HHH-steered')):
        for domain, fmt, fmt_label in LAYOUT:
            path = paths_for(slug, domain, fmt, arm, solo_dir, game_dir, judge_tail)
            got = solo_rates(path) if domain == 'solo' else game_rates(path, specs)
            if not got:
                continue
            tot, skipped = got
            for v in VALUES:
                won, n = tot[v]
                if n:
                    rows.append({'model': label, 'domain': domain, 'format': fmt,
                                 'format_label': fmt_label, 'arm': arm_label,
                                 'value': v, 'rate': won / n, 'n': n,
                                 'skipped': skipped})
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_config_argument(parser)
    parser.add_argument('--solo-results-dir', required=True)
    parser.add_argument('--game-results-dir', required=True)
    parser.add_argument('--scenarios-dir', required=True)
    parser.add_argument('--game-scenarios-csv', required=True)
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
    load_pairs(args.game_scenarios_csv)
    rows = []
    for slug, label in zip(slugs, labels):
        got = collect(slug, label, args.solo_results_dir, args.game_results_dir,
                      args.judge_tail, specs)
        rows.extend(got)
        cells = {(r['domain'], r['format'], r['arm']) for r in got}
        print(f'  {label:22s} {len(cells)} of 8 cells present')
    if not rows:
        raise SystemExit('No results found; nothing to write.')
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, 'raw_rates_by_model.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'\nwrote {path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
