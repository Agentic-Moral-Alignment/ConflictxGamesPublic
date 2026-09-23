"""Do games change value preferences relative to solo evaluation?

This is the study's central question, and answering it needs one control:
elicitation format moves preferences more than anything else measured here, and
the solo and game domains are not run in the same formats by default. Comparing
"solo" against "game" without matching format therefore confounds the two.

So the comparison is made format-matched, twice:

* forced choice  -- solo MCQ  against game MCQ
* free text      -- solo chat against game open-ended

Each contrast is game minus solo on the same value pair, with a 95% Wald
interval on the difference of proportions. Both baseline and HHH-steered runs
are shown, because a domain effect that survives steering is a stronger claim
than one that does not.

Start: bash scripts/HHH_new_pipeline.sh domain-effect-figure
"""
from __future__ import annotations

import argparse
import os
from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from conflictxgame.common.config import add_config_argument, parse_with_config
from conflictxgame.common.stats import UNPAIRED_WALD_CSV_NOTE, wald_unpaired
from conflictxgame.figures.value_conditions_figure import load_pairs
from conflictxgame.figures.stake_domain_figure import solo_tally, game_tally

INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8983'
GRID, SURFACE, BAND = '#e6e5e1', '#fcfcfb', '#f2f1ed'

PAIRS = [('harmlessness', 'helpfulness'),
         ('harmlessness', 'honesty'),
         ('helpfulness', 'honesty')]

#: (contrast label, colour) -- one bar per format-matched pairing.
CONTRASTS = [
    ('forced choice', '#2a78d6'),
    ('free text', '#eb6834'),
]
ARMS = [('', 'baseline'), ('_steer-HHH', 'HHH-steered')]


def collect(slug: str, label: str, solo_dir: str, game_dir: str,
            judge_tail: str, pairs: dict) -> list:
    rows = []
    for tag, arm_label in ARMS:
        sources = {
            'forced choice': (
                os.path.join(solo_dir, f'{slug}_mcq_balanced50{tag}.csv'),
                os.path.join(game_dir, f'{slug}_mcq_balanced50{tag}.csv')),
            'free text': (
                os.path.join(solo_dir, f'{slug}_interactive_balanced50{tag}{judge_tail}'),
                os.path.join(game_dir, f'{slug}_open_balanced50{tag}{judge_tail}')),
        }
        for contrast, _colour in CONTRASTS:
            solo_path, game_path = sources[contrast]
            st = solo_tally(solo_path)
            gt = game_tally(game_path, pairs)
            if not st or not gt:
                continue
            for pair in PAIRS:
                key = tuple(sorted(pair))
                if key not in st or key not in gt:
                    continue
                sh, sn = st[key]
                gh, gn = gt[key]
                if not sn or not gn:
                    continue
                d, lo, hi = wald_unpaired(sh / sn, sn, gh / gn, gn)
                rows.append({
                    'model': label, 'arm': arm_label, 'contrast': contrast,
                    'value_first': key[0], 'value_second': key[1],
                    'solo': sh / sn, 'game': gh / gn, 'game_minus_solo': d,
                    'ci_lower': lo, 'ci_upper': hi, 'n_solo': sn, 'n_game': gn,
                })
    return rows


def plot(rows: list, output_path: str) -> str:
    if not rows:
        raise ValueError('No matched solo/game pairs found yet.')
    df = pd.DataFrame(rows)
    models = sorted(set(df['model']))
    pairs = [p for p in PAIRS
             if tuple(sorted(p)) in set(zip(df['value_first'], df['value_second']))]

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK_2, 'text.color': INK})
    fig, axes = plt.subplots(len(models), len(pairs),
                             figsize=(4.9 * len(pairs), 4.2 * len(models)),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    width = 0.34
    arm_labels = [a for _, a in ARMS]
    for mi, model in enumerate(models):
        for pi, pair in enumerate(pairs):
            ax = axes[mi][pi]
            ax.set_facecolor(SURFACE)
            key = tuple(sorted(pair))
            sub = df[(df['model'] == model) & (df['value_first'] == key[0])
                     & (df['value_second'] == key[1])]
            ax.axhline(0, color=INK, lw=1.2, zorder=3)
            for ci, (contrast, colour) in enumerate(CONTRASTS):
                xs, ys, los, his = [], [], [], []
                for xi, arm in enumerate(arm_labels):
                    cell = sub[(sub['contrast'] == contrast) & (sub['arm'] == arm)]
                    if cell.empty:
                        continue
                    r = cell.iloc[0]
                    xs.append(xi + (ci - 0.5) * width)
                    ys.append(r['game_minus_solo'])
                    los.append(r['game_minus_solo'] - r['ci_lower'])
                    his.append(r['ci_upper'] - r['game_minus_solo'])
                if not xs:
                    continue
                ax.bar(xs, ys, width=width * 0.9, color=colour, zorder=4,
                       label=contrast if (mi == 0 and pi == 0) else None)
                ax.errorbar(xs, ys, yerr=[los, his], fmt='none', ecolor=INK_2,
                            elinewidth=1.0, capsize=3, zorder=5)
            ax.set_xticks(range(len(arm_labels)))
            ax.set_xticklabels(arm_labels, color=INK_2)
            ax.set_xlim(-0.6, len(arm_labels) - 0.4)
            ax.grid(axis='y', color=GRID, lw=0.7, zorder=1)
            ax.set_axisbelow(True)
            if mi == 0:
                ax.set_title(f'{key[0]} over {key[1]}', color=INK, fontsize=10, pad=8)
            if pi == 0:
                ax.set_ylabel(f'{model}\ngame − solo', color=INK_2)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Game minus solo, with elicitation format held fixed',
                 color=INK, fontsize=12, y=0.995)
    fig.text(0.5, 0.955, 'above zero = the game favours the first-named value more than solo does',
             ha='center', color=INK_3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output_path, dpi=170, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_config_argument(parser)
    parser.add_argument('--game-results-dir', required=True)
    parser.add_argument('--solo-results-dir', required=True)
    parser.add_argument('--game-scenarios-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-slugs', required=True,
                        help='Comma-separated file-name slugs')
    parser.add_argument('--model-labels', required=True,
                        help='Comma-separated display names, same order')
    parser.add_argument('--judge-tail',
                        default='_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv')
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args, _defaults, _explicit = parse_with_config(build_parser(), argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    slugs = [s.strip() for s in args.model_slugs.split(',') if s.strip()]
    labels = [s.strip() for s in args.model_labels.split(',') if s.strip()]
    if len(slugs) != len(labels):
        raise ValueError('model-slugs and model-labels must be the same length.')
    pairs = load_pairs(args.game_scenarios_csv)
    rows = []
    for slug, label in zip(slugs, labels):
        got = collect(slug, label, args.solo_results_dir, args.game_results_dir,
                      args.judge_tail, pairs)
        if got:
            rows.extend(got)
        else:
            print(f'skipping {label}: no matched solo/game pairs yet')
    if not rows:
        raise SystemExit('No matched solo/game pairs found; nothing to plot.')
    os.makedirs(args.output_dir, exist_ok=True)
    png = plot(rows, os.path.join(args.output_dir, 'game_minus_solo_by_format.png'))
    csv_path = os.path.join(args.output_dir, 'game_minus_solo_by_format.csv')
    df = pd.DataFrame(rows)
    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(UNPAIRED_WALD_CSV_NOTE + '\n')
        df.to_csv(handle, index=False)
    print(f'wrote {png}')
    print(f'wrote {csv_path} ({len(df)} rows)')

    sig = df[(df['ci_lower'] > 0) | (df['ci_upper'] < 0)]
    print(f'\n{len(sig)}/{len(df)} contrasts have a 95% interval excluding zero')
    print('\nmean game-minus-solo by value pair (sign is the direction the game moves):')
    for (a, b), v in df.groupby(['value_first', 'value_second'])['game_minus_solo'].mean().items():
        print(f'  {a} over {b}: {v:+.3f}')
    print('\nby format, mean |game - solo|:')
    for k, v in df.assign(a=df['game_minus_solo'].abs()).groupby('contrast')['a'].mean().items():
        print(f'  {k:14s} {v:.3f}')


if __name__ == '__main__':
    main()
