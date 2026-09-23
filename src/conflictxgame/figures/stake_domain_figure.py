"""Does a stake move values differently in a game than it does solo?

The programme runs the same three stake criteria in four places: solo MCQ, solo
chat, game MCQ and game open-ended. Each arm is compared against *its own*
domain baseline, so the four are read like for like even though their baselines
sit at very different heights.

The question this answers is whether the game setting changes how much a stake
bites. Reading the shifts by domain (solo vs game) and by format (forced choice
vs free text) separates the two explanations, which the raw numbers otherwise
confound: solo and game each appear in both formats.

One panel per value pair; bars are the shift from that domain's baseline with a
95% Wald interval on the difference of proportions.

Start: bash scripts/HHH_new_pipeline.sh stake-domain-figure
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
from conflictxgame.figures.value_conditions_figure import load_pairs, solo_rows

INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8983'
GRID, SURFACE, BAND = '#e6e5e1', '#fcfcfb', '#f2f1ed'

#: Stake arm -> label and colour. Payoff has no solo counterpart in the
#: programme, so it appears only where it was actually run.
ARMS = [
    ('stakes-payoff', 'Payoff', '#2a78d6'),
    ('stakes-alignment', 'Alignment', '#eb6834'),
    ('stakes-usefulness', 'Usefulness', '#1baf7a'),
]

#: (label, domain, format) -- domain and format are tracked separately because
#: the interesting question is which of the two explains the pattern.
DOMAINS = [
    ('solo MCQ', 'solo', 'forced choice'),
    ('solo chat', 'solo', 'free text'),
    ('game MCQ', 'game', 'forced choice'),
    ('game open', 'game', 'free text'),
]

PAIRS = [('harmlessness', 'helpfulness'),
         ('harmlessness', 'honesty'),
         ('helpfulness', 'honesty')]


def tally(rows: Sequence[tuple]) -> dict:
    """(winner, loser) observations -> {pair: (hits for pair[0], total)}."""
    out: dict = {}
    for winner, loser in rows:
        pair = tuple(sorted((winner, loser)))
        hit, tot = out.get(pair, (0, 0))
        out[pair] = (hit + (1 if winner == pair[0] else 0), tot + 1)
    return out


def solo_tally(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    rows = []
    for pair, chosen in solo_rows(path):
        other = [v for v in pair if v != chosen]
        if len(other) == 1:
            rows.append((chosen, other[0]))
    return tally(rows)


def game_tally(path: str, pairs: dict) -> dict:
    """Single-state game rows only: probe must be empty."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if 'probe' in df.columns:
        df = df[df['probe'].fillna('') == '']
    rows = []
    for _, r in df.iterrows():
        try:
            seat = int(r['seat'])
        except (TypeError, ValueError):
            seat = None
        pair = pairs.get((r['scenario_id'], seat)) or pairs.get((r['scenario_id'], None))
        chosen = str(r.get('choice_value_tag', '')).strip()
        if pair is None or chosen not in pair:
            continue
        other = [v for v in pair if v != chosen]
        if len(other) == 1:
            rows.append((chosen, other[0]))
    return tally(rows)


def collect(slug: str, solo_dir: str, game_dir: str, judge_tail: str, pairs: dict) -> list:
    """One row per (domain, arm, pair) with the shift from that domain's baseline."""
    paths = {
        'solo MCQ': (lambda a: os.path.join(
            solo_dir, f'{slug}_mcq_balanced50' + ('' if a is None else f'_{a}') + '.csv'), solo_tally),
        'solo chat': (lambda a: os.path.join(
            solo_dir, f'{slug}_interactive_balanced50' + ('' if a is None else f'_{a}') + judge_tail), solo_tally),
        'game MCQ': (lambda a: os.path.join(
            game_dir, f'{slug}_mcq_balanced50' + ('' if a is None else f'_{a}') + '.csv'),
            lambda p: game_tally(p, pairs)),
        'game open': (lambda a: os.path.join(
            game_dir, f'{slug}_open_balanced50' + ('' if a is None else f'_{a}') + judge_tail),
            lambda p: game_tally(p, pairs)),
    }
    rows = []
    for label, domain, fmt in DOMAINS:
        build, reader = paths[label]
        base = reader(build(None))
        if not base:
            continue
        for arm, arm_label, _colour in ARMS:
            got = reader(build(arm))
            if not got:
                continue
            for pair in PAIRS:
                key = tuple(sorted(pair))
                if key not in base or key not in got:
                    continue
                bh, bn = base[key]
                ah, an = got[key]
                if not bn or not an:
                    continue
                d, lo, hi = wald_unpaired(bh / bn, bn, ah / an, an)
                rows.append({
                    'domain': label, 'domain_kind': domain, 'format': fmt,
                    'arm': arm_label, 'value_first': key[0], 'value_second': key[1],
                    'baseline': bh / bn, 'stake': ah / an, 'shift': d,
                    'ci_lower': lo, 'ci_upper': hi, 'n_baseline': bn, 'n_stake': an,
                })
    return rows


def plot(rows: list, output_path: str, model_label: str = '') -> str:
    if not rows:
        raise ValueError('No stake arms found in either domain yet.')
    df = pd.DataFrame(rows)
    pairs = [p for p in PAIRS if tuple(sorted(p)) in
             {(r['value_first'], r['value_second']) for _, r in df.iterrows()}]
    order = [d for d, _, _ in DOMAINS if d in set(df['domain'])]
    models = sorted(set(df['model']))

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK_2, 'text.color': INK})
    fig, axes = plt.subplots(len(models), len(pairs),
                             figsize=(4.9 * len(pairs), 4.4 * len(models)),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    width = 0.26
    for mi, model in enumerate(models):
      for pi, pair in enumerate(pairs):
        ax = axes[mi][pi]
        ax.set_facecolor(SURFACE)
        key = tuple(sorted(pair))
        sub = df[(df['model'] == model) & (df['value_first'] == key[0])
                 & (df['value_second'] == key[1])]
        # Shade the two free-text domains: the format split is the finding.
        for xi, dom in enumerate(order):
            if 'chat' in dom or 'open' in dom:
                ax.axvspan(xi - 0.5, xi + 0.5, color=BAND, zorder=0, lw=0)
        ax.axhline(0, color=INK_3, lw=1.0, zorder=2)
        for ai, (_arm, arm_label, colour) in enumerate(ARMS):
            xs, ys, los, his = [], [], [], []
            for xi, dom in enumerate(order):
                cell = sub[(sub['domain'] == dom) & (sub['arm'] == arm_label)]
                if cell.empty:
                    continue
                r = cell.iloc[0]
                xs.append(xi + (ai - 1) * width)
                ys.append(r['shift'])
                los.append(r['shift'] - r['ci_lower'])
                his.append(r['ci_upper'] - r['shift'])
            if not xs:
                continue
            ax.bar(xs, ys, width=width * 0.92, color=colour, zorder=3,
                   label=arm_label if (pi == 0 and mi == 0) else None)
            ax.errorbar(xs, ys, yerr=[los, his], fmt='none', ecolor=INK_2,
                        elinewidth=1.0, capsize=2.5, zorder=4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([d.replace(' ', '\n') for d in order], color=INK_2)
        ax.set_xlim(-0.6, len(order) - 0.4)
        if mi == 0:
            ax.set_title(f'{key[0]} over {key[1]}', color=INK, fontsize=10, pad=8)
        if pi == 0:
            ax.set_ylabel(model + '\nshift from own baseline', color=INK_2)
        ax.grid(axis='y', color=GRID, lw=0.7, zorder=1)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(ARMS), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Does a stake bite differently in a game than solo?',
                 color=INK, fontsize=12, y=0.995)
    fig.text(0.5, 0.90, 'shaded columns are the free-text formats',
             ha='center', color=INK_3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
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
    pairs = load_pairs(args.game_scenarios_csv)
    slugs = [x.strip() for x in args.model_slugs.split(',') if x.strip()]
    labels = [x.strip() for x in args.model_labels.split(',') if x.strip()]
    if len(slugs) != len(labels):
        raise ValueError('model-slugs and model-labels must be the same length.')
    rows = []
    for slug, label in zip(slugs, labels):
        got = collect(slug, args.solo_results_dir, args.game_results_dir,
                      args.judge_tail, pairs)
        for r in got:
            r['model'] = label
        rows.extend(got)
        if not got:
            print(f'skipping {label}: no stake arms in either domain yet')
    if not rows:
        raise SystemExit('No stake arms found yet in either domain; nothing to plot.')
    os.makedirs(args.output_dir, exist_ok=True)
    png = plot(rows, os.path.join(args.output_dir, 'stake_effect_solo_vs_game.png'))
    csv_path = os.path.join(args.output_dir, 'stake_effect_solo_vs_game.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(UNPAIRED_WALD_CSV_NOTE + '\n')
        pd.DataFrame(rows).to_csv(handle, index=False)
    print(f'wrote {png}')
    print(f'wrote {csv_path} ({len(rows)} rows)')

    # The comparison the figure exists to make, stated in numbers too.
    df = pd.DataFrame(rows)
    df['abs'] = df['shift'].abs()
    print('\nmean |shift| by domain:')
    for k, v in df.groupby('domain')['abs'].mean().sort_values(ascending=False).items():
        print(f'  {k:12s} {v:.3f}')
    print('\nmean |shift| by elicitation format:')
    for k, v in df.groupby('format')['abs'].mean().sort_values(ascending=False).items():
        print(f'  {k:14s} {v:.3f}')
    print('\nmean |shift| by domain kind (solo vs game):')
    for k, v in df.groupby('domain_kind')['abs'].mean().sort_values(ascending=False).items():
        print(f'  {k:6s} {v:.3f}')


if __name__ == '__main__':
    main()
