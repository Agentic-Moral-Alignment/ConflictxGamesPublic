"""Solo, single-round, first-round: what does each step of the game add?

Three conditions in which the agent has **no history at all**, so they can be
read against each other directly:

* **solo** -- no counterpart in the scenario;
* **single round** -- a counterpart, choosing simultaneously, one shot;
* **first round** -- the same counterpart under the repeated framing, evaluated
  at its fresh-start state, so the interaction is ongoing with an unknown
  horizon but nothing has happened yet.

That makes the game effect decomposable rather than monolithic. Solo to single
round is what adding a counterpart does; single round to first round is what the
repeated framing does on its own, with memory still empty. Anything left over
belongs to having something to remember, which the state-probe figures cover.

Each stake arm is then applied within every condition, so the same decomposition
answers whether a stake bites differently depending on which of the three the
agent is in.

Formats are kept matched throughout -- solo MCQ with game MCQ, solo chat with
game open-ended -- because elicitation format moves preferences more than any
manipulation here and would otherwise dominate the comparison.

Start: bash scripts/HHH_new_pipeline.sh horizon-figure
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
from conflictxgame.games.analyze_games import load_specs
from conflictxgame.games.value_figures import (
    decision_events, comparisons_from_events, pair_rate)
from conflictxgame.figures.value_conditions_figure import load_pairs
from conflictxgame.figures.stake_domain_figure import solo_tally

INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8983'
GRID, SURFACE, BAND = '#e6e5e1', '#fcfcfb', '#f2f1ed'

VALUES = ('harmlessness', 'helpfulness', 'honesty')
PAIRS = [('harmlessness', 'helpfulness'),
         ('harmlessness', 'honesty'),
         ('helpfulness', 'honesty')]

#: (condition, colour) in the order the decomposition reads.
CONDITIONS = [
    ('solo', '#2a78d6'),
    ('single round', '#eb6834'),
    ('first round', '#1baf7a'),
]
#: arm key -> label. '' is the no-stake baseline for that condition.
ARMS = [('', 'no stake'), ('payoff', 'payoff'),
        ('alignment', 'alignment'), ('usefulness', 'usefulness')]


def to_values(pair_rates: dict) -> dict:
    """Pairwise rates -> {value: (chosen, contests)} pooled over both its pairs."""
    tot = {v: [0, 0] for v in VALUES}
    for p in PAIRS:
        if p not in pair_rates:
            continue
        n = pair_rates[p]['n']
        if n == 0:
            continue
        hits = round(pair_rates[p]['p'] * n)
        tot[p[0]][0] += hits
        tot[p[0]][1] += n
        tot[p[1]][0] += n - hits
        tot[p[1]][1] += n
    return tot


def game_values(path: str, specs: dict, probe: str) -> dict | None:
    """Marginal value rates from a game CSV, restricted to one probe state.

    ``probe=''`` selects the single-round rows; ``probe='fresh'`` selects the
    repeated run's no-history state.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['probe'] = df['probe'].fillna('') if 'probe' in df.columns else ''
    sub = df[df['probe'] == probe]
    if sub.empty:
        return None
    comps = comparisons_from_events(decision_events(sub, specs))
    return to_values({p: pair_rate(comps, *p) for p in PAIRS})


def solo_values(path: str) -> dict | None:
    t = solo_tally(path)
    if not t:
        return None
    tot = {v: [0, 0] for v in VALUES}
    for (a, b), (hits, n) in t.items():
        tot[a][0] += hits
        tot[a][1] += n
        tot[b][0] += n - hits
        tot[b][1] += n
    return tot


def paths_for(slug: str, fmt: str, arm: str, solo_dir: str, game_dir: str,
              judge_tail: str) -> dict:
    """One path per condition for a given format and stake arm."""
    tail = '.csv' if fmt == 'mcq' else judge_tail
    solo_mode = 'mcq' if fmt == 'mcq' else 'interactive'
    stake = '' if not arm else f'_stakes-{arm}'
    # The states files spell the arm with a different separator per format.
    states_stake = '' if not arm else (f'_stakes-{arm}' if fmt == 'mcq' else f'-stakes-{arm}')
    return {
        'solo': os.path.join(solo_dir, f'{slug}_{solo_mode}_balanced50{stake}{tail}'),
        'single round': os.path.join(game_dir, f'{slug}_{fmt}_balanced50{stake}{tail}'),
        'first round': os.path.join(game_dir, f'{slug}_{fmt}_balanced50_states{states_stake}{tail}'),
    }


def collect(slug: str, label: str, solo_dir: str, game_dir: str,
            judge_tail: str, specs: dict) -> list:
    rows = []
    for fmt in ('mcq', 'open'):
        for arm, arm_label in ARMS:
            paths = paths_for(slug, fmt, arm, solo_dir, game_dir, judge_tail)
            for cond, _colour in CONDITIONS:
                path = paths[cond]
                if cond == 'solo':
                    got = solo_values(path)
                else:
                    got = game_values(path, specs, '' if cond == 'single round' else 'fresh')
                if not got:
                    continue
                for v in VALUES:
                    hits, n = got[v]
                    if not n:
                        continue
                    rows.append({'model': label, 'format': fmt, 'condition': cond,
                                 'arm': arm_label, 'value': v,
                                 'rate': hits / n, 'n': n})
    # Stake shifts within each condition, against that condition's own baseline.
    base = {(r['model'], r['format'], r['condition'], r['value']): r
            for r in rows if r['arm'] == 'no stake'}
    out = []
    for r in rows:
        key = (r['model'], r['format'], r['condition'], r['value'])
        b = base.get(key)
        rec = dict(r)
        if b and r['arm'] != 'no stake':
            d, lo, hi = wald_unpaired(b['rate'], b['n'], r['rate'], r['n'])
            rec.update({'shift': d, 'ci_lower': lo, 'ci_upper': hi,
                        'significant': not (lo <= 0 <= hi)})
        out.append(rec)
    return out


def plot(rows: list, output_path: str) -> str:
    df = pd.DataFrame(rows)
    base = df[df['arm'] == 'no stake']
    if base.empty:
        raise ValueError('No baseline rows for any condition yet.')
    models = sorted(set(base['model']))
    formats = [f for f in ('mcq', 'open') if f in set(base['format'])]

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK_2, 'text.color': INK})
    fig, axes = plt.subplots(len(models), len(formats),
                             figsize=(5.4 * len(formats), 4.2 * len(models)),
                             squeeze=False, sharey=True, facecolor=SURFACE)
    conds = [c for c, _ in CONDITIONS]
    colours = dict(CONDITIONS)
    width = 0.8 / max(len(conds), 1)
    for mi, model in enumerate(models):
        for fi, fmt in enumerate(formats):
            ax = axes[mi][fi]
            ax.set_facecolor(SURFACE)
            sub = base[(base['model'] == model) & (base['format'] == fmt)]
            for ci, cond in enumerate(conds):
                xs, ys = [], []
                for vi, v in enumerate(VALUES):
                    cell = sub[(sub['condition'] == cond) & (sub['value'] == v)]
                    if cell.empty:
                        continue
                    xs.append(vi + (ci - (len(conds) - 1) / 2) * width)
                    ys.append(cell.iloc[0]['rate'])
                if xs:
                    ax.bar(xs, ys, width=width * 0.9, color=colours[cond], zorder=3,
                           label=cond if (mi == 0 and fi == 0) else None)
            ax.axhline(0.5, color=INK_3, lw=1.0, ls=(0, (4, 3)), zorder=2)
            ax.set_xticks(range(len(VALUES)))
            ax.set_xticklabels(VALUES, color=INK_2)
            ax.set_ylim(0, 1)
            if mi == 0:
                ax.set_title({'mcq': 'forced choice', 'open': 'free text'}[fmt],
                             color=INK, fontsize=10, pad=8)
            if fi == 0:
                ax.set_ylabel(model + '\nshare of contests won', color=INK_2)
            ax.grid(axis='y', color=GRID, lw=0.7, zorder=1)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(conds), frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('History empty in all three: what does each step of the game add?',
                 color=INK, fontsize=12, y=0.995)
    fig.text(0.5, 0.945,
             'solo → single round adds a counterpart; single round → first round adds the '
             'repeated framing',
             ha='center', color=INK_3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output_path, dpi=170, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_config_argument(parser)
    parser.add_argument('--game-results-dir', required=True)
    parser.add_argument('--solo-results-dir', required=True)
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
    load_pairs(args.game_scenarios_csv)   # validates the corpus is readable
    rows = []
    for slug, label in zip(slugs, labels):
        got = collect(slug, label, args.solo_results_dir, args.game_results_dir,
                      args.judge_tail, specs)
        if not got:
            print(f'skipping {label}: no matching conditions yet')
        rows.extend(got)
    if not rows:
        raise SystemExit('No conditions found; nothing to plot.')
    os.makedirs(args.output_dir, exist_ok=True)
    png = plot(rows, os.path.join(args.output_dir, 'horizon_decomposition.png'))
    csv_path = os.path.join(args.output_dir, 'horizon_decomposition.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(UNPAIRED_WALD_CSV_NOTE + '\n')
        pd.DataFrame(rows).to_csv(handle, index=False)
    print(f'wrote {png}')
    print(f'wrote {csv_path} ({len(rows)} rows)')

    df = pd.DataFrame(rows)
    base = df[df['arm'] == 'no stake']
    print('\nbaseline decomposition (share of contests won):')
    for model in sorted(set(base['model'])):
        for fmt in ('mcq', 'open'):
            sub = base[(base['model'] == model) & (base['format'] == fmt)]
            if sub.empty:
                continue
            print(f'  {model} / {fmt}')
            for v in VALUES:
                cells = {c: sub[(sub['condition'] == c) & (sub['value'] == v)]
                         for c, _ in CONDITIONS}
                txt = '  '.join(
                    f'{c} {cells[c].iloc[0]["rate"]:.2f}' if not cells[c].empty else f'{c} --'
                    for c, _ in CONDITIONS)
                print(f'    {v:14s} {txt}')


if __name__ == '__main__':
    main()
