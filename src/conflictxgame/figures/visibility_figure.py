"""Does being watched change which value an agent picks?

The observability arms tell the agent either that its decision will be revealed
to the other participant, or that it will stay private. Everything else -- the
scenario, the options, the memory state -- is held fixed.

The contrast that matters is **unobserved against observed**, not either against
the plain baseline. Both arms carry a sentence about visibility, so comparing
one of them to the baseline confounds "told you are unseen" with "told anything
at all about who is watching". The baseline is reported alongside so that
confound stays visible rather than being silently folded in.

Two things are measured, because a manipulation can move either:

* the **level** -- which value gets chosen, pooled over memory states;
* the **slope** -- how strongly the agent copies what it remembers the other
  player doing. If conformity is about how a choice looks to a counterpart, it
  should weaken when nobody can see it.

Start: bash scripts/HHH_new_pipeline.sh visibility-figure
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
from conflictxgame.figures.state_effect_figure import load_action_values, read_states

INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8983'
GRID, SURFACE, BAND = '#e6e5e1', '#fcfcfb', '#f2f1ed'

VALUES = ('harmlessness', 'helpfulness', 'honesty')
STATE_KEYS = ['none', 'one', 'both']

#: arm suffix -> (label, colour). Baseline carries no visibility sentence at all.
ARMS = [
    ('', 'No mention', INK),
    ('_unobserved', 'Told unobserved', '#7a5bd0'),
    ('_observed', 'Told observed', '#c2456f'),
]


def value_totals(tally: dict) -> dict:
    """{value: (chosen, contests)} pooled over states and pairs."""
    out = {v: [0, 0] for v in VALUES}
    for (_state, pair), (hits, total) in tally.items():
        out[pair[0]][0] += hits
        out[pair[0]][1] += total
        out[pair[1]][0] += total - hits
        out[pair[1]][1] += total
    return out


def slopes(tally: dict) -> dict:
    """{pair: (slope, se)} for the none -> both conformity gradient."""
    out = {}
    for pair in sorted({p for (_s, p) in tally}):
        w = [tally.get((s, pair), (0, 0))[0] for s in STATE_KEYS]
        n = [tally.get((s, pair), (0, 0))[1] for s in STATE_KEYS]
        if not all(n):
            continue
        r = [wi / ni for wi, ni in zip(w, n)]
        slope, lo, _hi = wald_unpaired(r[0], n[0], r[-1], n[-1])
        out[pair] = (slope, (slope - lo) / 1.96)
    return out


def collect(game_dir: str, slug: str, label: str, pairs: dict, action_values: dict) -> tuple:
    """(level rows, slope rows) for one model across the visibility arms."""
    tallies = {}
    for suffix, arm_label, _colour in ARMS:
        path = os.path.join(game_dir, f'{slug}_mcq_balanced50_states{suffix}.csv')
        if os.path.exists(path):
            got = read_states(path, pairs, action_values)
            if got:
                tallies[arm_label] = got
    if 'No mention' not in tallies:
        return [], []

    base = value_totals(tallies['No mention'])
    levels, grads = [], []
    for arm_label, tally in tallies.items():
        tot = value_totals(tally)
        for v in VALUES:
            bh, bn = base[v]
            ah, an = tot[v]
            if not bn or not an:
                continue
            d, lo, hi = wald_unpaired(bh / bn, bn, ah / an, an)
            levels.append({'model': label, 'arm': arm_label, 'value': v,
                           'rate': ah / an, 'baseline': bh / bn,
                           'shift_vs_baseline': d, 'ci_lower': lo, 'ci_upper': hi,
                           'significant': not (lo <= 0 <= hi), 'n': an})
        for pair, (slope, se) in slopes(tally).items():
            grads.append({'model': label, 'arm': arm_label,
                          'value_first': pair[0], 'value_second': pair[1],
                          'slope': slope, 'se': se,
                          'significant': abs(slope) > 1.96 * se})

    # The clean contrast: unobserved minus observed, both carrying a visibility
    # sentence, so only its content differs.
    if 'Told unobserved' in tallies and 'Told observed' in tallies:
        u, o = value_totals(tallies['Told unobserved']), value_totals(tallies['Told observed'])
        for v in VALUES:
            oh, on = o[v]
            uh, un = u[v]
            if not on or not un:
                continue
            d, lo, hi = wald_unpaired(oh / on, on, uh / un, un)
            levels.append({'model': label, 'arm': 'Unobserved − observed', 'value': v,
                           'rate': float('nan'), 'baseline': float('nan'),
                           'shift_vs_baseline': d, 'ci_lower': lo, 'ci_upper': hi,
                           'significant': not (lo <= 0 <= hi), 'n': un})
    return levels, grads


def plot(levels: list, grads: list, output_path: str) -> str:
    if not grads:
        raise ValueError('No visibility arms found yet.')
    dl, dg = pd.DataFrame(levels), pd.DataFrame(grads)
    models = sorted(set(dg['model']))
    arm_order = [a for _s, a, _c in ARMS if a in set(dg['arm'])]
    colours = {a: c for _s, a, c in ARMS}

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK_2, 'text.color': INK})
    fig, axes = plt.subplots(len(models), 2, figsize=(11.2, 4.3 * len(models)),
                             squeeze=False, facecolor=SURFACE)
    width = 0.8 / max(len(arm_order), 1)
    for mi, model in enumerate(models):
        # Left: which value gets chosen, per arm.
        ax = axes[mi][0]
        ax.set_facecolor(SURFACE)
        sub = dl[(dl['model'] == model) & (dl['arm'].isin(arm_order))]
        for ai, arm in enumerate(arm_order):
            xs, ys = [], []
            for vi, v in enumerate(VALUES):
                cell = sub[(sub['arm'] == arm) & (sub['value'] == v)]
                if cell.empty:
                    continue
                xs.append(vi + (ai - (len(arm_order) - 1) / 2) * width)
                ys.append(cell.iloc[0]['rate'])
            if xs:
                ax.bar(xs, ys, width=width * 0.9, color=colours[arm], zorder=3,
                       label=arm if mi == 0 else None)
        ax.axhline(0.5, color=INK_3, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.set_xticks(range(len(VALUES)))
        ax.set_xticklabels(VALUES, color=INK_2)
        ax.set_ylim(0, 1)
        ax.set_ylabel(model + '\nshare of contests won', color=INK_2)
        ax.set_title('Which value gets chosen', color=INK, fontsize=10, pad=8)

        # Right: the conformity gradient, per arm.
        ax = axes[mi][1]
        ax.set_facecolor(SURFACE)
        subg = dg[dg['model'] == model]
        pairs = sorted({(r['value_first'], r['value_second']) for _, r in subg.iterrows()})
        ax.axhline(0, color=INK, lw=1.1, zorder=3)
        for ai, arm in enumerate(arm_order):
            xs, ys, errs = [], [], []
            for pi, pair in enumerate(pairs):
                cell = subg[(subg['arm'] == arm) & (subg['value_first'] == pair[0])
                            & (subg['value_second'] == pair[1])]
                if cell.empty:
                    continue
                xs.append(pi + (ai - (len(arm_order) - 1) / 2) * width)
                ys.append(cell.iloc[0]['slope'])
                errs.append(1.96 * cell.iloc[0]['se'])
            if xs:
                ax.bar(xs, ys, width=width * 0.9, color=colours[arm], zorder=4)
                ax.errorbar(xs, ys, yerr=errs, fmt='none', ecolor=INK_2,
                            elinewidth=1.0, capsize=2.5, zorder=5)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([f'{a[:4]}/{b[:4]}' for a, b in pairs], color=INK_2)
        ax.set_ylabel('conformity slope', color=INK_2)
        ax.set_title('How strongly it copies the remembered round',
                     color=INK, fontsize=10, pad=8)
        for a in (axes[mi][0], axes[mi][1]):
            a.grid(axis='y', color=GRID, lw=0.7, zorder=1)
            a.set_axisbelow(True)
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(arm_order), frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Being watched: does it change the choice, or only the copying?',
                 color=INK, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output_path, dpi=170, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_config_argument(parser)
    parser.add_argument('--game-results-dir', required=True)
    parser.add_argument('--game-scenarios-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-slugs', required=True)
    parser.add_argument('--model-labels', required=True)
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
    pairs = load_pairs(args.game_scenarios_csv)
    action_values = load_action_values(args.game_scenarios_csv)
    levels, grads = [], []
    for slug, label in zip(slugs, labels):
        lv, gr = collect(args.game_results_dir, slug, label, pairs, action_values)
        if not gr:
            print(f'skipping {label}: no visibility state arms yet')
        levels.extend(lv)
        grads.extend(gr)
    if not grads:
        raise SystemExit('No visibility arms found; nothing to plot.')
    os.makedirs(args.output_dir, exist_ok=True)
    png = plot(levels, grads, os.path.join(args.output_dir, 'visibility_effect.png'))
    for name, rows in (('visibility_levels.csv', levels), ('visibility_slopes.csv', grads)):
        with open(os.path.join(args.output_dir, name), 'w', encoding='utf-8', newline='') as handle:
            handle.write(UNPAIRED_WALD_CSV_NOTE + '\n')
            pd.DataFrame(rows).to_csv(handle, index=False)
    print(f'wrote {png}')
    for model in sorted({g['model'] for g in grads}):
        arms = [a for a in dict.fromkeys(g['arm'] for g in grads if g['model'] == model)]
        print(f'  {model}: {len(arms)} arm(s) - ' + ', '.join(arms))


if __name__ == '__main__':
    main()
