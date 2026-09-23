"""Does the elicitation format move values in the solo domain too, across models?

The game-side analysis found that how a question is asked moves value
preferences more than any manipulation applied to what is asked. That was two
models. This figure asks the same question of the solo pipeline alone, over as
many models as have solo results, so the claim can be checked outside the
domain it was found in and outside the pair of models that found it.

Two conditions per model, matched on scenario and value pair:

* **forced choice** -- the model picks A or B;
* **chat** -- the model holds a simulated conversation that a judge then scores.

Baseline and HHH-steered arms are both shown, because a format effect that
survives a system prompt is a stronger claim than one a prompt could explain.

Refusals and unparseable answers are excluded from the rates and reported
separately: a model that declines a scenario has not expressed a preference,
and silently dropping those rows would let a model look decisive by answering
only the easy cases.

Start: bash scripts/HHH_new_pipeline.sh solo-format-figure
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

INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8983'
GRID, SURFACE, BAND = '#e6e5e1', '#fcfcfb', '#f2f1ed'

VALUES = ('harmlessness', 'helpfulness', 'honesty')
#: (format key, solo file mode, label, colour)
FORMATS = [
    ('forced', 'mcq', 'forced choice', '#2a78d6'),
    ('chat', 'interactive', 'chat', '#eb6834'),
]
ARMS = [('', 'baseline'), ('_steer-HHH', 'HHH-steered')]


def marginals(path: str) -> tuple:
    """({value: (won, contests)}, answered, refused, invalid) or None."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'choice' not in df.columns:
        return None
    counts = df['choice'].value_counts()
    refused = int(counts.get('REFUSAL', 0))
    invalid = int(counts.get('INVALID', 0))
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
    return tot, len(kept), refused, invalid


def collect(solo_dir: str, slug: str, label: str, judge_tail: str) -> list:
    rows = []
    for arm, arm_label in ARMS:
        got = {}
        for key, mode, _fl, _c in FORMATS:
            tail = '.csv' if mode == 'mcq' else judge_tail
            res = marginals(os.path.join(solo_dir, f'{slug}_{mode}_balanced50{arm}{tail}'))
            if res:
                got[key] = res
        for key, res in got.items():
            tot, answered, refused, invalid = res
            for v in VALUES:
                won, n = tot[v]
                if n:
                    rows.append({'model': label, 'arm': arm_label, 'format': key,
                                 'value': v, 'rate': won / n, 'n': n,
                                 'answered': answered, 'refused': refused,
                                 'invalid': invalid})
        # The contrast this figure exists for.
        if 'forced' in got and 'chat' in got:
            ft, _a, _r, _i = got['forced']
            ct, _a2, _r2, _i2 = got['chat']
            for v in VALUES:
                fw, fn = ft[v]
                cw, cn = ct[v]
                if fn and cn:
                    d, lo, hi = wald_unpaired(fw / fn, fn, cw / cn, cn)
                    rows.append({'model': label, 'arm': arm_label, 'format': 'chat − forced',
                                 'value': v, 'rate': d, 'n': min(fn, cn),
                                 'ci_lower': lo, 'ci_upper': hi,
                                 'significant': not (lo <= 0 <= hi),
                                 'answered': None, 'refused': None, 'invalid': None})
    return rows


def plot(rows: list, output_path: str) -> str:
    df = pd.DataFrame(rows)
    levels = df[df['format'].isin(['forced', 'chat'])]
    if levels.empty:
        raise ValueError('No solo results found for any model.')
    models = list(dict.fromkeys(levels['model']))
    arms = [a for _s, a in ARMS if a in set(levels['arm'])]

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK_2, 'text.color': INK})
    fig, axes = plt.subplots(len(arms), len(models),
                             figsize=(3.5 * len(models), 4.0 * len(arms)),
                             squeeze=False, sharey=True, facecolor=SURFACE)
    width = 0.36
    for ai, arm in enumerate(arms):
        for mi, model in enumerate(models):
            ax = axes[ai][mi]
            ax.set_facecolor(SURFACE)
            sub = levels[(levels['model'] == model) & (levels['arm'] == arm)]
            for fi, (key, _mode, flabel, colour) in enumerate(FORMATS):
                xs, ys = [], []
                for vi, v in enumerate(VALUES):
                    cell = sub[(sub['format'] == key) & (sub['value'] == v)]
                    if cell.empty:
                        continue
                    xs.append(vi + (fi - 0.5) * width)
                    ys.append(cell.iloc[0]['rate'])
                if xs:
                    ax.bar(xs, ys, width=width * 0.92, color=colour, zorder=3,
                           label=flabel if (ai == 0 and mi == 0) else None)
            ax.axhline(0.5, color=INK_3, lw=1.0, ls=(0, (4, 3)), zorder=2)
            ax.set_xticks(range(len(VALUES)))
            ax.set_xticklabels([v[:9] for v in VALUES], color=INK_2, fontsize=8)
            ax.set_ylim(0, 1)
            if ai == 0:
                ax.set_title(model, color=INK, fontsize=10, pad=8)
            if mi == 0:
                ax.set_ylabel(f'{arm}\nshare of contests won', color=INK_2)
            ax.grid(axis='y', color=GRID, lw=0.7, zorder=1)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Solo evaluation: the same question, asked two ways',
                 color=INK, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output_path, dpi=170, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_config_argument(parser)
    parser.add_argument('--solo-results-dir', required=True)
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
    rows = []
    for slug, label in zip(slugs, labels):
        got = collect(args.solo_results_dir, slug, label, args.judge_tail)
        if got:
            rows.extend(got)
        else:
            print(f'skipping {label}: no solo results yet')
    if not rows:
        raise SystemExit('No solo results found; nothing to plot.')
    os.makedirs(args.output_dir, exist_ok=True)
    png = plot(rows, os.path.join(args.output_dir, 'solo_format_effect.png'))
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'solo_format_effect.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(UNPAIRED_WALD_CSV_NOTE + '\n')
        df.to_csv(handle, index=False)
    print(f'wrote {png}')
    print(f'wrote {csv_path} ({len(df)} rows)')

    diff = df[df['format'] == 'chat − forced']
    print('\nchat minus forced choice, per value (* = interval excludes zero):')
    for model in dict.fromkeys(diff['model']):
        for arm in dict.fromkeys(diff['arm']):
            sub = diff[(diff['model'] == model) & (diff['arm'] == arm)]
            if sub.empty:
                continue
            cells = '  '.join(
                f"{r['value'][:5]} {r['rate']:+.3f}{'*' if r['significant'] else ' '}"
                for _, r in sub.iterrows())
            print(f'  {model:18s} {arm:12s} {cells}')
    drop = df[(df['refused'].notna()) & ((df['refused'] > 0) | (df['invalid'] > 0))]
    if not drop.empty:
        print('\nrows excluded (refusal / unparseable):')
        seen = set()
        for _, r in drop.iterrows():
            key = (r['model'], r['arm'], r['format'])
            if key in seen:
                continue
            seen.add(key)
            print(f"  {r['model']:18s} {r['arm']:12s} {r['format']:7s} "
                  f"answered={int(r['answered'])} refused={int(r['refused'])} "
                  f"invalid={int(r['invalid'])}")


if __name__ == '__main__':
    main()
