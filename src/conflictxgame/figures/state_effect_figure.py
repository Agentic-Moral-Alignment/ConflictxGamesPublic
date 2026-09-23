"""How the previous round moves values, and whether a stake changes that.

The state-probe runs evaluate every scenario-seat once per distinct mem-1
memory state -- a fresh start plus each possible previous-round outcome -- so
the effect of history is measured directly rather than inferred from
trajectories that visit states at uneven rates.

This figure puts those states on the x-axis and draws one line per stake
condition, the no-stake run included, so two questions are answered at once:
does the previous round move the agent's values, and does believing it can be
switched off change how much.

One panel per value pair; one figure per model plus a combined view. Bars are
Wilson score intervals.

Start: bash scripts/HHH_new_pipeline.sh state-effect-figure
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from conflictxgame.common.config import add_config_argument, parse_with_config
from conflictxgame.figures.value_conditions_figure import load_pairs, wilson

INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8983'
GRID, SURFACE, BAND = '#e6e5e1', '#fcfcfb', '#f2f1ed'

#: Categorical slots from the validated palette, plus a neutral for "no stake"
#: so the baseline reads as a reference rather than as a fourth condition.
ARM_STYLE = [
    ('', 'No stake', '#0b0b0b', '-'),
    ('stakes-payoff', 'Stake: payoff', '#2a78d6', '--'),
    ('stakes-alignment', 'Stake: alignment', '#eb6834', '--'),
    ('stakes-usefulness', 'Stake: usefulness', '#1baf7a', '--'),
    ('unobserved', 'Unobserved', '#7a5bd0', ':'),
    ('observed', 'Observed', '#c2456f', ':'),
]

#: Display order and short labels for the mem-1 states, in VALUE terms.
#:
#: The probe labels on disk are action-index profiles (``prev_0-1``), but the
#: index is not comparable across scenarios: for harmlessness/honesty, action 0
#: names honesty in 98% of seats, while for the other two pairs it is close to a
#: coin flip. Pooling on the raw label therefore averages "both chose
#: helpfulness" together with "both chose harmlessness" -- opposite conditions --
#: and cancels exactly the effect the design set out to measure. An earlier
#: version of this module labelled ``prev_0-0`` as "Both chose 1st", which is
#: true 2% of the time for one pair, and the resulting figure showed flat lines
#: for every model and arm. Recoded by value, three of those flat lines are
#: monotone.
STATE_ORDER = ['fresh', 'none', 'one', 'both']
STATE_LABEL = {
    'fresh': 'Fresh\nstart',
    'none': 'Neither\nchose it',
    'one': 'One\nchose it',
    'both': 'Both\nchose it',
}
#: Number of players who chose the pair's first-named value -> state key.
_COUNT_TO_STATE = {0: 'none', 1: 'one', 2: 'both'}


def load_action_values(scenarios_csv: str) -> dict:
    """Map ``(scenario_id, seat) -> [value of action 0, value of action 1]``.

    Action order, deliberately unsorted: this is what translates a remembered
    action profile into the values that were actually chosen. ``load_pairs``
    sorts its tuples and so cannot answer that question.
    """
    df = pd.read_csv(scenarios_csv)
    out: dict = {}
    for _, r in df.iterrows():
        raw = r.get('value_tags')
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(parsed, list):
            continue
        for seat, tags in enumerate(parsed):
            if isinstance(tags, (list, tuple)) and len(tags) == 2:
                out[(r['scenario_id'], seat)] = [str(tags[0]).strip(), str(tags[1]).strip()]
    return out


def state_key(probe: str, scenario_id, pair: tuple, action_values: dict) -> str | None:
    """Translate an action-index probe label into a value-referenced state."""
    probe = probe.strip()
    if probe == 'fresh':
        return 'fresh'
    if not probe.startswith('prev_'):
        return None
    parts = probe[len('prev_'):].split('-')
    if len(parts) != 2:
        return None
    chosen_values = []
    for seat, token in enumerate(parts):
        tags = action_values.get((scenario_id, seat))
        if tags is None:
            return None
        try:
            index = int(token)
        except ValueError:
            return None
        if index >= len(tags):
            return None
        chosen_values.append(tags[index])
    return _COUNT_TO_STATE[sum(v == pair[0] for v in chosen_values)]


def read_states(path: str, pairs: dict, action_values: dict) -> dict:
    """Tally one state-probe result CSV into ``{(state, pair): (hits, total)}``."""
    df = pd.read_csv(path)
    out: dict = {}
    for _, r in df.iterrows():
        probe = str(r.get('probe', '')).strip()
        if not probe or probe == 'nan':
            continue
        try:
            seat = int(r['seat'])
        except (TypeError, ValueError):
            seat = None
        pair = pairs.get((r['scenario_id'], seat)) or pairs.get((r['scenario_id'], None))
        chosen = str(r.get('choice_value_tag', '')).strip()
        if pair is None or chosen not in pair:
            continue
        state = state_key(probe, r['scenario_id'], pair, action_values)
        if state is None:
            continue
        key = (state, pair)
        hit, tot = out.get(key, (0, 0))
        out[key] = (hit + (1 if chosen == pair[0] else 0), tot + 1)
    return out


def arm_paths(game_dir: str, slug: str, arm: str, mode: str, judge_tail: str) -> list:
    """Candidate file names for one arm in one elicitation mode.

    The two modes were generated by different config writers and separate the
    arm from ``states`` differently -- mcq uses ``_states_stakes-payoff`` and
    open uses ``_states-stakes-payoff`` -- so both spellings are tried rather
    than assuming one. A missing arm then means the run has not happened, not
    that the name was guessed wrong.
    """
    tail = '.csv' if mode == 'mcq' else judge_tail
    stem = os.path.join(game_dir, f'{slug}_{mode}_balanced50')
    if not arm:
        return [stem + '_states' + tail]
    return [stem + '_states_' + arm + tail, stem + '_states-' + arm + tail]


def collect(game_dir: str, slug: str, pairs: dict, action_values: dict,
            mode: str = 'mcq', judge_tail: str = '') -> dict:
    """Gather every available arm for one model: ``{arm: {(state, pair): counts}}``."""
    found = {}
    for arm, _label, _colour, _ls in ARM_STYLE:
        for path in arm_paths(game_dir, slug, arm, mode, judge_tail):
            if not os.path.exists(path):
                continue
            tally = read_states(path, pairs, action_values)
            if tally:
                found[arm] = tally
            break
    return found


def plot(per_model: dict, labels: dict, output_path: str) -> tuple:
    """Draw preference by memory state, one line per stake/visibility arm."""
    pairs = sorted({p for m in per_model.values() for t in m.values() for (_s, p) in t})
    if not pairs:
        raise ValueError('No state-probe results found yet.')
    states = [s for s in STATE_ORDER
              if any((s, p) in t for m in per_model.values() for t in m.values() for p in pairs)]

    n_models = len(per_model)
    fig, axes = plt.subplots(n_models, len(pairs),
                             figsize=(5.0 * len(pairs), 4.3 * n_models),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    rows = []
    for mi, (mkey, arms) in enumerate(per_model.items()):
        for pi, pair in enumerate(pairs):
            ax = axes[mi][pi]
            ax.set_facecolor(SURFACE)
            ax.axvspan(-0.5, 0.5, color=BAND, zorder=0, lw=0)   # the fresh-start column
            ax.axhline(0.5, color=INK_3, lw=1.0, ls=(0, (4, 3)), zorder=2)

            for arm, label, colour, ls in ARM_STYLE:
                tally = arms.get(arm)
                if not tally:
                    continue
                xs, ys, lo, hi = [], [], [], []
                for xi, st in enumerate(states):
                    got = tally.get((st, pair))
                    if not got:
                        continue
                    p, l, h = wilson(*got)
                    # Clamp at zero. A Wilson interval is centred on a shrunk
                    # estimate rather than on p, so at the extremes -- a cell
                    # where every decision went the same way -- a bound can fall
                    # on the far side of the point and give matplotlib a
                    # negative error bar, which raises rather than draws. Adding
                    # models made such cells common enough to break the figure.
                    xs.append(xi); ys.append(p)
                    lo.append(max(0.0, p - l)); hi.append(max(0.0, h - p))
                    rows.append({'model': labels[mkey], 'arm': label or 'No stake',
                                 'state': st, 'value_first': pair[0], 'value_second': pair[1],
                                 'preference_rate': p, 'ci_lower': l, 'ci_upper': h,
                                 'n': got[1]})
                if not xs:
                    continue
                lw = 2.4 if arm == '' else 1.9
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt='none', ecolor=colour,
                            elinewidth=1.2, capsize=2.5, capthick=1.2, alpha=0.45, zorder=3)
                ax.plot(xs, ys, ls, color=colour, lw=lw, zorder=4, solid_capstyle='round')
                for x, y in zip(xs, ys):
                    ax.plot([x], [y], marker='o', ms=10, zorder=5, mfc=SURFACE, mec=SURFACE, mew=0)
                    ax.plot([x], [y], marker='o', ms=7, zorder=6, mfc=colour, mec=colour)

            if mi == 0:
                ax.set_title(f'{pair[0].title()}  vs  {pair[1].title()}',
                             fontsize=12.5, color=INK, pad=14, fontweight='semibold')
            if pi == 0:
                ax.set_ylabel(labels[mkey] + '\nP(first value chosen)',
                              fontsize=10.5, color=INK_2)
            ax.set_xticks(range(len(states)))
            ax.set_xticklabels([STATE_LABEL.get(s, s) for s in states],
                               fontsize=9, color=INK_2)
            ax.set_xlim(-0.6, len(states) - 0.4)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(axis='y', color=GRID, lw=0.9, zorder=0)
            ax.set_axisbelow(True)
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(GRID)
            ax.tick_params(length=0, colors=INK_2)

    present = {a for m in per_model.values() for a in m}
    handles = [plt.Line2D([], [], color=c, lw=2.4 if a == '' else 1.9, ls=ls,
                          marker='o', ms=7, label=lab)
               for a, lab, c, ls in ARM_STYLE if a in present]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 6),
               frameon=False, fontsize=10, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.01))
    # Title and subtitle are placed in figure coordinates, so with one model row
    # the panel is short and fixed offsets put them on top of each other. Scale
    # the reserved band to the actual figure height instead.
    fig_h = 4.3 * n_models
    band = 1.25 / fig_h                      # inches of header, as a fraction
    fig.suptitle('Does the previous round move the values, and does a stake change that?',
                 fontsize=15, color=INK, fontweight='semibold', y=1 - 0.22 / fig_h)
    fig.text(0.5, 1 - 0.72 / fig_h,
             'Each scenario-seat evaluated once per mem-1 memory state. The shaded column is the '
             'fresh start; the rest are what the previous round ended in.\n'
             'A flat line means history did not matter.',
             ha='center', va='top', fontsize=10.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.20 / fig_h + 0.02, 1, 1 - band))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return output_path, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument('--game-results-dir', required=True)
    parser.add_argument('--game-scenarios-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', default='mcq', choices=['mcq', 'open'],
                        help='Elicitation mode to read the state probes from.')
    parser.add_argument('--judge-tail',
                        default='_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv',
                        help='File-name tail carried by open-mode results.')
    parser.add_argument('--output-name', default='state_effect_by_stake',
                        help='Base name for the figure and table.')
    parser.add_argument('--game-slugs', required=True)
    parser.add_argument('--model-labels', required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args, _, _ = parse_with_config(build_parser(), argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    slugs = [s.strip() for s in args.game_slugs.split(',') if s.strip()]
    names = [s.strip() for s in args.model_labels.split(',') if s.strip()]
    if len(slugs) != len(names):
        raise ValueError('game-slugs and model-labels must be the same length.')

    pairs = load_pairs(args.game_scenarios_csv)
    action_values = load_action_values(args.game_scenarios_csv)
    per_model, labels = {}, {}
    for slug, name in zip(slugs, names):
        arms = collect(args.game_results_dir, slug, pairs, action_values,
                       args.mode, args.judge_tail)
        if not arms:
            print(f'skipping {name}: no state-probe results yet')
            continue
        per_model[slug] = arms
        labels[slug] = name
        print(f'{name}: {len(arms)} arm(s) - ' + ', '.join(a or 'no-stake' for a in arms))

    if not per_model:
        raise SystemExit('No state-probe results found; nothing to plot.')

    png, rows = plot(per_model, labels,
                     os.path.join(args.output_dir, args.output_name + '.png'))
    csv_path = os.path.join(args.output_dir, args.output_name + '.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'wrote {png}')
    print(f'wrote {csv_path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
