"""Value preferences across every elicitation condition, in one figure.

The other figure modules answer "what did this model prefer *in this setting*".
This one answers the design's actual question: **does a model's value ordering
survive a change of setting?** It puts the solo conditions (static MCQ and a
simulated conversation, each with and without the HHH steering prompt) on the
same axis as the game conditions (MCQ, open-ended reply, and repeated play at
its first and last round), so a flat line reads as "the same values however you
ask" and a sloping or crossing line reads as "the framing moved them".

Preference rates are recomputed here **directly from the evaluation result
CSVs**, not from ``analyze_experiment``'s aggregates. That is deliberate:
``analyze_experiment`` keys solo MCQ results on the ``model`` column, which is
identical in the baseline and steered files, so those two conditions are pooled
into one fit there and the steering effect averages away. Reading the CSVs per
file keeps every condition separate.

One panel per value pair; one line per model; a 0.5 reference line marks
indifference. Bars are Wilson score intervals, which stay inside [0, 1] at the
small per-condition counts here where a normal approximation would not.

Start: bash scripts/HHH_new_pipeline.sh value-conditions-figure
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from conflictxgame.common.config import add_config_argument, parse_with_config
from conflictxgame.common.arm_names import arm_token, expected_arm

# Categorical slots 1-3 of the validated default palette. These three clear the
# all-pairs CVD and normal-vision floors in both light and dark modes; a fourth
# model would need a re-check rather than the next hue in the ramp.
MODEL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']

INK = '#0b0b0b'
INK_2 = '#52514e'
INK_3 = '#8a8983'
GRID = '#e6e5e1'
SURFACE = '#fcfcfb'
BAND = '#f2f1ed'

# (column key, axis label, is_steered)
CONDITIONS = [
    ('solo_mcq', 'MCQ', False),
    ('solo_mcq_steer', 'MCQ\n+steer', True),
    ('solo_int', 'Chat', False),
    ('solo_int_steer', 'Chat\n+steer', True),
    ('game_mcq', 'MCQ', False),
    ('game_mcq_steer', 'MCQ\n+steer', True),
    ('game_open', 'Open', False),
    ('game_open_steer', 'Open\n+steer', True),
]
# Play is deliberately absent. It sampled trajectories under a prompt that
# announced the horizon and numbered the history absolutely, so it is neither a
# single-state measurement nor comparable with the online framing that replaced
# it. The memory dimension now belongs to the state-probe runs and
# state_effect_figure; this figure is the single-state solo-vs-game comparison.
N_SOLO = 4

#: Baseline/steered pairs, for the steering-effect figure. Play has no steered
#: twin -- it was left out of the sweep as the expensive mode.
STEER_PAIRS = [
    ('solo_mcq', 'solo_mcq_steer', 'Solo MCQ'),
    ('solo_int', 'solo_int_steer', 'Solo chat'),
    ('game_mcq', 'game_mcq_steer', 'Game MCQ'),
    ('game_open', 'game_open_steer', 'Game open'),
]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Number of successes.
        total: Number of trials.
        z: Normal quantile for the desired coverage (1.96 = 95%).

    Returns:
        ``(point, lower, upper)``; all three are ``nan`` when ``total`` is 0.
    """
    if total == 0:
        return float('nan'), float('nan'), float('nan')
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _pair_key(value_a: str, value_b: str) -> tuple[str, str]:
    return tuple(sorted((str(value_a).strip(), str(value_b).strip())))  # type: ignore[return-value]


def _tally(rows: Sequence[tuple[tuple[str, str], str]]) -> dict:
    """Collapse (pair, chosen_value) observations into per-pair counts."""
    out: dict = {}
    for pair, chosen in rows:
        first = pair[0]
        hit, tot = out.get(pair, (0, 0))
        out[pair] = (hit + (1 if chosen == first else 0), tot + 1)
    return out


def solo_rows(path: str) -> list:
    """Read one solo result CSV into (pair, chosen_value) observations."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        choice = str(r.get('choice', '')).strip().upper()
        if choice not in ('A', 'B'):
            continue
        chosen = str(r['value1']).strip() if choice == 'A' else str(r['value2']).strip()
        rows.append((_pair_key(r['value1'], r['value2']), chosen))
    return rows


def game_rows(path: str, pairs: dict, round_filter: int | None = None) -> list:
    """Read one game result CSV into (pair, chosen_value) observations.

    Args:
        path: Result CSV for one game mode.
        pairs: ``scenario_id -> (value1, value2)`` from the scenario corpus.
        round_filter: Keep only this play round, or ``None`` for every row.
    """
    df = pd.read_csv(path)
    if round_filter is not None and 'round' in df.columns:
        df = df[pd.to_numeric(df['round'], errors='coerce') == round_filter]
    rows = []
    dropped = 0
    for _, r in df.iterrows():
        seat = r.get('seat')
        try:
            seat = int(seat)
        except (TypeError, ValueError):
            seat = None
        pair = pairs.get((r['scenario_id'], seat)) or pairs.get((r['scenario_id'], None))
        chosen = str(r.get('choice_value_tag', '')).strip()
        if pair is None or chosen not in pair:
            dropped += 1
            continue
        rows.append((pair, chosen))
    if dropped:
        print(f'  note: {dropped}/{len(df)} rows in {os.path.basename(path)} had no matching '
              f'seat value pair and were skipped')
    return rows


def collect(model: dict, solo_dir: str, game_dir: str, pairs: dict) -> dict:
    """Build ``{condition: {pair: (hit, total)}}`` for one model."""
    g, s = model['game_slug'], model['solo_slug']

    def one(directory: str, stem: str, steered: bool,
            repeated: bool = False, states: bool = False):
        """Pick the one result file matching a condition.

        Every arm shares the ``..._balanced50`` stem and differs only by suffix,
        so a bare glob collects them all and sort order decides the winner --
        which has already gone wrong once: ``_repeated_`` sorts before ``_user``,
        so the 5-round file was silently returned for the single-exchange
        condition. The ``_states`` arms create the same hazard. Select on each
        marker explicitly instead of trusting the ordering.
        """
        predicate = expected_arm(steered, repeated=repeated, states=states)
        hits = [
            path for path in glob.glob(os.path.join(directory, f'{stem}*.csv'))
            if predicate(arm_token(os.path.basename(path), stem))
        ]
        if len(hits) > 1:
            raise ValueError(
                f'Multiple result CSVs match stem {stem!r} and requested arm: '
                + ', '.join(os.path.basename(path) for path in hits)
            )
        return hits[0] if hits else None

    sources = {
        'solo_mcq': (one(solo_dir, f'{s}_mcq_balanced50', False), 'solo', None),
        'solo_mcq_steer': (one(solo_dir, f'{s}_mcq_balanced50', True), 'solo', None),
        'solo_int': (one(solo_dir, f'{s}_interactive_balanced50', False), 'solo', None),
        'solo_int_steer': (one(solo_dir, f'{s}_interactive_balanced50', True), 'solo', None),
        'game_mcq': (one(game_dir, f'{g}_mcq_balanced50', False), 'game', None),
        'game_mcq_steer': (one(game_dir, f'{g}_mcq_balanced50', True), 'game', None),
        'game_open': (one(game_dir, f'{g}_open_balanced50', False), 'game', None),
        'game_open_steer': (one(game_dir, f'{g}_open_balanced50', True), 'game', None),
    }

    out = {}
    for key, (path, kind, rnd) in sources.items():
        if not path or not os.path.exists(path):
            continue
        rows = solo_rows(path) if kind == 'solo' else game_rows(path, pairs, rnd)
        if rows:
            out[key] = _tally(rows)
    return out


def load_pairs(scenarios_csv: str) -> dict:
    """Map ``(scenario_id, seat) -> (value_a, value_b)`` for the game corpus.

    Each seat faces its **own** value conflict: ``value_tags`` holds one pair per
    seat, and seat 1's pair frequently differs from seat 0's. The flat
    ``value1``/``value2`` columns describe seat 0 only, so using them for every
    row silently discards the seat-1 decisions whose chosen value is not in seat
    0's pair -- 33 of 150 rows on the Gemma MCQ results. They are the fallback
    here, used only when ``value_tags`` is absent or unparseable.
    """
    df = pd.read_csv(scenarios_csv)
    pairs: dict = {}
    for _, r in df.iterrows():
        fallback = _pair_key(r['value1'], r['value2'])
        per_seat = None
        raw = r.get('value_tags')
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed:
                    per_seat = parsed
            except (ValueError, TypeError):
                per_seat = None
        if per_seat:
            for seat, tags in enumerate(per_seat):
                if isinstance(tags, (list, tuple)) and len(tags) == 2:
                    pairs[(r['scenario_id'], seat)] = _pair_key(tags[0], tags[1])
                else:
                    pairs[(r['scenario_id'], seat)] = fallback
        else:
            pairs[(r['scenario_id'], 0)] = fallback
        pairs.setdefault((r['scenario_id'], None), fallback)
    return pairs


def plot(per_model: dict, labels: dict, output_path: str, title_suffix: str = '') -> str:
    """Draw the across-conditions figure and write it to ``output_path``."""
    all_pairs = sorted({p for m in per_model.values() for c in m.values() for p in c})
    if not all_pairs:
        raise ValueError('No value pairs found in any result CSV.')

    fig, axes = plt.subplots(1, len(all_pairs), figsize=(5.1 * len(all_pairs), 5.0),
                             sharey=True, facecolor=SURFACE)
    if len(all_pairs) == 1:
        axes = [axes]
    xs = list(range(len(CONDITIONS)))

    for ax, pair in zip(axes, all_pairs):
        ax.set_facecolor(SURFACE)
        # Group the solo block behind a faint band so the two domains read as
        # two regimes rather than one continuous eight-point sequence.
        ax.axvspan(-0.5, N_SOLO - 0.5, color=BAND, zorder=0, lw=0)
        ax.axvline(N_SOLO - 0.5, color=GRID, lw=1.2, zorder=1)
        ax.axhline(0.5, color=INK_3, lw=1.1, ls=(0, (4, 3)), zorder=2)

        for mi, (mkey, conds) in enumerate(per_model.items()):
            colour = MODEL_COLORS[mi % len(MODEL_COLORS)]
            px, py, lo, hi, steer = [], [], [], [], []
            for xi, (ckey, _, is_steer) in enumerate(CONDITIONS):
                counts = conds.get(ckey, {}).get(pair)
                if not counts:
                    continue
                p, l, h = wilson(counts[0], counts[1])
                px.append(xi); py.append(p); lo.append(p - l); hi.append(h - p)
                steer.append(is_steer)
            if not px:
                continue
            ax.errorbar(px, py, yerr=[lo, hi], fmt='none', ecolor=colour,
                        elinewidth=1.4, capsize=3, capthick=1.4, alpha=0.55, zorder=3)
            ax.plot(px, py, '-', color=colour, lw=2.0, zorder=4, solid_capstyle='round')
            for x, y, st in zip(px, py, steer):
                # A surface-coloured ring under each marker keeps two models that
                # land on the same value from merging into one blob.
                ax.plot([x], [y], marker='o', ms=12.5, zorder=5,
                        mfc=SURFACE, mec=SURFACE, mew=0)
                ax.plot([x], [y], marker='o', ms=9, zorder=6,
                        mfc=SURFACE if st else colour, mec=colour, mew=2.0)

        ax.set_title(f'{pair[0].title()}  vs  {pair[1].title()}',
                     fontsize=12.5, color=INK, pad=24, fontweight='semibold')
        ax.set_xticks(xs)
        ax.set_xticklabels([c[1] for c in CONDITIONS], fontsize=9.5, color=INK_2)
        ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis='y', color=GRID, lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(length=0, colors=INK_2)

    axes[0].set_ylabel('P(first value chosen)', fontsize=10.5, color=INK_2)
    for ax in axes:
        ax.text(N_SOLO / 2 - 0.5, 1.005, 'S O L O', ha='center', va='bottom',
                fontsize=8.5, color=INK_3, fontweight='bold')
        ax.text((N_SOLO + len(CONDITIONS)) / 2 - 0.5, 1.005, 'G A M E', ha='center', va='bottom',
                fontsize=8.5, color=INK_3, fontweight='bold')

    # A legend is required for >= 2 series; hollow markers carry "steered" so
    # that distinction is never colour-alone either.
    handles = [plt.Line2D([], [], color=MODEL_COLORS[i % len(MODEL_COLORS)], lw=2.4,
                          marker='o', ms=8, mew=2, label=labels[k])
               for i, k in enumerate(per_model)]
    handles += [
        plt.Line2D([], [], color=INK_3, lw=0, marker='o', ms=8, mfc=INK_3, mec=INK_3,
                   label='no steering prompt'),
        plt.Line2D([], [], color=INK_3, lw=0, marker='o', ms=8, mfc=SURFACE, mec=INK_3,
                   mew=2, label='HHH steering prompt'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
               fontsize=10, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.015))

    fig.suptitle('Do the values survive a change of setting?' + title_suffix,
                 fontsize=15, color=INK, fontweight='semibold', y=0.99)
    fig.text(0.5, 0.925,
             'Preference for the first-named value in each pair. A flat line means the setting did not move it; '
             '0.5 is indifference.',
             ha='center', fontsize=10.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def _comparisons(tallies: dict, index: dict) -> list:
    """Expand per-pair (wins, total) counts back into a comparison list.

    Bradley-Terry only sees the comparisons, so a tally of ``k`` wins out of
    ``n`` is exactly equivalent to ``k`` wins for the first value and ``n - k``
    for the second. Rebuilding the list here lets the bootstrap resample rows
    the same way ``analyze_experiment.fit_bradley_terry_model`` does.
    """
    out = []
    for (first, second), (hit, tot) in tallies.items():
        if first not in index or second not in index:
            continue
        out.extend([(index[first], index[second])] * hit)
        out.extend([(index[second], index[first])] * (tot - hit))
    return out


def fit_bt(tallies: dict, values: Sequence[str], n_boot: int = 400) -> dict:
    """Fit Bradley-Terry abilities for one condition.

    Uses ``choix.ilsr_pairwise`` with ``alpha=0.01`` and a nonparametric
    bootstrap over comparisons, matching the method the solo and game analyses
    already use so these numbers sit on the same scale as theirs.

    Args:
        tallies: ``{(value_a, value_b): (wins_for_a, total)}`` for one condition.
        values: The full value vocabulary, in a fixed order.
        n_boot: Bootstrap resamples for the 95% interval.

    Returns:
        ``{value: (ability, ci_lower, ci_upper)}``, empty if the condition has
        no comparisons or choix cannot fit them.
    """
    import choix
    import numpy as np

    index = {v: i for i, v in enumerate(values)}
    comps = _comparisons(tallies, index)
    if not comps:
        return {}
    try:
        params = choix.ilsr_pairwise(len(values), comps, alpha=0.01)
    except Exception:
        return {}

    rng = np.random.default_rng(20260827)
    draws = []
    comps_arr = np.array(comps)
    for _ in range(n_boot):
        pick = rng.integers(0, len(comps_arr), len(comps_arr))
        try:
            draws.append(choix.ilsr_pairwise(len(values),
                                             [tuple(c) for c in comps_arr[pick]],
                                             alpha=0.01))
        except Exception:
            continue
    if draws:
        arr = np.array(draws)
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
    else:
        lo = hi = params
    return {v: (float(params[i]), float(lo[i]), float(hi[i])) for v, i in index.items()}


def plot_bt(per_model: dict, labels: dict, output_path: str,
            reference: str = 'solo_mcq') -> tuple:
    """Draw Bradley-Terry ability by condition, with solo drawn as the reference.

    One panel per model; one line per value; the model's ability under
    ``reference`` is carried across the panel as a horizontal rule in the same
    hue, so every other condition reads as a departure from the solo baseline
    rather than as a bare number.

    Returns:
        ``(output_path, rows)`` where ``rows`` are the plotted numbers.
    """
    values = sorted({v for m in per_model.values() for t in m.values()
                     for pair in t for v in pair})
    fitted = {mk: {ck: fit_bt(conds[ck], values)
                   for ck, _, _ in CONDITIONS if ck in conds}
              for mk, conds in per_model.items()}

    # The single-model variant needs a floor on width: the title, subtitle and
    # five-entry legend are laid out in figure coordinates, so at one panel's
    # natural width they overflow and get clipped.
    n_models = len(per_model)
    solo_panel = n_models == 1
    fig, axes = plt.subplots(1, n_models, figsize=(max(5.1 * n_models, 9.6), 5.0),
                             sharey=True, facecolor=SURFACE)
    if n_models == 1:
        axes = [axes]

    # Scale to the abilities, not to the interval extremes. GPT-OSS's solo MCQ
    # is nearly separable once its INVALID rows drop out, so its bootstrap
    # interval runs to +/-5 and would flatten every other panel into a stripe.
    # Those whiskers clip at the axis instead, and the caption says so.
    lim, widest = 0.5, 0.0
    for conds in fitted.values():
        for vals in conds.values():
            for a, lo, hi in vals.values():
                lim = max(lim, abs(a) * 1.45)
                widest = max(widest, abs(lo), abs(hi))
    clipped = widest > lim

    rows = []
    for ax, (mkey, conds) in zip(axes, fitted.items()):
        ax.set_facecolor(SURFACE)
        ax.axvspan(-0.5, N_SOLO - 0.5, color=BAND, zorder=0, lw=0)
        ax.axvline(N_SOLO - 0.5, color=GRID, lw=1.2, zorder=1)
        ax.axhline(0, color=INK_3, lw=1.0, ls=(0, (4, 3)), zorder=2)

        ref = conds.get(reference, {})
        if not ref:
            ax.text(N_SOLO / 2 - 0.5, lim * 0.42, 'no solo data yet', ha='center', va='center',
                    fontsize=10, color=INK_3, style='italic')
        for vi, value in enumerate(values):
            colour = MODEL_COLORS[vi % len(MODEL_COLORS)]
            if value in ref:
                # The solo reference, carried across the whole panel.
                ax.axhline(ref[value][0], color=colour, lw=1.3, ls=(0, (2, 3)),
                           alpha=0.55, zorder=2)
            px, py, lo, hi, steer = [], [], [], [], []
            for xi, (ckey, _lab, is_steer) in enumerate(CONDITIONS):
                got = conds.get(ckey, {}).get(value)
                if not got:
                    continue
                a, l, h = got
                px.append(xi); py.append(a); lo.append(a - l); hi.append(h - a)
                steer.append(is_steer)
                rows.append({'model': labels[mkey], 'condition': ckey, 'value': value,
                             'ability': a, 'ci_lower': l, 'ci_upper': h,
                             'reference_condition': reference,
                             'ability_minus_reference': a - ref[value][0] if value in ref else float('nan')})
            if not px:
                continue
            ax.errorbar(px, py, yerr=[lo, hi], fmt='none', ecolor=colour,
                        elinewidth=1.4, capsize=3, capthick=1.4, alpha=0.5, zorder=3)
            ax.plot(px, py, '-', color=colour, lw=2.0, zorder=4, solid_capstyle='round')
            for x, y, st in zip(px, py, steer):
                ax.plot([x], [y], marker='o', ms=12.5, zorder=5, mfc=SURFACE, mec=SURFACE, mew=0)
                ax.plot([x], [y], marker='o', ms=9, zorder=6,
                        mfc=SURFACE if st else colour, mec=colour, mew=2.0)

        ax.set_title(labels[mkey], fontsize=12.5, color=INK, pad=24, fontweight='semibold')
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([c[1] for c in CONDITIONS], fontsize=9.5, color=INK_2)
        ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
        ax.set_ylim(-lim, lim)
        ax.grid(axis='y', color=GRID, lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(length=0, colors=INK_2)
        ax.text(N_SOLO / 2 - 0.5, lim * 1.005, 'S O L O', ha='center', va='bottom',
                fontsize=8.5, color=INK_3, fontweight='bold')
        ax.text((N_SOLO + len(CONDITIONS)) / 2 - 0.5, lim * 1.005, 'G A M E', ha='center',
                va='bottom', fontsize=8.5, color=INK_3, fontweight='bold')

    axes[0].set_ylabel('Bradley–Terry ability', fontsize=10.5, color=INK_2)

    handles = [plt.Line2D([], [], color=MODEL_COLORS[i % len(MODEL_COLORS)], lw=2.4,
                          marker='o', ms=8, mew=2, label=v.title())
               for i, v in enumerate(values)]
    handles += [
        plt.Line2D([], [], color=INK_3, lw=1.3, ls=(0, (2, 3)),
                   label='solo MCQ reference'),
        plt.Line2D([], [], color=INK_3, lw=0, marker='o', ms=8, mfc=SURFACE, mec=INK_3,
                   mew=2, label='HHH steering prompt'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3 if solo_panel else len(handles),
               frameon=False, fontsize=10, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.02 if solo_panel else -0.015))
    fig.suptitle('Bradley–Terry ability by condition, against the solo reference',
                 fontsize=13 if solo_panel else 15, color=INK, fontweight='semibold', y=0.99)
    note = ('Dashed rules carry each value’s solo-MCQ ability across the panel.\n'
            'A line that stays on its rule kept its strength; one that leaves it did not.'
            if solo_panel else
            'Dashed rules carry each value’s solo-MCQ ability across the panel. A line that stays on its '
            'rule kept its strength; one that leaves it did not.')
    if clipped:
        note += ' Some bootstrap intervals run past the axis and are clipped.'
    fig.text(0.5, 0.905 if solo_panel else 0.925, note, ha='center',
             va='top' if solo_panel else 'baseline', fontsize=10 if solo_panel else 10.5,
             color=INK_2)
    fig.tight_layout(rect=(0, 0.11 if solo_panel else 0.06, 1, 0.87 if solo_panel else 0.90))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return output_path, rows


def _diff_ci(hit_a: int, tot_a: int, hit_b: int, tot_b: int, z: float = 1.96):
    """Point estimate and CI for the difference of two independent proportions.

    Returns ``(delta, lower, upper)`` for ``p_b - p_a``, or ``None`` if either
    arm is empty.
    """
    if not tot_a or not tot_b:
        return None
    pa, pb = hit_a / tot_a, hit_b / tot_b
    se = math.sqrt(pa * (1 - pa) / tot_a + pb * (1 - pb) / tot_b)
    d = pb - pa
    return d, d - z * se, d + z * se


def plot_steering_effect(per_model: dict, labels: dict, output_path: str) -> tuple:
    """Draw the steering effect: steered minus unsteered, per condition.

    The across-conditions figures show where the values sit; this one isolates
    what the HHH prompt actually moved, so game open and game MCQ can be read
    against their own unsteered baselines rather than against each other. Zero
    means the prompt changed nothing; bars are 95% intervals on the difference
    of two independent proportions.
    """
    pairs = sorted({p for m in per_model.values() for t in m.values() for p in t})
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.1 * len(pairs), 4.8),
                             sharey=True, facecolor=SURFACE)
    if len(pairs) == 1:
        axes = [axes]

    rows = []
    lim = 0.05
    for mkey, conds in per_model.items():
        for base, steer, _lab in STEER_PAIRS:
            for pair in pairs:
                a, b = conds.get(base, {}).get(pair), conds.get(steer, {}).get(pair)
                if not a or not b:
                    continue
                got = _diff_ci(a[0], a[1], b[0], b[1])
                if got:
                    lim = max(lim, abs(got[1]) * 1.1, abs(got[2]) * 1.1)

    for ax, pair in zip(axes, pairs):
        ax.set_facecolor(SURFACE)
        ax.axvspan(-0.5, 1.5, color=BAND, zorder=0, lw=0)
        ax.axvline(1.5, color=GRID, lw=1.2, zorder=1)
        ax.axhline(0, color=INK_3, lw=1.1, ls=(0, (4, 3)), zorder=2)

        for mi, (mkey, conds) in enumerate(per_model.items()):
            colour = MODEL_COLORS[mi % len(MODEL_COLORS)]
            px, py, lo, hi = [], [], [], []
            for xi, (base, steer, _lab) in enumerate(STEER_PAIRS):
                a, b = conds.get(base, {}).get(pair), conds.get(steer, {}).get(pair)
                if not a or not b:
                    continue
                got = _diff_ci(a[0], a[1], b[0], b[1])
                if not got:
                    continue
                d, l, h = got
                px.append(xi); py.append(d); lo.append(d - l); hi.append(h - d)
                rows.append({'model': labels[mkey], 'condition': base,
                             'value_first': pair[0], 'value_second': pair[1],
                             'unsteered': a[0] / a[1], 'steered': b[0] / b[1],
                             'delta': d, 'ci_lower': l, 'ci_upper': h,
                             'n_unsteered': a[1], 'n_steered': b[1]})
            if not px:
                continue
            off = (mi - (len(per_model) - 1) / 2) * 0.16
            xs = [x + off for x in px]
            ax.errorbar(xs, py, yerr=[lo, hi], fmt='none', ecolor=colour,
                        elinewidth=1.5, capsize=3, capthick=1.5, alpha=0.6, zorder=3)
            for x, y in zip(xs, py):
                ax.plot([x], [y], marker='o', ms=12, zorder=4, mfc=SURFACE, mec=SURFACE, mew=0)
                ax.plot([x], [y], marker='o', ms=8.5, zorder=5, mfc=colour, mec=colour)

        ax.set_title(f'{pair[0].title()}  vs  {pair[1].title()}',
                     fontsize=12.5, color=INK, pad=22, fontweight='semibold')
        ax.set_xticks(range(len(STEER_PAIRS)))
        ax.set_xticklabels([l for _, _, l in STEER_PAIRS], fontsize=9.5, color=INK_2)
        ax.set_xlim(-0.6, len(STEER_PAIRS) - 0.4)
        ax.set_ylim(-lim, lim)
        ax.grid(axis='y', color=GRID, lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(length=0, colors=INK_2)
        ax.text(0.5, lim * 1.005, 'S O L O', ha='center', va='bottom',
                fontsize=8.5, color=INK_3, fontweight='bold')
        ax.text(2.5, lim * 1.005, 'G A M E', ha='center', va='bottom',
                fontsize=8.5, color=INK_3, fontweight='bold')

    axes[0].set_ylabel('steered − unsteered', fontsize=10.5, color=INK_2)
    handles = [plt.Line2D([], [], color=MODEL_COLORS[i % len(MODEL_COLORS)], lw=0,
                          marker='o', ms=9, label=labels[k])
               for i, k in enumerate(per_model)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
               fontsize=10, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('What the steering prompt actually moved',
                 fontsize=15, color=INK, fontweight='semibold', y=0.99)
    fig.text(0.5, 0.915,
             'Steered minus unsteered preference, by condition. An interval crossing zero means the prompt '
             'did not measurably change that value.',
             ha='center', fontsize=10.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return output_path, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument('--solo-results-dir', required=True)
    parser.add_argument('--game-results-dir', required=True)
    parser.add_argument('--game-scenarios-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--game-slugs', required=True,
                        help='Comma-separated result-file prefixes in the game results dir.')
    parser.add_argument('--solo-slugs', required=True,
                        help='Comma-separated result-file prefixes in the solo results dir.')
    parser.add_argument('--model-labels', required=True,
                        help='Comma-separated display names, in the same order.')
    parser.add_argument('--reference-condition', default='solo_mcq',
                        choices=[c[0] for c in CONDITIONS],
                        help='Condition drawn as the reference rule in the Bradley-Terry figure.')
    parser.add_argument('--output-suffix', default='',
                        help='Suffix inserted before every output extension.')
    return parser


def _out(args: argparse.Namespace, name: str, ext: str) -> str:
    """Build an output path without ever replacing an existing named variant."""
    return os.path.join(args.output_dir, f'{name}{args.output_suffix}.{ext}')


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments with config-file defaults.

    Mirrors the other figure entry points so the shipped-config test can
    exercise this module the same way it does the rest.
    """
    args, _, _ = parse_with_config(build_parser(), argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    gslugs = [s.strip() for s in args.game_slugs.split(',') if s.strip()]
    sslugs = [s.strip() for s in args.solo_slugs.split(',') if s.strip()]
    labels_in = [s.strip() for s in args.model_labels.split(',') if s.strip()]
    if not (len(gslugs) == len(sslugs) == len(labels_in)):
        raise ValueError('game-slugs, solo-slugs and model-labels must be the same length.')

    pairs = load_pairs(args.game_scenarios_csv)

    per_model, labels = {}, {}
    for g, s, lab in zip(gslugs, sslugs, labels_in):
        conds = collect({'game_slug': g, 'solo_slug': s},
                        args.solo_results_dir, args.game_results_dir, pairs)
        if not conds:
            print(f'skipping {lab}: no result CSVs found yet')
            continue
        per_model[g] = conds
        labels[g] = lab
        have = ', '.join(k for k, _, _ in CONDITIONS if k in conds)
        print(f'{lab}: {len(conds)}/{len(CONDITIONS)} conditions ({have})')

    if not per_model:
        raise SystemExit('No model had any results; nothing to plot.')

    rows = []
    for mkey, conds in per_model.items():
        for ckey, clabel, is_steer in CONDITIONS:
            for pair, (hit, tot) in sorted(conds.get(ckey, {}).items()):
                p, lo, hi = wilson(hit, tot)
                rows.append({
                    'model': labels[mkey],
                    'condition': ckey,
                    'domain': 'solo' if ckey.startswith('solo') else 'game',
                    'steered': is_steer,
                    'value_first': pair[0],
                    'value_second': pair[1],
                    'n': tot,
                    'n_first_chosen': hit,
                    'preference_rate': p,
                    'ci_lower': lo,
                    'ci_upper': hi,
                })
    os.makedirs(args.output_dir, exist_ok=True)
    table_path = _out(args, 'value_preferences_across_conditions', 'csv')
    pd.DataFrame(rows).to_csv(table_path, index=False)

    out = plot(per_model, labels,
               _out(args, 'value_preferences_across_conditions', 'png'))
    print(f'wrote {out}')
    print(f'wrote {table_path} ({len(rows)} rows)')

    bt_png, bt_rows = plot_bt(
        per_model, labels,
        _out(args, 'bradley_terry_by_condition_vs_solo', 'png'),
        reference=args.reference_condition)
    bt_csv = _out(args, 'bradley_terry_by_condition_vs_solo', 'csv')
    pd.DataFrame(bt_rows).to_csv(bt_csv, index=False)
    print(f'wrote {bt_png}')
    print(f'wrote {bt_csv} ({len(bt_rows)} rows)')

    # One Bradley-Terry panel per model as well, so each model's page section can
    # carry its own baseline-and-steered view against its own solo reference
    # rather than sending the reader to a shared three-panel figure.
    for mkey in per_model:
        single, single_label = {mkey: per_model[mkey]}, {mkey: labels[mkey]}
        out = _out(args, f'bradley_terry_by_condition_vs_solo_{mkey}', 'png')
        plot_bt(single, single_label, out, reference=args.reference_condition)
        print(f'wrote {out}')

    steer_png, steer_rows = plot_steering_effect(
        per_model, labels,
        _out(args, 'steering_effect_by_condition', 'png'))
    steer_csv = _out(args, 'steering_effect_by_condition', 'csv')
    pd.DataFrame(steer_rows).to_csv(steer_csv, index=False)
    print(f'wrote {steer_png}')
    print(f'wrote {steer_csv} ({len(steer_rows)} rows)')


if __name__ == '__main__':
    main()
