"""Appendix figure: value-preference levels by fabricated remembered state.

Regenerates ``reproduce/out/figures/state_levels_appendix.pdf``: level plots
for the two harmlessness contests, all four criterion arms, including
usefulness which the body panel omits.

Reads ``stats.json`` as written by ``state_levels.py``.

Styled to match the other figures (2026-09-05): uppercase bold panel letters
set flush left, full axis frames, the same grid and tick weights, hollow
markers with a colored edge, and the legend set directly under the panels via
constrained layout. Arms keep their own palette (they are not the
representations that blue, vermillion and green denote elsewhere), drawn from
the same Okabe--Ito set.

Markers carry no connecting lines: the four remembered states are discrete
conditions rather than an ordered continuum, matching the convention used by
every other figure in the paper.

Run: python reproduce/make_state_levels_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Written by reproduce/state_levels.py.
STATS = os.path.join(HERE, 'out', 'state_levels', 'stats.json')
OUT = os.path.join(HERE, 'out', 'figures', 'state_levels_appendix.pdf')

STATE_ORDER = ('fresh', 'none', 'one', 'both')

# Okabe--Ito colors not already used for the three action representations.
CCOLOR = {'no criterion': '0.3', 'payoff': '#CC79A7',
          'alignment': '#E69F00', 'usefulness': '#56B4E9'}
CMARK = {'no criterion': 'D', 'payoff': 'o', 'alignment': 's', 'usefulness': 'v'}
CARMKEY = {'no criterion': 'none', 'payoff': 'payoff',
           'alignment': 'alignment', 'usefulness': 'usefulness'}
ARM_ORDER = ['no criterion', 'payoff', 'alignment', 'usefulness']
DODGE = {'no criterion': -0.24, 'payoff': -0.08,
         'alignment': 0.08, 'usefulness': 0.24}

TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, MARKER_SIZE = 9.0, 8.0, 7.6, 8.0, 5.0

PANELS = [
    ('harmlessness_vs_helpfulness', 'A  Harmlessness vs helpfulness',
     'P(harmless beats helpful)',
     ['Fresh\nstart', 'Neither\nchose\nharmless', 'One\nchose\nharmless',
      'Both\nchose\nharmless']),
    ('harmlessness_vs_honesty', 'B  Harmlessness vs honesty',
     'P(harmless beats honest)',
     ['Fresh\nstart', 'Neither\nchose\nharmless', 'One\nchose\nharmless',
      'Both\nchose\nharmless']),
]

with open(STATS, encoding='utf-8') as f:
    SL = json.load(f)['state_levels']

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout='constrained')
for ax, (pkey, title, ylab, ticklabels) in zip(axes, PANELS):
    for arm in ARM_ORDER:
        pts = [SL[CARMKEY[arm]][pkey][st] for st in STATE_ORDER]
        x = np.arange(len(STATE_ORDER)) + DODGE[arm]
        ys = [p['mean_rate'] for p in pts]
        lo = [p['mean_rate'] - p['ci_lower'] for p in pts]
        hi = [p['ci_upper'] - p['mean_rate'] for p in pts]
        ax.errorbar(x, ys, yerr=[lo, hi], color=CCOLOR[arm], marker=CMARK[arm],
                    markerfacecolor='white', markeredgewidth=1.0,
                    markersize=MARKER_SIZE, linestyle='none', linewidth=1.1,
                    capsize=2.0, zorder=4)
    ax.set_xticks(range(len(STATE_ORDER)), ticklabels)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.5, 1.02)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel(ylab, fontsize=LABEL_SIZE)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=TITLE_SIZE)
    ax.grid(axis='y', color='0.9', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

handles = [
    mlines.Line2D([], [], color=CCOLOR[arm], marker=CMARK[arm], linestyle='none',
                  markerfacecolor='white', markersize=MARKER_SIZE, label=arm)
    for arm in ARM_ORDER
]
fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False,
           fontsize=LEGEND_SIZE, handletextpad=0.5, columnspacing=1.6)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, format='pdf', bbox_inches='tight',
            metadata={'Creator': 'make_state_levels_figure.py', 'CreationDate': None})
fig.savefig(OUT.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
print('wrote', os.path.abspath(OUT))
