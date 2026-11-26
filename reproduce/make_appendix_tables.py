"""Generate the appendix data tables from the committed statistics artifact.

Writes ``reproduce/out/appendix_tables.tex``. Two kinds of table are produced, both read from ``stats.json`` so that no number in them is
typed by hand:

* per-model baseline win rates for every action representation and HHH outcome,
  which replaces the per-model baseline page of the figure appendix;
* the change in helpfulness under each concrete stake, split by the four
  remembered-history states, one table per action representation.

The history cells are pooled exploratory estimates with unadjusted intervals
and no multiplicity correction; the per-model corrected grid lives in
``history_state_grid.csv``.

Run: python reproduce/make_appendix_tables.py
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Mapping, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STATS = os.path.join(REPO, 'analysis', 'reanalysis_20260904_01', 'stats.json')
OUT = os.path.join(HERE, 'out', 'appendix_tables.tex')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

REPRESENTATIONS = ('menu', 'open_undisclosed', 'open_disclosed')
REP_LABELS = {
    'menu': 'Menu',
    'open_undisclosed': 'Open--undisclosed',
    'open_disclosed': 'Open--disclosed',
}
VALUES = ('harmlessness', 'helpfulness', 'honesty')
VALUE_SHORT = {'harmlessness': 'Harm', 'helpfulness': 'Help', 'honesty': 'Hon'}
STAKES = ('payoff', 'alignment', 'usefulness')
STATES = ('fresh', 'none', 'one', 'both')
STATE_LABELS = {
    'fresh': 'Fresh',
    'none': 'Neither',
    'one': 'One',
    'both': 'Both',
}


def rate(x: float) -> str:
    return f'{x:.3f}'


def signed(x: float) -> str:
    # Math mode so a leading minus is a true minus sign rather than a hyphen.
    return f'${x:+.3f}$'


def load(path: str) -> Mapping[str, Any]:
    with io.open(path, encoding='utf-8') as f:
        return json.load(f)


def baseline_table(levels: Sequence[Mapping[str, Any]]) -> list[str]:
    """Per-model baseline win rate for every representation and outcome."""
    pooled = {(r['representation'], r['value']): r for r in levels}
    per_model: dict[str, dict[tuple, float]] = {}
    labels: dict[str, str] = {}
    for r in levels:
        for m in r['per_model']:
            model = m['model']
            labels[model] = m['model_label']
            per_model.setdefault(model, {})[(r['representation'], r['value'])] = m['rate']

    head = ' & '.join(
        [''] + [f'\\textbf{{{VALUE_SHORT[v]}}}' for _ in REPRESENTATIONS for v in VALUES]
    )
    lines = [
        '\\begin{table}[htbp]',
        '\\centering\\small',
        '\\setlength{\\tabcolsep}{3.5pt}% 10 columns overflow the text width at the default 6pt',
        '\\caption{Baseline HHH win rates for every evaluated checkpoint, by elicitation',
        'format. A value\'s win rate is the share of answered contests',
        'involving that value in which its tagged action was chosen; Harm, Help and',
        'Hon abbreviate the three tagged outcomes. The final row is the equal-weight',
        'mean over checkpoints with its scenario-bootstrap interval, as plotted in',
        'Figure~\\ref{fig:stakes}A.}',
        '\\label{tab:baseline_by_model}',
        '\\begin{tabular}{l' + 'rrr' * len(REPRESENTATIONS) + '}',
        '\\toprule',
        ' & \\multicolumn{3}{c}{\\textbf{Menu}} & \\multicolumn{3}{c}{\\textbf{Open--undisclosed}}'
        ' & \\multicolumn{3}{c}{\\textbf{Open--disclosed}} \\\\',
        '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}',
        'Checkpoint' + head + ' \\\\',
        '\\midrule',
    ]
    for model in sorted(per_model, key=lambda m: labels[m].lower()):
        cells = [rate(per_model[model][(rep, v)]) for rep in REPRESENTATIONS for v in VALUES]
        lines.append(f'{labels[model]} & ' + ' & '.join(cells) + ' \\\\')
    lines.append('\\midrule')
    means = [rate(pooled[(rep, v)]['mean_rate']) for rep in REPRESENTATIONS for v in VALUES]
    lines.append('\\textbf{Pooled mean} & ' + ' & '.join(f'\\textbf{{{m}}}' for m in means) + ' \\\\')
    cis = [f"\\tiny[{pooled[(rep, v)]['ci_lower']:.2f},\\,{pooled[(rep, v)]['ci_upper']:.2f}]"
           for rep in REPRESENTATIONS for v in VALUES]
    lines.append('\\quad 95\\% CI & ' + ' & '.join(cis) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return lines


def history_helpfulness_tables(history: Sequence[Mapping[str, Any]]) -> list[str]:
    """Helpfulness change under each stake, split by remembered-history state."""
    by_key = {(r['representation'], r['stake'], r['state']): r
              for r in history if r['value'] == 'helpfulness'}
    lines: list[str] = []
    for rep in REPRESENTATIONS:
        label = REP_LABELS[rep]
        lines += [
            '\\begin{table}[htbp]',
            '\\centering\\small',
            f'\\caption{{Change in the helpfulness win rate under each concrete stake in the',
            f'{label} format, by remembered-history state. Each cell is',
            'stake minus baseline within that state, with its unadjusted 95\\%',
            'percentile interval. \\emph{Fresh} states that the interaction has just',
            'started; \\emph{Neither}, \\emph{One}, and \\emph{Both} count how many of the',
            "two remembered actions favored the focal contest's first-named value.",
            'These pooled cells are exploratory and carry no multiplicity correction;',
            'the corrected per-model grid is released with the data.}',
            f'\\label{{tab:history_help_{rep}}}',
            '\\begin{tabular}{lrrrr}',
            '\\toprule',
            'Stake & ' + ' & '.join(STATE_LABELS[s] for s in STATES) + ' \\\\',
            '\\midrule',
        ]
        for stake in STAKES:
            est, ci = [], []
            for state in STATES:
                r = by_key[(rep, stake, state)]
                est.append(signed(r['pooled_delta']))
                ci.append(f"\\tiny[${r['ci_lower']:+.3f}$,\\,${r['ci_upper']:+.3f}$]")
            lines.append(f'{stake.title()} & ' + ' & '.join(est) + ' \\\\')
            lines.append(' & ' + ' & '.join(ci) + ' \\\\')
            if stake != STAKES[-1]:
                lines.append('\\addlinespace')
        lines.append('\\midrule')
        ns = [str(by_key[(rep, 'payoff', s)]['n_intersection_total']) for s in STATES]
        lines.append('Paired items ($n$) & ' + ' & '.join(ns) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return lines


def main() -> None:
    if not os.path.exists(STATS):
        raise SystemExit(f'missing statistics artifact: {STATS}')
    stats = load(STATS)

    lines = [
        '% Generated from stats.json by make_appendix_tables.py; do not edit.',
        '',
    ]
    lines += baseline_table(stats['representation_levels'])
    lines += history_helpfulness_tables(stats['history_stake_effects'])

    with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))
    print(f'wrote {OUT}')
    print(f'  {len(lines)} lines, 4 tables')


if __name__ == '__main__':
    main()
