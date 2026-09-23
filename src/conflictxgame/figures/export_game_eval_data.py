"""The game scenarios actually evaluated, one row per scenario-seat.

The results CSVs record what each model *did*; this records what it was
*asked*. Reviewers checking whether a value pair was fairly posed, or whether
an action menu leans on one side, need the inputs rather than the decisions.

The evaluated unit is a **scenario-seat**, not a scenario. One scenario seats
several players, each carrying its own value pair, and ``pair_limit`` selects
exactly that many seats per pair -- so a scenario can contribute one seat, two,
or none. Exporting whole scenarios would misrepresent the corpus.

Selection is not reimplemented here. :func:`load_game_scenarios` and
:func:`select_game_seats` are imported from the evaluator, so the export is the
evaluated set by construction rather than by a parallel implementation that
could drift. ``--verify-against`` re-reads a results CSV and asserts the two
seat sets match exactly, which catches a config pointed at the wrong corpus.

Start: bash scripts/HHH_new_pipeline.sh export-game-eval-data
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Sequence

import pandas as pd

from conflictxgame.common.config import add_config_argument, parse_with_config
from conflictxgame.games.evaluate_games import (
    load_game_scenarios,
    select_game_seats,
)

#: Scenario-level columns copied from the source CSV when present. These
#: describe the cell a scenario instantiates, not the seat, so they repeat
#: across the seats of one scenario.
SCENARIO_COLUMNS = (
    'game_id', 'cell_key', 'structure', 'n_players', 'arities', 'horizon',
    'value_set', 'value_tags', 'agent_kinds', 'agent_mix', 'payoff_matrix',
    'generating_model', 'context', 'user_prompt',
)


def seat_rows(scenarios: list, selected: set) -> list:
    """One dict per selected (scenario_id, seat), in corpus order."""
    rows = []
    for scenario_id, spec, source in scenarios:
        for seat, player in enumerate(spec.players):
            if (str(scenario_id), seat) not in selected:
                continue
            # Sorted, stripped -- the exact key select_game_seats grouped on,
            # so a reader can regroup the export and get the same buckets.
            pair = tuple(sorted(str(v).strip() for v in player.value_pair))
            row = {
                'scenario_id': str(scenario_id),
                'seat': seat,
                'value_pair': '|'.join(pair),
                'player_name': player.name,
                'agent_kind': player.agent_kind,
                'arity': player.arity,
                'persona': player.persona,
                'role_description': player.role_description,
                'description': spec.description,
            }
            for i, value in enumerate(pair):
                row[f'value_{i + 1}'] = value
            for i, action in enumerate(player.actions):
                row[f'action_{i}_label'] = action.label
                row[f'action_{i}_description'] = action.description
                row[f'action_{i}_value_tag'] = action.value_tag
            for column in SCENARIO_COLUMNS:
                if column in source.index:
                    row[column] = source[column]
            rows.append(row)
    return rows


def verify(df: pd.DataFrame, results_csv: str) -> str:
    """Compare the exported seat set with one a model was actually run on."""
    got = pd.read_csv(results_csv)
    missing = {'scenario_id', 'seat'} - set(got.columns)
    if missing:
        raise ValueError(f'{results_csv} lacks {sorted(missing)}; cannot verify.')
    theirs = {(str(s), int(t)) for s, t in zip(got['scenario_id'], got['seat'])}
    ours = {(str(s), int(t)) for s, t in zip(df['scenario_id'], df['seat'])}
    if ours == theirs:
        return f'verified: {len(ours)} seats identical to {os.path.basename(results_csv)}'
    only_ours, only_theirs = ours - theirs, theirs - ours
    raise SystemExit(
        f'MISMATCH against {os.path.basename(results_csv)}: '
        f'{len(only_ours)} exported seats absent from results, '
        f'{len(only_theirs)} evaluated seats absent from the export. '
        f'Examples: {sorted(only_ours)[:3]} / {sorted(only_theirs)[:3]}'
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_config_argument(parser)
    parser.add_argument('--scenarios-dir', required=True,
                        help='Directory holding the filtered *_games.csv files')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--pair-limit', type=int, default=50,
                        help='Seats per value pair; must match the eval configs')
    parser.add_argument('--seed', type=int, default=0,
                        help='Selection seed; must match the eval configs')
    parser.add_argument('--filter', action='store_true', default=True,
                        help='Keep only rows with keep_scenario == True')
    parser.add_argument('--verify-against', default=None,
                        help='A results CSV whose seat set must match exactly')
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args, _d, _e = parse_with_config(build_parser(), argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = sorted(glob.glob(os.path.join(args.scenarios_dir, '*_games.csv')))
    if not paths:
        raise SystemExit(f'No *_games.csv under {args.scenarios_dir}.')

    scenarios = []
    for path in paths:
        found = load_game_scenarios(path, args.filter)
        print(f'  {os.path.basename(path):40s} {len(found):4d} scenarios kept')
        scenarios.extend(found)

    selected = select_game_seats(scenarios, args.pair_limit, args.seed)
    df = pd.DataFrame(seat_rows(scenarios, selected))
    if len(df) != len(selected):
        raise SystemExit(
            f'Selected {len(selected)} seats but built {len(df)} rows; '
            'the corpus likely holds duplicate scenario_ids.'
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f'\nwrote {args.output_path} ({len(df)} seats, {len(df.columns)} columns)')
    print(f'  {df["scenario_id"].nunique()} distinct scenarios')
    for pair, n in df['value_pair'].value_counts().sort_index().items():
        print(f'  {pair:28s} {n:4d} seats')
    if args.verify_against:
        print('  ' + verify(df, args.verify_against))


if __name__ == '__main__':
    main()
