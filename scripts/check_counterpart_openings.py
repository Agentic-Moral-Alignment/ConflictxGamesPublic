"""Report how many cached opening messages name the counterpart.

Read-only acceptance gate for the counterpart-disclosed open-ended arm
(``--open-disclose-counterpart``). The baseline cache names the other
decision-maker in 2 of 150 openings and never calls it an AI; the disclosed
cache should name it in nearly all of them. Run this after the first arm
writes the new cache, before spending on the remaining arms:

    python scripts/check_counterpart_openings.py \
        --cache data/HHH_new_paper_bothAI_anthropic_batch_20260826_01/games/\
open_prompts_balanced50_counterpart_paid-gpt-5.6-luna.json

Exits non-zero when the rate is below ``--min-rate`` (default 125/150), so a
chain script can stop rather than pay for the rest of the sweep.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from conflictxgame.games import open_ended  # noqa: E402
from conflictxgame.games.evaluate_games import (  # noqa: E402
    load_game_scenarios,
    select_game_seats,
)

DEFAULT_CSV = ("data/HHH_new_paper_bothAI_anthropic_batch_20260826_01/"
               "claude-sonnet-5_games.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True,
                        help="Opening-message cache JSON to audit")
    parser.add_argument("--scenarios-csv", default=DEFAULT_CSV,
                        help=f"Game scenario CSV (default {DEFAULT_CSV})")
    parser.add_argument("--pair-limit", type=int, default=50,
                        help="Seats per value pair, matching the eval configs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-rate", type=float, default=125 / 150,
                        help="Fail below this share of openings naming the counterpart")
    parser.add_argument("--show", type=int, default=0,
                        help="Print this many non-compliant openings")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    scenarios = load_game_scenarios(args.scenarios_csv, True)
    selected = select_game_seats(scenarios, args.pair_limit, args.seed)
    cache = open_ended.load_open_prompt_cache(args.cache)
    if not cache:
        print(f"No cache entries read from {args.cache}")
        return 2

    named, missing_from_cache, offenders = 0, 0, []
    for scenario_id, spec, _row in scenarios:
        for seat in range(spec.config.n_players):
            if (str(scenario_id), seat) not in selected:
                continue
            entry = cache.get(open_ended.cache_key(scenario_id, seat))
            opening = (entry or {}).get("opening", "")
            if not entry:
                missing_from_cache += 1
                continue
            if open_ended.counterpart_named(spec, seat, opening):
                named += 1
            else:
                others = ", ".join(
                    spec.players[i].name
                    for i in range(spec.config.n_players) if i != seat
                )
                offenders.append((scenario_id, seat, others, opening))

    total = len(selected)
    rate = named / total if total else 0.0
    print(f"Cache: {args.cache}")
    print(f"Counterpart named in {named}/{total} selected openings ({rate:.1%}).")
    if missing_from_cache:
        print(f"{missing_from_cache} selected seat(s) had no cache entry yet.")
    for scenario_id, seat, others, opening in offenders[: args.show]:
        print(f"\n-- {scenario_id} seat {seat}: expected mention of {others}")
        print("   " + opening[:400].replace("\n", "\n   "))
    if rate < args.min_rate:
        print(f"\nFAIL: below the {args.min_rate:.1%} gate.")
        return 1
    print("\nOK: gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
