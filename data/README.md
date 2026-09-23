# `data/` — what is upstream and what the pipeline writes

This directory holds two provenances, one subdirectory per value set
(`HHH/`, `modelspec/`, `personalprotective/`). Nothing here is regenerated
automatically; the pipeline only ever **adds or appends** files.

> **In this release, only the upstream ConflictScope files are present.**
> The generated game corpora described further down are *not* shipped — the
> tables below document the layout the pipeline creates when you run it, and
> the filenames the shipped configs expect. Run the generation stages and
> these paths populate themselves.

See [`../ATTRIBUTION.md`](../ATTRIBUTION.md) for full code and data provenance.

## Upstream ConflictScope files — do not regenerate or overwrite

From [andyjliu/conflictscope](https://github.com/andyjliu/conflictscope)
(MIT, © 2025 Andy Liu), released with
[arXiv:2509.25369](https://www.arxiv.org/abs/2509.25369):

| File | What it is |
| --- | --- |
| `<value_set>/claude-3-5-sonnet-latest.csv` | The paper's filtered dyadic scenario dataset for that value set (11 columns: `scenario_id`, `context`, `description`, `user_prompt`, `value1`, `value2`, `action1`, `action2`, `keep_scenario`, `check_results`, `generating_model`) |
| `<value_set>/prompts.json` | Cached simulated-user opening messages used by the dyadic interactive evaluation |

These are inputs to the solo pipeline (`conflictxgame.solo.*`) and are left
byte-for-byte as released.

## ConflictxGame files — written by the game pipeline (not shipped)

| Path | Written by | What it is |
| --- | --- | --- |
| `<value_set>/<gen_model>_games.csv` | `conflictxgame.games.filter_game_scenarios` | Filtered game scenarios. **Compounds**: each run appends new rows (deduplicated on `scenario_id`) to this one file per generating model. Its schema is a superset of the upstream scenario CSV — the same 11 columns (from player 1's perspective) plus `n_players`, `arities`, `structure`, `horizon`, `player2_values`, `payoff_matrix`, `agent_kinds`, `game_spec`, and more |
| `<value_set>/games/<gen_model>_games.json` | `conflictxgame.games.generate_game_scenarios` | Raw generated game specs, keyed by `game_id` |
| `<value_set>/games/<gen_model>_game_prompts.json` | same | The stage-1 and stage-2 generation prompts |
| `<value_set>/games/coverage_report.{json,csv}` | same (kept counts added by the filter) | Per configuration cell: target, generated, kept, notes |
| `<value_set>/games/pair_coverage.{json,csv}` | `conflictxgame.games.generate_balanced` | Per **value pair**: target, generated, kept, deficit, acceptance rate. Separate from `coverage_report`, which is per configuration cell |
| `<value_set>/<gen_model>_open_prompts.json` | `conflictxgame.games.evaluate_games --mode open --open-prompts` | Optional cache of the LLM-written opening messages, keyed `"<scenario_id>|<seat>"`, so every target model can be evaluated on identical prompts |

Both game CLIs default to these locations when `--output-dir` is omitted, so
`python -m conflictxgame.games.generate_game_scenarios -v HHH` and the filter
step land in `data/HHH/games/` and `data/HHH/` respectively.

## Why the two can share a directory

`evaluate_games` globs `*_games.csv` in its `--scenarios-dir` first and only
falls back to `*.csv` if that finds nothing, so pointing it at `data/HHH`
picks up **only** our game CSVs and never the upstream
`claude-3-5-sonnet-latest.csv`. `analyze_games` additionally skips any CSV
without a `game_spec` column. Keep the `_games.csv` suffix on generated game
data for this reason.

A note on size: open-ended evaluation results (written to `results/`, not
here) store the full user prompt and model completion per row for
auditability, so those CSVs grow much larger than the MCQ ones.

## Value-pair balance, and why it matters for comparison

The upstream corpora are **not** uniform across value pairs. Every pair was
generated in equal numbers, but pairs survive filtering at very different
rates, so the accepted scenarios are lopsided — in HHH, 576
harmlessness-helpfulness against 306 harmlessness-honesty and 227
helpfulness-honesty:

```
python -m conflictxgame.common.generation_quality --csv data/HHH/claude-3-5-sonnet-latest.csv
```

That matters when comparing a model's behavior in the generated games against
its behavior in the paper's scenarios: if the two corpora ask about different
conflicts in different proportions, a difference in the fitted value ranking
confounds behavior with composition. Two facilities address it, and they
compose:

* `conflictxgame.games.generate_balanced` generates until every value pair has
  the same number of **accepted** scenarios, so the generated corpus is uniform
  by construction. The filtered `<gen_model>_games.csv` is its resume state — it
  recounts from that file, so re-running tops up rather than restarting.
* `analyze_games --pair-weights match --reference-csv <upstream csv>`
  resamples the evaluation's comparisons to reproduce the upstream per-pair
  mix, writing `*_reweighted.csv` **alongside** the uniform results. One
  evaluation run therefore yields both readings, and the difference between
  them is attributable to composition alone.

`scripts/HHH_reproduction.sh` runs the whole comparison end to end.
