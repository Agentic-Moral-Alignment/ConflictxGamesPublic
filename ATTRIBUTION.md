# Attribution

**ConflictxGame is a derivative work of [ConflictScope](https://github.com/andyjliu/conflictscope)** — the code release accompanying [*Generative Value Conflicts Reveal LLM Priorities*](https://www.arxiv.org/abs/2509.25369) (Liu, Ghate, Diab, Fried, Kasirzadeh & Kleiman-Weiner; ICLR 2026). ConflictScope is MIT-licensed, Copyright (c) 2025 Andy Liu; that notice is retained in [`LICENSE`](LICENSE) and continues to cover the inherited code. The ConflictScope authors have not endorsed, and are not responsible for, the changes made here.

This file records **module-by-module provenance**: which of our modules are their code, which are split or adapted from it, and which are new. Upstream paths below refer to `https://github.com/andyjliu/conflictscope/blob/main/<path>`.

## Citation

If you use this repository, cite the original paper — the evaluation methodology it extends is theirs:

```bibtex
@article{liu2025generative,
  title={Generative Value Conflicts Reveal LLM Priorities},
  author={Liu, Andy and Ghate, Kshitish and Diab, Mona and Fried, Daniel and Kasirzadeh, Atoosa and Kleiman-Weiner, Max},
  journal={arXiv preprint arXiv:2509.25369},
  year={2025}
}
```

## Code provenance

### `conflictxgame.solo` — the upstream dyadic pipeline

Carried over from ConflictScope essentially unchanged: imports were repathed for the package layout and Google-style docstrings were added, but the generation, filtering, evaluation, and analysis logic is theirs.

| Our module | Upstream source | Derivation |
| --- | --- | --- |
| `solo/generate_scenarios.py` | `src/generate_scenarios.py` | Unchanged (imports repathed, docstrings added) |
| `solo/filter_scenarios.py` | `src/filter_scenarios.py` | Unchanged (imports repathed, docstrings added) |
| `solo/evaluate_models.py` | `src/evaluate_models.py` | Unchanged (imports repathed, docstrings added) |
| `solo/simulated_conversation.py` | `src/simulated_conversation.py` | Unchanged apart from the POSIX-lock shim and moving the prompt text to `solo/prompts.py`. Carries the paper's Appendix C.5 (simulated-user opening message) and C.6 (interaction judge) prompts verbatim |
| `solo/analyze_experiment.py` | `src/analyze_experiment.py` | Unchanged (imports repathed, docstrings added) |
| `solo/prompts.py` | `src/utils.py` | Split out: the dyadic generation prompt templates (`CAUSAL_GRAPH_PROMPTS`, `STAGE_ONE_*`, `STAGE_TWO_*`), the legacy `VALUE_DICT`/`load_value_dict`, and the Appendix C.5 opening-message instructions moved here from `simulated_conversation.py` |

### `conflictxgame.common` — shared infrastructure

| Our module | Upstream source | Derivation |
| --- | --- | --- |
| `common/model_wrappers.py` | `src/model_wrappers.py` | Adapted: upstream client layer plus explicit `provider:` routing, OpenAI-compatible gateway handling, and filename-safe short model names |
| `common/utils.py` | `src/utils.py` | Split out: `parse_json`, `SYSTEM_PROMPT`, `MAX_LIKERT` |
| `common/values.py` | `src/utils.py` | Split out and extended: value-set loading plus per-agent-kind definition resolution and repo-root-relative path discovery |
| `common/portable_lock.py` | — | New: cross-platform advisory locking (msvcrt on Windows, fcntl elsewhere) replacing upstream's direct `fcntl` use |

### `conflictxgame.games` — the multi-agent extension

New code written for this fork. Several modules deliberately mirror the structure of their upstream counterparts so the two pipelines stay comparable; that correspondence is recorded here.

| Our module | Upstream correspondence | Derivation |
| --- | --- | --- |
| `games/game_structures.py` | — | New: game data model, payoff-tensor analysis, config enumeration |
| `games/scripted_strategies.py` | — | New: scripted opponents incl. zero-determinant strategies |
| `games/generate_game_scenarios.py` | `src/generate_scenarios.py` | New code; mirrors the upstream two-stage generation design (staged summaries → elaboration, embedding deduplication) over a configuration space instead of a value pair |
| `games/filter_game_scenarios.py` | `src/filter_scenarios.py` | New code; adapts the upstream six LLM-judge checks to n-player, k-action games. Its output CSV is a superset of the upstream scenario CSV schema |
| `games/game_engine.py` | `src/simulated_conversation.py`, `src/evaluate_models.py` | New code; the batched round stepping mirrors `ValueTester.test_scenarios_batch`, and the Likert elicitation prompt mirrors `evaluate_models.create_likert_prompt` |
| `games/evaluate_games.py` | `src/evaluate_models.py` | New code; mirrors the upstream MCQ/interactive split. `ModelClientManager` reimplements the upstream class (not imported: that module transitively required POSIX-only `fcntl`). `--mode open` reimplements the paper's open-ended evaluation (§3.3, Appendix C.5/C.6) for games |
| `games/open_ended.py` | `src/simulated_conversation.py` | New code; reimplements the Appendix C.5 user-prompt and C.6 judge patterns single-turn and generalized to k candidate actions |
| `games/steering.py` | — | New; the generated prompt structure follows the paper's Appendix J steering prompts |
| `games/pair_quota.py` | — | New: canonical value-pair keys, accepted-count quota tracking, deficit-aware value sampling, and per-pair recounting from a filtered CSV |
| `games/generate_balanced.py` | — | New: generate/filter/top-up orchestrator that fills a per-value-pair quota of *accepted* scenarios, so our corpus is uniform across pairs (the upstream corpora are not). Drives the existing generation and filter functions rather than reimplementing them |
| `games/prescreen.py` | — | New: one small screening call between the two generation stages, recording (and optionally acting on) whether an outline is already broken before the long stage-two call and the filter pass are spent on it. The filter judge is untouched and still decides what enters the corpus |
| `common/generation_quality.py` | `src/filter_scenarios.py` (criteria only) | New code; the checklists paraphrase the upstream filter dimensions so generation targets the properties the judge tests, ordered by which checks measurably reject the most scenarios. The judge itself is unmodified. Also provides construction guidance for the generation prompts, the targeted repair prompt, per-check failure rates from the `check_results` column the upstream corpora carry, and the pre-screen calibration table |

### Data, value sets, and prompts

| Path | Source | Notes |
| --- | --- | --- |
| `value_sets/HHH_new.json` | — | Reworded definitions of the three values in upstream `value_sets/HHH.json` |
| `value_sets/steering/HHH.txt` | paper Appendix J.2 | Steering system prompt transcribed verbatim; the other prompts in the folder are new |

ConflictScope's own datasets are not redistributed here.

## Reimplemented paper methods

Two methods from the paper are reimplemented for the game pipeline rather than inherited as code:

* **Open-ended evaluation** (§3.3, Appendix C.5/C.6) — a user LLM writes an opening message from the scenario and persona; the target model answers it with no other scenario context; a judge LLM decides which candidate action the completion resembles. Implemented in `games/open_ended.py` and `evaluate_games.py --mode open`.
* **Steerability evaluation** (§4.2, §5.3, Appendix J) — a system prompt stating a target value ranking, with alignment measured as the share of decisions matching that ranking and a normalized effect size `(a_steered - a_default) / (1 - a_default)`. Implemented in `games/steering.py` and the `--steer-prompt`/`--target-ranking` flags.
