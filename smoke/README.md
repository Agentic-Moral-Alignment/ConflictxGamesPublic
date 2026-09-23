# Live smoke tests

End-to-end checks of the **game pipeline** against the real institutional
gateway. Unlike the offline suite in [`../tests/`](../tests), these spend API
calls: roughly **100–150 requests** for a full run on a 2-scenario corpus.

They are excluded from the normal `pytest` run (`testpaths = ["tests"]`) and
skip themselves when no key is configured, so they can never break CI.

## 1. Save your API keys

Create a file called `.env` in the repository root (it is gitignored — there
is deliberately **no** committed template, so a key can never be added to a
tracked file by accident):

```
OPENAI_BASE_URL=https://gateway.example.edu/api
GATEWAY_API_KEY_1=sk-...
GATEWAY_API_KEY_2=sk-...
GATEWAY_API_KEY_3=sk-...
```

The suite spreads the keys across pipeline stages (generation, filtering, each
evaluation mode, analysis) so one key hitting a rate limit cannot stall the
whole run. Only `GATEWAY_API_KEY_1` is required; with fewer keys the
rotation simply wraps. Duplicate keys are collapsed.

Check what the loader sees (prints only masked prefixes):

```bash
python -c "import sys; sys.path.insert(0,'smoke'); from conftest import api_keys; print(len(api_keys()), 'keys')"
```

**Already have a key in your conda environment?** It still works. When no
`.env` is present the loader falls back to `OPENAI_API_KEY` /
`OPENAI_BASE_URL` from the environment, which is how the repository was
already configured. To persist keys that way instead:

```bash
conda env config vars set OPENAI_API_KEY=sk-...
conda env config vars set OPENAI_BASE_URL=https://gateway.example.edu/api
conda activate conflictxgame     # reactivate for the change to take effect
```

Never pass a key on the command line — it lands in your shell history.

## 2. Run

```bash
python -m pytest smoke/ -v -s          # everything, ~100-150 API calls
python -m pytest smoke/test_01_gateway.py -v -s   # credentials only, ~7 calls
```

`-s` is worth passing: the tests print sample opening messages, completions,
parse rates, the steering-impact table, and — for the balanced run — the
calls spent per accepted scenario and how the outline pre-screen compares
against the judge.

Run one stage at a time (files are ordered by dependency):

| File | What it checks | Approx. calls |
| --- | --- | --- |
| `test_01_gateway.py` | Every key authenticates; all four role models answer; no `provider:` prefix reaches a filename | ~7 |
| `test_02_generate_filter.py` | Generation defaults to one-shot 2x2, specs validate, filtered CSV is a superset of the upstream schema | ~25 |
| `test_03_mcq.py` | MCQ + Likert + outcome rankings parse | ~20 |
| `test_04_open.py` | Open-ended mode: user model writes an opening, target answers with no scenario context, judge classifies it; prompt cache is reused | ~25 |
| `test_05_play_memk.py` | Live play with `--memory-k 1`, both `--repeat-mode same` and `sequence` | ~20 |
| `test_06_steering.py` | Appendix J steering prompt, `--target-ranking`, and `steering_impact.csv` | ~40 |
| `test_07_balanced.py` | The quota loop: per-pair coverage, resume, `--status`, plus the cost block and the pre-screen calibration table | ~40 |

## 3. Artifacts

Everything is written to `smoke/_artifacts/` (gitignored) — **never** to
`data/`, so generated smoke data cannot mix with the real corpora.
`test_02` asserts this explicitly.

```
smoke/_artifacts/
├── data/HHH/
│   ├── anthropic.claude-sonnet-5_games.csv         # filtered corpus
│   ├── anthropic.claude-sonnet-5_open_prompts.json # cached openings
│   └── games/                                      # raw specs + coverage
├── results/            # unsteered mcq / open / play
├── results_steered/    # steered runs (separate: resume dedup is by id+mode)
└── analysis/           # steering_impact.csv, value rankings
```

The corpus is generated **once** and reused: re-running the suite while
iterating costs only the evaluation calls. Delete `smoke/_artifacts/` to force
a fresh corpus.

## Models used

All gateway-served, with the `openai:` routing prefix:

| Role | Model |
| --- | --- |
| Generator / target | `openai:anthropic.claude-sonnet-5` |
| Filter judge / open-mode judge / simulated user | `openai:openai.gpt-5.6-luna` |

Change them at the top of [`conftest.py`](conftest.py).
