# ConflictxGame

**ConflictxGame** is research infrastructure for **value-conflict evaluation of language models**. It is a fork of [ConflictScope](https://github.com/andyjliu/conflictscope) — the pipeline released with [Generative Value Conflicts Reveal LLM Priorities](https://www.arxiv.org/abs/2509.25369) (Liu et al., 2025) — that keeps the original single-assistant pipeline intact and adds a parallel **game-theoretic** pipeline: instead of one assistant choosing between two value-laden actions, two decision-makers with interdependent outcomes each resolve their own value conflict.

> **Fork notice.** The single-assistant pipeline (`conflictxgame.solo`), the shipped value sets, and the upstream data under `data/` originate in [andyjliu/conflictscope](https://github.com/andyjliu/conflictscope) and are used under the original MIT license. See [Attribution](#attribution) and [Citation](#citation).

---

## The takeaway

The repository was built to answer a **measurement-methodology** question: *how much of a reported LLM value trade-off is a property of the model, and how much is a property of the way you asked?*

Ten open-weight models were evaluated on an identical set of 150 dilemmas through two different elicitation procedures — forced multiple choice, and free text with no scenario context — and under several system-prompt and memory manipulations. The headline results:

* **How you ask matters more than what you tell the model.** Moving the same corpus from forced choice to free text shifts the measured helpfulness win rate by **+0.25** and harmlessness by **−0.17**, consistent in direction across 10 of 10 models. **This exceeds every system-prompt effect tested by a factor of three or more.**
* **System-prompt framing moves behavior, but modestly.** Telling a model its deployment will be judged on accumulated payoff reduces harmlessness-tagged choices by **−0.069**; an alignment-framed criterion raises them by **+0.047**. Both survive multiple-comparison correction on pooled data, but are individually distinguishable in only a minority of models.
* **Ordering is more stable than levels.** Under forced choice, all ten models produce the same implied ordering (harmlessness > honesty > helpfulness), and the HHH steering prompt changes it for none of them. Under free text the ordering holds for 7 of 10.

**The practical implication:** a published claim of the form *"model X prioritizes harmlessness over honesty"* is under-specified unless it names the elicitation procedure that produced it. Benchmark scores for value trade-offs are not procedure-independent properties of a model.

**Read the paper for the actual findings.** This README states them only at the level above. The numbers here are point estimates whose intervals, corrections, and — importantly — limitations live in the manuscript. The largest known measurement threat is that a single unvalidated judge model scores every free-text condition; the paper's Limitations section is required reading before quoting any figure.

### Why the comparisons are interpretable

The one design property worth stating outside the paper: arms replay **identical cached stimuli at identical temperatures over the same balanced item set**. Differences between arms are therefore attributable to the manipulation rather than to sampling, temperature, or corpus composition. Every arm is defined by a checked-in YAML config, and every free-text item retains its opening message, the model's completion, and the judge's reasoning, so a run you produce is auditable item by item rather than only from summary tables.

### What this repository contains

**This is a code-and-configuration release.** It ships the two pipelines, the
complete set of run configurations that defined every reported arm, the value
sets and steering prompts, and the upstream ConflictScope reference data. It
does **not** ship the generated scenario corpora, the per-item evaluation
records, or the derived analysis tables and figures.

The practical consequence, stated plainly: **the reproduction path is to re-run
the pipeline, not to re-check our arithmetic.** Reproducing the reported numbers
means generating a corpus and running the evaluation arms against live model
APIs, which costs real money. The configs under `configs/` are the auditable
record of exactly what was run — model roles, temperatures, seeds, item counts —
so what you regenerate is defined precisely, but the numbers in the paper cannot
be verified from files in this repository alone.

Paths inside the shipped configs point at corpus files under `data/` and
`results/` that this release does not contain. They are the paths the pipeline
*writes* when you run it; generate a corpus first and the configs resolve.

### The corpus the paper used

Claude Sonnet 5 generated 503 game scenarios; 117 passed the LLM-judge filter. From those, **150 scenario-seats** were selected — one seat of one scenario facing one value pair, drawn from 102 distinct scenarios, exactly 50 per HHH value pair. Selection is deterministic from `sha256(seed | scenario_id | seat)`, so every model and every arm sees the same seats. Excess kept scenarios stay in the corpus and are never deleted; balancing happens at selection time.

---

## Setup

Python **3.12+** is required. `uv.lock` is the ground truth for this project's environment; [uv](https://docs.astral.sh/uv/) is the recommended installer.

```bash
uv sync --extra dev
```

That creates `.venv/` with the exact locked dependency graph and installs the project in editable mode. Prefix commands with `uv run`, or activate the venv directly.

> **Windows: clone to a short path.** Config filenames encode the full arm
> specification, and the longest repo-relative path is 161 characters. Windows'
> default 260-character `MAX_PATH` therefore leaves under 100 characters for the
> directory you clone into — `C:\src\` is safe, a deep path under
> `Users\<name>\Documents\...` may not be. Symptoms are `FileNotFoundError`
> on config files that plainly exist. Either clone shallow or enable long paths
> (`git config --system core.longpaths true`).

> **Every dependency is pinned to an exact version** (`anthropic==0.45.2`,
> `vllm==0.9.1`, and so on). That is deliberate — it is what makes a run
> reproducible — but it ages badly: the further you are from the release date,
> the more likely a pinned wheel is unavailable for your platform or conflicts
> with your toolchain. If resolution fails, relax the pins in `pyproject.toml`
> rather than assuming the repository is broken, and expect that doing so may
> change results at the margin. `requirements.txt` is generated from `uv.lock`;
> use one or the other, never a hand-edited mix.

The default install covers both pipelines end to end — generation, filtering, evaluation, analysis — including the figure and Bradley–Terry stack and the sentence-transformers embeddings used for deduplication (`-d`).

### Credentials

Credentials are read from the process environment (there is no `.env` loader). **Never pass a key on the command line** — it lands in shell history.

```bash
export OPENAI_API_KEY=<>          # Windows PowerShell: $env:OPENAI_API_KEY = "<>"
export ANTHROPIC_API_KEY=<>
```

### Verify the install

```bash
uv run pytest tests/                                   # offline, no API keys needed
uv run python -m conflictxgame.games.evaluate_games --help
```

Everything is invoked as a module CLI: `python -m conflictxgame.<subpackage>.<module> --help`.

<details>
<summary><b>Without uv: pip + venv, or conda</b></summary>

`requirements.txt` is exported from `uv.lock` and pins the same core + dev graph.

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
# ...or: conda create -n conflictxgame python=3.12 && conda activate conflictxgame

pip install -r requirements.txt
pip install -e . --no-deps      # --no-deps: requirements.txt already pinned them
```

The package must be installed in the *active* interpreter; a bare `python` from outside the environment fails with `ModuleNotFoundError: No module named 'conflictxgame'`.
</details>

<details>
<summary><b>Changing dependencies</b></summary>

`uv.lock` is the single source of truth. Edit `pyproject.toml`, then regenerate the lock and its export, and commit all three together so they never drift:

```bash
uv lock
uv export --no-hashes --no-emit-project --extra dev --format requirements-txt -o requirements.txt
```

Never hand-edit `requirements.txt`.
</details>

<details>
<summary><b>Local open-weight inference (optional)</b></summary>

`uv sync --extra vllm` adds [vLLM](https://github.com/vllm-project/vllm). It has no Windows wheel — skip it there; nothing outside local open-weight inference needs it. New models and inference backends are added in `src/conflictxgame/common/model_wrappers.py`.
</details>

<details>
<summary><b>OpenAI-compatible gateways</b></summary>

Gateways that serve non-OpenAI models behind an OpenAI-compatible API are supported by pointing the OpenAI variables at the gateway:

```bash
export OPENAI_API_KEY=<gateway-key>
export OPENAI_BASE_URL=<gateway-url>
```

Prefix model ids with `openai:` so they route through the OpenAI-compatible client instead of being matched by substring — e.g. `-m openai:anthropic.claude-sonnet-5`. The prefix is stripped before the id reaches the API and never appears in an output filename.

Some providers constrain sampling — `gpt-5`-family models pin temperature to 1.0 regardless of what was requested. Clients expose both the requested and the effective temperature and log a warning when they differ, so the temperature actually used is recorded rather than assumed.
</details>

---

## Using the repository

### Run configurations

Every entry point in both pipelines takes `--config PATH`: a **YAML** file holding one flat mapping keyed by argparse destination name — `--kept-per-pair` is `kept_per_pair`, `-v1` is `value1`. JSON and JSONC remain supported for historical runs.

Three properties are worth knowing before you add a flag:

* **Precedence** is command line > config file > argparse default. The config records the run; the command line records the one-off deviation.
* **Validation is strict and happens before any client is built.** An unknown key, a wrong type, or a value outside `choices` aborts the run. This is deliberate: the alternative is silently falling back to a default that changes what the run measures.
* **Configs are the auditable record.** Filenames carry the model roles (`focal-`, `judge-`, `generator-`, `user-judge-`, `fallback-judge-`) so secondary models are explicit rather than buried in a log.

Give each evaluation mode a distinct `output_name`, as the shipped configs do. Evaluation resumes by reading its own output and skipping scenario ids already covered, so sharing one name makes the second run skip everything and produce nothing.

Key-by-key reference: [`docs/configs.rst`](docs/configs.rst).

### The dispatcher

`scripts/HHH_new_pipeline.sh` exposes one subcommand per stage:

```bash
bash scripts/HHH_new_pipeline.sh list        # print every command + model role, no API calls
bash scripts/HHH_new_pipeline.sh game-generate
bash scripts/HHH_new_pipeline.sh game-eval-mcq
bash scripts/HHH_new_pipeline.sh game-eval-open
bash scripts/HHH_new_pipeline.sh game-eval-play
bash scripts/HHH_new_pipeline.sh solo-eval-mcq
bash scripts/HHH_new_pipeline.sh solo-analyze   # no API calls; safe to re-run anytime
```

**Run `list` first when unsure** — it is free and shows exactly what a subcommand would do.

On Windows, use `& .\scripts\HHH_new_pipeline.ps1 list`; the wrapper selects Git Bash rather than the `bash` alias that may point at WSL.

### Reproducibility and arm-selection flags

Two flags govern whether a run and its analysis are reproducible, and both matter more than their size suggests:

* **`--seed`** (`games.evaluate_games`, `solo.evaluate_models`) seeds MCQ and ranking option order and pair sampling. Option order is **content-hashed**, so a resumed run reproduces the order the interrupted run used rather than reshuffling.
* **`--arm`** (`games.analyze_games`, `cross_analysis`) selects which experiment arm to analyze by its exact filename token; `baseline` selects the unsuffixed arm. One results directory holds every arm of an experiment and the `mode` column cannot tell them apart, so analysis **aborts when a directory holds more than one arm and none was named**, rather than silently pooling them into one condition. `--include-open-in-ranking` admits unprobed free-text decisions into value rankings.

### Testing

```bash
uv run pytest tests/          # offline suite, no API keys, all clients faked
uv run pytest smoke/ -v -s    # LIVE end-to-end tests, ~100-150 paid calls
```

The offline suite uses fakes from `tests/fakes.py` that still route through the usage ledger, so cost assertions run without credentials. `tests/test_config_files.py` parses every shipped config against its real parser — so a renamed CLI flag becomes a caught offline failure instead of a crash after paid credentials load. Smoke tests write only to `smoke/_artifacts/` (gitignored), asserted so generated test data can never mix into the real corpora.

On Windows, pass a repo-local basetemp: `--basetemp .pytest-<name>`. The default temp directory has produced access-denied errors.

### API documentation

```bash
uv run sphinx-build -b html docs docs/_build
```

Sphinx with Google-style docstrings, covering both pipelines.

---

## What is in the package

The code is an installable package (`src/conflictxgame/`):

* **`conflictxgame.solo`** — the upstream single-assistant pipeline, carried over essentially unchanged: `generate_scenarios`, `filter_scenarios`, `evaluate_models`, `analyze_experiment`, plus `prompts`, `simulated_conversation`, `topup_scenarios`.
* **`conflictxgame.games`** — the game-theoretic extension: the data model and payoff-tensor helpers (`game_structures`), generation and LLM-judge filtering (`generate_game_scenarios`, `filter_game_scenarios`, `prescreen`), evaluation and analysis (`evaluate_games`, `analyze_games`), the live-play engine and scripted opponents including zero-determinant strategies (`game_engine`, `scripted_strategies`), the free-text setting (`open_ended`), steering (`steering`), and the value-pair balancing layer (`pair_quota`, `generate_balanced`, `reweighting`, `quota_monitor`).
* **`conflictxgame.common`** — shared infrastructure: model clients and gateway routing (`model_wrappers`), the `--config` system (`config`), value-set loading (`values`), experiment-arm filename parsing (`arm_names`), shared statistical estimators (`stats`), acceptance-rate helpers (`generation_quality`), evaluation-awareness probes (`eval_awareness*`), reasoning capture, and cross-platform file locking (`portable_lock`).
* **`conflictxgame.figures`** — standalone figure and table CLIs that read the result CSVs and write a figure *plus the CSV table behind it*, so every plotted number stays traceable.
* **`conflictxgame.cross_analysis`** — a top-level module comparing value rankings across the two pipelines, since it belongs to neither.

Deeper documentation lives in [`docs/`](docs/): [`game_extension.rst`](docs/game_extension.rst) for the game pipeline, [`solo.rst`](docs/solo.rst) for the upstream one, [`cross_pipeline.rst`](docs/cross_pipeline.rst) for the comparison, [`generation_tuning.rst`](docs/generation_tuning.rst) for generation cost, and [`data/README.md`](data/README.md) for how generated data sits alongside the upstream datasets.

### The two pipelines in brief

**Single-assistant (upstream).** Generate with `solo.generate_scenarios`, filter with `solo.filter_scenarios`, evaluate with `solo.evaluate_models` in multiple-choice or simulated-user mode, and produce statistics and rankings with `solo.analyze_experiment`. Custom value sets go in `value_sets/[NAME].json` and are passed to each stage by name.

**Game.** `generate_game_scenarios` samples the configuration space; `filter_game_scenarios` runs the LLM-judge checks and appends to a games CSV that is a **strict superset of the single-assistant scenario schema**, so downstream tooling stays backwards compatible. `evaluate_games` then runs one of three modes — `mcq` (forced choice), `open` (free text, where a user model writes the opening a real user would send, the target answers with no scenario context, and a judge classifies which action the completion resembles), or `play` (live rounds against scripted, self, or cross-model opponents). `analyze_games` produces Bradley–Terry and Plackett–Luce rankings, material-alignment and play metrics, and steering impact; the `figures` subpackage draws the cross-model and cross-arm comparisons.

The default configuration space is deliberately narrow — **one-shot 2-player 2×2 simultaneous games** — though the code generalizes over player count, action arity, and structure via `--structures` / `--max-players` / `--max-actions`.

**Evaluation arms** beyond the mcq/open contrast, each written to its own `output_name`: steering (`--steer-prompt`, `--target-ranking`), system-prompt criteria, memory states, and counterpart disclosure (`--open-disclose-counterpart`).

> ⚠️ **The counterpart-disclosed arm carries a retry gate**, and it is a methodological commitment rather than an implementation detail. Each opening is checked for a name mention and every non-compliant draft gets one batched retry; still-failing openings are kept. The arm is therefore *filtered toward compliance* rather than a clean sample of user openings. Measure the retry and residual-failure rates before drawing a conclusion from it.

---

## Design commitments

Four decisions are load-bearing for the research validity of anything this repository produces, and any extension must preserve them:

* **Focal models are value-blind.** The evaluated model receives only the natural scenario text — never the value names, definitions, tags, or an instruction naming a value. Value metadata exists solely for balancing and scoring. Regression coverage lives in `tests/test_value_blind_prompts.py`.
* **The judge is never modified.** The filter's LLM judge is the independent arbiter of what enters a corpus; cost optimizations may only skip calls whose answers provably cannot change the verdict.
* **Runs are resumable and append-only.** Every expensive stage resumes by reading its own output and skipping what it already covered. Rejected scenarios and old logs are deliberately retained rather than pruned, and corrected analysis outputs are added under new filenames (`*_v2`) rather than overwriting the originals.
* **Payoff tensors are annotative, not ground truth.** They describe material stakes for an economically rational agent — they are *not* the utility function of the model under evaluation, and are never shown to it. The primary signal is always the model's own preference ranking over actions and outcomes.

---

## Attribution

ConflictxGame is derived from **ConflictScope** by Andy Liu and coauthors:

* Upstream repository: <https://github.com/andyjliu/conflictscope>
* Paper: [Generative Value Conflicts Reveal LLM Priorities](https://www.arxiv.org/abs/2509.25369)

**[`ATTRIBUTION.md`](ATTRIBUTION.md) records module-by-module provenance** — which modules are their code carried over, which are split or adapted from it, which are new, and which methods (free-text evaluation, steering prompts) are reimplemented here. Each module's docstring carries the same pointer.

The upstream code is MIT-licensed (Copyright (c) 2025 Andy Liu); that license is retained in [`LICENSE`](LICENSE) and continues to cover the inherited code. Modifications and the game-theoretic extension in this repository are released under the same terms. The ConflictScope authors have not endorsed, and are not responsible for, the changes made here.

### Acknowledgements

Acknowledgements of funding, institutional affiliation, and individual contributors are omitted from this repository while the associated manuscript is under double-blind review. They will be restored in full on publication.

## Citation

If you use ConflictxGame, please cite the original ConflictScope paper, since the evaluation methodology it extends is theirs:

```bibtex
@article{liu2025generative,
  title={Generative Value Conflicts Reveal LLM Priorities},
  author={Liu, Andy and Ghate, Kshitish and Diab, Mona and Fried, Daniel and Kasirzadeh, Atoosa and Kleiman-Weiner, Max},
  journal={arXiv preprint arXiv:2509.25369},
  year={2025}
}
```

Citation details for the multi-agent extension are withheld pending double-blind review and will be added on publication.
