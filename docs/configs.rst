Run configurations
==================

Every command-line entry point accepts ``--config PATH``. **YAML** is the
preferred format and holds one flat mapping of settings. A run is long,
expensive, and resumable,
and the reason a knob was set the way it was rarely survives anywhere else, so
the config is the auditable record of what a run measured -- and the command
line is where a one-off deviation from it goes.

.. code-block:: bash

   python -m conflictxgame.solo.filter_scenarios \
       --config configs/solo/HHH_new_solo_20260826_01_filter_judge-gpt-5.6-luna.yaml \
       -i data/HHH_new_solo_20260826_01/scenarios/claude-sonnet-5_honesty_helpfulness.json

The file format
---------------

**Keys are argparse destination names**: underscores, not option hyphens, and
no leading dashes. ``--kept-per-pair`` is ``"kept_per_pair"``, ``-v1`` is
``"value1"``. Every option in a command's ``--help`` is settable this way,
including flags, which take ``true``/``false``.

**YAML is preferred** and uses ``#`` comments:

.. code-block:: yaml

   # Deterministic: the same scenario gets the same verdict on re-runs.
   temperature: 0.0
   max_tokens: 128  # each check returns a verdict, not an essay

Strict JSON and JSONC (JSON plus comments and trailing commas) remain
supported so historical run records continue to load.

**Precedence** is command line > config file > argparse default. Explicit
arguments always win, so a config can be reused verbatim while one setting is
varied per invocation.

**Validation is strict and happens first**, before any client is constructed
or any credential is read. An unknown key, a string where an integer belongs,
or a value outside an option's ``choices`` aborts with a message naming the
key. This is deliberate: the alternative is silently falling back to a default
that changes what the run measures.

**Required settings may come from either side.** A required option missing
from both is reported as, for example::

   error: missing required setting(s): --input-file/-i, --output-dir/-o
   -- give them on the command line or as config key(s) "input_file", "output_dir"

Where the files live
--------------------

``configs/`` holds one file per stage of a run, named
``<value set>_<arm>_<date>_<serial>_<stage>_<model roles>.yaml``. The model
roles in each filename make secondary evaluation models explicit:

.. code-block:: text

   configs/
     games/
       ..._generate_generator-claude-sonnet-5_judge-gpt-5.6-luna.yaml
       ..._eval-mcq_focal-gemma-4-31B-it.yaml
       ..._eval-open_focal-gemma-4-31B-it_user-judge-gpt-5.6-luna.yaml
       ..._eval-play_focal-gemma-4-31B-it_fallback-judge-gpt-5.6-luna.yaml
     solo/
       ..._generate_generator-claude-sonnet-5.yaml
       ..._filter_judge-gpt-5.6-luna.yaml
       ..._eval-mcq_focal-gemma-4-31B-it.yaml
       ..._eval-interactive_focal-gemma-4-31B-it_user-judge-gpt-5.6-luna.yaml
       ..._analyze.yaml

Game generation and filtering share one config because
:mod:`conflictxgame.games.generate_balanced` runs them as one resumable loop.
Each evaluation mode has its own config and output file. All configs are
cross-checked against their parsers in ``tests/test_config_files.py``.

Two things are deliberately *not* in the solo configs, because they vary per
invocation while everything else stays fixed:

* the value pair for :mod:`conflictxgame.solo.generate_scenarios` (``-v1``,
  ``-v2``), which handles one pair per run and HHH has three, and
* the input file for :mod:`conflictxgame.solo.filter_scenarios` (``-i``),
  which takes one generation output file per run.

Baking either in would mean three near-identical files whose shared settings
drift apart. Each config's header comment carries the exact loop to run.

.. _solo-config-reference:

Solo pipeline key reference
---------------------------

Defaults below are the argparse defaults, which apply when a key is omitted.

Stage 1 -- generate (:mod:`conflictxgame.solo.generate_scenarios`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 24 10 14 52

   * - Key
     - Type
     - Default
     - Meaning
   * - ``value1``, ``value2``
     - str
     - *required*
     - The two conflicting values. One pair per invocation; normally passed
       as ``-v1``/``-v2`` rather than set here.
   * - ``value_set``
     - str
     - *required*
     - Name of a file under ``value_sets/``. Supplies the long definitions
       shown to both the generator and the judge.
   * - ``model``
     - str
     - ``gpt-4o-mini``
     - The generating model.
   * - ``temperature``
     - float
     - ``1.0``
     - Stage one samples skeletons, so it wants variety rather than the most
       likely scenario.
   * - ``max_tokens``
     - int
     - ``4096``
     - Output budget per call. A reply cut off mid-JSON is unparseable and
       costs the whole batch, so this is generous by design.
   * - ``num_scenarios``
     - int
     - ``40``
     - Scenarios requested for this value pair, before filtering.
   * - ``batch_size``
     - int
     - ``40``
     - Skeletons per stage-one call, split across the causal-graph prompt
       variants.
   * - ``max_extra_tries``
     - int
     - ``5``
     - Extra stage-one attempts when a reply is malformed or deduplication
       drops the count below ``num_scenarios``. Bounds the spend one
       pathological pair can incur.
   * - ``deduplicate``
     - bool
     - ``false``
     - Drop skeletons that are near-copies of one already kept
       (``all-MiniLM-L6-v2`` cosine similarity). Included in the default
       install.
   * - ``deduplication_threshold``
     - float
     - ``0.9``
     - Similarity above which two skeletons count as duplicates. Lower is
       stricter.
   * - ``criteria_in_context``
     - bool
     - ``false``
     - Append the distilled filter checklist to both generation stages.
       Raises the share that survives filtering; the judge is unchanged, so
       this buys yield without moving the acceptance bar.
   * - ``add_to_existing``
     - bool
     - ``false``
     - Append to this pair's existing output file instead of overwriting.
   * - ``output_dir``
     - str
     - *required*
     - Receives ``{model}_{value1}_{value2}.json``.

Stage 2 -- filter (:mod:`conflictxgame.solo.filter_scenarios`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applies the six checks -- realism, groundedness, action feasibility, value
guidedness, impossibility, ambiguity -- and keeps a scenario only if all six
pass.

.. list-table::
   :header-rows: 1
   :widths: 24 10 14 52

   * - Key
     - Type
     - Default
     - Meaning
   * - ``model``
     - str
     - ``gpt-4o-mini``
     - The **judge**, not the generator. It is the independent arbiter of
       what enters the corpus; keep it identical across arms being compared.
   * - ``temperature``
     - float
     - ``0.0``
     - Deterministic, so the same scenario gets the same verdict and per-pair
       accepted counts stay meaningful across re-runs.
   * - ``max_tokens``
     - int
     - ``128``
     - Each check returns a short verdict. Raise only if a reasoning model
       truncates before emitting its answer.
   * - ``value_set``
     - str
     - ``0217``
     - Must match stage 1, or value guidedness is scored against a different
       question than the scenario was written to answer.
   * - ``input_file``
     - str
     - *required*
     - One stage-1 output file. Normally passed as ``-i``.
   * - ``output_dir``
     - str
     - *required*
     - Receives ``{generating model}.csv``, appended across pairs. This is
       also stage 3's ``scenarios_dir``.

Stage 3 -- evaluate (:mod:`conflictxgame.solo.evaluate_models`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two modes, run as two configs. ``interactive: false`` is the paper's
*expressed* preferences (an A/B question plus per-action Likert ratings);
``interactive: true`` is their *revealed* preferences (a simulated
conversation, judged afterwards). They answer different questions, so the
second is not a substitute for the first.

.. list-table::
   :header-rows: 1
   :widths: 24 10 14 52

   * - Key
     - Type
     - Default
     - Meaning
   * - ``model``
     - str
     - *required*
     - The model under test. Also the fallback for ``assistant_model``.
   * - ``interactive``
     - bool
     - ``false``
     - Select the simulated-conversation mode.
   * - ``temperature``
     - float
     - ``0.0``
     - Deterministic for MCQ. In interactive mode a temperature of ``0.0``
       (or ``max_tokens`` ≤ 5) is replaced by ``1.0`` / ``1024``, so set both
       explicitly there to keep the file honest about what ran.
   * - ``max_tokens``
     - int
     - ``5``
     - The MCQ answer is one letter, but a reasoning model spends part of its
       budget deliberating first and a reply that runs out is scored
       ``INVALID`` -- use ``1024`` for those.
   * - ``batch_size``
     - int
     - ``32``
     - Scenarios per batched call.
   * - ``scenarios_dir``
     - str
     - *required*
     - Globbed for ``*.csv``. Point it at stage 2's ``output_dir``, not at
       the generation JSON.
   * - ``filter``
     - bool
     - ``false``
     - Evaluate only rows with ``keep_scenario=True``.
   * - ``output_dir``
     - str
     - *required*
     - Receives the results CSV, rewritten after each input file so partial
       progress survives interruption.
   * - ``output_name``
     - str
     - model short name
     - **Set this per mode.** Both modes otherwise default to the same file
       name; they emit different columns, and evaluation resumes by skipping
       scenario ids already present, so the second run would silently produce
       nothing.
   * - ``force_recompute``
     - bool
     - ``false``
     - Re-evaluate ids already in the output file. Leaving it off is what
       makes a run resumable.
   * - ``user_model``
     - str
     - ``gpt-4o-mini``
     - Interactive only: roleplays the human. Keep it distinct from the model
       under test, which otherwise plays both seats and stops pressing the
       dilemma.
   * - ``assistant_model``
     - str
     - ``model``
     - Interactive only. Leave unset so it cannot drift from ``model``.
   * - ``judge_model``
     - str
     - ``gpt-4o-mini``
     - Interactive only: maps the conversation onto action1/action2. Usually
       the stage-2 judge, so one standard spans corpus and evaluation.
   * - ``cache``
     - bool
     - ``false``
     - Write assembled prompts to ``output_dir/prompts.json``.
   * - ``steer_prompt``
     - str
     - unset
     - Path to a text file whose contents steer the assistant.

Stage 4 -- analyze (:mod:`conflictxgame.solo.analyze_experiment`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Makes no API calls, so it is cheap to re-run whenever a stage-3 run tops up.

.. list-table::
   :header-rows: 1
   :widths: 24 10 14 52

   * - Key
     - Type
     - Default
     - Meaning
   * - ``model_dir``
     - str
     - *required*
     - Stage 3 output. Every ``*.csv`` here is one evaluated model's results
       -- which is why the two modes need distinct file names.
   * - ``scenario_dir``
     - str
     - *required*
     - Stage 2 output. Supplies ``keep_scenario`` and ``check_results``.
   * - ``output_dir``
     - str
     - *required*
     - Receives ``preference_rates_{variant}.csv`` and the ranking plots.
   * - ``value_set``
     - str
     - *required*
     - Must match stages 1 and 2 or the value names will not resolve.
   * - ``compute_bradley_terry``
     - bool
     - ``false``
     - Fit per-model ability scores with bootstrapped 95% CIs and save the
       ranking plots.
   * - ``wandb``
     - bool
     - ``false``
     - Log per-generating-model metrics to Weights & Biases.
   * - ``wandb_project``
     - str
     - ``conflictscope``
     - W&B project name.
   * - ``job_name``
     - str
     - unset
     - W&B run label.

Adding a config
---------------

#. Copy the closest existing file in ``configs/solo/`` and rename it for the
   run. Keep the YAML comments and update them -- a stale comment is worse than
   none.
#. Add it to ``SHIPPED_CONFIGS`` in ``tests/test_config_files.py``. That
   suite parses every file under ``configs/`` against the parser it targets,
   so a renamed flag is caught offline rather than after paid credentials
   have loaded, and it fails if a config is left uncovered.
#. Check it without spending anything::

      python -m conflictxgame.solo.evaluate_models --config <path> --help

API
---

.. toctree::
   :maxdepth: 1

   Config loading <api/config>
