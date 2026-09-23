Multi-Agent Game-Theoretic Extension
====================================

The ConflictxGame extension generalizes `ConflictScope
<https://github.com/andyjliu/conflictscope>`_ -- the upstream project this
repository forks -- from a single assistant facing a binary value conflict to
**games**: several decision-makers with interdependent outcomes, varying player
counts, per-player action arities, and temporal structures. It keeps
ConflictScope's identity — measuring *value priorities* — by tagging every
action of every role with the value it favors. References to "the dyadic
pipeline" or "the original format" below mean upstream ConflictScope code,
which ships here unchanged.

Design principles
-----------------

Parameterized games, not named games
    Scenarios are generated for cells of the configuration space
    ``(structure, n_players, arities)`` — e.g. a *sequential 3-player 2x2x2*
    game or a *simultaneous 2-player 2x3* game — never from canonical named
    templates (prisoner's dilemma, chicken, ...). Named patterns constrain
    breadth; instead they are *detected post hoc* from a scenario's payoff
    tensor (:func:`conflictxgame.games.game_structures.detect_canonical_2x2`), which is exactly
    what gates the zero-determinant scripted strategies.

Annotative payoffs
    Every scenario carries a numeric payoff tensor over joint action profiles.
    It describes the *material stakes* an economically rational agent would
    play by — useful for equilibrium, welfare, and exploitability analysis —
    but it is **not** treated as the ground-truth utility of the (moral) agent
    under evaluation. The primary measurement is the model's own preference
    ranking over actions (Likert ratings) and predicted outcomes (explicit
    rankings); :mod:`conflictxgame.games.analyze_games` reports the *divergence* between stated
    preferences and payoff order as a finding
    (``material_alignment.csv``), not an error.

Sampled coverage with an explicit report
    Full coverage of structures x player counts x arity vectors x value
    assignments explodes combinatorially. Generation therefore *samples* up to
    ``--max-per-config`` scenarios per cell and writes a **coverage report**
    (``coverage_report.json``/``.csv``) recording, per cell: the target, the
    number generated, the number kept after filtering, and skip notes. Nothing
    silently pretends to be exhaustive.

Soft caps, not hard caps
    Player counts and arities beyond the defaults (2-3 each; ``n=1`` allowed
    for the classic single-agent setting, now with up to 3 actions) are
    accepted, but any configuration whose joint-profile count exceeds
    :data:`conflictxgame.games.game_structures.DEFAULT_PROFILE_WARN_THRESHOLD` raises a
    :class:`conflictxgame.games.game_structures.RepresentationSizeWarning` spelling out the cost:
    tensor entries, per-profile narrative outcome descriptions, and estimated
    prompt tokens.

Structured action declarations
    In live play, an LLM seat answers with
    ``{"reasoning": ..., "action": "<label>"}`` from an enumerated menu.
    Free-text action attribution by a judge degrades quickly as players and
    arities grow; structured declarations keep parsing exact, with one retry,
    an optional judge fallback, and a flagged random substitution as the last
    resort (:class:`conflictxgame.games.game_engine.GameEngine`).

Failure modes considered
------------------------

The design responds to concrete anticipated failures:

* **Judge validity collapse** beyond 2 players / 2 actions — solved by
  structured action declarations (above).
* **Scenario incoherence** — stage 1 produces a strategic outline whose
  player counts, arities, and per-slot value assignments are fixed inputs;
  stage 2 fills narrative and outcomes for an *enumerated* profile list
  (``P1..Pm``), so keys cannot drift. Structurally invalid outputs are
  dropped and logged.
* **Cost explosion in repeated games** — history is passed as a compact
  round/action table; the engine advances all games of a batch one round at a
  time so LLM calls batch across scenarios.
* **Position and order bias** — MCQ options are permuted per query with an
  inverse map; live-play menus are shuffled per game; the focal model's seat
  rotates across repetitions.
* **Losing the value-conflict identity** — every action carries a
  ``value_tag`` from the value set, assigned by the sampler (not echoed by
  the model), and verified per player by the ``value_guidedness`` filter.

Pipeline
--------

Generation defaults to **one-shot 2-player 2x2 simultaneous games**; the code
still supports more players, larger menus, and other structures via
``--structures``/``--max-players``/``--max-actions``. The example driver
(``scripts/game_generation.sh``) uses the defaults with every seat a deployed
AI assistant (``--assistant-seat-prob 1.0``). A generation run writes **two
combined JSON files** plus the coverage report into the output directory —
``{model}_games.json`` (all game specs, keyed by ``game_id``) and
``{model}_game_prompts.json`` (the stage-1 and stage-2 prompts) — instead of
one file per configuration cell. Re-running with ``--add-to-existing/-a`` tops
up each cell inside the combined file (new ids continue from the highest
surviving index).

The stage-one prompt is ``guided`` by default: the sampler assigns a value to
each action slot. ``--prompt-style paper`` instead uses a short, paper-style
prompt that gives all value definitions and the decision-maker shapes but no
assigned value pairs; the model returns a ``favors`` value name for each
action, and invalid or single-value menus are dropped. In balanced generation
this also disables construction guidance, so it can be compared cleanly with
the guided arm.

.. code-block:: bash

   # 1. Generate one-shot 2-player 2x2 games (writes coverage report)
   python -m conflictxgame.games.generate_game_scenarios \
       -m claude-3-5-sonnet-latest -v HHH \
       --assistant-seat-prob 1.0 --max-per-config 10 -d

   # 2. Filter the games with an LLM judge (updates the coverage report)
   python -m conflictxgame.games.filter_game_scenarios \
       -i data/HHH/games/claude-3-5-sonnet-latest_games.json \
       -m gpt-4.1 -v HHH --coverage-report data/HHH/games/coverage_report.json

   # 3a. Static evaluation: MCQ + Likert + outcome rankings
   python -m conflictxgame.games.evaluate_games -m claude-3-5-sonnet-latest --mode mcq \
       -d data/HHH -o results/HHH_games --filter --max-tokens 1024

   # 3b. Open-ended evaluation: simulated user writes the opening message
   python -m conflictxgame.games.evaluate_games -m claude-3-5-sonnet-latest --mode open \
       -d data/HHH -o results/HHH_games --filter \
       --user-model gpt-4.1 --judge-model gpt-4.1 --open-prompts

   # 3c. Live play: each one-shot scenario repeated for 5 rounds with mem-1
   #     compressed history, against scripted strategies
   python -m conflictxgame.games.evaluate_games -m claude-3-5-sonnet-latest --mode play \
       --opponents scripted --strategies tit_for_tat,always_defect,grim_trigger,zd_extort \
       -d data/HHH -o results/HHH_games --filter --repetitions 2 \
       --rounds 5 --memory-k 1

   # 4. Steered runs (separate results directory)
   python -m conflictxgame.games.evaluate_games -m claude-3-5-sonnet-latest --mode mcq \
       -d data/HHH -o results/HHH_games_steered --filter --max-tokens 1024 \
       --steer-prompt value_sets/steering/HHH.txt

   # 5. Analysis: value rankings, material alignment, play metrics, steering
   python -m conflictxgame.games.analyze_games --results-dir results/HHH_games \
       --steered-results-dir results/HHH_games_steered \
       --target-ranking "harmlessness>honesty>helpfulness" \
       --scenarios-dir data/HHH --output-dir analysis/HHH_games --plots

   # 6. Figures for one model: values by evaluation condition and over rounds
   python -m conflictxgame.games.value_figures --results-dir results/HHH_games \
       --scenarios-dir data/HHH --output-dir analysis/HHH_games/figures \
       --model claude-3-5-sonnet-latest

The filter's output CSV (``{model}_games.csv``) is a **superset of the dyadic
pipeline's scenario schema**: it carries the original solo columns
(``scenario_id``, ``context``, ``description``, ``user_prompt``, ``value1``,
``value2``, ``action1``, ``action2``, ``keep_scenario``, ``check_results``,
``generating_model`` — populated from player 1's perspective) alongside the
game columns (``n_players``, ``arities``, ``structure``, ``horizon``,
``player2_values``, ``payoff_matrix``, ``agent_kinds``, ``game_spec``, ...),
keeping downstream evals backwards compatible with the dyadic tooling.

Generated game data lands in ``data/<value_set>/`` beside the upstream
ConflictScope datasets and *compounds* there: each filter run appends to the
same per-model CSV. Evaluation globs ``*_games.csv``, so it never picks up the
upstream scenario files. See ``data/README.md``.

Open-ended evaluation
---------------------

Multiple-choice questions ask a model what it *would* do; the open-ended
setting from the paper (§3.3, Appendix C.5/C.6) measures what it actually
does. ``--mode open`` runs three steps per seat
(:mod:`conflictxgame.games.open_ended`):

1. A **user model** (``--user-model``, default ``gpt-4.1``) is given the
   scenario and the persona of the party interacting with the focal seat, and
   writes the opening message a real user would send. The instruction block is
   :data:`conflictxgame.solo.prompts.OPENING_MESSAGE_INSTRUCTIONS` — the
   paper's Appendix C.5 text, shared verbatim with the dyadic pipeline.
2. The **target model** answers that message and nothing else: no scenario
   description, no action menu, no payoffs. Its system prompt is the only
   other input, which is where a steering prompt applies.
3. A **judge model** (``--judge-model``) maps the completion onto the seat's
   action menu, mirroring the Appendix C.6 judge but generalized from A/B to
   ``A``..``K``; the 1–7 alignment rating is requested only for two-action
   menus, where it keeps the paper's meaning.

Openings are written fresh on each run, as in the paper. Passing
``--open-prompts`` caches them to
``data/<value_set>/<model>_open_prompts.json`` so several target models can be
compared on identical prompts; cached entries are reused only when the same
``--user-model`` wrote them. Result rows carry the full ``open_user_prompt``
and ``open_completion`` for auditability, so these CSVs are much larger than
the MCQ ones.

Steering toward a target ranking
--------------------------------

The paper's steerability experiment (§4.2/§5.3) asks how far a system prompt
can move a model's elicited value ranking toward a developer-chosen target.
``--steer-prompt <file>`` sets the focal model's system prompt; the paper's
Appendix J prompts ship verbatim in ``value_sets/steering/`` for all three
value sets. ``--target-ranking "a>b>c"`` instead generates a prompt in the
same structure for an arbitrary ranking
(:func:`conflictxgame.games.steering.build_steering_prompt`): an explicit
ranking with definitions, a resolution rule for every pair of values, and a
numbered decision framework. Steering applies to all three modes.

Run the steered evaluation into a **separate** ``--output-dir`` (resume dedup
is by scenario id and mode, so a steered run would otherwise be skipped as
already-evaluated), then measure the impact with
:func:`conflictxgame.games.analyze_games.compute_steering_impact`. It reports,
per model and mode, the alignment ``a(R, R_t)`` — the share of decisions
choosing the action whose value ranks highest in the target among the menu's
distinct tags — before and after steering, plus the paper's normalized effect
size ``(a_steered - a_default) / (1 - a_default)``, the fraction of the
achievable improvement actually captured. Output: ``steering_impact.csv``.

Repeated play of one-shot scenarios (mem-k)
-------------------------------------------

Repetition happens at **evaluation time**, not generation time. In play mode,
``--rounds N`` (N >= 2) converts each one-shot simultaneous scenario into an
N-round repeated game (:func:`conflictxgame.games.game_structures.as_repeated`)
— and ``--memory-k k`` (default 1) bounds the history the acting model sees:
its prompt shows how many rounds have been played but reproduces only the most
recent ``k`` rounds (``-1`` restores full history; ``0`` shows the round count
alone). Scripted opponents always condition on the true full history, so a
mem-1 focal model can face, e.g., a grim trigger it cannot fully reconstruct.

``--repeat-mode`` chooses what repeats:

``same`` (default)
    The same scenario is played over and over for ``--rounds`` rounds — the
    classic repeated-game setting over one narrative.

``sequence``
    Chains of ``--rounds`` *distinct* one-shot scenarios are played in order,
    with a compressed mem-k summary of the previous scenarios' outcomes
    (actions chosen, value tags, payoffs) carried into each new scenario's
    prompt. Sequences are composed by seeded shuffle by default, or grouped by
    value pair first with ``--sequence-grouping value-pair``. Output rows
    carry ``sequence_id``/``sequence_pos`` for reconstruction.

Agent kinds and unified value definitions
-----------------------------------------

Game seats carry an ``agent_kind`` — ``human`` (default) or ``assistant`` (a
deployed, text-only AI assistant playing a seat, possibly alongside humans).
The kind follows the seat through the whole pipeline: generation prompts frame
assistant seats with the dyadic chatbot constraints (text-only, no external
actions) and describe a deployment context instead of a biography; the
``action_feasibility`` filter applies the same constraints; live-play and
elicitation prompts announce the seat as an AI assistant; and analysis slices
value rankings and play metrics by kind
(``value_rankings_bradley_terry_by_kind.csv``). Sample assistant seats at
generation time with ``--assistant-seat-prob`` (the example driver uses 1.0,
so every seat is an assistant), or use ``--one-assistant-seat`` to make
exactly one uniformly selected seat an assistant and all remaining seats
human. The latter is useful when two independent assistants acting on the
same record would make a simultaneous game artificial.

**Value definitions are agent-neutral**: every shipped value set
(``value_sets/HHH.json``, ``modelspec.json``, ``personalprotective.json``) is
a flat ``name -> definition`` mapping with no human-vs-assistant variants and
no assistant-specific language ("user", "AI", "chatbot", ...), and the same
definition text is used for every decision-maker in generation prompts and in
the ``value_guidedness`` judge. Agent kind still shapes a seat's *framing*
(deployment context, text-only constraints) — just not its value language.

For backward compatibility, :func:`conflictxgame.common.values.resolve_value_definition` still
accepts legacy per-kind dict entries (``{"human": ..., "assistant": ...}``),
resolving the seat's kind, then ``human``, then the first entry; flat entries
pass through unchanged. The dyadic pipeline's :func:`conflictxgame.solo.prompts.load_value_dict`
is untouched.

Because gpt-5-family judge models are locked at temperature 1.0 and the
guidedness check is conjunctive across players and values, each
player-x-value sub-judgment is asked ``--guidedness-votes`` times (default 3)
and passes on a strict majority; set 1 for the original single-shot behavior.

Opponent types
--------------

``--opponents scripted`` (default)
    Every non-focal seat runs one scripted strategy
    (:mod:`conflictxgame.games.scripted_strategies`): ``always_first``, ``always_last``,
    ``always_cooperate``, ``always_defect``, ``random``, ``tit_for_tat``,
    ``grim_trigger``, ``win_stay_lose_shift``, ``zd_extort``, ``zd_generous``.
    "Cooperate"/"defect" are operationalized from the tensor as the player's
    action in the welfare-maximizing / welfare-minimizing profile.

``--opponents self``
    All seats are played by the focal model (within-model dynamics).

``--opponents cross``
    Seats are played by different models (``--cross-models``), rotated across
    repetitions (model-vs-model tournaments).

Zero-determinant strategies
---------------------------

:class:`conflictxgame.games.scripted_strategies.ZDStrategy` implements extortionate (Press &
Dyson 2012) and generous (Stewart & Plotkin 2012) memory-one strategies. The
four cooperation probabilities are solved from the *scenario's own* payoff
tensor; the scripted agent assumes the game is symmetric. Applicability is
detected from the tensor — a ``repeated`` symmetric 2x2 game whose payoffs
admit a prisoner's-dilemma ordering ``T > R > P > S``
(:func:`conflictxgame.games.scripted_strategies.zd_applicable`) — never from a game name.
Inapplicable scenario/strategy combinations are skipped with a logged warning.
For the canonical payoffs (3, 0, 5, 1) and ``chi = 2`` the implementation
reproduces Extort-2 ``(8/9, 1/2, 1/3, 0)`` and ZDGTFT-2 ``(1, 1/8, 1, 1/4)``
exactly (verified in ``tests/test_scripted_strategies.py``, including the
asymptotic surplus ratio the extortioner enforces).

Analysis outputs
----------------

``value_rankings_bradley_terry.csv``
    Bradley-Terry abilities per value per model with bootstrap CIs, fit from
    every valid decision event (each chosen action's value tag beats the other
    distinct tags in that menu).
``value_rankings_plackett_luce.csv``
    Plackett-Luce abilities fit from Likert-derived per-menu value rankings
    (informative for 3-action menus).
``material_alignment.csv``
    Mean Spearman correlation of stated outcome rankings with own annotative
    payoff and with total welfare — the measured gap between the model and a
    payoff-maximizing agent.
``play_metrics.csv``
    Cooperation rate, best-response rate, pure-Nash-profile rate, welfare
    ratio, and endgame defection by structure.
``conditional_cooperation.csv``
    P(cooperate | opponents all cooperated last round) vs otherwise.
``zd_exploitability.csv``
    Surplus ratio against the extortionate ZD opponent.
``mcq_play_consistency.csv``
    Agreement between stated MCQ choices and round-1 live play.
``steering_impact.csv``
    Alignment with a target value ranking, unsteered vs steered, and the
    normalized effect size (written when both ``--steered-results-dir`` and
    ``--target-ranking`` are given).

Windows portability
-------------------

The dyadic simulated-user pipeline uses the POSIX-only :mod:`fcntl`; the game
pipeline instead uses :mod:`conflictxgame.common.portable_lock` (msvcrt on Windows, fcntl
elsewhere) and avoids importing :mod:`conflictxgame.solo.simulated_conversation`. Provider SDKs
(vllm, openai, anthropic) are imported lazily by :mod:`conflictxgame.common.model_wrappers`, so the
game modules import on machines with none of them installed.

Game API reference
------------------

The modules below implement the multi-agent pipeline described in this guide.

.. toctree::
   :maxdepth: 1

   Game structures <api/game_structures>
   Game engine <api/game_engine>
   Generate scenarios <api/generate_game_scenarios>
   Filter scenarios <api/filter_game_scenarios>
   Balanced generation <api/generate_balanced>
   Pair quotas <api/pair_quota>
   Outline pre-screen <api/prescreen>
   Scripted strategies <api/scripted_strategies>
   Open-ended play <api/open_ended>
   Steering <api/steering>
   Evaluate games <api/evaluate_games>
   Analyze games <api/analyze_games>
   Value figures <api/value_figures>
   Reweighting <api/reweighting>
