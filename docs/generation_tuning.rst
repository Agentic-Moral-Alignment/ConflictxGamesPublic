Generation cost and yield tuning
================================

Most generated scenarios are rejected by the filter (the upstream corpora kept
17--31%), which is what makes filling a quota expensive. A rejected 2x2 game
costs a short stage-1 call, a *long* stage-2 call, and up to 18 judge calls --
twelve of them value-guidedness votes.

This page covers the mechanisms that reduce that cost, how to measure whether
they work on your own models, and the calibration gates to clear before
trusting any of them to discard work. For the balanced-generation workflow
itself, see :doc:`game_extension`.

Mechanisms
----------

``generate_balanced`` ships three mechanisms on by default, aimed at both
halves of the problem: fewer rejections, and cheaper rejections.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Mechanism
     - Flag
     - What it does
   * - Construction guidance
     - ``--generation-guidance checklist,recipes,stage2`` (``none`` to disable)
     - Prompt text only. The checklist restates the filter's criteria; the
       recipes say how to *build* a menu that is exclusive by construction and
       a dilemma whose two sides carry comparable named costs -- which is what
       the two most expensive checks actually reject.
   * - Paper-style outlines
     - ``--prompt-style paper``
     - A shorter stage-one prompt that gives the value definitions and seat
       shapes but no assigned pairs, recipe, or checklist. The generator labels
       each action with ``favors``; quotas are then filled passively from the
       accepted labels. It is intended for A/B testing against the guided
       prompt, not assumed to be better without data.
   * - Short-circuited filter
     - on; ``--full-checks`` to disable
     - Stops asking questions once the answer cannot change the verdict, and
       skips the non-fatal payoff check for scenarios already rejected. A
       rejection can cost one judge call instead of eighteen.
   * - Outline pre-screen
     - ``--prescreen shadow`` (default), ``enforce``, ``off``
     - One small call between the two generation stages, asking whether the
       outline is already broken. ``shadow`` only records the answer;
       ``enforce`` drops flagged outlines before the long stage-2 call is
       spent. Realism is always answered but remains shadow-only unless
       explicitly named in ``--prescreen-criteria``.

One repair pass over rejected scenarios (``--revise-failures N``) is still
available but **off by default**: measured salvage did not cover the long reply
plus full filter pass each attempt costs. Enabling it forces ``--full-checks``,
since a repair has to be told every check a scenario failed.

What the judge guarantee covers
-------------------------------

**The judge is never modified.** Its prompts, its vote rule, and
``keep_scenario`` are untouched, so it remains the independent arbiter of what
enters the corpus. Two clarifications about what that guarantee does and does
not cover:

* Short-circuiting edits the filter's *orchestration*, not the judge.
  ``keep_scenario`` is a conjunction over the fatal checks, so neither the order
  they run in nor whether the already-doomed ones run at all can change a
  verdict; only calls whose answers could not have mattered are skipped. The
  accepted corpus is identical.
* The pre-screen does change which scenarios are *attempted* -- in ``enforce``
  mode, a flagged outline never reaches the judge. That is a real change to the
  corpus, which is why it defaults to ``shadow`` and why the calibration gate
  below exists.

Reading the cost report
-----------------------

``pair_coverage.json`` records a ``cost`` block alongside the quota: calls and
tokens per role, and per **accepted** scenario. That last number is the one to
compare mechanisms on.

An acceptance rate can be raised by attempting fewer, safer scenarios --
enforcing the pre-screen raises it mechanically, since discarded outlines never
reach the judge to be counted as rejections -- so the report carries both
denominators (``kept_per_filtered_scenario`` and
``kept_per_attempted_outline``) plus cost per kept, which cannot be moved that
way.

Finding where yield leaks
-------------------------

To see where yield leaks before tuning anything:

.. code-block:: bash

   python -m conflictxgame.common.generation_quality \
       --csv data/HHH/claude-3-5-sonnet-latest.csv

The ``n_sole_cause`` column is the actionable one: those scenarios failed
exactly one check, so fixing that check converts them into accepted scenarios.
Across the three shipped corpora, roughly 44% of all generated scenarios failed
exactly one check, and the two dominant causes are "the dilemma has an obvious
answer" and "the actions aren't mutually exclusive".

Run this on a corpus generated with ``--full-checks``: a short-circuited pass
stops at the first failure, so it cannot attribute a sole cause (those rows are
reported under ``n_unevaluated`` instead).

Calibrating the pre-screen
--------------------------

Before trusting the pre-screen to discard anything, check it against the judge
on shadow-mode data:

.. code-block:: bash

   python -m conflictxgame.common.generation_quality \
       --prescreen --csv data/HHH/<model>_games.csv

Two columns decide it, and they pull against each other:

``precision``
   Of the outlines it flagged, the share the judge also rejected. Compare
   against ``base_reject_rate``, what random flagging would score.

``kept_lost_rate``
   Of the scenarios the judge *accepted*, the share the screen had flagged --
   the good scenarios enforcement would throw away unexamined.

A reasonable gate is precision >= 0.8 and ``kept_lost_rate`` <= 0.10 over at
least 40 flagged outlines; otherwise narrow ``--prescreen-criteria`` to the
criteria that clear it.

Measuring it on your own models
-------------------------------

To measure any of this on your own models rather than taking it on faith,
``scripts/acceptance_ab.sh`` runs matched arms (no guidance / checklist only /
full guidance / plus near-miss) with everything else held fixed, and reports
kept rates with Wilson intervals, cost per accepted scenario, the
rejection-cause table, and the pre-screen calibration.

Set ``AUDIT_JUDGE`` to a model from a different family to re-judge each arm's
corpus as a check on judge-overfitting: acceptance that rose because the
generator learned one judge's tastes is not an improvement.

Acceptance is a coin flip per scenario, so sample size matters -- detecting
25% -> 40% takes roughly 150 scenarios per arm, 25% -> 50% about 60.

Why the mechanisms cost input tokens, not output
------------------------------------------------

Both prompt mechanisms are deliberately chosen to cost input tokens or one
extra small call, never a longer *reply*. Asking the generator to narrate a
self-check inflates the reply instead, and a reply that runs out of output
budget mid-JSON is unparseable -- which is how an early version of this lost
whole batches.

One operational caveat: a generation call returns every scenario of its batch
in a single JSON reply, so a reply truncated by ``--max-tokens`` loses the whole
batch -- and reasoning models spend part of that budget deliberating before they
emit anything. ``generate_balanced`` therefore sends **one scenario per call**
by default (``--batch-size 1``) with a generous ``--max-tokens`` (16384), so a
truncated reply costs one scenario instead of a batch. Raise ``--batch-size`` to
bundle scenarios into fewer, longer calls.

If a round reports generating nothing, the message says which cause it was --
API failures, unparseable replies, or the pre-screen having flagged every
outline -- so you are not left tuning the wrong knob.

The standalone generation CLIs keep their original prompts unless you opt in
with ``--criteria-in-context``. ``--prompt-style paper`` is the exception: it
deliberately uses the smaller paper-style stage-one prompt and returns
model-selected ``favors`` tags instead of a slot-assignment table.
