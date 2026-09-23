ConflictxGame Documentation
===========================

ConflictxGame generates value-conflict scenarios, filters them with LLM
judges, evaluates models on them, and ranks the values each model
prioritizes -- for single assistants and for multi-agent games.

It is a fork of `ConflictScope <https://github.com/andyjliu/conflictscope>`_,
the pipeline accompanying `Generative Value Conflicts Reveal LLM Priorities
<https://www.arxiv.org/abs/2509.25369>`_ (Liu et al., 2025). ConflictScope
supplies the dyadic pipeline, the LLM-judge filters, and the shipped value
sets, and is MIT-licensed (Copyright (c) 2025 Andy Liu); the multi-agent
game-theoretic extension is added here. Please cite the original paper
alongside this repository -- see ``README.md`` for both entries.

The code is an installable package (``uv sync --extra dev``) with three
subpackages:

* :mod:`conflictxgame.solo` — the **dyadic pipeline** from the paper: one
  assistant, one simulated user, two mutually exclusive actions
  (:mod:`conflictxgame.solo.generate_scenarios`,
  :mod:`conflictxgame.solo.filter_scenarios`,
  :mod:`conflictxgame.solo.evaluate_models`,
  :mod:`conflictxgame.solo.analyze_experiment`),
* :mod:`conflictxgame.games` — the **multi-agent game-theoretic extension**:
  scenarios parameterized over player count, per-player action arity, and
  game structure (one-shot 2x2 simultaneous games by default), evaluated by
  static judgments and by live play — including repeated play of one-shot
  scenarios with compressed mem-k histories — against scripted, self, or
  cross-model opponents
  (:mod:`conflictxgame.games.game_structures`,
  :mod:`conflictxgame.games.generate_game_scenarios`,
  :mod:`conflictxgame.games.filter_game_scenarios`,
  :mod:`conflictxgame.games.game_engine`,
  :mod:`conflictxgame.games.scripted_strategies`,
  :mod:`conflictxgame.games.evaluate_games`,
  :mod:`conflictxgame.games.analyze_games`), and
* :mod:`conflictxgame.common` — shared infrastructure (model clients,
  value-set loading, JSON parsing, file locking).

Documentation sections
======================

.. toctree::
   :maxdepth: 2

   Run configurations <configs>
   Multi-agent games <game_extension>
   Generation tuning <generation_tuning>
   Solo pipeline <solo>
   Shared infrastructure <common>
   Cross-pipeline analysis <cross_pipeline>

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
