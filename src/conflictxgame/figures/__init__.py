"""Figure and table generation across evaluation arms.

Each module is a standalone CLI (``python -m conflictxgame.figures.<name>
--config ...``) that reads the evaluation CSVs under ``results/`` and writes a
figure plus the CSV table behind it, so every plotted number stays auditable.

The modules divide by the research question each answers:

* :mod:`~conflictxgame.figures.value_conditions_figure` -- value choice by
  elicitation condition; the shared loader (``load_pairs``) several siblings build on.
* :mod:`~conflictxgame.figures.stake_domain_figure`,
  :mod:`~conflictxgame.figures.domain_effect_figure` -- stake and domain effects.
* :mod:`~conflictxgame.figures.state_effect_figure`,
  :mod:`~conflictxgame.figures.visibility_figure` -- memory-state and
  observability probes.
* :mod:`~conflictxgame.figures.horizon_figure`,
  :mod:`~conflictxgame.figures.solo_format_figure` -- horizon decomposition and
  the solo/game format contrast.
* :mod:`~conflictxgame.figures.comparison_figures`,
  :mod:`~conflictxgame.figures.summary_tables`,
  :mod:`~conflictxgame.figures.raw_rates` -- cross-arm summaries.
* :mod:`~conflictxgame.figures.export_game_eval_data` -- flattens game
  evaluation results for external analysis.

Per-model game figures live in :mod:`conflictxgame.games.value_figures`, which
is scoped to one model across game modes and rounds rather than across arms.
Cross-pipeline ranking comparison lives in :mod:`conflictxgame.cross_analysis`.
"""
