"""ConflictxGame: value-conflict scenario generation and evaluation for LLMs.

Subpackages:
    * :mod:`conflictxgame.solo` — the original dyadic (single-decision)
      pipeline: generate, filter, evaluate, and analyze scenarios where one
      AI assistant chooses between two value-laden actions.
    * :mod:`conflictxgame.games` — the game-theoretic pipeline: one-shot 2x2
      (by default) multi-agent scenarios with payoff tensors, scripted
      opponents, and repeated/mem-k evaluation.
    * :mod:`conflictxgame.common` — shared infrastructure: model clients,
      value-set loading, JSON parsing, and cross-platform file locking.
"""

__version__ = "0.1.0"
