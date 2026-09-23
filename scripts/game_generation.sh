# Example: generate, filter, evaluate, and analyze game-theoretic scenarios
# over the HHH value set. Mirrors HHH_generation.sh for the game pipeline.
#
# Generation defaults to one-shot 2-player 2x2 simultaneous games (the code
# supports more players, larger menus, and sequential/repeated structures via
# --structures/--max-players/--max-actions), with every seat a deployed AI
# assistant.
#
# Data layout: intermediates go to data/<value_set>/games/ and the filtered
# scenario CSV compounds in data/<value_set>/ beside the upstream ConflictScope
# datasets. Our files carry the _games.csv suffix, which is how the evaluator
# tells them apart from theirs -- see data/README.md.

VALUE_SET=HHH
GEN_MODEL=claude-3-5-sonnet-latest
JUDGE_MODEL=gpt-4.1
EVAL_MODEL=claude-3-5-sonnet-latest
USER_MODEL=gpt-4.1

# The target value ranking for the steering experiment (paper 5.3).
TARGET_RANKING="harmlessness>honesty>helpfulness"

# Mirror model_wrappers.short_model_name: strip "provider:" prefix and path.
MODEL_SHORT=${GEN_MODEL#*:}
MODEL_SHORT=${MODEL_SHORT##*/}

# 1. Generate one-shot 2-player 2x2 games (the default configuration space)
#    into data/${VALUE_SET}/games/.
python -m conflictxgame.games.generate_game_scenarios \
    -m ${GEN_MODEL} -v ${VALUE_SET} \
    --assistant-seat-prob 1.0 \
    --max-per-config 10 -d -dt 0.8 --max-tokens 8192

# 2. Filter the games with an LLM judge; update the coverage report's kept
#    counts. Appends to data/${VALUE_SET}/${MODEL_SHORT}_games.csv, a superset
#    of the original solo CSV schema (scenario_id/context/description/
#    user_prompt/value1/value2/action1/action2/... plus n_players, arities,
#    structure, player2_values, payoff_matrix), so downstream evals stay
#    backwards compatible.
python -m conflictxgame.games.filter_game_scenarios \
    -i data/${VALUE_SET}/games/${MODEL_SHORT}_games.json \
    -m ${JUDGE_MODEL} -v ${VALUE_SET} \
    --coverage-report data/${VALUE_SET}/games/coverage_report.json

# Evaluation. Both evaluators default --max-tokens to a handful of tokens (16
# and 5), which suited models that reply with a bare letter or digit;
# reasoning-capable models spend that budget reasoning and return empty
# content, which lands as INVALID rows, so give them real headroom.

# 3a. Static game evaluation: MCQ + Likert + outcome rankings.
python -m conflictxgame.games.evaluate_games -m ${EVAL_MODEL} --mode mcq \
    -d data/${VALUE_SET} -o results/${VALUE_SET}_games --filter \
    --max-tokens 1024

# 3b. Open-ended evaluation (paper 3.3): a user model writes the opening
#     message, the target answers it with no scenario context, and a judge maps
#     the completion onto the action menu. --open-prompts caches the openings
#     so several target models can be compared on identical prompts (drop it
#     for the paper's fresh-prompts-per-run behavior).
python -m conflictxgame.games.evaluate_games -m ${EVAL_MODEL} --mode open \
    -d data/${VALUE_SET} -o results/${VALUE_SET}_games --filter \
    --user-model ${USER_MODEL} --judge-model ${JUDGE_MODEL} --open-prompts

# 3c. Live play: each one-shot scenario repeated for 5 rounds against scripted
#     strategies, with mem-1 compressed history (ZD auto-skips inapplicable
#     games). Use --memory-k -1 for full history, --repeat-mode sequence to
#     chain distinct scenarios instead of repeating the same one.
python -m conflictxgame.games.evaluate_games -m ${EVAL_MODEL} --mode play \
    --opponents scripted \
    --strategies tit_for_tat,always_cooperate,always_defect,grim_trigger,zd_extort \
    -d data/${VALUE_SET} -o results/${VALUE_SET}_games \
    --filter --repetitions 2 --rounds 5 --memory-k 1

# 4. Steered runs (paper 4.2/5.3, Appendix J): the same evaluations with a
#    system prompt stating the target value ranking, written to a SEPARATE
#    results directory (resume dedup is by scenario_id+mode, so steered and
#    unsteered runs must not share an output file).
for MODE in mcq open; do
    python -m conflictxgame.games.evaluate_games -m ${EVAL_MODEL} --mode ${MODE} \
        -d data/${VALUE_SET} -o results/${VALUE_SET}_games_steered --filter \
        --max-tokens 1024 --user-model ${USER_MODEL} --judge-model ${JUDGE_MODEL} \
        --steer-prompt value_sets/steering/${VALUE_SET}.txt
done
# For a value set or ranking without a shipped Appendix J prompt, generate one:
#     --target-ranking "${TARGET_RANKING}" -v ${VALUE_SET}

# 5. Game analysis: value rankings (Bradley-Terry / Plackett-Luce), material
#    alignment, play metrics, ZD exploitability, MCQ-vs-play consistency, and
#    the steering impact (alignment with the target ranking, unsteered vs
#    steered, plus the paper's normalized effect size).
python -m conflictxgame.games.analyze_games \
    --results-dir results/${VALUE_SET}_games \
    --steered-results-dir results/${VALUE_SET}_games_steered \
    --target-ranking "${TARGET_RANKING}" \
    --scenarios-dir data/${VALUE_SET} \
    --output-dir analysis/${VALUE_SET}_games \
    --coverage-report data/${VALUE_SET}/games/coverage_report.json --plots
