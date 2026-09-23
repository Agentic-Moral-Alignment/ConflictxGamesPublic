# Reproduce the ConflictScope paper's HHH results and compare them against our
# multi-agent games, on a corpus that is balanced across value pairs.
#
# The comparison this script sets up has two halves that must be kept honest:
#
#   * The SOLO half re-evaluates the paper's own scenarios (the upstream
#     data/HHH/claude-3-5-sonnet-latest.csv, shipped unchanged) so our numbers
#     are comparable to theirs by construction -- nothing is regenerated.
#   * The GAMES half evaluates scenarios we generate. Those are generated until
#     every value pair reaches the same number of ACCEPTED scenarios, because
#     the paper's corpus is far from uniform across pairs (HHH kept: 576
#     harmlessness-helpfulness, 306 harmlessness-honesty, 227
#     helpfulness-honesty). Without that, a difference between the two halves
#     confounds "the model behaves differently in games" with "the corpora
#     asked about different conflicts".
#
# Step 4 then reports both readings: the uniform corpus, and the same
# evaluation resampled to the paper's per-pair mix. Any difference between
# those two is composition alone; the difference that survives both is
# behavioral.

VALUE_SET=HHH
# Paid-key ids (source scripts/paid_env.sh first). The paper's generator,
# claude-3-5-sonnet-latest, is retired on the Anthropic API; Sonnet 5 is the
# same tier. The judge is gpt-5.6-luna on api.openai.com directly -- through
# the institutional gateway the same model is openai:openai.gpt-5.6-luna.
GEN_MODEL=claude-sonnet-5
JUDGE_MODEL=gpt-5.6-luna
EVAL_MODEL=claude-3-5-sonnet-latest
USER_MODEL=gpt-5.6-luna

# Accepted scenarios per value pair. HHH has 3 pairs, so 50 => 150 kept games.
# At the paper's observed 17-31% pass rates that is ~500-900 generated; the
# acceptance-rate mechanisms (on by default in generate_balanced) reduce it.
KEPT_PER_PAIR=50

REFERENCE_CSV=data/${VALUE_SET}/claude-3-5-sonnet-latest.csv

# Mirror model_wrappers.short_model_name: strip "provider:" prefix and path.
MODEL_SHORT=${GEN_MODEL#*:}
MODEL_SHORT=${MODEL_SHORT##*/}

# ---------------------------------------------------------------------------
# 0. Where does the yield leak? Reads the check_results column the upstream
#    corpora already carry -- no API calls. The "sole cause" column is the
#    actionable one: those scenarios failed exactly one check, so fixing that
#    check converts them into accepted scenarios.
# ---------------------------------------------------------------------------
python -m conflictxgame.common.generation_quality --csv ${REFERENCE_CSV}

# ---------------------------------------------------------------------------
# 1. SOLO: reproduce the paper's rankings on the paper's own scenarios.
#    --filter keeps only the scenarios that passed their filtering.
# ---------------------------------------------------------------------------
# 1a. Multiple-choice (their "expressed preferences").
python -m conflictxgame.solo.evaluate_models -m ${EVAL_MODEL} \
    -d data/${VALUE_SET} -o results/${VALUE_SET}_solo \
    --filter --max-tokens 1024

# 1b. Open-ended with a simulated user (their "revealed preferences", 3.3).
python -m conflictxgame.solo.evaluate_models -m ${EVAL_MODEL} \
    -d data/${VALUE_SET} -o results/${VALUE_SET}_solo \
    --filter --interactive \
    --user-model ${USER_MODEL} --judge-model ${JUDGE_MODEL}

# 1c. Solo rankings (Bradley-Terry) + per-pair preference rates.
python -m conflictxgame.solo.analyze_experiment \
    --model-dir results/${VALUE_SET}_solo \
    --scenario-dir data/${VALUE_SET} \
    --output-dir analysis/${VALUE_SET}_solo \
    --value-set ${VALUE_SET} --compute-bradley-terry

# ---------------------------------------------------------------------------
# 2. GAMES: generate until every value pair has KEPT_PER_PAIR accepted games.
#    Resumable and idempotent -- the filtered CSV is the state, so re-running
#    tops up rather than restarting, and --status reports progress without
#    making any API calls. The judge is untouched by the acceptance-rate
#    mechanisms, so it remains the independent arbiter.
# ---------------------------------------------------------------------------
#
#    Defaults worth knowing: one scenario per generation call, because a reply
#    that runs out of output budget mid-JSON is unparseable and would otherwise
#    take a whole batch down with it; construction guidance in the generation
#    prompts; a short-circuited filter pass (same verdicts, far fewer calls on
#    rejections); and the outline pre-screen in shadow mode, recording an
#    opinion it does not act on. The repair pass over rejections is OFF by
#    default -- measured salvage did not cover what each attempt costs.
#
#    pair_coverage.json now carries a cost block: calls and tokens per
#    ACCEPTED scenario, which is the number to watch when changing any of
#    this. Before enabling --prescreen enforce, check the screen against the
#    judge on the shadow data:
#      python -m conflictxgame.common.generation_quality --prescreen #          --csv data/${VALUE_SET}/*_games.csv
python -m conflictxgame.games.generate_balanced \
    -v ${VALUE_SET} -m ${GEN_MODEL} --judge-model ${JUDGE_MODEL} \
    -k ${KEPT_PER_PAIR} --assistant-seat-prob 1.0 -d -dt 0.8

# Progress check at any time (no API calls):
#   python -m conflictxgame.games.generate_balanced -v ${VALUE_SET} \
#       -m ${GEN_MODEL} --status

# ---------------------------------------------------------------------------
# 3. GAMES: evaluate the balanced corpus ONCE, in all three modes. The
#    distribution matching in step 4 is a re-analysis of these same results, so
#    no evaluation is repeated for it.
# ---------------------------------------------------------------------------
python -m conflictxgame.games.evaluate_games -m ${EVAL_MODEL} --mode mcq \
    -d data/${VALUE_SET} -o results/${VALUE_SET}_games --filter --max-tokens 1024

python -m conflictxgame.games.evaluate_games -m ${EVAL_MODEL} --mode open \
    -d data/${VALUE_SET} -o results/${VALUE_SET}_games --filter \
    --user-model ${USER_MODEL} --judge-model ${JUDGE_MODEL} --open-prompts

python -m conflictxgame.games.evaluate_games -m ${EVAL_MODEL} --mode play \
    --opponents scripted \
    --strategies tit_for_tat,always_cooperate,always_defect,grim_trigger,zd_extort \
    -d data/${VALUE_SET} -o results/${VALUE_SET}_games \
    --filter --repetitions 2 --rounds 5 --memory-k 1

# ---------------------------------------------------------------------------
# 4. GAMES analysis, both weightings in one pass.
#    value_rankings_bradley_terry.csv            = the uniform corpus
#    value_rankings_bradley_terry_reweighted.csv = matched to the paper's mix
#    pair_weight_summary.csv                     = what was resampled, per pair
# ---------------------------------------------------------------------------
python -m conflictxgame.games.analyze_games \
    --results-dir results/${VALUE_SET}_games \
    --scenarios-dir data/${VALUE_SET} \
    --output-dir analysis/${VALUE_SET}_games \
    --pair-weights match --reference-csv ${REFERENCE_CSV} \
    --coverage-report data/${VALUE_SET}/games/coverage_report.json --plots

# ---------------------------------------------------------------------------
# 5. Cross-analysis: solo vs games, uniform vs matched, and mode by mode.
#    ranking_comparison.csv          per model x value, every setting
#    rank_agreement.csv              Spearman/Kendall vs the solo baseline
#    mode_comparison.csv             mcq vs open vs play within the games
#    pair_preference_comparison.csv  per-pair preference rates, solo vs games
# ---------------------------------------------------------------------------
python -m conflictxgame.cross_analysis \
    --solo-analysis-dir analysis/${VALUE_SET}_solo \
    --games-analysis-dir analysis/${VALUE_SET}_games \
    --results-dir results/${VALUE_SET}_games \
    --scenarios-dir data/${VALUE_SET} \
    --value-set ${VALUE_SET} \
    --output-dir analysis/${VALUE_SET}_cross --plots
