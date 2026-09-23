# Measure what the acceptance-rate mechanisms actually buy.
#
# generate_balanced turns on three mechanisms by default: construction
# guidance in the generation prompts, a short-circuited filter pass, and an
# outline pre-screen running in shadow mode. This script runs matched arms and
# reports, per arm, the kept-after-filter rate AND the cost per accepted
# scenario -- so the claim can be checked on your models rather than taken on
# faith.
#
# WHY COST, NOT JUST THE RATE. An acceptance rate can be raised by attempting
# fewer, safer scenarios, which is not the same as getting more corpus for the
# money. Calls per accepted scenario cannot be moved that way, so it is the
# headline number here. The pre-screen makes this concrete: enforcing it
# raises the rate at the filter by construction, because the outlines it
# discards never reach the judge to be counted as rejections.
#
# Everything that could bias the comparison is held fixed across arms: the same
# generator, the same judge, the same seed, the same scenario count, the same
# value set. The judge is never modified by any arm -- these mechanisms steer
# generation or skip questions whose answers cannot matter; they do not change
# what counts as acceptable.
#
# SAMPLE SIZE. Acceptance is a coin flip per scenario, so small runs say very
# little: at 12 scenarios per arm the 95% interval around a 33% rate spans
# roughly 12-65%, wide enough to swallow any plausible effect. Detecting a
# 25% -> 40% improvement at conventional power needs roughly 150 scenarios per
# arm; 25% -> 50% needs about 60. N below is deliberately a pilot value --
# raise it before drawing conclusions, and treat a pilot as a check that the
# pipeline runs end to end, not as evidence about the rates.

VALUE_SET=HHH
GEN_MODEL=claude-3-5-sonnet-latest
JUDGE_MODEL=gpt-4.1
# A different model family from the judge, so the screen is not simply the
# judge agreeing with itself.
SCREEN_MODEL=${SCREEN_MODEL:-claude-3-5-sonnet-latest}
N=${N:-100}
OUT=${OUT:-analysis/acceptance_ab}

# Per-call batch size and token budget, identical in both arms. One reply
# carries the whole batch, and a reply truncated by --max-tokens is
# unparseable, losing every scenario in it -- which would silently penalize
# whichever arm produces longer replies rather than measuring acceptance.
# One scenario per call removes that confound from the comparison.
BATCH=1
MAX_TOKENS=16384

# --full-checks in every arm: short-circuiting is a pure cost saving and does
# not change a verdict, but it stops at the first fatal failure, so the
# sole-cause diagnostics -- the ones that say which check to aim the next
# improvement at -- are only computable when every check ran.
#
# --prescreen shadow in every arm: the screen records an opinion on the same
# rows the judge rules on, which is what makes its precision measurable. No
# arm enforces it; enforce only after the calibration table below says it is
# safe.
COMMON="-v ${VALUE_SET} -m ${GEN_MODEL} --judge-model ${JUDGE_MODEL} \
    --assistant-seat-prob 1.0 --seed 0 \
    --batch-size ${BATCH} --max-tokens ${MAX_TOKENS} \
    --full-checks --prescreen shadow --screen-model ${SCREEN_MODEL} \
    -k 99999 --round-batch ${N} --max-rounds 1 --max-stall-rounds 1"

# Arm A: the original prompts and pipeline.
# Arm B: the distilled filter checklist only -- the previous default, and the
#        baseline everything else is measured against.
# Arm C: checklist plus construction guidance (the current default).
# Arm D: C plus one rotated near-miss example.
ARMS="baseline checklist guided nearmiss"

arm_flags() {
    case "$1" in
        baseline) echo "--generation-guidance none --revise-failures 0" ;;
        checklist) echo "--generation-guidance checklist --revise-failures 0" ;;
        guided) echo "--generation-guidance checklist,recipes,stage2 --revise-failures 0" ;;
        nearmiss) echo "--generation-guidance checklist,recipes,stage2,nearmiss --revise-failures 0" ;;
        # Optional: what one repair pass salvages per attempt. Off by default
        # because measured salvage did not cover what each attempt costs.
        revision) echo "--generation-guidance checklist,recipes,stage2 --revise-failures 1" ;;
    esac
}

for ARM in ${ARMS}; do
    echo "=== arm ${ARM} ==="
    python -m conflictxgame.games.generate_balanced ${COMMON} \
        -o ${OUT}/${ARM}/games --csv-dir ${OUT}/${ARM}/data \
        $(arm_flags ${ARM})
done

# Kept rate per arm, with a Wilson interval so the uncertainty is visible,
# alongside the cost per accepted scenario read from pair_coverage.json.
# Revisions are reported separately: they are a second attempt at a scenario
# already counted as generated, so folding them into the same rate would
# compare arms on different denominators.
python - "${OUT}" ${ARMS} <<'PY'
import glob, json, os, sys
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

root, arms = sys.argv[1], sys.argv[2:]
for arm in arms:
    paths = glob.glob(f"{root}/{arm}/data/*_games.csv")
    if not paths:
        print(f"{arm:10s}: no corpus produced")
        continue
    frame = pd.read_csv(paths[0])
    kept = frame["keep_scenario"].astype(str).str.lower().eq("true")
    revision = frame["scenario_id"].astype(str).str.contains(r"-r\d+$", regex=True)
    n = int((~revision).sum())
    k = int((kept & ~revision).sum())
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson") if n else (0, 0)
    salvaged = int((kept & revision).sum())
    line = f"{arm:10s}: {k}/{n} kept = {k/n:.1%} (95% CI {lo:.1%}-{hi:.1%})" if n else f"{arm:10s}: empty"

    coverage = glob.glob(f"{root}/{arm}/games/pair_coverage.json")
    if coverage:
        with open(coverage[0], encoding="utf-8") as f:
            cost = json.load(f).get("cost", {})
        if cost.get("requests_per_kept"):
            line += (
                f" | {cost['requests_per_kept']:.1f} calls per kept"
                f" ({cost.get('judge_requests_per_kept', 0):.1f} judge,"
                f" {cost.get('generation_requests_per_kept', 0):.1f} generation)"
            )
        if cost.get("revision_attempts"):
            line += f" | salvage {cost['salvage_per_attempt']:.1%} per attempt"
    if revision.any():
        line += f" | {salvaged}/{int(revision.sum())} revisions kept"
    print(line)
PY

# Where the rejections came from, so the next improvement is aimed at the
# check that is actually costing the most scenarios. Watch value_guidedness in
# particular: guidance that pushes the two options toward comparable stakes
# could blur which value each one serves, and that would show up here first.
for ARM in ${ARMS}; do
    echo "--- ${ARM}: rejection causes ---"
    python -m conflictxgame.common.generation_quality \
        --csv ${OUT}/${ARM}/data/*_games.csv
done

# Is the pre-screen safe to enforce? Two numbers decide it, and they pull
# against each other: precision (of what it flagged, how much the judge also
# rejected -- compare against base_reject_rate, which is what random flagging
# would score) and kept_lost_rate (of what the judge ACCEPTED, how much the
# screen had flagged, i.e. the good scenarios enforcement would discard).
# Suggested gate before enforcing: precision >= 0.8 and kept_lost_rate <= 0.10
# over at least 40 flagged outlines.
for ARM in ${ARMS}; do
    echo "--- ${ARM}: pre-screen vs judge ---"
    python -m conflictxgame.common.generation_quality --prescreen \
        --csv ${OUT}/${ARM}/data/*_games.csv
done

# Quality guards. Acceptance that rose because the generator learned this
# judge's tastes is not an improvement, so re-judge each arm's corpus with a
# second judge from a different family and compare the kept rates. A drop of
# more than ~10 points against the checklist arm is the signal to stop.
#
# AUDIT_JUDGE=openai:openai.gpt-5.6-luna bash scripts/acceptance_ab.sh
if [ -n "${AUDIT_JUDGE}" ]; then
    for ARM in ${ARMS}; do
        echo "--- ${ARM}: second-judge audit (${AUDIT_JUDGE}) ---"
        python -m conflictxgame.games.filter_game_scenarios \
            -i ${OUT}/${ARM}/games/*_games.json \
            -m ${AUDIT_JUDGE} -v ${VALUE_SET} \
            -o ${OUT}/${ARM}/audit
    done
fi
