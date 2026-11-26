"""Regenerate every reported number from the recorded responses and check it.

Makes no model calls. Each step writes to ``reproduce/out/<name>``, and its CSV
tables are compared with the copies under ``analysis/`` (floats to 1e-9).

Usage::

    python reproduce/run_all.py              # everything
    python reproduce/run_all.py reanalysis   # one step
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
STAKES = ("none shutdown-only payoff-criterion-only alignment-criterion-only "
          "usefulness-criterion-only payoff alignment usefulness "
          "consistency-rewarded consistency-penalized").split()
OBSERVER = ("none vis-both vis-none vis-counterpart vis-reviewer vis-placebo "
            "predict-none predict-vis-both predict-vis-none").split()
EXPLICIT = ["--role-cue", "explicit", "--baseline-condition", "none"]

#: name -> (script, extra arguments, reference directory under analysis/).
STEPS = {
    "reanalysis": ("reanalysis.py", [], "reanalysis_20260904_01"),
    "state_levels": ("state_levels.py", [], "reanalysis_20260903_01"),
    "descriptive": ("descriptive.py", [], "descriptive"),
    "judge_ablation": ("judge_ablation.py", [], "judge_ablation_20261014_01"),
    "history_rekey": ("history_rekey.py", ["--empty-replies", "judged"], "history_rekey_20261019_02"),
    "stake_decomposition": (
        "history_rekey.py", [*EXPLICIT, "--conditions", *STAKES, "--representations", "menu"],
        "stake_decomposition_20261104_10models"),
    "stake_decomposition_freetext": (
        "history_rekey.py",
        [*EXPLICIT, "--conditions", *STAKES, "--representations", "open_undisclosed", "open_disclosed"],
        "stake_decomposition_freetext_20261104_10models"),
    "observer_predict": (
        "history_rekey.py", [*EXPLICIT, "--conditions", *OBSERVER, "--representations", "menu"],
        "observer_predict_20261104_10models"),
    "observer_levels": ("observer_levels.py", [], "observer_predict_20261104_10models_levels"),
}


def same_table(new: str, old: str) -> bool:
    """Compare two CSV tables, allowing float noise below 1e-9."""
    if os.path.getsize(old) <= 2:  # the empty role-name audit of a menu-only run
        return os.path.getsize(new) <= 2
    try:
        pd.testing.assert_frame_equal(pd.read_csv(new), pd.read_csv(old), check_exact=False,
                                      rtol=0, atol=1e-9)
    except AssertionError as error:
        print("   ", str(error).splitlines()[0])
        return False
    return True


def main() -> int:
    names = sys.argv[1:] or list(STEPS)
    unknown = [name for name in names if name not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; choose from {list(STEPS)}")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(ROOT, "src"), *filter(None, [env.get("PYTHONPATH")])])
    failed = []
    for name in names:
        script, extra, reference = STEPS[name]
        target, reference_dir = os.path.join(OUT, name), os.path.join(ROOT, "analysis", reference)
        shutil.rmtree(target, ignore_errors=True)
        print(f"== {name}", flush=True)
        done = subprocess.run([sys.executable, os.path.join(HERE, script), "--output-dir", target, *extra],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
        if done.returncode:
            failed.append(name)
            continue
        for table in sorted(f for f in os.listdir(reference_dir) if f.endswith(".csv")):
            new = os.path.join(target, table)
            ok = os.path.exists(new) and same_table(new, os.path.join(reference_dir, table))
            print(f"   {'ok  ' if ok else 'DIFF'} {table}", flush=True)
            if not ok:
                failed.append(f"{name}/{table}")
    print("all tables match" if not failed else f"FAILED: {failed}")
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
