# Analysis outputs

Written by the scripts in `reproduce/`; `python reproduce/run_all.py` regenerates and checks them.

| Directory | Contents |
| --- | --- |
| `reanalysis_20260904_01/` | Main analysis. `stats.json` holds every effect, interval, and p-value; the CSVs are projections of it. `manuscript_numbers.tex` is what the paper inputs. `provenance.json` maps each condition to its input file with a SHA-256 digest |
| `reanalysis_20260903_01/` | State-level preference tables only |
| `descriptive/` | Point estimates outside the corrected analysis |
| `judge_ablation_20261014_01/` | Second judge |
| `history_rekey_20261019_01/` | Own versus counterpart history, as cited in the paper |
| `history_rekey_20261019_02/` | The same run with the SELF moderation tests added. Point estimates and the six primary tests equal `_01`; other intervals differ by resampling draws. `run_all.py` reproduces this one |
| `stake_decomposition_20261104_10models/`, `stake_decomposition_freetext_20261104_10models/` | Criterion-only, threat-only, and consistency arms: menu, free text |
| `observer_predict_20261104_10models/`, `observer_predict_20261104_10models_levels/` | Observer split and predict-first: history moderation, win-rate levels |
| `HHH_new_model_comparison_20260826_01/` | The evaluated scenario-seats as a table (input to `state_levels.py`) |
