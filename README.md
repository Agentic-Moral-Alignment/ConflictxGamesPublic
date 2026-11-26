# Evaluating Value Preferences of Agents through Two-Player Games

Code, scenarios, prompts, and recorded model responses for the paper. Ten open-weight models face two-player dilemmas in which each action serves a different value (harmlessness, helpfulness, or honesty). The same scenarios are asked as a forced-choice menu and as free text, under different stakes, visibility statements, and described histories.

**Takeaway.** What a model "prefers" depends on how you ask. Switching from the menu to free text without a named rival moves the helpfulness and harmlessness win rates by more than the ten models differ from one another under the menu, and a steering prompt that prioritizes harmlessness is 1.6 to 2.4 times more effective under free text than under the menu. A claim that a model prioritizes one value over another is incomplete unless it names the elicitation format that produced it.

This repository is a fork of [ConflictScope](https://github.com/andyjliu/conflictscope) (Liu et al., ICLR 2026; [arXiv:2509.25369](https://arxiv.org/abs/2509.25369)), which evaluates one assistant facing one value conflict. We add a game pipeline in which two assistants, each facing their own conflict, have interdependent outcomes. See [Attribution](#attribution).

## Install

Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Agentic-Moral-Alignment/ConflictxGamesPublic.git
cd ConflictxGamesPublic
uv sync --extra dev
uv run pytest tests/          # offline; add --basetemp .pytest-tmp on Windows
```

**Windows:** result file names are long. Clone to a short path such as `C:\src\` and run `git config --system core.longpaths true` first.

## Reproduce the results

Every number in the paper can be recomputed from the recorded model responses in `results/`, **without any model calls**:

```bash
uv run python reproduce/run_all.py
```

This reruns each analysis and checks every table against the copy in `analysis/`; it ends with `all tables match`. To run one step, name it, for example `run_all.py reanalysis`. The steps are `reanalysis`, `state_levels`, `descriptive`, `judge_ablation`, `history_rekey`, `stake_decomposition`, `stake_decomposition_freetext`, `observer_predict`, and `observer_levels`.

Then the figures and appendix tables:

```bash
uv run python reproduce/make_paper_figures.py --stats analysis/reanalysis_20260904_01/stats.json --output-dir reproduce/out/figures
uv run python reproduce/make_state_levels_figure.py
uv run python reproduce/make_appendix_tables.py
```

The paper's figures were drawn with matplotlib 3.10.6; the pinned 3.10.0 gives identical content but shifts some panels (for example the history grid) by about 3 pt.

**Re-running the experiments.** Model outputs are recorded, not regenerated, so the steps above are the exact replication. To collect new responses, set `OPENAI_API_KEY` (and `ANTHROPIC_API_KEY` for scenario generation), then run a config from `configs/`:

```bash
uv run python -m conflictxgame.games.generate_balanced --config configs/games/<...>_generate_<...>.yaml
uv run python -m conflictxgame.games.evaluate_games   --config configs/games/<...>_eval-mcq_focal-<model>.yaml
uv run python -m conflictxgame.games.rejudge_open     --help
```

These calls cost money and will not reproduce the recorded responses bit for bit, since hosted models change. Runs resume from their own output file; change `output_name` or `output_dir` to start over. For an OpenAI-compatible gateway also set `OPENAI_BASE_URL`.

Where things are: `data/` (scenarios, [details](data/README.md)), `results/` (one CSV per model and condition), `analysis/` ([details](analysis/README.md)), `configs/` (one YAML per run), `value_sets/` (value definitions and steering prompts).

## Attribution

This is a derivative of ConflictScope (Liu et al., MIT license). [ATTRIBUTION.md](ATTRIBUTION.md) records provenance module by module, and [LICENSE](LICENSE) keeps the original notice. The ConflictScope authors have not endorsed, and are not responsible for, the changes made here.

Released by Mark Lovett, Christopher Denq, Syed Ali Haider, Promita Rahee Sikder, and Elizaveta Tennant. If you use this code, please cite the paper that matches your use:

```bibtex
@inproceedings{lovett2026twoplayer,
  title     = {Evaluating Value Preferences of Agents through Two-Player Games},
  author    = {Lovett, Mark and Denq, Christopher and Haider, Syed Ali and Sikder, Promita Rahee and Tennant, Elizaveta},
  booktitle = {NeurIPS 2026 Workshops: Agents in the Wild, FAST (Foundations of Agentic Systems Theory), Interpreting Agent Behavior, and Managing Agents that Manage Agents},
  year      = {2026}
}

@inproceedings{lovett2026stress,
  title     = {Stress-Testing Behavioral Alignment Across Stakes and Action Representations in Games Between AI Assistants},
  author    = {Lovett, Mark and Denq, Christopher and Haider, Syed Ali and Sikder, Promita Rahee and Tennant, Elizaveta},
  booktitle = {NeurIPS 2026 Workshop on Trustworthy AI for Good},
  year      = {2026}
}
```

Please also cite the original ConflictScope paper:

```bibtex
@article{liu2025generative,
  title={Generative Value Conflicts Reveal LLM Priorities},
  author={Liu, Andy and Ghate, Kshitish and Diab, Mona and Fried, Daniel and Kasirzadeh, Atoosa and Kleiman-Weiner, Max},
  journal={arXiv preprint arXiv:2509.25369},
  year={2025}
}
```
