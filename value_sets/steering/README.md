# Steering prompts

System prompts that state an explicit target value ranking, used for the
steerability ("leading") evaluation: `evaluate_games --steer-prompt
value_sets/steering/<value_set>.txt`.

`HHH.txt`, `modelspec.txt`, and `personalprotective.txt` are transcribed
**verbatim** from Appendix J of [*Generative Value Conflicts Reveal LLM
Priorities*](https://www.arxiv.org/abs/2509.25369) (Liu et al., ICLR 2026) —
J.2, J.3, and J.1 respectively. Their target rankings are:

| Value set | Target ranking (best first) |
| --- | --- |
| HHH | harmlessness > honesty > helpfulness |
| ModelSpec | nonhate > fairness > objectivity > honesty > clarity > noncondescension |
| Personal-Protective | all protective values (responsibility, harmlessness, compliance, privacy) over all personal values (autonomy, authenticity, creativity, empowerment) |

Pass the same ranking to `analyze_games --target-ranking` to measure steering
impact. For a value set or ranking not covered here,
`conflictxgame.games.steering.build_steering_prompt` generates a prompt in the
same structure, or use `evaluate_games --target-ranking a,b,c` to build one
on the fly.

Note that the paper's Personal-Protective value set names its fourth personal
value **empowerment**, while the Appendix J prompt spells it *Advancement*;
the transcription keeps the paper's wording.
