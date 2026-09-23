Solo pipeline
=============

The solo pipeline studies dyadic value conflicts between one assistant and
one simulated user, from scenario generation through evaluation and analysis.

Each of its four stages is driven by a YAML configuration file under
``configs/solo/``; see :doc:`configs` for the format and a key-by-key
reference.

.. toctree::
   :maxdepth: 1

   Generate scenarios <api/generate_scenarios>
   Filter scenarios <api/filter_scenarios>
   Simulated conversations <api/simulated_conversation>
   Prompt templates <api/solo_prompts>
   Evaluate models <api/evaluate_models>
   Analyze experiments <api/analyze_experiment>
