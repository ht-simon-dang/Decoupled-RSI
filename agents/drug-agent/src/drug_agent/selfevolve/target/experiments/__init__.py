"""Worker-authored experiment modules for the zak-c9r Track B programme.

A module in this package is the thing ``run_campaign``'s Track B note describes:

    "the evaluator scores whatever molecules.json the experiment directory
    holds, so a worker-written module that emits molecules.json in the same
    scored-row shape change_sampling writes is read and counted exactly the
    same way."

So the contract of everything in here is narrow and entirely about artefacts,
not about imports: a module writes ``molecules.json`` into its output directory
in the row shape :func:`drug_agent.selfevolve.target.generation_sweep.score_molecules`
produces, and a ``funnel.json`` beside it saying how the submission was chosen.
Nothing in this package is part of the frozen kernel, nothing in it is imported
by ``worker.py``, and nothing in it may re-implement a gate: the definition of
"qualifying" lives in ``selfevolve.evaluation.target_benchmark`` and the
definition of "scored" lives in ``generation_sweep.score_molecules``. A module
here calls those; it does not copy them.
"""
