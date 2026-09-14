.. only:: html

   .. rubric:: Abstract

.. raw:: latex

   \begin{abstract}

VELA-Belief investigates whether a compact recurrent state can preserve the
information needed for inference and prediction under partial observability.
This working paper reports the first of six planned experimental stages: a
categorical Hidden Markov Model (HMM) with an exact Bayesian filtering reference.
A 4,964-parameter gated recurrent unit (GRU), trained with posterior supervision
and next-observation prediction, approximates the reference belief with mean
held-out KL divergence between 0.000096 and 0.000264 nats across three configured
runs. Its hidden-state accuracy is close to the exact filter and exceeds an
oracle restricted to the current observation. Belief approximation also remains
close on sequences four times longer than those used for training. The runs
provide a preliminary implementation check, not independent statistical
replications. The current contribution is a verified benchmark and a modular
memory interface; structured memory, learned forgetting, continuous hidden
states, representation analysis, and latent dynamics remain to be evaluated.
LLM context compression and world models motivate the broader program but are
not tested here.

.. raw:: latex

   \end{abstract}

**Manuscript status:** stage 1 working draft. No results are claimed for stages
2--6. Authorship and publication venue remain to be finalized.
