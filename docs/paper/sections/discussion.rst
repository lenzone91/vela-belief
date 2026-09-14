Discussion and current conclusion
-----------------------------------------

Stage 1 establishes a working posterior-learning baseline with explicit numerical
and causal checks. A small supervised GRU reproduces the belief of a fixed
two-state HMM closely and uses history more effectively than an observation-only
oracle. The modular implementation makes this result a starting point for
comparisons, rather than evidence for a particular future memory design.

Several limitations constrain interpretation. The tested HMM has only one free
posterior probability, whereas the recurrent vector has 32 coordinates. This is
compression relative to an observation history of growing length, not evidence
of a smaller representation than the exact belief. No minimal-capacity result is
claimed. There is no parameter-distribution shift, learned fixed-window baseline,
unsupervised training result, or independent multi-seed uncertainty estimate.
The model is trained to decode an exact teacher posterior, and its long-sequence
evaluation does not isolate retention of very old evidence.

Before drawing broader conclusions, subsequent stages should use disjoint
experiment-level data streams, additional parameter settings, matched baselines,
and sequence-level uncertainty estimates that respect temporal dependence.
The final paper should integrate all six stages around a single question: under
what conditions does persistent latent memory preserve useful uncertainty about
the hidden state, and at what cost? At present, only the controlled first step
has been demonstrated.
