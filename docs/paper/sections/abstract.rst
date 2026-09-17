.. only:: html

   .. rubric:: Abstract

.. raw:: latex

   \begin{abstract}

Sequential systems need to preserve information whose usefulness can outlast
the context in which it was observed. VELA-Belief studies a bounded persistent
latent memory that could organize this information into evolving state slots.
The proposed lifecycle routes new evidence, updates or allocates slots,
consolidates redundant representations, and forgets information according to
its estimated future predictive value. We develop the connection to Bayesian
belief states and task-dependent compression, position the proposal relative
to recurrent memory, latent context compression, KV-cache methods, and
object-centric models, and specify two application interfaces: a structured
state for world models and an overflow memory alongside an LLM's exact recent
context. Three central hypotheses concern the benefits of structure, sparse
writes, and repeated consolidation under matched resource budgets. Only the
first experimental stage is currently implemented. On a fixed two-state HMM,
a 4,964-parameter supervised GRU achieves held-out posterior KL between
0.000096 and 0.000264 nats across three configured, statistically dependent
runs. This validates the initial benchmark, not the proposed slot architecture.
The paper provides a nine-stage research map, explicit failure modes, and
protocols for testing retention, uncertainty, cost, specialization, and drift
over repeated memory updates.

.. raw:: latex

   \end{abstract}

**Manuscript status:** expanded research design with stage 1 results; stages
2--9 are prospective. The conceptual source is the repository's ``patch.md``.
Authorship and publication venue remain to be finalized.
