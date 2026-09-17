.. _discussion:

Discussion: what would make this memory useful?
------------------------------------------------

The conceptual change is from storing a compact trace to maintaining a bounded
state whose units can be reorganized. This perspective connects Bayesian
filtering, learned context compression, and structured world modeling. Its
appeal is that repeated evidence could enrich an existing unit rather than
consume another permanent record, while consolidation could recover capacity
without discarding all the information in one row. Its difficulty is that
identity, future usefulness, and redundancy are themselves uncertain quantities
that must be learned.

What the current evidence establishes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 supplies a functioning supervised posterior-learning benchmark. A small
GRU approximates the exact filter on a fixed two-state HMM, uses history more
effectively than an observation-only oracle, and remains accurate on tested
sequences longer than its training horizon. Its 32-coordinate state is much
larger than the one free probability of the exact posterior. The experiment
therefore establishes neither minimal memory nor an advantage of structured
storage. It also does not isolate retention of very old evidence, since the
mixing HMM can itself forget distant observations.

The reported runs use overlapping seed offsets across experiments, so their
variation cannot provide independent statistical uncertainty. There is no
implemented slot controller, merge operator, LLM adapter, or world-model rollout
in the current repository. The paper's architecture and application sections
are explicit proposals for those components.

Failure modes that shape the design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sparse router can collapse onto one row. A persistence loss can freeze stale
facts. A diversity penalty can separate representations that should share
information. A merge can conflate two similar entities, and a consistency loss
can fail to notice because the reader ignores the damaged attribute. A utility
predictor can favor frequent queries while losing rare instructions. An LLM
adapter can preserve perplexity while failing exact retrieval. A world model
can produce convincing short rollouts while being miscalibrated under a new
action policy. These are concrete outcomes for the experiments, not exceptions
to a presumed successful architecture.

A bounded memory also needs a bounded claim. It can be updated for arbitrarily
many steps without increasing its slot count, but this does not mean it retains
arbitrarily much information or remains accurate indefinitely. Compression is
relative to tasks, precision, and changing distributions. If an application
needs a complete auditable history, an additional archive is a different system
component with its own storage budget.

What would constitute an advance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The strongest result would locate a reproducible region of the quality,
memory, compute, and stability tradeoff where structured maintenance is useful,
and explain the mechanism through controlled interventions. It might show that
sparse writes protect independent factors, that learned merging helps under
redundancy and capacity pressure, or that a frozen reader can exploit repeated
memory updates at a worthwhile total cost. These gains need not occur together
or in every domain.

A careful negative result is informative too. If a matched GRU or recurrent
compressor performs as well, if merging fails after many events, or if re-prefill
erases latency savings, the simplest supported system should be preferred.
The project's scientific contribution should be the evidence and the resulting
understanding of memory organization, rather than the number of mechanisms
included. The next concrete step is stage 2: a minimal slot baseline on tasks
with enough independent information to make structure testable.
