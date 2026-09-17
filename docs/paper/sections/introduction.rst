.. _introduction:

Introduction: from remembering a sequence to maintaining a state
----------------------------------------------------------------

Why the latest observation is not enough
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sensor reading, a sentence, or a camera frame is usually an incomplete view of
an evolving situation. A noisy measurement can be consistent with several hidden
regimes. A conversation can refer to an entity described thousands of tokens
earlier. An object can disappear behind an obstacle while continuing to exist.
In each case, useful prediction requires combining current evidence with a
representation of the past. The difficult part is deciding what that
representation should preserve as the history grows.

Consider a running example. A stream first mentions a red car owned by Alice.
Later, it states that Alice's car is electric. Still later, a reliable update
reports that she sold it to Bob. A useful memory should connect the first two
observations, revise the ownership relation after the third, and retain enough
temporal information to distinguish current ownership from historical ownership.
It should not confuse this car with another red car. If an exact vehicle
identifier will be queried later, a broad semantic description is insufficient.
This example combines identity, accumulation, revision, uncertainty, and precise
recall; these are different demands on a finite memory.

Three straightforward responses expose the design problem. Retaining the entire
history preserves the source material but increases storage and attention costs.
Keeping only a recent window bounds those costs but removes old evidence when it
crosses an arbitrary boundary. Recurrently compressing everything into one vector
bounds the state size but gives limited direct control over which information is
rewritten, protected, or consolidated. None of these observations establishes
that a structured alternative will perform better. They motivate controlled
experiments in which the reasons for remembering are observable.

The progression of the VELA hypothesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VELA-Belief began with a minimal question: can a learned recurrent state preserve
the posterior information required to infer a hidden state? Its first experiment
uses a categorical Hidden Markov Model (HMM), for which exact causal inference
is available [Rabiner1989]_. This choice separates uncertainty about the correct
answer from uncertainty about the learned memory. It also supplies a working
baseline before more complicated memory operations are introduced.

The conceptual revision in ``patch.md`` extends that question. If several
persistent factors matter independently, can memory organize them into evolving
units, update only the affected units, allocate space for new factors, combine
redundant units, and release information whose future value is small? The target
is a bounded matrix of latent slots rather than an ever-growing archive:

.. math::

   o_{0:t}\longrightarrow M_t\in\mathbb{R}^{K\times d_m}
   \longrightarrow\text{predictions, beliefs, or decisions}.

The number of rows :math:`K` and their width :math:`d_m` are fixed deployment
budgets. A row is a candidate unit of persistent information, not necessarily an
object or a human-readable concept. The hypothesis is that explicit memory
operations, coupled to predictive objectives, can produce useful organization
within that budget. Merely reshaping a vector into a matrix cannot do this:
the routing, interaction pattern, learning signal, and evaluation must make the
structural claim meaningful.

.. figure:: _generated/memory-cycle.*
   :name: memory-cycle
   :width: 100%
   :alt: New evidence is encoded, routed, written or allocated, consolidated,
         and forgotten selectively; the bounded memory is read by task heads.

   Proposed memory lifecycle. Arrows describe information flow, not implemented
   stage 1 modules. Consolidation and deletion are conditional maintenance
   operations and need not execute after every observation.

Two uses of the same memory principle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a **world model**, the memory is an inferred state of a partially observed
environment. It must preserve uncertainty, support action-conditioned prediction,
and eventually enable useful imagined trajectories. Observations can arrive
continuously; inference updates the state whenever new evidence is available.

For a **language model**, the proposed deployment is more specific: VELA is a
long-term overflow memory. The normal Transformer context is used until older
content must leave the active window. Only that evicted content is then folded
into a bounded latent memory, while recent tokens remain available verbatim.
Subsequent overflow events update the same memory. The LLM still reads the
memory between events, but the proposed writer is triggered by eviction.
This makes repeated compression, rather than a single successful encoding of a
document, a central evaluation condition.

These applications share a formulation, not a demonstrated transferable model.
They need different encoders, task distributions, readouts, and training
objectives. In particular, a frozen LLM will not automatically understand
arbitrary latent slots, and a good state estimator is not automatically a
generative dynamics model. Sections :ref:`llm-overflow` and :ref:`world-models`
make these interfaces explicit.

What this paper contributes today
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The manuscript combines an implemented baseline with a detailed research
proposal. The distinction is important when interpreting every diagram and
equation that follows.

.. list-table:: Evidence and scope
   :header-rows: 1
   :widths: 22 39 39

   * - Status
     - Content
     - What it supports
   * - Implemented
     - Categorical HMM, exact filter, observation-only oracle, supervised GRU,
       explicit streaming state, archived metrics and correctness tests.
     - A reproducible starting point for measuring posterior preservation.
   * - Proposed
     - Slots, sparse routing, allocation, consolidation, forgetting, and
       objectives for predictive organization.
     - A concrete design space and falsifiable experimental program.
   * - Not yet evaluated
     - Semantic specialization, LLM overflow integration, latent world-model
       rollouts, and practical efficiency gains.
     - Questions for later stages; no performance or novelty result yet.

We retain the historical name **VELA — Value-Aware Evolving Latent Agent** for
the broader project. In this paper, *VELA memory* denotes its proposed persistent
structured belief component. The phrase *value-aware evolving latent memory*
describes the revised direction without implying that a complete agent has been
implemented. Here, value primarily means usefulness for future prediction under
a specified task distribution; decision-dependent value is a later extension.

How to read the manuscript
~~~~~~~~~~~~~~~~~~~~~~~~~~

The conceptual path is problem, precedents, mathematical target, memory
operations, learning pressures, and application interfaces. The stage 1 section
then reports the existing evidence. The evaluation section explains how to test
claims that this small HMM cannot establish. Finally, the nine-stage roadmap
turns each additional mechanism into an experiment with dependencies,
comparisons, deliverables, and a decision criterion. A glossary and a traceability
table connect the revised project context to the corresponding sections.
