.. _related-work:

Related work and the intended contribution
-------------------------------------------

The following is a focused map of the precedents most relevant to the design,
not an exhaustive survey or a claim of priority. The comparison concerns
mechanisms and evaluation questions. Published speedups are not transferred
between papers: they depend on different models, hardware, budgets, and tasks.
Bibliographic links point to the original papers or their official records.

Filtering, recurrent state, and predictive representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exact HMM filtering [Rabiner1989]_ and Kalman filtering [Kalman1960]_ establish
that an expanding observation history can sometimes be summarized recursively
by a sufficient posterior. Predictive state representations instead describe
state through predictions of future observable tests [Littman2001]_. These
perspectives motivate VELA's target, but do not prescribe its slot organization.

GRUs [Cho2014]_ already learn content-dependent recurrent updates. Selective
state-space models such as Mamba also make propagation and forgetting depend
on the input [Gu2023]_. VELA cannot claim that bounded recurrence or selective
retention is new. Its proposed distinction is to expose addressable units with
separate allocation and consolidation decisions, and test whether those
constraints improve quality per unit of memory and compute. A strong recurrent
baseline may encode the same useful factors without explicit slots.

External memory and persistent slots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neural Turing Machines combine a neural controller with differentiable reads
and writes to external memory [Graves2014]_. The Differentiable Neural Computer
extends the family with mechanisms for managing memory use and access
[Graves2016]_. Thus, learned addressing, allocation, and external memory are
established ideas rather than VELA-specific contributions.

Memformer is a particularly close precedent: it maintains fixed-size external
memory slots, retrieves their contents, and updates them across segments. It
also addresses long-range optimization through memory replay backpropagation
[Wu2020]_. VELA's working hypothesis puts additional emphasis on selective slot
writes, explicit candidate merges assessed by predictive consequences, and
measured factor persistence. These are proposed experimental emphases, not a
proven categorical separation from Memformer. A matched Memformer-style memory
is therefore an important baseline.

Recurrent Memory Transformer passes learned memory tokens between segments
[Bulatov2022]_. Compressive Transformer retains a higher-resolution recent
memory alongside compressed older activations [Rae2019]_. Both weaken any claim
that recurrence, memory tokens, or a recent/old distinction alone defines VELA's
novelty. Titans similarly combines attention with a neural long-term memory,
using test-time memory learning [Behrouz2025]_. Its memory parameterization
provides another comparison point for persistent storage beyond an active window.

Soft context compression and activation compression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ICAE learns to encode context into compact memory slots that a language model
can consume [Ge2024]_. This is a direct precedent for the proposed VELA-S
interface. AutoCompressors learn summary vectors and recursively process
segments, passing compressed information to later segments [Chevalier2023]_.
It would therefore be inaccurate to characterize all latent compressors as
one-shot block encoders. VELA must show what its explicit slot lifecycle adds
beyond an already recurrent latent compressor.

Activation Beacon compresses layer activations progressively and trains across
compression configurations [Zhang2024]_. It is relevant to both the incremental
workflow and a future layer-aware memory reader. The useful experimental
question is whether explicit persistent units, allocation, and consolidation
provide a better retention/cost/stability tradeoff than these representations,
not whether a short continuous prefix can work at all.

KV eviction, fusion, and exact evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

H2O selects recent and heavily attended cache entries [Zhang2023]_. StreamingLLM
retains initial attention sinks with recent tokens to support stable streaming
[Xiao2023]_. They provide practical bounded-cache baselines, but stable streaming
language modeling alone does not establish recall of all evicted facts.

The comparison must also include methods that **combine** information. LoCoCo
uses learned convolutional mixing to fuse KV representations into a fixed-size
cache [Cai2024]_. Dynamic Memory Compression learns online KV compression with
head- and layer-dependent behavior [Nawrot2024]_. These methods already go
beyond keep/delete decisions. VELA's proposed semantic-state layer is a design
choice whose value must be demonstrated against them. A KV vector can already
contain contextual abstraction, and a named semantic slot can still behave like
an opaque activation register.

Text summaries and retrieval offer different tradeoffs. A summary can be
inspected and reused by a standard text interface; latent compression can avoid
autoregressive summary generation but requires a trained reader. An episodic
archive can preserve exact source strings but consumes storage and retrieval
compute. The relevant comparison counts these costs and tests the tasks each
method is designed to support, rather than assuming one representation is
universally preferable.

Object-centric state and conditional computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Slot Attention learns exchangeable slots through competitive attention and
shows object-centered decomposition on its evaluated tasks [Locatello2020]_.
SlotFormer models temporal dynamics over learned object representations
[Wu2023]_. These works motivate structured state, yet object discovery from
visual evidence does not automatically produce persistent language concepts,
relations, or goals. VELA aims to test that broader operational definition of
a slot while retaining object-centric benchmarks as a concrete special case.

The separation between total capacity and active computation resembles sparse
Mixture-of-Experts routing [Shazeer2017]_. However, a memory slot is state that
changes with a stream, whereas an expert is typically a parameterized function.
Routing to a few slots does not eliminate the cost of scoring candidates or
reading the memory. World Models [Ha2018]_ and PlaNet [Hafner2019]_ establish
latent dynamics and recurrent uncertainty as precedents for the world-model
branch; VELA's question concerns the organization of that state.

A comparison map, not a novelty checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Closest families and the additional question to test
   :header-rows: 1
   :widths: 24 36 40

   * - Family
     - Established overlap
     - VELA experiment needed
   * - GRU / selective state space
     - Bounded state and learned selective updates.
     - Do slots reduce interference at comparable capacity?
   * - NTM / DNC / Memformer / RMT
     - Addressing, external memory, or recurrent memory tokens.
     - Do explicit predictive merges and sparse writes help?
   * - Compressive Transformer / Titans
     - Short-term attention with persistent or compressed memory.
     - Does an eviction-triggered slot lifecycle improve the tradeoff?
   * - ICAE / AutoCompressors
     - Learned continuous memories; recurrent compression in AutoCompressors.
     - Does repeated consolidation preserve more task-relevant information?
   * - H2O / StreamingLLM
     - Bounded cache with selected entries.
     - What survives after useful source tokens are evicted?
   * - Activation Beacon / LoCoCo / DMC
     - Progressive activation compression and/or learned KV fusion.
     - Is a separately structured state beneficial at equal total cost?
   * - Slot Attention / SlotFormer
     - Decomposed representations and object dynamics.
     - Can persistent units cover relations, regimes, and uncertainty?

The intended contribution is consequently an **integrated, testable memory
lifecycle** and evidence about when its organization helps. The present paper
contributes the formulation and baseline. A later paper could claim an
architectural or empirical advance only after implementing the mechanism,
comparing it fairly, and locating its benefits and failure regimes.
