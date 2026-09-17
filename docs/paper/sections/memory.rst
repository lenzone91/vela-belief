.. _structured-memory:

Proposed memory: route, update, allocate, consolidate, forget
-------------------------------------------------------------

State and readout
~~~~~~~~~~~~~~~~~~

The proposed state is a matrix with occupancy and small auxiliary metadata:

.. math::

   \mathcal M_t=(M_t,c_t,\eta_t),\qquad
   M_t=[m_t^1;\ldots;m_t^K]\in\mathbb R^{K\times d_m},\qquad
   c_t\in\{0,1\}^K.

Here :math:`\eta_t` can include age, usage statistics, and estimated utility.
Its size must also be bounded. Empty rows are masked in reads and matching;
otherwise a zero vector could accidentally behave as a meaningful memory.
The matrix is storage; a readout produces the task representation:

.. math::

   r_t=f_\theta(M_t,c_t,Q_t),\qquad
   \hat p_t=D_\theta(r_t,Q_t).

A pooled readout is the simplest baseline. Query-conditioned attention allows
one task to read ownership and another to read location from the same state.
Interactions between slots can encode relationships, but dense interactions
increase cost and may undermine the intended locality. A first implementation
should compare shared per-slot updates with a small interaction module before
adding a full Transformer over memory.

ROUTE: find relevant existing information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encode the observation as :math:`x_t=E_\theta(o_t)` and score occupied slots:

.. math::

   q_t=q_\theta(x_t),\qquad k_t^j=k_\theta(m_{t-1}^j),\qquad
   s_t^j=\frac{q_t^\top k_t^j}{\sqrt{d_k}},\qquad
   \alpha_t^j=\frac{\exp(s_t^j/\tau)}
   {\sum_{i:c_{t-1}^i=1}\exp(s_t^i/\tau)}.

Only occupied slots enter the denominator. When no slot is occupied, allocation
is handled directly. A dense baseline updates all occupied slots. A sparse
variant selects :math:`\mathcal S_t=\operatorname{Top}_r(\alpha_t)` and
renormalizes the weights on that set. With :math:`K=256` and :math:`r=4`, at
most four existing rows receive the expensive write; the router can still score
all 256 rows. Low-temperature softmax makes weights concentrated but not
literally sparse. Hard top-k selection supplies exact sparsity, with a discrete
selection boundary and different gradient behavior.

Crucially, relative attention weights do not measure absolute semantic match.
If every slot is a poor match, softmax still sums to one and can be sharply
peaked. A threshold on :math:`\max_j\alpha_t^j` alone is therefore an unreliable
novelty detector. The proposed allocator should use a calibrated match logit,
a learned new-slot option, or a novelty predictor trained on controlled data.
For example, jointly score occupied slots and a NEW alternative:

.. math::

   p(j\mid x_t,M)=\operatorname{softmax}
   (s_t^1,\ldots,s_t^{K_{\mathrm{occ}}},s_t^{\mathrm{new}})_j.

Calibration and dependence on occupancy still require testing. High confidence
can mean either a good match or a systematically overconfident router.

UPDATE: revise a persistent unit without rewriting everything
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an existing selected row, a gated residual update is a useful starting point:

.. math::

   g_t^k=\sigma(G_\theta(m_{t-1}^k,x_t)),\qquad
   \widetilde m_t^k=F_\theta(m_{t-1}^k,x_t),

.. math::

   m_t^{k,+}=m_{t-1}^k+
   \widetilde\alpha_t^k(1-g_t^k)\odot
   (\widetilde m_t^k-m_{t-1}^k).

Unselected rows remain unchanged during this write operation. The gate can
retain an old attribute while incorporating a new one, but the equation alone
does not ensure coherent entity binding. A shared update network is a clean
baseline because adding more slots then increases state capacity without
necessarily adding one parameter set per slot.

In the car example, observing that Alice's car is electric should enrich an
existing representation if identity is established. A later ownership change
should revise the current relation while retaining historical ownership only
if relevant queries require it. Contradictions are not always corrections:
uncertain sources may require retaining alternatives with confidence and time
information. A benchmark must specify whether latest evidence is authoritative,
noisy, or merely another claim.

ALLOCATE: admit a factor not represented adequately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the NEW option is selected and an empty slot :math:`k` exists,

.. math::

   m_t^{k,+}=A_\theta(x_t),\qquad c_t^{k,+}=1.

Allocation initializes metadata as well as content. With no free row, the
controller must choose between a validated consolidation, replacement of a
low-value row, partial integration into an existing row, or rejection of the
new item. Rejection is a necessary baseline: always evicting something to store
new information can destroy important old evidence for irrelevant noise.

The conceptual lifecycle puts allocation before consolidation, but an actual
implementation can run maintenance before allocation when capacity is full.
Specify a deterministic ordering for ties and competing writes. Batched blocks
may contain multiple references to the same factor, so parallel writes need
an aggregation rule or an explicitly ordered update. Otherwise results can
depend on incidental batching rather than the information stream.

CONSOLIDATE: recover capacity by reducing predictive redundancy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two rows may have been allocated separately before later evidence reveals that
they refer to the same entity or encode overlapping information. A candidate
merge constructs :math:`m^*=C_\theta(m^i,m^j,\eta^i,\eta^j)` and proposes a
state :math:`M^{i,j\to *}` with one occupied row released. This is distinct from
simply deleting one row: the output can retain complementary attributes.

.. figure:: _generated/consolidation.*
   :name: consolidation-figure
   :width: 100%
   :alt: Two candidate car slots are merged, future-query predictions are
         compared before and after, and one slot is released only if accepted.

   Illustrative consolidation. Labels describe the intended information, not
   decoded outputs of an implemented model. Identity evidence is required;
   sharing a color or owner alone does not prove two mentions denote one car.

Cosine similarity can cheaply propose candidates but is not a sufficient
criterion for merging. Similar vectors can represent different entities, and
geometrically different vectors can support equivalent predictions. A functional
criterion compares readouts on a distribution of future questions:

.. math::
   :label: merge-distortion

   \Delta_{ij}=\mathbb E_{Q\sim\mathcal D_Q}
   D_{\mathrm{KL}}\!\left(
   p_{\bar\theta}(Y\mid M,Q)\ \Vert\
   p_\theta(Y\mid M^{i,j\to *},Q)\right).

The pre-merge prediction is a stopped-gradient or frozen reference during
training. A small :math:`\Delta_{ij}` suggests that the tested predictions are
preserved; it does not prove that all useful information survived. If both
readouts ignore a rare identifier, this criterion misses its loss. Include
held-out delayed queries and a grounded task loss alongside consistency.
Testing every pair with a full downstream model is costly; use a cheap candidate
scorer followed by a limited number of functional checks, and charge those
checks to the compute budget.

A merge operator should be order-invariant when the inputs carry no temporal
roles, or explicitly encode which source is newer. It need not be associative:
merging :math:`i` with :math:`j` first may change a later merge with :math:`k`.
Randomize and evaluate merge order, track cumulative distortion, and preserve
metadata needed to interpret contradictions. A recovered empty row is useful
only if the retained predictions justify the information loss.

FORGET: remove information with low expected future value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A conceptual definition of a row's predictive utility is the loss increase
caused by its removal:

.. math::

   u_k=\mathbb E_{Q,Y}\left[
   \ell(Y,D(M\setminus k,Q))-\ell(Y,D(M,Q))\right].

Positive utility indicates that removal harms the evaluated tasks. This is a
counterfactual training or analysis target, not an oracle available online.
A deployed controller estimates it from observable features, such as age,
usage, surprise, source reliability, redundancy, and context. Recent attention
is only one feature. A code mentioned once can have high future utility even
if it has never yet been retrieved.

Utility is conditional on the other rows. Redundant slots can each have low
individual removal cost even though deleting both is harmful. Maintenance
must therefore recompute or approximate utility sequentially rather than
independently removing every low-scoring slot. Deletion resets occupancy and
metadata; replacement must not leak the old row through stale keys or readout
caches. Exact-item retention can instead use a separate bounded episodic store,
as discussed in :ref:`llm-overflow`.

One update cycle in operational form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following is proposed pseudocode, not an API currently in the repository.
It makes the capacity invariant explicit while leaving the learned decisions
open to ablation.

.. code-block:: text

   input: bounded state (slots, occupied, metadata), new evidence
   x = encode(evidence)
   route = score_existing_slots_and_new_option(state, x)
   if route selects existing information:
       update at most r selected occupied slots
   else:
       if no free slot:
           try one budgeted, validated merge
       if still no free slot:
           compare replacement with rejecting the new item
       if a slot is available:
           allocate and initialize all associated metadata
   if maintenance is scheduled:
       apply budgeted consolidation / forgetting decisions sequentially
   assert occupied_count <= K
   return updated state, task readout

All decisions use the available prefix. Future targets may train the controller
but cannot enter its inference-time state. Full and chunked execution should
agree when they use the same observation order and maintenance schedule.
Changing block boundaries can change a blockwise compressor; that sensitivity
is an additional experimental variable rather than an automatic correctness
failure.
