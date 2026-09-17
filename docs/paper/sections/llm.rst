.. _llm-overflow:

LLM application: a long-term overflow memory
---------------------------------------------

What changes when the context fills
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before the first eviction, the proposed system uses the LLM's ordinary active
context. At an overflow event :math:`n`, a block :math:`X_{\mathrm{evicted},n}`
would otherwise be discarded. VELA updates its persistent state:

.. math::

   M_n=U_\theta(M_{n-1},E_\theta(X_{\mathrm{evicted},n})).

The remaining recent tokens are kept verbatim. A projected memory prefix and
those tokens then condition the model:

.. math::
   :label: llm-budget

   [P_\theta(M_n),X_{\mathrm{recent}}],\qquad P+W+G\le C.

Here :math:`P` is the number of soft memory tokens, :math:`W` the active text
length including any pinned instructions, and :math:`G` reserved generation
space. In the simplest one-token-per-slot implementation, :math:`P=K`.
If retrieved episodic excerpts are inserted, their token count must also enter
the inequality. Memory tokens consume context positions rather than creating
free extra positions outside the window.

For an illustrative capacity :math:`C=128{,}000`, evicting
:math:`B=16{,}000` tokens from a full text window leaves 112,000 verbatim tokens.
Adding 256 memory tokens uses 112,256 positions, leaving 15,744 before any
additional generation reserve. These are arithmetic examples, not the
specification or measured limits of an integrated model. Eviction can be
triggered early enough to respect the chosen generation budget.

.. figure:: _generated/overflow.*
   :name: overflow-figure
   :width: 100%
   :alt: At successive overflow events, old blocks update one bounded latent
         memory, while the recent verbatim suffix remains in the active window.

   Proposed LLM flow across overflow events. The writer processes evicted
   evidence; the reader consumes memory and recent text. Repeated arrows do
   not imply that all old information remains exactly recoverable.

The encoder may consume token embeddings, hidden activations, or representations
computed while a block was still active. These choices differ in cost and
causal content. Never encode text that occurs after the memory snapshot being
evaluated. If evicted-block activations already depended on an older memory,
reusing them can duplicate previous evidence; compare raw-block encoding with
contextual encoding and test repeated mentions explicitly.

VELA-S: soft tokens for a frozen reader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest reader maps each occupied row into the LLM embedding space:

.. math::

   P_\theta:\mathbb R^{d_m}\to\mathbb R^{d_{\mathrm{model}}},\qquad
   p_{\mathrm{student}}(y)=
   p_{\mathrm{LLM}}(y\mid P_\theta(M_n),X_{\mathrm{recent}}).

The LLM weights can be frozen while the writer and projection are trained.
This preserves its parameter values, but it does not make the integration
training-free or available through every text API. The runtime needs input
embedding access and gradients through the frozen model to the memory prefix.
Freezing parameters removes their optimizer state; it does not remove the
activation memory needed for this backward pass. A ``no_grad`` block around
the student reader would sever the required gradient path.

ICAE provides a precedent for learned memory slots [Ge2024]_. Compatibility,
however, depends on the particular reader, positional scheme, projection, and
training distribution. Compare a linear projection, a small nonlinear adapter,
and any reader adaptation explicitly. If LoRA or other reader weights are
trained, report that variant separately from a strictly frozen LLM.

A cache-consistency constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changing a prefix is not generally compatible with reusing every old KV entry
for the remaining suffix. Those entries were computed using the previous
prefix and positional assignments. Recent *tokens* can remain exact while
their cached *activations* require recomputation.

A scientifically clean first implementation updates memory and re-prefills the
retained suffix under the new prefix with a specified position policy. Measure
this cost at each event. A faster implementation might retain some stale
activations or use a specialized cache update, but it must be described as a
separate approximation and compared against recomputation. The same issue
arises at the first insertion of memory tokens. Position IDs, rotary embeddings,
causal masks, pinned instructions, and cache length all need an explicit policy;
changing them silently can confound the memory experiment.

Distillation from an available full context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a training example whose history and target suffix fit within the teacher's
window, split the history into old and recent portions. The teacher sees both;
the student sees compressed old information and the exact recent portion:

.. math::

   p_{\mathrm{full},j}=p_{\mathrm{LLM}}
   (y_j\mid X_{\mathrm{old}},X_{\mathrm{recent}},y_{<j}),

.. math::

   p_{\mathrm{mem},j}=p_{\mathrm{LLM}}
   (y_j\mid P_\theta(M),X_{\mathrm{recent}},y_{<j}).

A candidate objective combines behavioral distillation and grounded targets:

.. math::

   \mathcal L_{\mathrm{LLM}}=
   \sum_j D_{\mathrm{KL}}(p_{\mathrm{full},j}\Vert p_{\mathrm{mem},j})
   -\lambda_{\mathrm{ans}}\sum_j\log p_{\mathrm{mem},j}(y_j^*)
   +\mathcal L_{\mathrm{struct}}.

Teacher distributions are detached. Both branches receive the same query and
target prefix; neither writer receives a future answer. The target suffix must
also fit the teacher budget. Matching full-vocabulary distributions can be
expensive; any approximation using top logits or sampled tokens changes the
objective and should be documented.

A teacher with capacity :math:`C` cannot supply full-context targets for an
arbitrary 10-million-token history. First train multiple smaller eviction events
within a teacher-readable sequence, then evaluate longer streams using generated
ground-truth queries or another explicitly described reference. A larger teacher
changes the comparison. Repeated short compressions can teach recurrence but do
not certify generalization to indefinitely many updates.

VELA-KV: a layer-aware reader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A later design projects the same persistent state into keys and values for each
layer:

.. math::

   P_\theta^{(\ell)}(M_n)=
   (K_{\mathrm{mem}}^{(\ell)},V_{\mathrm{mem}}^{(\ell)}),
   \qquad\ell=1,\ldots,L.

This gives each layer direct access to memory but requires the appropriate
head dimensions, positional conventions, masks, and cache layout. If each layer
stores :math:`P` memory positions with :math:`H_{\mathrm{kv}}` KV heads and
head dimension :math:`d_h`, the projected payload alone is approximately

.. math::

   B_{\mathrm{KVmem}}=2LPH_{\mathrm{kv}}d_h b_{\mathrm{elt}}

bytes, where :math:`b_{\mathrm{elt}}` is bytes per element. This is in addition
to the semantic state, recent KV cache, and adapter parameters. Grouped-query
attention uses the KV-head count rather than the query-head count. Direct KV
projection may reduce some prefix-processing cost, but does not automatically
eliminate suffix-cache inconsistency or guarantee a latency improvement.

Semantic and episodic memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some tasks require exact identifiers, quotations, numbers, or code fragments.
A proposed hybrid stores the pair

.. math::

   \mathcal M_n=(M_n^{\mathrm{semantic}},E_n^{\mathrm{episodic}}),\qquad
   \operatorname{bytes}(\mathcal M_n)\le B_{\mathrm{total}}.

This is a pair of stores, not an arithmetic sum of incompatible representations.
The semantic component accumulates predictive abstractions; the episodic
component retains a bounded set of exact records or retrievable references.
A reference is useful only if its backing storage still exists, and that
storage must be counted. An unbounded external archive changes the bounded
memory claim even if its index is small.

The router could decide whether evidence needs exact preservation, semantic
integration, both, or neither. Evaluate this decision using separate semantic
and exact-match tasks. In the car example, an ownership update and a vehicle
identifier may require different storage policies. Exact recent tokens alone
do not solve exact recall once the relevant token has itself been evicted.

Why repeated consolidation is the defining stress test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A one-time compression test measures the first transition. Deployment repeatedly
reuses already compressed information:

.. math::

   M_0\to M_1\to\cdots\to M_N.

Evaluate facts introduced before event 1, facts introduced later, updates to
old facts, and queries requiring evidence from several events. Vary
:math:`N`, block size, slot capacity, and the number of simultaneously relevant
facts independently. Track retention, stale-answer rate, lost uncertainty,
occupancy, and cumulative merge count. Compare the recurrent memory with a
one-shot recompression of the full available source where that reference is
feasible. This identifies loss due to repeated updating separately from loss
due to the final capacity bottleneck.

A latent memory could avoid the autoregressive generation needed for a text
summary and may encode information compactly, but its encoder, maintenance,
projection, and re-prefill costs can dominate. Performance is a measured
end-to-end property, not an automatic consequence of fewer stored vectors.
