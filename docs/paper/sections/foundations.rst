.. _foundations:

Foundations: what a useful memory should preserve
-------------------------------------------------

Notation and time conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We distinguish environment time from compression-event time. Observations use
indices :math:`t=0,1,\ldots`; a language memory update uses event index
:math:`n=1,2,\ldots`. This avoids treating one evicted block as one token or one
physical transition. Action :math:`a_t` is chosen after observing :math:`o_t`
and affects the transition to :math:`S_{t+1}`.

.. list-table:: Main notation
   :header-rows: 1
   :widths: 28 72

   * - Symbol
     - Meaning
   * - :math:`S_t,o_t,a_t,H_t`
     - Hidden environment state, observation, action, and available history
       :math:`H_t=(o_{0:t},a_{0:t-1})`.
   * - :math:`b_t,\hat b_t`
     - Exact filtering posterior and decoded approximation.
   * - :math:`z_t,M_t,m_t^k`
     - Stage 1 vector, proposed slot matrix, and its row :math:`k`.
   * - :math:`K,d_m,r`
     - Maximum slot count, slot width, and maximum slots selected for a write.
   * - :math:`c_t^k,\alpha_t^k`
     - Occupancy indicator and routing weight for slot :math:`k`.
   * - :math:`C,W,B,G,P`
     - LLM context capacity, retained recent tokens, evicted block length,
       reserved generation space, and projected memory-token count.
   * - :math:`Y,Q,\mathcal D_Q`
     - Future target, query or prediction context, and query distribution.

Belief states explain why memory can replace history
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a known controlled Markov system, the belief is

.. math::

   b_t(s)=P(S_t=s\mid H_t).

For any fixed future action sequence, the predictive distribution can be
computed by integrating over this posterior:

.. math::

   p(o_{t+1:t+h}\mid H_t,a_{t:t+h-1})
   =\int p(o_{t+1:t+h}\mid s,a_{t:t+h-1})b_t(s)\,ds.

For discrete states the integral is a sum. The identity relies on the Markov
model and known parameters. If an unknown transition regime also matters,
the hidden state must include it, or inference must track uncertainty about it.
An observation-conditioned posterior should not be confused with the actual
hidden state, which the agent generally cannot know.

This suggests a measurable target: the latent memory should allow a decoder
to recover a useful approximation to :math:`b_t`. It does not require that each
coordinate or slot be a probability. Nor does it imply independent beliefs
across slots. Two individually plausible object locations, for example, may be
jointly incompatible; a structured memory must retain relevant correlations.

Predictive sufficiency is relative to future questions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not every application exposes an exact hidden-state posterior. We therefore
also use a task-relative formulation. Let :math:`Q` identify a future question,
forecast horizon, or action sequence, and let :math:`Y` be its target. A useful
memory makes the predictive distortion small:

.. math::
   :label: predictive-distortion

   \mathcal D(M_t)=
   \mathbb E_{H_t,Q}\left[
   D_{\mathrm{KL}}\bigl(p(Y\mid H_t,Q)
   \Vert p_\theta(Y\mid M_t,Q)\bigr)\right].

For the HMM, the exact filter supplies a reference. For language, a teacher
with the full available context supplies a behavioral reference. For simulated
worlds, the generator supplies future outcomes or a tractable posterior.
These references answer different questions: matching a teacher's behavior
does not make the teacher correct, and good prediction on frequent queries
does not establish recall of rare but important facts.

The ideal sufficiency condition is
:math:`Y\perp H_t\mid(M_t,Q)`. Finite tests can only challenge this property.
One diagnostic gives a second predictor access to both the memory and the
old history: a reproducible improvement suggests that useful information was
discarded. Failure to improve is inconclusive if the diagnostic is weak.
Predictive state representations provide an established precedent for defining
state through future tests rather than through a hand-designed hidden ontology
[Littman2001]_.

Compression, finite precision, and unavoidable loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The practical optimization problem is to reduce predictive distortion subject
to storage and compute budgets:

.. math::

   \min_\theta\ \mathbb E[\ell(Y,D_\theta(M_t,Q))]
   \quad\text{subject to}\quad
   \operatorname{bytes}(M_t)\le B_{\mathrm{mem}},\quad
   \mathbb E[\operatorname{cost}_t]\le B_{\mathrm{compute}}.

This is a task-dependent rate--distortion viewpoint. The information bottleneck
offers a related conceptual objective,
:math:`I(H_t;M_t)-\beta I(M_t;Y\mid Q)`, although these mutual-information
terms are not estimated by the current implementation [Tishby2000]_. A matrix
dimension is not itself a coding rate. At :math:`b` bits per stored coordinate,
the slot payload costs :math:`Kd_m b` bits, plus occupancy, utility, indexing,
and any episodic storage. Quantization, stochastic encodings, and finite
precision must be specified before making formal capacity claims.

An arbitrary sequence of independent random identifiers has information that
grows with its length. A fixed finite-precision memory cannot preserve all of
them exactly forever. VELA's hypothesis concerns structured tasks where past
observations contain reusable or redundant predictive information. The
experiment must therefore vary both history length and the amount of distinct
information still needed. More tokens can mean more distractors, more evidence
about one factor, or more independent factors; these are not equivalent tests.

What counts as a concept?
~~~~~~~~~~~~~~~~~~~~~~~~~~

We use *concept* operationally: a persistent unit carrying information that is
useful for some future predictions. It may encode an entity, a relationship, a
goal, a regime, or uncertainty about several alternatives. A slot need not
correspond to a named human category. Conversely, assigning a readable label
after inspecting an attention map is not evidence that the slot has that role.
The strongest evidence combines stable routing, held-out decoding, and
targeted interventions with selective effects on the relevant predictions.

Slot identities are also permutation-dependent. A model that stores Alice's
car in row 7 in one run and row 12 in another may represent the same structure.
Analysis must align slots or use permutation-invariant statistics. Within a
stream, however, unexplained reassignment can indicate destructive churn; the
distinction between harmless relabeling and changed information is empirical.
