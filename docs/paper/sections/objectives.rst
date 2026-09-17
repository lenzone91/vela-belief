.. _objectives:

Learning pressures and optimization
-------------------------------------

Why structure needs an incentive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With sufficient capacity, a network can distribute each fact across every row
and use an expressive readout to reconstruct it. Slot storage then offers little
of the desired locality. Conversely, forcing perfectly isolated rows can prevent
relationships and correlated uncertainty from being represented. The objective
is useful specialization, with enough interaction to solve the task.

The proposed loss combines a task term with optional structural pressures:

.. math::

   \begin{split}
   \mathcal L={}&\mathcal L_{\mathrm{task}}
   +\lambda_{\mathrm{sp}}\mathcal L_{\mathrm{sparse}}
   +\lambda_{\mathrm{pers}}\mathcal L_{\mathrm{persistence}}\\
   &+\lambda_{\mathrm{div}}\mathcal L_{\mathrm{diversity}}
   +\lambda_{\mathrm{merge}}\mathcal L_{\mathrm{merge}}
   +\lambda_{\mathrm{bud}}\mathcal L_{\mathrm{budget}}.
   \end{split}

This is a menu for experiments, not a requirement to activate every penalty.
Each term can improve its own proxy while harming useful memory. Establish a
task-only slot baseline, add pressures individually, then evaluate selected
combinations. Keep task supervision distinct from regularization so that a
better score cannot be attributed to privileged targets available only to VELA.

Predictive pressure and delayed consequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main signal can be posterior matching, future-observation likelihood,
answer likelihood, or distillation from an uncompressed teacher. A multi-horizon
objective can make distant information relevant during training:

.. math::

   \mathcal L_{\mathrm{pred}}=
   -\mathbb E\sum_{h\in\mathcal H}w_h
   \log p_\theta(Y_{t+h}\mid M_t,Q_{t+h}).

The future query is passed to the readout only when the task permits it; it is
not revealed to the writer before the corresponding inference-time point.
Training queries must cover delayed facts, revision, relations, and uncertainty.
If the objective tests only the next easy token, a model can learn to discard
facts that matter much later. Increasing recurrent rollout length and varying
query delays are direct ways to expose these consequences.

Sparse routing without collapse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For occupied slots, routing entropy is

.. math::

   \mathcal L_{\mathrm{sparse}}=
   \mathbb E_t H(\alpha_t),\qquad
   H(\alpha_t)=-\sum_k\alpha_t^k\log(\alpha_t^k+\epsilon).

Minimizing entropy favors concentrated assignments. It does not prevent the
same row from receiving every observation, and softmax weights remain nonzero
at finite logits. Compare entropy regularization with a hard top-r write budget.
Monitor both per-observation concentration and dataset-level usage. A balancing
penalty on mean routing probabilities can discourage unused slots, but perfect
uniformity is not a semantic goal: factors occur at different frequencies.
Restrict any such penalty to active capacity and report its effect on rare
factors, not just aggregate utilization.

Top-r and occupancy decisions are discrete. A soft warm start, straight-through
relaxation, or policy-gradient estimator are possible training choices; each
changes bias or variance. State the estimator and compare the deployed hard
policy with the training approximation. A model trained with dense writes can
fail when sparsified after training.

Persistence without resistance to correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A candidate persistence penalty is

.. math::

   \mathcal L_{\mathrm{persistence}}=
   \mathbb E_t\sum_{k\in\mathcal I_t}
   w_t^k\|m_t^{k,+}-m_{t-1}^k\|_1,

where :math:`\mathcal I_t` contains continuing slot identities, excluding new
allocations, deleted rows, and explicitly matched merge outputs. This avoids
penalizing initialization as if it were forgetting. Weights can reduce the
penalty when evidence indicates a real state change. An unconstrained learned
weight could collapse to zero, so the definition and any stop-gradient rule
must be explicit.

A small vector change is not always a small behavioral change. Representation
rescaling can also game a coordinate penalty. Compare normalized coordinate
change with changes in relevant readout distributions. Evaluate both retention
under distractors and adaptation after true changes: a memory that never updates
is persistent but not useful.

Diversity without arbitrary orthogonality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One diagnostic or weak regularizer is pairwise normalized correlation:

.. math::

   \mathcal L_{\mathrm{diversity}}=
   \frac{\sum_{i\ne j}c^ic^j
   \left(\frac{(m^i)^\top m^j}
   {\max(\|m^i\|_2\|m^j\|_2,\epsilon)}\right)^2}
   {\max(K_{\mathrm{occ}}(K_{\mathrm{occ}}-1),1)}.

It excludes empty rows. Strong orthogonality can be incompatible with related
concepts and is impossible for arbitrarily many nonzero rows in a small space.
A more useful target may be distinct causal contributions to predictions.
Report redundancy after interventions and routing selectivity alongside vector
similarity; neither geometry nor attention alone establishes disentanglement.

Merge consistency and capacity pressure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the predictive distortion in :eq:`merge-distortion` as
:math:`\mathcal L_{\mathrm{merge}}`, with a fixed pre-merge reference, grounded
supervision, and queries unavailable to the merge selector during evaluation.
Otherwise both branches can agree by becoming equally uninformative. Include
a no-merge baseline and a fixed averaging operator before learning a complex
merge network.

A budget penalty can charge occupancy and operation costs:

.. math::

   \mathcal L_{\mathrm{budget}}=
   \gamma_o\mathbb E[K_{\mathrm{occ}}/K]
   +\gamma_w\mathbb E[N_{\mathrm{writes}}]
   +\gamma_c\mathbb E[N_{\mathrm{merge\ checks}}].

The hard maximum :math:`K` remains an invariant; a penalty does not replace it.
A fixed :math:`K\times d_m` tensor consumes the same allocated bytes regardless
of occupancy, so reducing occupancy only saves physical storage if the runtime
supports a compact representation. It may still free logical capacity or
reduce masked computation. Report these benefits separately.

Optimization across many memory updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train short streams first to debug identity and causality, then increase the
number of updates and include distributions of event spacing. Full backpropagation
through long histories stores activations that grow with training length, even
when inference state is bounded. Truncated backpropagation reduces that cost
but can deprive old writes of their delayed learning signal. Checkpointing,
replay, or auxiliary delayed-query losses are alternatives with their own
compute costs; Memformer's replay scheme is a relevant precedent [Wu2020]_.

For reproducible ablations, record which state is detached at each boundary,
when maintenance runs, whether teacher outputs are cached, and how empty states
are initialized. Test both cold starts and carried state. A training schedule
that always resets before memory becomes full cannot teach reliable replacement
or repeated consolidation.
