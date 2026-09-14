Stage 1: discrete hidden-state inference
------------------------------------------------

Environment and exact reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default HMM has two hidden states and two observation symbols, with

.. math::

   \pi=(0.5,0.5),\qquad
   A=\begin{pmatrix}0.97&0.03\\0.03&0.97\end{pmatrix},\qquad
   B=\begin{pmatrix}0.7&0.3\\0.3&0.7\end{pmatrix}.

Here :math:`A_{ij}=P(S_{t+1}=j\mid S_t=i)` and
:math:`B_{io}=P(O_t=o\mid S_t=i)`. The persistent regimes and overlapping
emissions make history useful. The generator samples :math:`S_0\sim\pi` and
then emits :math:`O_0`; there is no transition before the first observation.

The reference performs causal filtering, not smoothing. Its prediction and
conditioning steps are

.. math::

   \bar b_0=\pi,\qquad \bar b_t=b_{t-1}A\quad(t>0),\qquad
   b_t(j)=\frac{\bar b_t(j)B_{j o_t}}
                  {\sum_k\bar b_t(k)B_{k o_t}},
   \qquad q_t=b_tAB.

The implementation uses float64 log-space normalization to support long,
unlikely sequences and structural zeros. Impossible sequences raise an error.
The reference probabilities :math:`q_t` provide an additional check on the
learned next-observation distribution.

We also evaluate an observation-only oracle. It computes
:math:`P(S_t\mid O_t)` from the unconditional marginal :math:`\pi A^t` and
the current emission, then predicts the next observation. This reference knows
the system parameters and time index, but forgets previous observations. It
isolates the value of history; it is not a trained fixed-window neural baseline.

Experimental protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The encoder has 16 dimensions and the GRU state has 32. The two linear heads
bring the total to 4,964 trainable parameters. Training uses Adam with learning
rate 0.003, default betas :math:`(0.9,0.999)`, no weight decay, and gradient-norm
clipping at 1.0. Each of 800 updates processes 64 independent sequences with
64 input observations. Full backpropagation is applied within each sequence;
the memory resets between batches.

Each sampled sequence has one additional observation. Inputs are positions
0--63 and next-observation targets are positions 1--64. The exact filter sees
only the input prefix. This alignment gives every input position a prediction
target without leaking future observations into the memory or belief target.

Validation uses 256 fixed sequences, evaluated every 100 updates. The checkpoint
with the lowest weighted validation loss is selected, rather than the one with
the lowest test error or lowest belief KL alone. The selected model is evaluated
on 512 fresh sequences of length 64 and 512 of length 256. A separate tail score
uses positions 64--255 of the latter, retaining the state inferred from the
preceding prefix. Long-sequence scores use the same predictions for the full
sequence and its tail.

Experiments use seeds 42, 43, and 44. Within a run, model initialization uses
the experiment seed; data generators use offsets +1, +2, +3, and +4 for training,
validation, test, and long test respectively. Test data do not participate in
checkpoint selection. However, offsets overlap across adjacent experiment
seeds. In particular, one run's length-64 test can share prefixes with another
run's length-256 test. Consequently, these runs are not statistically independent
replications. We report their individual values without confidence intervals
or significance claims.

Evaluation measures posterior KL, posterior mean absolute error, hidden-state
accuracy, the multiclass Brier score, next-observation KL against the exact
predictive distribution, and next-observation negative log-likelihood (NLL).
Probability losses are measured in nats. Scores average over sequences and time;
the Brier score sums over classes before averaging. Tiny negative numerical KL
values are clamped to zero. Logarithms of predicted probabilities use a float64
positive floor, so an exact zero prediction receives a large finite penalty.

Results
~~~~~~~~~~~~~~~

.. include:: _generated/tables.rst

The learned posterior is close to the exact reference in all three reported
runs. The observation-only oracle has much larger belief KL and lower state
accuracy, consistent with the importance of accumulating noisy evidence in
this environment. The learned predictive head also approaches the exact
filter's NLL, although it is not constrained to implement the known dynamics.

.. figure:: _generated/stage1.*
   :name: stage1-figure
   :width: 100%

   Individual run results. Left: belief KL on a logarithmic scale, including
   evaluation beyond the training horizon. Right: hidden-state accuracy on the
   length-64 test sets. Connecting lines aid reading and are not a fitted trend;
   seeds have no ordered scientific interpretation. The exact filter and GRU
   accuracies are nearly coincident. No uncertainty intervals are inferred from
   these three dependent runs.

Low tail error shows that the learned update remains useful beyond the training
sequence length for this fixed HMM. It does not establish indefinite memory
retention: the posterior of this mixing system can itself forget distant
observations. A dedicated delayed-information task is needed to assess long-term
retention. Similarly, tiny finite-sample reversals in accuracy do not indicate
that a learned model improves on exact inference under the assumed system.

Software verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reviewed implementation passes 38 parametrized test cases. These tests
address correctness and integration separately from the three full benchmark
runs:

* **Reference inference:** compare filtering beliefs and sequence likelihoods
  against exhaustive hidden-path enumeration on a small asymmetric,
  nonstationary HMM. Check next-observation probabilities, causal prefixes,
  structural zeros, impossible sequences, and 2,000-step unlikely sequences.
* **Data generation:** check seeded reproducibility and empirical initial,
  transition, and emission probabilities over 30,000 sampled sequences. Reject
  invalid distributions and observation indices.
* **Memory semantics:** compare full-sequence, chunked, and one-step execution;
  verify batch isolation and causal outputs. Check gradient flow and deliberate
  state detachment. A small test memory with structured state verifies the
  interface, without claiming a trained slot-memory result.
* **Objectives and metrics:** verify next-observation alignment, averaging over
  time, finite gradients with zero target probabilities, and analytical metric
  values on hand-constructed distributions.
* **Integration:** run a short learning test, reproduce checkpoint evaluation,
  preserve caller RNG state, reject invalid configuration, exercise a
  three-state/four-symbol configuration, and execute the command-line workflow.

Ruff lint and formatting checks complement these tests. A short learning test
requires held-out belief KL below 0.04 and below half the observation-only
reference. It guards against training regressions; it is not the source of the
full experimental results above. No GPU execution or distributed training has
been validated.

Reproducibility record
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The archived runs use Python 3.11.2 and PyTorch 2.14.0+cpu with deterministic
algorithms and one CPU thread. Each run records its resolved configuration,
validation history, selected checkpoint, metrics, and one posterior trace.
Checkpoints contain inference weights, not the optimizer and generator state
needed to resume training. Reloading them reproduces held-out evaluation in
the tested environment. Bitwise identity across library releases or platforms
is not assumed [PyTorchRepro]_.

.. include:: _generated/provenance.rst

The source snapshot contains all reported metrics and configurations. It is
versioned with the manuscript; figures and tables are regenerated by Sphinx
without training a model or fetching external data.
