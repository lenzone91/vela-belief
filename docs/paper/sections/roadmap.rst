Stages 2--6: planned experiments
----------------------------------------

The following sections define the questions that will extend this paper.
They are prospective protocols, not completed experiments. Each stage should
add its method, controlled comparisons, results, and limitations to this
manuscript while retaining stage 1 as a regression baseline.

Stage 2: structured memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace the recurrent vector with :math:`K` memory slots and a readout into
the common latent interface. Compare slot counts, slot widths, and update
mechanisms with the GRU under stated parameter, memory, and computational
budgets. Include trained fixed-window and recurrent baselines. Test whether
slots improve posterior quality or efficiency, rather than interpreting
attention patterns alone as evidence of specialization. Report both successes
and null results.

Stage 3: learned forgetting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Introduce explicit replacement or deletion and compare it with gated recurrence
and slot memory without that mechanism. Use environments that independently
control how long evidence remains relevant and when previously useful evidence
becomes obsolete. Measure retention of delayed information, adaptation after
changes, and errors caused by premature forgetting. Matched ablations should
separate the effect of forgetting from additional model capacity.

Stage 4: continuous hidden states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Move first to a linear Gaussian state-space model with a known Gaussian initial
state, using a Kalman filter as the posterior reference. Evaluate both posterior
mean and covariance with suitable distribution heads and scoring rules. Only
then extend to nonlinear or stochastic-volatility settings, where an approximate
reference has its own error. Separate reference approximation error from learned
representation error.

Stage 5: representation analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use held-out probes, controlled interventions, memory ablations, and capacity
sweeps to examine which posterior quantities the state retains. Evaluate
representation stability and whether information about earlier history improves
prediction after conditioning on the learned state. A decodable posterior alone
does not establish an approximately Markovian learned representation. Avoid
interpreting visually separated latent clusters as sufficient evidence.

Stage 6: latent transition model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn :math:`p_\phi(z_{t+1}\mid z_t)` and assess multi-step observation rollouts,
calibration, and error accumulation. Compare against direct observation
prediction and reference-model simulations. A transition over learned belief
states must account for future observation uncertainty; accurate one-step
decoding is not by itself a valid generative rollout model.

For action-driven extensions, condition inference on the action history and
use :math:`p_\phi(z_{t+1}\mid z_t,a_t)` with suitable observation and reward
decoders. Planning and language context compression require separate benchmarks
and are not prerequisites for claiming completion of the current six-stage
belief-state study. Their evaluation should follow once the relevant interface
and task are implemented.
