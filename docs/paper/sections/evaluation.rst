.. _evaluation:

Evaluation: claims, budgets, and failure modes
-----------------------------------------------

Three hypotheses and their possible rejection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**H1: structured memory.** At comparable deployed state and training resources,
multiple addressable slots preserve independently changing factors better than
one recurrent state. Evidence would be a reproducible advantage in posterior
quality, delayed-query accuracy, or adaptation with controlled capacity. Equal
or worse performance across the useful budget range weakens this hypothesis;
more visually appealing slots do not rescue it.

**H2: sparse routing.** Restricting writes to relevant rows improves efficiency,
stability, or specialization without unacceptable predictive loss. Measure wall
time as well as active rows, and intervention selectivity as well as routing
entropy. Dense scoring, scattered GPU operations, or poor load balance may erase
the theoretical saving. A speedup on tiny synthetic states need not transfer
to an LLM pipeline.

**H3: learned consolidation.** Predictively informed merging preserves more
useful information under sustained capacity pressure than replacement alone.
The critical comparison holds state size and input stream fixed and varies
merge availability. A useful result combines regained capacity with acceptable
error after many events. Better performance after one merge but rapid later
drift would not support the long-horizon hypothesis.

Tasks that isolate different demands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Proposed benchmark axes
   :header-rows: 1
   :widths: 25 38 37

   * - Task family
     - Controlled variation
     - Main diagnostic
   * - Discrete filtering
     - Number of states, observation ambiguity, transition rates, factor count.
     - Posterior KL, Brier score, predictive NLL.
   * - Delayed recall
     - Query delay, distractor amount, number of independent facts.
     - Recall versus delay and active information load.
   * - Factor updates
     - Entity count, aliases, attribute revisions, source reliability.
     - Correct binding, stale facts, calibrated uncertainty.
   * - Consolidation
     - Redundancy, identity evidence, merge order, repeated merge count.
     - Distortion per event, recovered capacity, cumulative loss.
   * - Continuous filtering
     - Noise, missing observations, hidden regime changes.
     - Mean and covariance quality, adaptation lag.
   * - LLM overflow
     - Eviction count, block size, recent window, semantic/exact query mix.
     - Answer quality, exact recall, instruction retention, runtime.
   * - World-model rollout
     - Occlusion, action sequence, rollout horizon, task change.
     - Calibration, future prediction, eventual planning return.

The two-state HMM has only one independent posterior probability and can be
solved with a very small state. Retain it as a regression test, then use
factorial HMMs or multi-entity simulators where independently changing factors
create an actual organization problem. When an exact joint filter is tractable,
use it; otherwise retain small exact instances and clearly label approximate
references on larger instances.

For LLM tasks, a single needle test is insufficient. RULER provides a precedent
for varying retrieval, multi-hop tracing, and aggregation [Hsieh2024]_. Extend
such controlled queries with changing facts and repeated overflow. Query an old
item only after its source has left the recent window; otherwise the reader
may bypass memory entirely. Vary nominal stream lengths such as 128k, 256k,
1M, and eventually 10M tokens only as compute permits, and report actual event
counts. These are prospective scales, not supported capacities today.

Baselines and information parity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For synthetic tasks, include exact references where available, observation-only
and fixed-window predictors, GRU and LSTM models, a small causal Transformer,
a selective state-space baseline where appropriate, dense slot memory, and
sparse memory without allocation or merging. Separate analytic oracles that
know environment parameters from learned models that see only observations.
Apply the same privileged posterior targets to learned variants in a supervised
comparison; run prediction-only learning as a separate experiment.

For overflow, include rolling context, a truncation policy, textual summaries,
selected KV eviction, an ICAE-style encoder, AutoCompressors, Activation Beacon,
LoCoCo, DMC, and VELA variants when implementations and compatible checkpoints
are available. Do not label an approximate reimplementation as the published
method without qualification. Use two comparison tracks: a fixed frozen
backbone with a common adapter budget, and adapted systems with their training
cost and reader changes disclosed. Some published methods require adaptation
and cannot enter a strictly frozen-reader track unchanged.

Match the query set, source stream, tokenizer, active context policy, precision,
and hardware wherever applicable. Report separate sweeps for state bytes,
parameter count, and compute if all three cannot be matched simultaneously.
A slot payload, a set of soft tokens, and a layer-wise KV cache are different
units; comparing their row counts alone is misleading.

What bounded state does and does not imply about cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For dense keys computed by linear projections and sparse writes, an illustrative
per-observation cost is

.. math::

   O(Kd_m d_k)+O(Kd_k)+O(r\,C_U)+O(C_{\mathrm{read}}),

where :math:`C_U` is one slot-update cost. Cached keys can reduce the projection
work when only a few slots change, but that cache consumes storage. All-pairs
merge scoring costs :math:`O(K^2d_m)` for simple similarities before any task
readout checks. Scheduling it every :math:`J` observations amortizes the cost
without removing its peak latency. Restricting candidate pairs can help but
can miss important merges.

The state remains bounded in sequence length at fixed :math:`K,d_m`; total work
still grows with the number of processed observations. Training activation
storage can grow with unroll length. For a full-attention Transformer, KV state
is linear in cached length and each new token attends over that cache; full
prefill attention has quadratic arithmetic in length, even if an optimized
kernel avoids materializing a quadratic attention matrix. State storage,
activation storage, and arithmetic are distinct quantities.

For an LLM stream with :math:`N` evictions and :math:`T` processed tokens,
measure an amortized event cost in addition to ordinary decoding:

.. math::

   \overline C_{\mathrm{event}}=
   \frac{1}{T}\sum_{n=1}^{N}
   (C_{\mathrm{encode},n}+C_{\mathrm{write},n}
   +C_{\mathrm{maint},n}+C_{\mathrm{refill},n}).

Report peak accelerator memory, persistent bytes, compression FLOPs if
available, prefill time, per-token decode time, throughput, and eviction-event
latency including tail percentiles. Synchronize device timing, include warmup,
and disclose batch size. Count teacher generation and adapter training
separately from deployment. No efficiency curve is fabricated for this paper.

Interventions and cumulative stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mask one slot, swap it between matched histories, prevent a selected write,
or replay an event with and without a merge. Measure changes in the relevant
predictions and in unrelated predictions. Selective effects support localization;
global degradation suggests distributed or entangled storage. Strong probes
can decode information from many redundant places, so combine probing with
interventions and simple probe baselines.

Plot old-fact retention, current-fact accuracy, and uncertainty against memory
event count. Log allocation, deletion, merge acceptance, routing usage, and
slot identity changes. Track both average and worst-group outcomes: high mean
accuracy can conceal systematic loss of rare facts. Separate logical slot
turnover from harmless permutation alignment. Evaluate cumulative errors after
hundreds of updates before moving to larger token scales.

Statistical and reproducibility protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use disjoint experiment-level random streams for model initialization, training,
validation, and test generation. Reserve system parameters, entity templates,
and query compositions for out-of-distribution evaluation. Select checkpoints
and hyperparameters on validation data only. Pair methods on the same held-out
streams and estimate uncertainty across independent runs and sequences; tokens
within a sequence are not independent samples. Bootstrap whole sequences or
use another explicitly justified dependence-aware procedure.

Archive resolved configurations, seeds, generator versions, checkpoints,
operation traces, measurement scripts, and environment versions. Report tuning
budgets and unsuccessful settings as well as the selected configuration. The
existing stage 1 snapshot remains unchanged: its adjacent seed offsets overlap
across runs, so it cannot retroactively supply independent confidence intervals.
