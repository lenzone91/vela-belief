.. _roadmap:

Nine-stage research map
------------------------

This revision replaces the former six-stage outline with nine stages matching
``patch.md``. Continuous-state inference and representation analysis remain in
the program as cross-cutting tracks, especially in stages 6 and 7. Stages 2--9
are planned; their equations and acceptance criteria are not reported results.
A stage is complete when it answers its question with archived evidence,
including an unfavorable answer. Progress does not require declaring every
new mechanism beneficial.

.. figure:: _generated/roadmap.*
   :name: roadmap-figure
   :width: 100%
   :alt: Baseline leads to slots, sparse routing, allocation, consolidation,
         and representation analysis, then branches to world models and LLM
         overflow; layer-aware KV integration follows the LLM branch.

   Dependencies of the revised program. Stages 7 and 8 are separate application
   branches after the shared memory study. Stage 9 depends on stage 8. Small
   reader-compatibility pilots can run earlier without establishing the full
   application claim.

.. list-table:: Deliverables at a glance
   :header-rows: 1
   :widths: 13 38 49

   * - Stage
     - Added capability
     - Evidence to archive
   * - 1 — done
     - Single-vector belief baseline.
     - Exact-reference checks, GRU metrics, explicit limitations.
   * - 2
     - Structured slot memory.
     - Matched capacity sweeps and memory-interface checks.
   * - 3
     - Sparse routing and specialization.
     - Write counts, runtime, quality, causal slot analysis.
   * - 4
     - Allocation and forgetting.
     - Capacity-pressure traces, delayed retention, adaptation.
   * - 5
     - Consolidation and merge.
     - Freed capacity, per-merge and cumulative distortion.
   * - 6
     - Representation pressures.
     - Single-term ablations, interactions, stable specialization.
   * - 7
     - World-model state.
     - Calibrated filtering, rollouts, then controlled planning.
   * - 8
     - LLM overflow memory / VELA-S.
     - Frozen-reader integration and repeated-overflow comparisons.
   * - 9
     - Layer-aware reader / VELA-KV.
     - Quality and end-to-end cost against VELA-S and KV methods.

Stage 1 — single-vector belief baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question.** Can the implementation learn a causal posterior approximation and
use history better than the observation-only reference? **Current answer.** The
archived GRU experiments support this on the fixed two-state HMM, with the
limitations in :ref:`stage1`. The deliverable is a working reference pipeline,
not evidence for semantic slots.

Preserve its configuration and metric snapshot when adding later models. Add
new runs for independent seed streams and larger HMM families rather than
silently replacing the original results. The present streaming contract,
causality tests, posterior alignment, and checkpoint reconstruction are the
regression boundary for all later stages.

Stage 2 — structured slot memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question.** Does explicit factor capacity improve inference or retention?
Implement a dense slot state with a shared gated update, occupancy initially
fixed, and a simple pooled or attention readout. Begin without learned merging
or deletion so that the effect of the storage structure can be isolated.

Compare GRU, LSTM, dense slots, and attention-based slots using several
:math:`(K,d_m)` pairs at matched :math:`Kd_m`, then separate parameter and
runtime sweeps. Include :math:`K=1` as a control. Keep the small HMM for
correctness, and add independently changing factors or entities before drawing
conclusions about organization. A larger matrix should not win merely because
it contains more floats than the baseline.

**Deliverable and decision.** Archive posterior and delayed-query scores,
parameter counts, state bytes, readout ablations, and runtime. Require causal
prefix, batch-isolation, and chunk-equivalence checks for the new state.
If slots do not improve any meaningful tradeoff, examine the task's capacity
demand and readout bottleneck before introducing additional mechanisms.

Stage 3 — sparse routing and specialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question.** Can selective writes reduce interference or computation? Replace
dense writes with routing variants: ordinary softmax, lower temperature,
entropy regularization, hard top-r, and optionally a stated differentiable
relaxation. Keep the update network and state capacity fixed.

Sweep :math:`r` from one row to all rows. A larger example such as
:math:`K=256,r=4` is useful only after smaller configurations establish the
mechanism. Measure the number of scored, selected, and actually changed rows
separately. Include routing-only and update-only timing to locate overhead.
Probe both common and rare factors, with slot masking and controlled swaps.

**Deliverable and decision.** Produce quality/cost curves and evidence about
selective intervention effects. If low entropy simply routes everything to one
row, diagnose collapse before claiming specialization. If sparse arithmetic is
slower on the tested device, preserve that result and identify the budget range
where sparsity is still useful or reject the efficiency hypothesis there.

Stage 4 — allocation and learned forgetting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question.** Can memory admit new information without discarding valuable old
information prematurely? Introduce free rows, a calibrated NEW decision,
usage metadata, and explicit replacement or rejection. Begin with deterministic
free-slot selection and simple FIFO, least-recently-used, random, and
usage-based policies before learning a utility predictor.

Construct streams in which relevance lifetime, mention frequency, and query
delay vary independently. Include obsolete facts, rare facts queried after long
gaps, and periods where new evidence is irrelevant. Sweep active factor count
below, near, and above :math:`K`. Evaluate allocation precision, false novelty,
identity fragmentation, stale answers, and adaptation lag.

**Deliverable and decision.** Archive the memory-operation trace alongside
answers. Enforce the capacity invariant and reset all metadata on replacement.
An oracle that knows future queries can give an upper reference for eviction,
but the learned controller cannot see those queries early. A forgetting policy
must improve the retention/adaptation tradeoff, not only reduce occupancy.

Stage 5 — consolidation and merge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question.** Can redundant or complementary rows be combined without damaging
future use? Add a candidate-pair generator, merge operator, and acceptance rule.
Compare cosine-threshold merging, attention-based proposals, learned scoring,
fixed averaging, and predictive-consistency merging. Keep a no-merge control
with the same allocation and forgetting mechanisms.

Use alias-resolution streams where identity evidence arrives late, redundant
observations with complementary attributes, and deceptively similar but distinct
entities. Separate detection errors from merge-operator errors by supplying
oracle candidate identities in one diagnostic condition. Evaluate merges under
both spare capacity and forced pressure so that an unused merge mechanism
cannot appear successful without being exercised.

**Deliverable and decision.** Report recovered rows, acceptance rate, incorrect
entity fusions, immediate distortion, and retention after many maintenance
cycles. Use future queries held out from candidate selection. If repeated merges
accumulate unacceptable error, test conservative thresholds and episodic support
before claiming stable consolidation. Report order sensitivity and runtime.

Stage 6 — representation pressure and analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question.** Which learning pressures produce useful, stable organization?
Add the regularizers in :ref:`objectives` one at a time, then test selected
interactions. Persistence can oppose adaptation, diversity can oppose merging,
and balanced routing can oppose frequency-sensitive allocation. These tensions
are part of the experiment rather than reasons to sum every term by default.

Use held-out probes, intervention selectivity, permutation-aligned trajectories,
capacity sweeps, and predictors with residual access to history. Introduce a
continuous Gaussian task to test whether uncertainty beyond categorical labels
is retained; carry its full evaluation into stage 7. Compare task-only learning,
privileged belief supervision, and mixed objectives explicitly.

**Deliverable and decision.** Archive a regularization ablation matrix with
predictive scores, calibration, routing concentration, usage, and churn.
Retain only terms with a reproducible useful effect. The claim of organization
requires behavioral evidence; attractive slot plots are descriptive artifacts.
A null result can justify a simpler final architecture.

Stage 7 — structured state for a world model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question.** Does the memory support coherent action-conditioned futures?
Start with Kalman-reference inference, then switching or partially observed
multi-entity dynamics. Add an explicit generative transition with observation
and optional reward decoders, distinguishing posterior inference from prior
rollout as in :ref:`world-models`.

Compare a dense recurrent state-space model with the slot version under the
same observation encoder and training data. Evaluate object permanence,
hidden-variable tracking, covariance quality, and horizon-dependent open-loop
predictions. Teacher-forced one-step evaluation must be accompanied by rollouts
that never receive future observations.

**Deliverable and decision.** Archive calibrated rollout curves and failure
examples. Planning is a later substep with a shared planner and compute budget;
it should not hide poor model calibration behind task-specific optimization.
The LLM branch does not depend on a planning success, and world-model success
does not imply compatibility with a frozen language reader.

Stage 8 — frozen-LLM overflow memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question.** Can VELA-S preserve useful evicted information through repeated
overflows? First prove differentiable soft-token reading on short synthetic
facts with a frozen backbone. Then train a recurrent writer over several
evicted blocks while keeping recent text exact. Specify the position policy,
reserved generation space, and suffix re-prefill behavior.

Run the comparison tracks in :ref:`evaluation`, with rolling windows, text
summaries, eviction policies, latent compressors, and compatible activation
compressors. Include memory-free and one-shot-compression controls. Vary total
stream length, event count, block size, capacity, and query complexity.
Ground long-history tests in known answers when the full-context teacher
cannot process the history. Add a bounded episodic store as a separate ablation.

**Deliverable and decision.** Report answer accuracy, exact recall, old
instructions, factual revision, multi-event reasoning, cache bytes, peak memory,
and total event latency. Demonstrate that old-query performance actually depends
on memory by ablating its readout. Advance only with a clear account of which
information survives and at what cost; compatibility alone is not an advantage.

Stage 9 — layer-aware integration / VELA-KV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question.** Does a layer-specific projection improve the reader bottleneck
or deployment cost? Hold the semantic writer fixed initially and compare
soft-token input with direct KV projection. Then permit joint adapter training
in a separately reported experiment. Specify projection sharing across layers,
head layout, position handling, and whether suffix activations are recomputed.

Compare against VELA-S and relevant activation/KV methods at equal total bytes,
not equal slot counts. Test multiple backbones before claiming general
compatibility. Measure projection time, cache maintenance, decoding throughput,
and quality after repeated memory replacement. The frozen-reader assumption
must remain explicit for every variant.

**Deliverable and decision.** Archive a systems and quality comparison that
includes the semantic state, projected KV state, adapters, and active context.
A useful outcome can be a precise boundary: for example, a reader that helps
quality but is too costly at a given context size. Select the simplest reader
that achieves the desired measured tradeoff.

How the paper grows after each stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every completed stage, add a results section describing the question,
mechanism, training signal, information access, baselines, budgets, and failure
cases. Generate figures from versioned metrics and retain their provenance.
Replace that stage's prospective text with links to its evidence, then revise
the abstract, claim-status table, and discussion. Do not keep a favorable
hypothesis in the abstract after its controlled experiment rejects it.

The wider VELA modules remain a tentative path from belief to decisions:
``vela-belief`` to ``vela-american``, ``vela-battery``, ``vela-multienv``,
``vela-decision``, and ``vela-foundation``. They concern later optimal stopping,
control, decision-aware representations, and cross-environment learning.
This manuscript's nine stages concern memory; they are not completion claims
for those broader modules.
