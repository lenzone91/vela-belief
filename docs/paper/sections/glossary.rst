Glossary and traceability of the conceptual revision
-----------------------------------------------------

Terms used throughout the paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Belief state.** A posterior distribution over hidden states given available
observations and actions. A latent memory can approximate its information
without explicitly storing probabilities.

**Slot.** An addressable row of persistent latent state. Its interpretation is
learned and need not correspond to one human category.

**Routing.** Selecting which existing memory units should receive new evidence.
Relative routing confidence is different from absolute evidence of a match.

**Allocation.** Creating a new occupied unit when existing units cannot represent
new information adequately.

**Consolidation.** Combining existing representations while attempting to
preserve the predictions they support and free capacity.

**Forgetting.** Removing information, deliberately accepting some potential
loss according to an estimated future-value criterion.

**Semantic memory.** Latent information integrated across observations for
future use. The word does not certify interpretability.

**Episodic memory.** Retained source-specific records or references, potentially
supporting exact retrieval. Backing storage counts toward the system budget.

**Overflow event.** A point where old text must leave an LLM's active context
and the proposed writer folds it into persistent memory.

**Frozen reader.** A downstream model whose parameters are held fixed during
adapter training. Gradients through its computations may still be required.

**Predictive redundancy.** Approximate equivalence in the predictions supported
by two memory configurations over a stated query distribution.

**Specialization.** A stable, selectively useful role of a memory unit,
supported by behavior and interventions rather than a label or visualization.

Where the patch's ideas are developed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The numbering below refers to the conceptual note ``patch.md``. Mathematical
notation has been repaired and disambiguated; proposed mechanisms have been
expanded into protocols rather than treated as implemented capabilities.

.. list-table:: Coverage of the conceptual revision
   :header-rows: 1
   :widths: 20 38 42

   * - Patch topics
     - Main ideas
     - Manuscript location
   * - 1--2, 16, 27
     - Bounded slots, semantic units, predictive compression.
     - :ref:`foundations`, :ref:`structured-memory`.
   * - 3--10
     - Route, update, allocate, consolidate, forget.
     - :ref:`structured-memory`, including pseudocode and functional redundancy.
   * - 11--15
     - Predictive, sparse, persistence, diversity, merge, budget pressures.
     - :ref:`objectives`, with proxy failures and optimization choices.
   * - 17--18
     - Structured world-model state and object-centric connection.
     - :ref:`world-models`, :ref:`related-work`.
   * - 19--26
     - Overflow, frozen reader, soft tokens, distillation, KV projection,
       semantic/episodic split.
     - :ref:`llm-overflow`, including context and cache accounting.
   * - 28--31
     - Memformer, pruning, latent compressors, LoCoCo, DMC.
     - :ref:`related-work`, with recurrent and fusion precedents made explicit.
   * - 32--34, 38--39
     - Quality, cost, stability, repeated consolidation, hypotheses and limits.
     - :ref:`evaluation`, :ref:`discussion`.
   * - 35
     - Nine stages and progressive additions.
     - :ref:`roadmap`, with dependencies and stage deliverables.
   * - 36--37, 40
     - Revised definition, overall intuition, project identity.
     - :ref:`introduction`, abstract, and repository README.
