# VELA-Belief

**Learning a compact, persistent memory of what matters in a sequence.**

VELA-Belief explores how a model can summarize past observations in a latent state that is updated as new information arrives. The aim is to retain what is useful for inference, prediction, and eventually decision-making, without reprocessing the full history at every step.

It is the first research module of **VELA — Value-Aware Evolving Latent Agent**.

The approach is intended to be general-purpose: the same principle could support hidden-state estimation, **LLM context compression**, or the memory component of a **world model**. The first experiments focus on small synthetic systems where the information the model should remember can be measured precisely.

**Status:** stage 1 is implemented: a categorical Hidden Markov Model (HMM), an exact Bayesian filter, and a trainable single-vector recurrent model. Structured memory, language tasks, and world models remain research directions.

## Run the first experiment

Requires Python 3.11 or newer. The default benchmark runs on CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'

vela-belief train --config configs/hmm.toml --output runs/hmm-first
pytest
```

On Debian, if virtual-environment creation reports that `ensurepip` is unavailable, install the matching `python3-venv` package first.

The run directory contains the resolved configuration, training history, best validation checkpoint, test metrics, and an example posterior trace. Each run requires a new output directory. To reproduce the saved model's evaluation:

```bash
vela-belief evaluate --checkpoint runs/hmm-first/checkpoint.pt --output runs/hmm-first-eval
```

Change the HMM, latent size, loss weights, or training budget in [configs/hmm.toml](configs/hmm.toml). Use `--seed 43` to run a different seed. `python -m vela_belief` also works in place of `vela-belief`.

The initial model is `Embedding → GRU → belief and next-observation heads`. Its encoder, memory, and heads are independent components; memory state is passed explicitly for streaming and chunked inference. See the [architecture and experiment protocol](docs/architecture.md) and [stage 1 results](docs/stage1-results.md).

## Research article

The [working article](docs/paper/index.rst) develops the context, mathematical formulation, stage 1 experiments, and prospective protocols for stages 2–6. Sphinx generates HTML and LaTeX from the same source, with tables and a figure derived from the archived results.

```bash
.venv/bin/python -m pip install -e '.[docs]'
make article-html    # docs/_build/html/index.html
make article         # docs/_build/latex/vela-belief.pdf (requires Tectonic)
```

See [article build and editing instructions](docs/paper/README.md) for the LaTeX source, compiler setup, and how to extend the paper.

## Why persistent memory?

In a partially observable system, the latest observation rarely tells the whole story. A model needs information from earlier observations to estimate what is happening now.

For example, a noisy measurement may be ambiguous on its own, while a sequence of measurements can reveal the underlying regime. The goal is to learn a memory that preserves this useful evidence and updates it over time.

The central research question is:

> Can a compact latent state retain the information from the past needed to estimate hidden state and predict what comes next?

## How it works

The initial model has three components: an observation encoder, a recurrent memory update, and a task-specific decoder.

At time $t$, it receives an observation $o_t$ and updates its memory:

$$
x_t = E_\theta(o_t), \qquad z_t = U_\theta(z_{t-1}, x_t).
$$

Here, $x_t$ is the encoded observation and $z_t \in \mathbb{R}^{d_z}$ is the persistent latent state. Starting from an initial state $z_{-1}$, the model processes observations one at a time. A decoder then uses $z_t$ to estimate a hidden state, represent uncertainty, or predict a future observation.

The latent dimension $d_z$ controls the size of the memory. The model must learn what to incorporate, preserve, update, or forget within that budget. A fixed-size state does not guarantee that all useful information can be retained over arbitrary sequences; measuring this tradeoff is part of the project.

This first version uses **one latent vector**. Later experiments may replace it with multiple memory slots:

$$
M_t = U_\theta(M_{t-1}, x_t), \qquad z_t = f_\theta(M_t),
$$

where $M_t \in \mathbb{R}^{K \times d_m}$ contains $K$ slots and $f_\theta$ reads them into a downstream representation. Attention, gated writes, slot specialization, and learned forgetting are possible extensions.

Persistent state is also used by GRUs and LSTMs. The research question is whether the proposed memory structure and training objectives provide measurable benefits over these established recurrent models.

## What is a belief state?

A **belief state** is a probability distribution over the hidden state of a system, given the observations so far. For the initial experiments, which do not involve actions:

$$
b_t(s) = P(S_t = s \mid o_{0:t}),
$$

where $S_t$ is the hidden state and $o_{0:t}$ is the observation history through time $t$.

This distribution describes both what the model believes and how uncertain it is. Under the assumed state-space model, it summarizes the history needed for predicting the system's future.

VELA-Belief learns a latent vector $z_t$, rather than storing this distribution explicitly. A belief decoder makes the hypothesis testable:

$$
\hat b_t = D_\theta(z_t) \approx b_t.
$$

For a discrete system, the decoder outputs probabilities over the possible hidden states. The latent vector itself does not need to have the same dimension or interpretation as that distribution.

The aim is to preserve information about the posterior, including uncertainty, rather than only predict the most likely hidden state. Whether the learned representation does this reliably is an experimental question.

## First benchmarks

The project starts with controlled environments that make hidden-state inference easy to evaluate.

| Environment | What is hidden? | Reference |
| --- | --- | --- |
| **Hidden Markov Model (HMM)** | A discrete state that evolves according to Markov transition probabilities and generates observations | Exact Bayesian filtering with known model parameters |
| Linear Gaussian state-space model | A continuous state observed through noisy measurements | Kalman filtering with known parameters and a Gaussian initial state |
| Regime-switching stochastic process | A regime that changes the observed dynamics | Known simulated regimes; a reference filter where tractable |
| Stochastic volatility process | A time-varying volatility state | Known simulated volatility; approximate filtering where needed |

The implemented first benchmark is a minimal **Hidden Markov Model** with categorical observations. Its exact posterior provides a direct target for checking whether a learned memory approximates Bayesian filtering. The other environments are planned extensions.

Synthetic hidden states and reference posteriors may be used as training targets or for evaluation. They are not inputs to the observation-driven memory update.

## Training and evaluation

The HMM experiment supports three training signals, separately or in combination:

- **Next-observation prediction:** learn a predictive distribution $p_\theta(o_{t+1} \mid z_t)$ by minimizing negative log-likelihood.
- **Hidden-state estimation:** predict the simulated hidden state using a classification or regression loss.
- **Belief matching:** match the reference posterior, for example by minimizing $D_{\mathrm{KL}}(b_t \Vert \hat b_t)$ for a discrete HMM.

Later experiments may add memory regularization to encourage sparse slot use or control memory allocation.

Evaluation will measure:

- hidden-state estimation accuracy and posterior quality, including uncertainty;
- predictive performance on held-out sequences;
- retention of information over long horizons;
- quality versus memory size and computational cost;
- robustness to longer sequences, increased noise, and changes in system dynamics.

Stage 1 uses a GRU and compares it with the exact Bayesian filter and an observation-only reference that forgets past observations. Both references know the HMM parameters; the neural model receives only observations, with exact beliefs available as training targets.

Future baselines will include fixed-window MLPs, LSTMs, causal Transformers with a defined context budget, and simple recurrent state-space models. Kalman filtering will provide a reference for linear Gaussian systems. The information available to each model must remain explicit in comparisons.

## Potential applications

The general-purpose ambition concerns the **memory formulation**: encode new information, update a persistent state, and decode what a task needs. Different domains will require suitable encoders, decoders, and training objectives. A single model that transfers across these domains has not yet been demonstrated.

### LLM context compression

A possible extension is to encode tokens, text chunks, or LLM representations into a persistent latent memory. An LLM could then condition on a readout of that memory alongside recent text, rather than repeatedly processing the entire conversation or document.

This would be a form of **learned, task-dependent context compression**. Experiments would need to test which facts, relationships, and instructions survive compression, how reliably they can be retrieved, and whether the memory reduces computation at acceptable quality.

The current belief-state experiments provide a controlled way to study information retention. They do not yet establish performance on language tasks or lossless recovery of the original context.

### World models

A world model needs a representation of the current situation and a model of how it may evolve. VELA-Belief could provide the first component: a latent state inferred from partial observations.

An extension could add action-conditioned latent dynamics:

$$
p_\phi(z_{t+1} \mid z_t, a_t),
$$

along with observation and, where relevant, reward decoders. For environments without actions, the transition model would condition only on $z_t$.

In an action-driven setting, the memory update would also receive the previous action, and the reference belief would become $P(S_t \mid o_{0:t}, a_{0:t-1})$. This keeps state inference consistent with the agent's interaction history.

Such a model could support future-state prediction and eventually planning. Learning a useful belief representation is a first step; reliable rollouts and decision-making require separate training and evaluation.

## Experimental roadmap

1. **Minimal HMM — implemented:** synthetic environment, exact Bayesian filter, and single-vector recurrent model, with belief reconstruction, prediction, and longer-sequence evaluation.
2. **Structured memory:** introduce multiple slots and compare them with the single-vector model and recurrent baselines under comparable memory budgets.
3. **Learned forgetting:** test whether explicit replacement or deletion improves retention and adaptation.
4. **Continuous hidden states:** extend evaluation to linear Gaussian systems and stochastic volatility.
5. **Representation analysis:** use probes, ablations, and temporal interventions to examine what the memory retains and how it represents uncertainty.
6. **Latent dynamics and broader tasks:** test whether the learned state supports transition modelling, then explore world models and LLM context compression with dedicated benchmarks.

Success means demonstrating that the learned state preserves useful hidden-state information, approaches reference filtering quality on tractable systems, and offers measurable benefits in prediction, retention, or efficiency. Benefits from memory slots and explicit forgetting must be established through comparisons and ablations.

## The broader VELA project

VELA explores the path from observation history to decisions under uncertainty:

```text
Observation history → Latent belief state → Stochastic dynamics → Decision
```

The tentative module roadmap is:

```text
vela-belief
    ↓
vela-american
    ↓
vela-battery
    ↓
vela-multienv
    ↓
vela-decision
    ↓
vela-foundation
```

Later modules are intended to explore optimal stopping, stochastic control, decision-aware representations, and cross-environment pretraining. VELA-Belief establishes the foundation by asking what a model needs to remember and how to verify that it has learned to do so.
