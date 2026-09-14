
# VELA-Belief

**Learning persistent latent belief states in partially observable stochastic environments.**

VELA-Belief is the first research module of the broader **VELA — Value-Aware Evolving Latent Agent** project.

The goal of VELA is to learn compact latent representations of dynamic systems that are useful not only for prediction, but ultimately for sequential decision-making.

VELA-Belief focuses on the first question:

> **Can a model learn what must be remembered from the past in order to recover the hidden state of a partially observable stochastic system?**

---

## Motivation

In many sequential problems, the current observation is not sufficient to characterize the true state of the system.

A generic partially observable system can be written as

$$s_{t+1} \sim p(s_{t+1}\mid s_t,a_t),$$

$$o_t \sim p(o_t\mid s_t),$$

where:

- $s_t$ is the hidden state,
- $o_t$ is the available observation,
- $a_t$ is an optional action.

The theoretically relevant quantity for optimal prediction and control is the **belief state**

$$b_t(s) = p(s_t=s\mid o_{0:t},a_{0:t-1}).$$

In classical models, this belief state may be computed analytically or numerically using a Bayesian filter.

VELA-Belief explores whether a neural architecture with persistent memory can instead **learn an implicit belief state directly from data**.

---
## Core idea

Let $o_t \in \mathbb{R}^{d_o}$ denote the observation available to the model at time $t$. Rather than repeatedly reconstructing a representation of the system from a fixed historical window,

$$
(o_{t-L},\ldots,o_t)\longrightarrow z_t,
$$

VELA maintains a **persistent latent state**

$$
z_t \in \mathbb{R}^{d_z},
$$

which is updated sequentially as new observations become available.

When a new observation $o_t$ arrives, it is first mapped to an embedding

$$
x_t = E_{\theta_E}(o_t),
$$

with

$$
E_{\theta_E}:\mathbb{R}^{d_o}\rightarrow\mathbb{R}^{d_x},
$$

so that

$$
x_t\in\mathbb{R}^{d_x}.
$$

Here, $\theta_E$ denotes the learnable parameters of the observation encoder.

The latent state is then updated according to

$$
z_t = U_{\theta_U}(z_{t-1},x_t),
$$

where

$$
U_{\theta_U}:
\mathbb{R}^{d_z}\times\mathbb{R}^{d_x}
\rightarrow
\mathbb{R}^{d_z}
$$

is a learnable memory-update mechanism with parameters $\theta_U$.

The key difference from a fixed-context model is that $z_t$ persists through time. The model therefore does not need to reconstruct its representation of the system from scratch at every time step. Instead, it recursively updates an internal state intended to retain information from the past that remains relevant and discard information that no longer is.

Conceptually, the update mechanism should learn how to:

* incorporate information carried by the latest observation;
* update information already represented in the latent state;
* preserve relevant information over long horizons;
* combine redundant information;
* discard information that has become irrelevant.

In this first formulation, the latent state $z_t$ is itself the memory of the model. Its dimension $d_z$ therefore directly controls the capacity available to summarize the observation history.

After processing observations up to time $t$, the latent state can be written recursively as

$$
z_t
=
F_\theta(o_{0:t}),
$$

where

$$
o_{0:t}=(o_0,\ldots,o_t)
$$

denotes the full sequence of observations available up to time $t$, and $\theta$ collectively denotes the learnable parameters of the encoder and memory-update mechanism.

---

### Relation to the Bayesian belief state

Let $S_t$ denote the hidden state of the underlying system.

For a discrete hidden-state space, the optimal Bayesian belief state is

$$
b_t(s)
=
P(S_t=s\mid o_{0:t}).
$$

Thus, $b_t$ is itself a probability distribution over the possible values of $S_t$. If the hidden state can take $N_s$ discrete values, then

$$
b_t\in\Delta^{N_s-1},
$$

where $\Delta^{N_s-1}$ denotes the $(N_s-1)$-dimensional probability simplex.

The central hypothesis of VELA-Belief is **not necessarily that $z_t$ numerically equals $b_t$**. In general,

$$
z_t\in\mathbb{R}^{d_z}
$$

and

$$
b_t\in\Delta^{N_s-1}
$$

belong to different spaces.

Instead, the hypothesis is that $z_t$ can learn to contain the information represented by the Bayesian belief state.

One way to make this statement explicit is to introduce a learnable belief decoder

$$
\hat b_t = D_{\theta_D}(z_t),
$$

with

$$
D_{\theta_D}:
\mathbb{R}^{d_z}
\rightarrow
\Delta^{N_s-1},
$$

such that

$$
\boxed{
\hat b_t(s)
\approx
P(S_t=s\mid o_{0:t})
}
$$

for all relevant hidden states $s$.

The representation $z_t$ is therefore not required to have the same dimension or interpretation as the Bayesian belief state. It only needs to preserve sufficient information for the belief state to be recovered from it, and ultimately for future observations or decisions to be predicted accurately.

---

### Extension to structured memory

The initial VELA-Belief model uses a single latent vector

$$
z_t\in\mathbb{R}^{d_z}
$$

as its persistent memory.

Later versions may replace this vector with a structured multi-slot memory

$$
M_t
=
\begin{bmatrix}
(m_t^1)^\top\\
\vdots\\
(m_t^K)^\top
\end{bmatrix}
\in
\mathbb{R}^{K\times d_m},
$$

where each memory slot satisfies

$$
m_t^k\in\mathbb{R}^{d_m},
\qquad
k=1,\ldots,K.
$$

The total memory capacity is therefore distributed across $K$ latent slots rather than stored in a single vector.

In this setting, the memory update becomes

$$
M_t
=
U_\theta(M_{t-1},x_t),
$$

with

$$
U_\theta:
\mathbb{R}^{K\times d_m}
\times
\mathbb{R}^{d_x}
\rightarrow
\mathbb{R}^{K\times d_m}.
$$

Different slots may then learn to specialize in different persistent features of the system, while mechanisms such as attention, competition, merging, or learned forgetting can determine how new information is allocated across the memory.

A downstream latent representation may either be identified directly with the full memory,

$$
z_t := M_t,
$$

or obtained through a readout function,

$$
z_t = f_\theta(M_t),
$$

depending on the requirements of the downstream task.

The structured-memory formulation is therefore a generalization of the initial VELA-Belief architecture rather than a requirement for the first experiments.


---

## Research questions

VELA-Belief investigates the following questions.

### 1. Can persistent memory recover hidden state information?

Given only observations $o_{0:t}$, can the model infer latent variables that are not directly observable?

### 2. Does persistent memory outperform fixed context windows?

We compare the proposed architecture against models that repeatedly process bounded historical windows.

### 3. What should the model remember?

Can memory slots specialize in different persistent features of the environment?

### 4. Is explicit forgetting useful?

Does learning to discard old or irrelevant information improve state estimation?

### 5. Does the learned representation approximate the true belief state?

Whenever the exact Bayesian belief is available, the learned latent representation can be evaluated against it.

### 6. How much memory is necessary?

We study the relationship between:

- number of memory slots,
- latent dimension,
- environment complexity,
- hidden-state reconstruction quality.

---

## Initial environments

The project starts with controlled synthetic environments where the optimal hidden-state inference problem is known.

### Hidden Markov Models

A simple discrete latent regime:

$$
S_t \in \{1,\ldots,K\}
$$

with transition matrix

$$
P_{ij} = P(S_{t+1}=j\mid S_t=i).
$$

Observations are generated from regime-dependent distributions.

This setting provides an exact Bayesian filtering benchmark.

---

### Regime-switching stochastic processes

Example:

$$
X_{t+1} = \mu_{S_t} + \phi_{S_t}X_t + \sigma_{S_t}\varepsilon_{t+1}.
$$

The regime $S_t$ is hidden.

The model must infer the current regime from noisy observations and historical transitions.

---

### Stochastic volatility

A latent volatility process:

$$
X_{t+1} = \sigma_t \varepsilon_{t+1},
$$

with

$$
\log \sigma_{t+1}^2 = \mu + \phi \log \sigma_t^2 + \eta_{t+1}.
$$

The model observes returns but not volatility.

The learned memory should retain information useful for estimating the current volatility state.

---

### Linear Gaussian state-space models

$$
s_{t+1} = As_t+w_t,
$$

$$
o_t = Cs_t+v_t.
$$

The Kalman filter provides an exact reference belief state.

This environment is particularly useful for determining whether the learned representation captures both:

- the posterior mean,
- the posterior uncertainty.

---

## Baselines

VELA-Belief will initially compare against:

- fixed-window MLPs,
- GRUs,
- LSTMs,
- causal Transformers,
- exact Bayesian filtering,
- Kalman filtering,
- simple recurrent state-space models.

The goal is not merely to outperform recurrent neural networks.

The main objective is to understand whether an explicit persistent memory provides useful inductive biases for latent-state construction.

---

## Proposed architecture

The initial architecture consists of four components.

### Observation encoder

$$
x_t = E_\theta(o_t).
$$

### Memory reader

The new observation attends to the current memory:

$$
\alpha_t = \operatorname{softmax}\left(\frac{Q(x_t)K(M_t)^\top}{\sqrt d}\right).
$$

### Memory updater

The model updates one or more memory slots:

$$
M_{t+1} = U_\theta(M_t,x_t,\alpha_t).
$$

Possible future variants include:

- gated slot updates,
- competitive writes,
- explicit memory allocation,
- learned deletion scores,
- slot merging,
- differentiable memory eviction.

### Latent belief representation

$$
z_t = f_\theta(M_t).
$$

The latent state is then used to predict quantities associated with the hidden state or the next observation.

---

## Training objectives

Several objectives will be investigated.

### Next-observation prediction

$$
\mathcal L_{\text{pred}} = -\log p_\theta(o_{t+1}\mid z_t).
$$

### Hidden-state reconstruction

When the synthetic hidden state is available during training:

$$
\mathcal L_{\text{state}} = \ell(\hat s_t, s_t).
$$

### Belief matching

When the exact posterior distribution is available:

$$
\mathcal L_{\text{belief}} = D(\hat b_t, b_t),
$$

where \(D\) may be a KL divergence or another probabilistic distance.

### Memory regularization

Future experiments may penalize excessive memory usage or encourage sparse slot activation.

---

## Evaluation

The project will evaluate several dimensions independently.

### Hidden-state estimation

How accurately does the model recover the hidden regime or latent variable?

### Belief quality

How close is the learned posterior representation to the exact Bayesian belief?

### Predictive performance

Does the learned state improve prediction of future observations?

### Long-term memory

How well does the model retain information that remains relevant over long horizons?

### Memory efficiency

How much latent capacity is required?

### Robustness

How does performance degrade under:

- longer sequences,
- noisier observations,
- regime persistence changes,
- distribution shifts?

---

## Experimental roadmap

### Stage 1 — Minimal HMM

Build the smallest possible environment with a known exact belief state.

Goal:

> verify that a learned recurrent memory can reproduce Bayesian filtering behavior.

---

### Stage 2 — Memory slots

Replace a single recurrent hidden state with explicit memory slots.

Study:

- slot specialization,
- attention patterns,
- memory capacity.

---

### Stage 3 — Learned forgetting

Introduce explicit memory replacement or deletion mechanisms.

Study whether the model learns when information becomes irrelevant.

---

### Stage 4 — Continuous hidden states

Move from discrete HMMs to:

- Kalman systems,
- stochastic volatility,
- continuous regime representations.

---

### Stage 5 — Representation analysis

Analyze whether the learned latent space corresponds to meaningful hidden-state quantities.

Possible tools include:

- linear probing,
- latent-space visualization,
- mutual information estimates,
- slot ablations,
- temporal interventions.

---

### Stage 6 — Transition model

Once the latent belief state is sufficiently stable, learn

$$
p_\theta(z_{t+1}\mid z_t).
$$

This transition model will form the bridge toward the next VELA projects:

- `vela-american`,
- `vela-battery`,
- `vela-decision`.

---

## Success criteria

VELA-Belief will be considered successful if the learned memory can demonstrate that:

1. it reconstructs hidden-state information from partial observations;
2. it approaches Bayesian filtering performance on tractable environments;
3. it preserves useful information over longer horizons than bounded-context baselines;
4. explicit memory mechanisms provide measurable advantages over standard recurrent states;
5. the learned latent representation can be used as an approximately Markovian state for downstream transition modelling.

---

## Broader VELA roadmap

VELA-Belief is the first step in a larger research program.

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
````

The long-term objective is to build a model that learns

$$
\text{history}
\rightarrow
\text{latent state}
\rightarrow
\text{stochastic dynamics}
\rightarrow
\text{decision}.
$$

Rather than asking only

> What will happen next?

VELA ultimately aims to answer

> What information about the past matters, what can happen next, and what should be done about it?

---

## Status

Early research prototype.

The initial focus is intentionally narrow:

> **learn a persistent latent belief state in environments where the optimal belief is known and measurable.**

Later modules will introduce stochastic transition operators, optimal stopping, stochastic control, decision-aware representation learning and cross-environment pretraining.

---

## Project

VELA stands for:

**Value-Aware Evolving Latent Agent**

The broader project explores persistent latent memory and stochastic world models for sequential decision-making under uncertainty.


