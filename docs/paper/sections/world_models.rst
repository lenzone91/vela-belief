.. _world-models:

World-model application: structured state under uncertainty
------------------------------------------------------------

Filtering and imagination are different operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A world model needs both an inference process and a generative transition.
Inference incorporates an observation that has actually arrived:

.. math::

   M_t=U_\theta(M_{t-1},E_\theta(o_t),a_{t-1}).

Imagination predicts possible outcomes before observing them. The shorthand
:math:`p_\phi(M_{t+1}\mid M_t,a_t)` is useful, but learning it from posterior
states is not sufficient by itself. The next inferred belief depends on the
next observation, which is random from the current perspective. A rollout
must account for that observation uncertainty and preserve consistency between
latent transitions and emitted observations.

A candidate model separates persistent memory from stochastic innovation:

.. math::

   v_t\sim q_\theta(v_t\mid M_{t-1},a_{t-1},o_t),\qquad
   M_t=F_\theta(M_{t-1},a_{t-1},v_t).

The prior predicts :math:`v_t` without seeing :math:`o_t`:

.. math::

   v_t\sim p_\phi(v_t\mid M_{t-1},a_{t-1}),\qquad
   o_t\sim p_\phi(o_t\mid M_t,v_t).

This is a proposed factorization in the recurrent state-space family, with
PlaNet as an established precedent for deterministic and stochastic components
[Hafner2019]_. Reward and termination decoders can be added when the environment
provides them. The innovation symbol :math:`v_t` avoids confusing this variable
with the stage 1 GRU vector :math:`z_t` or the actual environment state
:math:`S_t`. Other factorizations are possible; each must specify when an
observation enters and how an unobserved future is sampled.

A candidate variational objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One starting objective combines observation likelihood with prior/posterior
consistency, optionally adding rewards and supervised belief targets:

.. math::

   \begin{split}
   \mathcal L_{\mathrm{WM}}=\sum_t\Bigl[
   &-\mathbb E_{q_\theta}\log p_\phi(o_t\mid M_t,v_t)\\
   &+\beta D_{\mathrm{KL}}\bigl(
   q_\theta(v_t\mid M_{t-1},a_{t-1},o_t)
   \Vert p_\phi(v_t\mid M_{t-1},a_{t-1})\bigr)\Bigr].
   \end{split}

The slot regularizers can be added only after the generative baseline works.
Likelihood alone can favor predictable visual details over task-critical hidden
factors, so reward prediction or targeted probes may be needed. Conversely,
a reward-only state can discard information needed when the task changes.
This is another instance of the dependence on the future-query distribution.

Why slots might help, and where they might fail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An occluded object should persist in the inferred state even without a current
visual feature. A relation should change when interacting objects change their
configuration. A latent regime may affect several entities at once. Slots offer
a way to localize some of these updates, and SlotFormer gives a concrete
precedent for dynamics over object representations [Wu2023]_. However, a
factorized storage layout does not imply independent dynamics: relations,
collisions, and correlated uncertainty require communication between rows.

Discrete allocation and merge events can also destabilize a transition model.
If the same physical situation receives different slot permutations at nearby
times, a coordinate-wise prediction loss penalizes a harmless rearrangement.
Use temporal identity tracking or permutation-invariant matching, and evaluate
in observation or task space as well as latent space. Interpret a small latent
error cautiously if the encoder itself changes during training.

A staged validation path
~~~~~~~~~~~~~~~~~~~~~~~~~

Begin with continuous linear Gaussian systems before high-dimensional video.
For a known Gaussian initial state,

.. math::

   S_{t+1}=AS_t+Ba_t+\epsilon_t,\quad
   o_t=HS_t+\nu_t,\quad
   \epsilon_t\sim\mathcal N(0,Q),\quad\nu_t\sim\mathcal N(0,R).

The Kalman filter provides posterior means and covariances [Kalman1960]_. A
mean-only decoder is insufficient: two states with identical means and
different covariance imply different uncertainty. Parameterize covariance with
a positive-definite construction and evaluate proper probabilistic scores.
In nonlinear switching or volatility models, document approximation error in
particle or other reference filters rather than treating them as exact.

Next use partial-observation tasks with controlled object disappearance,
reappearance, changing attributes, and latent regime changes. Test inferred
beliefs with real observations, then open-loop rollouts without future
observations. Report horizon-dependent prediction quality, calibration, and
invalid-state frequency. Only after these checks should planning be evaluated
with the same planner and compute allowance for competing state models.
Improved one-step inference can coexist with worse planning if model errors
compound or the planner exploits them.
