Problem and model
-------------------------

Belief as a measurable memory target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`S_t` denote a hidden state and :math:`O_t` an observation. In the
action-free setting studied here,

.. math::

   S_{t+1}\sim p(S_{t+1}\mid S_t),\qquad
   O_t\sim p(O_t\mid S_t).

The filtering belief is

.. math::
   :label: belief

   b_t(s)=P(S_t=s\mid O_{0:t}=o_{0:t}).

Under the assumed HMM, this posterior contains the history information needed
to predict future observations. It represents uncertainty over states, rather
than only the most likely state. A learned vector need not equal this
distribution; it should preserve enough information for a decoder to recover it.
Low average decoding error is evidence for this property on the evaluated
distribution, not a proof of sufficiency on arbitrary histories.

Recurrent architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The observation encoder and memory update are

.. math::

   x_t=E_\theta(o_t),\qquad z_t=U_\theta(z_{t-1},x_t),

with :math:`z_{-1}=0`. For stage 1, the encoder is an embedding table and
:math:`U_\theta` is a single-layer GRU. Two independent linear heads produce

.. math::

   \hat b_t=\operatorname{softmax}(W_bz_t+c_b),\qquad
   \hat q_t=\operatorname{softmax}(W_qz_t+c_q).

The first approximates :eq:`belief`; the second predicts the next observation.
The predictive head does not receive the HMM transition or emission matrices.
Both heads and the encoder are trained jointly with the memory.

The implementation separates the persistent state from the readout. Given
embeddings with shape :math:`B\times T\times E`, a memory returns latents with
shape :math:`B\times T\times D` and a final state with shape :math:`B\times\cdots`.
The GRU state is a vector, but a later implementation can return a matrix of
slots and expose the same readout shape. The caller owns the state; independent
sequences reset it explicitly. Full-sequence, chunked, and one-step inference
must agree. This contract is tested independently of the HMM task.

Training objective
~~~~~~~~~~~~~~~~~~~~~~~~~~

The available losses, averaged over batch and time, are

.. math::

   \mathcal{L}=
   \lambda_b\,\mathbb{E}\!\left[D_{\mathrm{KL}}(b_t\Vert\hat b_t)\right]
   -\lambda_q\,\mathbb{E}\!\left[\log\hat q_t(o_{t+1})\right]
   -\lambda_s\,\mathbb{E}\!\left[\log\hat b_t(s_t)\right].

The reported experiments use :math:`(\lambda_b,\lambda_q,\lambda_s)=(1,1,0)`.
Exact beliefs are privileged training targets. Hidden states are used for
evaluation; their optional training term is disabled. Neither hidden states nor
reference beliefs are inputs to the neural model. The current experiment
therefore evaluates supervised posterior learning, not unsupervised discovery
of a belief representation.
