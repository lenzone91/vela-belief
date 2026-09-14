Introduction and scope
------------------------------

In sequential problems, the latest observation may be insufficient to describe
the current situation. A model must combine new evidence with information from
the past. Reprocessing an expanding history is one option; maintaining a compact
state that is updated with each observation is another. The latter raises a
basic question: what must the model remember, and how can we verify that it has
remembered it?

VELA-Belief is the first module of VELA, **Value-Aware Evolving Latent Agent**.
The broader program connects observation history, latent state, stochastic
dynamics, and decisions under uncertainty. Its first task is deliberately
controlled: learning a useful state representation in environments where the
correct posterior can be computed. A finite HMM provides such a setting and
an established forward-inference reference [Rabiner1989]_.

The general-purpose ambition concerns the formulation, rather than a demonstrated
universal model. An encoder transforms each observation, a memory update retains
relevant information, and task-specific heads read the resulting state. In a
language setting, this could support learned context compression; the
in-context autoencoder is an example of existing work on compact latent memories
for language models [Ge2024]_. In an interactive environment, an inferred state
could feed a learned dynamics model, as recurrent representations do in work on
world models [Ha2018]_. Neither application follows automatically from success
on a small HMM.

Persistent memory is not itself a new mechanism. The first implementation uses
a standard GRU, following the family of gated recurrent models introduced by
Cho and colleagues [Cho2014]_. It establishes the reference against which later
memory mechanisms will be judged. We make three limited contributions at this
stage: a numerically checked HMM benchmark, a replaceable memory interface with
explicit streaming state, and an initial evaluation of posterior reconstruction
and predictive quality. We do not claim state-of-the-art performance, lossless
history compression, or an advantage over other trained recurrent architectures.
