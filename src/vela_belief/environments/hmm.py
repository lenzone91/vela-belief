"""Finite Hidden Markov Models with categorical observations (CPU, float64)."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HMMBatch:
    """Independent sequences, both tensors shaped [batch, time]."""

    observations: Tensor
    states: Tensor


def _probabilities(value: object, ndim: int, name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    if tensor.ndim != ndim or any(size == 0 for size in tensor.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}D probability array")
    if not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    sums = tensor.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-8, rtol=0):
        raise ValueError(f"{name} must sum to 1 along its last axis")
    return tensor


class CategoricalHMM:
    """A[i,j]=P(S[t+1]=j|S[t]=i), B[i,o]=P(O[t]=o|S[t]=i).

    Sample S[0] from initial, then emit O[0]. There is no transition before O[0].
    Parameters are known to the reference filter, not to the neural model.
    """

    def __init__(self, initial: object, transition: object, emission: object) -> None:
        self.initial = _probabilities(initial, 1, "initial")
        self.transition = _probabilities(transition, 2, "transition")
        self.emission = _probabilities(emission, 2, "emission")
        self.num_states = self.initial.numel()
        if self.transition.shape != (self.num_states, self.num_states):
            raise ValueError("transition must have shape [num_states, num_states]")
        if self.emission.shape[0] != self.num_states:
            raise ValueError("emission must have one row per state")
        self.num_observations = self.emission.shape[1]

    def sample(self, batch_size: int, length: int, *, generator: torch.Generator) -> HMMBatch:
        if batch_size < 1 or length < 1:
            raise ValueError("batch_size and length must be positive")
        states = torch.empty(batch_size, length, dtype=torch.long)
        observations = torch.empty_like(states)
        state = torch.multinomial(self.initial, batch_size, replacement=True, generator=generator)
        for t in range(length):
            if t:
                state = torch.multinomial(self.transition[state], 1, generator=generator).squeeze(
                    -1
                )
            states[:, t] = state
            observations[:, t] = torch.multinomial(
                self.emission[state], 1, generator=generator
            ).squeeze(-1)
        return HMMBatch(observations, states)

    def validate_observations(self, observations: Tensor) -> None:
        if observations.ndim != 2 or 0 in observations.shape:
            raise ValueError("observations must have nonempty shape [batch, time]")
        if observations.dtype != torch.long or observations.device.type != "cpu":
            raise ValueError("observations must be CPU int64 categorical indices")
        if (observations < 0).any() or (observations >= self.num_observations).any():
            raise ValueError("observation index outside the emission alphabet")
