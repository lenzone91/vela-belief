"""Memory contract: explicit batch-first state, separate from readout latents."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Memory(nn.Module, ABC):
    """Replaceable causal memory.

    Inputs: embeddings [B,T,E], optional state [B,...].
    Outputs: latents [B,T,D], final state [B,...].
    None resets the memory. State belongs to the caller, never to the module.
    A slot memory can return state [B,K,M] while keeping latents [B,T,D].
    Implementations must preserve full-sequence / chunked equivalence.
    """

    output_dim: int

    @abstractmethod
    def forward(self, embeddings: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class GRUMemory(Memory):
    """Single-vector baseline; a fused GRU also supports one-step streaming."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("memory dimensions must be positive")
        self.output_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, embeddings: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if embeddings.ndim != 3 or embeddings.shape[1] == 0:
            raise ValueError("embeddings must have shape [batch, nonempty time, embedding]")
        if state is not None and state.shape != (embeddings.shape[0], self.output_dim):
            raise ValueError("GRU state must have shape [batch, hidden_dim]")
        hidden = None if state is None else state.unsqueeze(0).contiguous()
        latents, final = self.gru(embeddings, hidden)
        return latents, final.squeeze(0)


def detach_state(state: Tensor) -> Tensor:
    """Cut the gradient between chunks only when truncated BPTT is intended."""
    return state.detach()
