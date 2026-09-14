"""Domain-independent composition of encoder, memory, and named heads."""

from dataclasses import dataclass

from torch import Tensor, nn

from vela_belief.models.memory import Memory


@dataclass(frozen=True)
class SequenceOutput:
    predictions: dict[str, Tensor]
    latents: Tensor
    state: Tensor


class SequenceModel(nn.Module):
    def __init__(self, encoder: nn.Module, memory: Memory, heads: dict[str, nn.Module]) -> None:
        super().__init__()
        self.encoder = encoder
        self.memory = memory
        self.heads = nn.ModuleDict(heads)

    def forward(self, observations: Tensor, state: Tensor | None = None) -> SequenceOutput:
        latents, state = self.memory(self.encoder(observations), state)
        predictions = {name: head(latents) for name, head in self.heads.items()}
        return SequenceOutput(predictions, latents, state)

    def step(self, observation: Tensor, state: Tensor | None = None) -> SequenceOutput:
        """Process one observation per batch item; time dimension stays length one."""
        return self(observation.unsqueeze(1), state)
