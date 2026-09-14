"""Categorical objectives averaged over batch AND time, in natural log units."""

import torch.nn.functional as F
from torch import Tensor

from vela_belief.config import LossConfig


def categorical_kl(logits: Tensor, target: Tensor) -> Tensor:
    target = target.to(dtype=logits.dtype)
    return F.kl_div(logits.log_softmax(-1), target, reduction="none").sum(-1).mean()


def hmm_loss(
    predictions: dict[str, Tensor],
    beliefs: Tensor,
    states: Tensor,
    next_observations: Tensor,
    weights: LossConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    components = {
        "belief": categorical_kl(predictions["belief"], beliefs),
        "prediction": F.cross_entropy(
            predictions["next_observation"].flatten(0, 1), next_observations.flatten()
        ),
        "state": F.cross_entropy(predictions["belief"].flatten(0, 1), states.flatten()),
    }
    total = sum(getattr(weights, name) * value for name, value in components.items())
    return total, components
