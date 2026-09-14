"""Common probability-based scores for learned and analytic predictors."""

import torch
from torch import Tensor


def _log_probs(probabilities: Tensor) -> Tensor:
    # Keep zero targets safe while still strongly penalizing zero predictions.
    return probabilities.double().clamp_min(torch.finfo(torch.float64).tiny).log()


def _kl(target: Tensor, prediction: Tensor) -> Tensor:
    target = target.double()
    # Roundoff may produce tiny negative values even when comparing p with p.
    return (
        (torch.special.xlogy(target, target) - target * _log_probs(prediction)).sum(-1).clamp_min(0)
    )


def categorical_metrics(
    beliefs: Tensor,
    next_probs: Tensor,
    reference_beliefs: Tensor,
    reference_next_probs: Tensor,
    states: Tensor,
    next_observations: Tensor,
) -> dict[str, float]:
    one_hot = torch.nn.functional.one_hot(states, beliefs.shape[-1]).double()
    scores = {
        "belief_kl": _kl(reference_beliefs, beliefs).mean(),
        "belief_mae": (reference_beliefs - beliefs).abs().mean(),
        "state_accuracy": (beliefs.argmax(-1) == states).double().mean(),
        "state_brier": (beliefs.double() - one_hot).square().sum(-1).mean(),
        "next_observation_kl": _kl(reference_next_probs, next_probs).mean(),
        "next_observation_nll": -_log_probs(next_probs)
        .gather(-1, next_observations.unsqueeze(-1))
        .mean(),
    }
    return {name: value.item() for name, value in scores.items()}
