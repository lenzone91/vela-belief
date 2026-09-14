"""Causal filtering in log space, including structural zero probabilities."""

from dataclasses import dataclass

import torch
from torch import Tensor

from vela_belief.environments.hmm import CategoricalHMM


@dataclass(frozen=True)
class FilterResult:
    beliefs: Tensor  # [batch, time, state], P(S[t] | O[:t+1])
    next_observation_probs: Tensor  # [batch, time, observation], P(O[t+1] | O[:t+1])
    log_likelihood: Tensor  # [batch], log P(O[:T])


def _condition(log_prior: Tensor, log_emission: Tensor) -> tuple[Tensor, Tensor]:
    joint = log_prior + log_emission
    evidence = torch.logsumexp(joint, dim=-1, keepdim=True)
    if not torch.isfinite(evidence).all():
        raise ValueError("observation sequence has zero probability under this HMM")
    return joint - evidence, evidence.squeeze(-1)


def exact_filter(hmm: CategoricalHMM, observations: Tensor) -> FilterResult:
    """Forward Bayesian filter (not a smoother): no future observation is read."""
    hmm.validate_observations(observations)
    batch, length = observations.shape
    log_transition = hmm.transition.log()
    log_emission = hmm.emission.log()
    log_belief = hmm.initial.log().expand(batch, -1)
    beliefs = []
    log_likelihood = torch.zeros(batch, dtype=torch.float64)
    for t in range(length):
        if t:
            log_belief = torch.logsumexp(log_belief.unsqueeze(-1) + log_transition, dim=1)
        log_belief, evidence = _condition(log_belief, log_emission[:, observations[:, t]].T)
        beliefs.append(log_belief.exp())
        log_likelihood += evidence
    posterior = torch.stack(beliefs, dim=1)
    return FilterResult(posterior, posterior @ hmm.transition @ hmm.emission, log_likelihood)


def observation_only(hmm: CategoricalHMM, observations: Tensor) -> FilterResult:
    """Oracle without history: P(S[t]|O[t]), using the time-dependent marginal.

    This reference knows the HMM parameters and time index but forgets past
    observations. Its log_likelihood is the sum of marginal observation scores,
    not the joint sequence likelihood returned by exact_filter.
    """
    hmm.validate_observations(observations)
    batch, length = observations.shape
    prior = hmm.initial.log()
    log_transition = hmm.transition.log()
    log_emission = hmm.emission.log()
    beliefs = []
    log_likelihood = torch.zeros(batch, dtype=torch.float64)
    for t in range(length):
        if t:
            prior = torch.logsumexp(prior.unsqueeze(-1) + log_transition, dim=0)
        posterior, evidence = _condition(prior, log_emission[:, observations[:, t]].T)
        beliefs.append(posterior.exp())
        log_likelihood += evidence
    posterior = torch.stack(beliefs, dim=1)
    return FilterResult(posterior, posterior @ hmm.transition @ hmm.emission, log_likelihood)
