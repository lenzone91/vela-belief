import itertools
import math

import pytest
import torch

from vela_belief.environments.hmm import CategoricalHMM
from vela_belief.references.hmm import exact_filter, observation_only


@pytest.fixture
def hmm():
    # Asymmetric and nonstationary: catches transition transposes and a spurious
    # transition before the first observation, unlike a symmetric default HMM.
    return CategoricalHMM([0.8, 0.2], [[0.7, 0.3], [0.1, 0.9]], [[0.6, 0.4], [0.2, 0.8]])


def enumerate_posterior(hmm, observations):
    joint = torch.zeros(hmm.num_states, dtype=torch.float64)
    for path in itertools.product(range(hmm.num_states), repeat=len(observations)):
        probability = hmm.initial[path[0]].item()
        for t, observation in enumerate(observations):
            if t:
                probability *= hmm.transition[path[t - 1], path[t]].item()
            probability *= hmm.emission[path[t], observation].item()
        joint[path[-1]] += probability
    return joint / joint.sum(), joint.sum().item()


def test_filter_matches_exhaustive_path_enumeration(hmm):
    observations = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 1]])
    result = exact_filter(hmm, observations)
    for b, sequence in enumerate(observations.tolist()):
        for t in range(len(sequence)):
            posterior, likelihood = enumerate_posterior(hmm, sequence[: t + 1])
            torch.testing.assert_close(result.beliefs[b, t], posterior)
            expected_next = sum(
                posterior[i] * hmm.transition[i, j] * hmm.emission[j]
                for i in range(hmm.num_states)
                for j in range(hmm.num_states)
            )
            torch.testing.assert_close(result.next_observation_probs[b, t], expected_next)
        assert result.log_likelihood[b].item() == pytest.approx(math.log(likelihood))


def test_filter_is_causal(hmm):
    observations = torch.tensor([[0, 1, 0, 0, 1]])
    prefix = exact_filter(hmm, observations[:, :3])
    full = exact_filter(hmm, observations)
    torch.testing.assert_close(prefix.beliefs, full.beliefs[:, :3])


def test_memoryless_uses_time_dependent_prior(hmm):
    observations = torch.tensor([[0, 1]])
    result = observation_only(hmm, observations)
    prior = torch.tensor([0.58, 0.42], dtype=torch.float64)
    expected = prior * torch.tensor([0.4, 0.8], dtype=torch.float64)
    torch.testing.assert_close(result.beliefs[0, 1], expected / expected.sum())


def test_sampling_is_reproducible_and_matches_distribution(hmm):
    first = hmm.sample(30_000, 2, generator=torch.Generator().manual_seed(7))
    second = hmm.sample(30_000, 2, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first.states, second.states)
    assert torch.equal(first.observations, second.observations)
    assert (first.states[:, 0] == 0).float().mean().item() == pytest.approx(0.8, abs=0.01)
    for state in range(2):
        mask = first.states[:, 0] == state
        emitted_zero = (first.observations[mask, 0] == 0).float().mean().item()
        next_zero = (first.states[mask, 1] == 0).float().mean().item()
        assert emitted_zero == pytest.approx(hmm.emission[state, 0].item(), abs=0.02)
        assert next_zero == pytest.approx(hmm.transition[state, 0].item(), abs=0.02)


def test_structural_zeros_and_impossible_sequence():
    hmm = CategoricalHMM([1, 0], [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    result = exact_filter(hmm, torch.zeros(2, 2000, dtype=torch.long))
    assert torch.isfinite(result.beliefs).all()
    assert (result.beliefs[..., 0] == 1).all()
    with pytest.raises(ValueError, match="zero probability"):
        exact_filter(hmm, torch.tensor([[0, 1]]))


def test_long_unlikely_sequence_does_not_underflow():
    hmm = CategoricalHMM([0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]], [[0.99, 0.01], [0.98, 0.02]])
    result = exact_filter(hmm, torch.ones(1, 2000, dtype=torch.long))
    assert result.log_likelihood.item() < -7000
    assert torch.isfinite(result.beliefs).all()
    torch.testing.assert_close(result.beliefs.sum(-1), torch.ones(1, 2000, dtype=torch.float64))


@pytest.mark.parametrize(
    "initial,transition,emission",
    [
        ([0.2, 0.2], [[1, 0], [0, 1]], [[1], [1]]),
        ([1, 0], [[1, 0]], [[1], [1]]),
        ([1, 0], [[1, 0], [0, 1]], [[1]]),
        ([1, 0], [[1.1, -0.1], [0, 1]], [[1], [1]]),
        ([float("nan"), 0], [[1, 0], [0, 1]], [[1], [1]]),
    ],
)
def test_invalid_parameters_are_rejected(initial, transition, emission):
    with pytest.raises(ValueError):
        CategoricalHMM(initial, transition, emission)


@pytest.mark.parametrize(
    "observations",
    [
        torch.tensor([[2]]),
        torch.tensor([[-1]]),
        torch.tensor([[0.0]]),
        torch.empty(2, 0, dtype=torch.long),
        torch.tensor([0]),
    ],
)
def test_invalid_observations_are_rejected(hmm, observations):
    with pytest.raises(ValueError):
        exact_filter(hmm, observations)
