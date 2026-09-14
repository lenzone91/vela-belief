import pytest
import torch

from vela_belief.config import LossConfig
from vela_belief.metrics import categorical_metrics
from vela_belief.objectives import categorical_kl, hmm_loss


def test_kl_averages_over_time_and_handles_zero_targets():
    target = torch.tensor([[[1.0, 0.0], [0.4, 0.6]]])
    logits = torch.tensor([[[2.0, -1.0], [0.0, 1.0]]], requires_grad=True)
    loss = categorical_kl(logits, target)
    repeated = categorical_kl(logits.repeat(3, 4, 1), target.repeat(3, 4, 1))
    torch.testing.assert_close(loss, repeated)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_losses_use_separate_prediction_targets():
    states = torch.tensor([[0, 1]])
    next_observations = 1 - states
    predictions = {
        "belief": torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]]),
        "next_observation": torch.tensor([[[-10.0, 10.0], [10.0, -10.0]]]),
    }
    target = torch.nn.functional.one_hot(states, 2).float()
    loss, components = hmm_loss(predictions, target, states, next_observations, LossConfig())
    assert loss.item() < 1e-6
    assert components["prediction"].item() < 1e-6


def test_probability_metrics_have_known_values():
    beliefs = torch.tensor([[[0.8, 0.2]]], dtype=torch.float64)
    next_probs = torch.tensor([[[0.25, 0.75]]], dtype=torch.float64)
    scores = categorical_metrics(
        beliefs, next_probs, beliefs, next_probs, torch.tensor([[0]]), torch.tensor([[1]])
    )
    assert scores["belief_kl"] == pytest.approx(0, abs=1e-12)
    assert scores["next_observation_kl"] == pytest.approx(0, abs=1e-12)
    assert scores["state_accuracy"] == 1
    assert scores["state_brier"] == pytest.approx(0.08)
    assert scores["next_observation_nll"] == pytest.approx(-torch.tensor(0.75).log().item())
