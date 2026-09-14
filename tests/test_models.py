import torch
from torch import nn

from vela_belief.config import ExperimentConfig
from vela_belief.experiments.hmm import build_model
from vela_belief.models.memory import Memory, detach_state
from vela_belief.models.sequence import SequenceModel


def test_full_sequence_chunks_and_steps_are_equivalent():
    torch.manual_seed(3)
    model = build_model(ExperimentConfig()).eval()
    observations = torch.randint(2, (3, 11))
    full = model(observations)
    first = model(observations[:, :4])
    second = model(observations[:, 4:], first.state)
    torch.testing.assert_close(full.state, second.state)
    torch.testing.assert_close(full.latents, torch.cat([first.latents, second.latents], dim=1))
    state = None
    predictions = []
    for observation in observations.unbind(1):
        output = model.step(observation, state)
        state = output.state
        predictions.append(output.predictions["belief"])
    torch.testing.assert_close(full.predictions["belief"], torch.cat(predictions, dim=1))
    torch.testing.assert_close(full.state, state)


def test_causality_reset_and_batch_isolation():
    model = build_model(ExperimentConfig()).eval()
    observations = torch.randint(2, (3, 9))
    full = model(observations)
    changed_future = observations.clone()
    changed_future[:, 5:] = 1 - changed_future[:, 5:]
    torch.testing.assert_close(full.latents[:, :5], model(changed_future).latents[:, :5])
    for b in range(3):
        torch.testing.assert_close(full.latents[b : b + 1], model(observations[b : b + 1]).latents)
    torch.testing.assert_close(full.latents, model(observations).latents)


def test_gradients_reach_encoder_memory_and_heads_and_can_be_detached():
    model = build_model(ExperimentConfig())
    output = model(torch.tensor([[0, 1, 0], [1, 0, 1]]))
    sum(logits.square().mean() for logits in output.predictions.values()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert not detach_state(output.state).requires_grad
    assert output.state.requires_grad


def test_structured_state_and_continuous_encoder_need_no_model_changes():
    class SlotMemory(Memory):
        output_dim = 3

        def forward(self, embeddings, state=None):
            if state is None:
                state = embeddings.new_zeros(embeddings.shape[0], 2, 3)
            latents = []
            for embedding in embeddings.unbind(1):
                state = state + embedding.unsqueeze(1)
                latents.append(state.mean(1))
            return torch.stack(latents, 1), state

    model = SequenceModel(nn.Linear(4, 3), SlotMemory(), {"continuous": nn.Linear(3, 2)})
    inputs = torch.randn(2, 5, 4)
    full = model(inputs)
    prefix = model(inputs[:, :2])
    suffix = model(inputs[:, 2:], prefix.state)
    assert full.state.shape == (2, 2, 3)
    assert full.predictions["continuous"].shape == (2, 5, 2)
    torch.testing.assert_close(full.latents[:, 2:], suffix.latents)
