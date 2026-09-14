import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from vela_belief.config import ExperimentConfig, TrainingConfig, load_config
from vela_belief.experiments.hmm import evaluate_checkpoint, load_checkpoint, make_batch, train
from vela_belief.references.hmm import exact_filter


def test_target_alignment():
    hmm = ExperimentConfig().hmm.build()
    batch = make_batch(hmm, 3, 5, torch.Generator().manual_seed(8))
    sample = hmm.sample(3, 6, generator=torch.Generator().manual_seed(8))
    assert torch.equal(batch.observations, sample.observations[:, :-1])
    assert torch.equal(batch.next_observations, sample.observations[:, 1:])
    torch.testing.assert_close(
        batch.reference.beliefs, exact_filter(hmm, sample.observations[:, :-1]).beliefs
    )


def test_training_learns_beliefs_and_checkpoint_reproduces_evaluation(tmp_path):
    config = ExperimentConfig(
        training=TrainingConfig(
            steps=160,
            batch_size=32,
            sequence_length=24,
            eval_every=80,
            validation_sequences=32,
            test_sequences=48,
            long_sequence_length=48,
        )
    )
    report = train(config, tmp_path / "run")
    learned = report["test"]["learned"]
    assert learned["belief_kl"] < 0.04
    assert learned["belief_kl"] < report["test"]["observation_only"]["belief_kl"] * 0.5
    assert learned["belief_kl"] < report["initial_validation"]["belief"] * 0.2
    reloaded = evaluate_checkpoint(tmp_path / "run/checkpoint.pt", tmp_path / "eval")
    for split in ("test", "long_test", "long_test_tail"):
        assert reloaded[split] == report[split]
    model, _ = load_checkpoint(tmp_path / "run/checkpoint.pt")
    assert not model.training
    assert model(torch.tensor([[0, 1]])).state.shape == (1, config.model.hidden_dim)
    assert len((tmp_path / "run/history.jsonl").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "run/metrics.json").read_text()) == report
    with pytest.raises(FileExistsError):
        train(config, tmp_path / "run")


def test_training_reproducibility_and_rng_isolation(tmp_path):
    config = ExperimentConfig(
        training=TrainingConfig(
            steps=2,
            batch_size=4,
            sequence_length=4,
            validation_sequences=4,
            test_sequences=4,
            long_sequence_length=8,
        )
    )
    before = torch.random.get_rng_state().clone()
    first = train(config, tmp_path / "first")
    assert torch.equal(before, torch.random.get_rng_state())
    second = train(config, tmp_path / "second")
    for split in ("test", "long_test", "long_test_tail"):
        assert first[split] == second[split]


@pytest.mark.parametrize(
    "contents",
    [
        "[oops]\nvalue=1",
        "[model]\nhiden_dim=4",
        "[training]\nsteps=0",
        "[loss]\nbelief=0\nprediction=0\nstate=0",
        "[training]\nlearning_rate=nan",
        "[training]\nlong_sequence_length=32",
        "[training]\nseed=-1",
        "[training]\nlearning_rate=true",
        "[loss]\nbelief=true",
        '[loss]\nprediction="1.0"',
    ],
)
def test_bad_config_fails_early(tmp_path, contents):
    path = tmp_path / "bad.toml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_config(path)


def test_three_states_and_four_observation_symbols(tmp_path):
    config = ExperimentConfig()
    config = replace(
        config,
        hmm=replace(
            config.hmm,
            initial=[0.2, 0.5, 0.3],
            transition=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
            emission=[[0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2], [0.2, 0.1, 0.4, 0.3]],
        ),
        training=replace(
            config.training,
            steps=2,
            batch_size=4,
            sequence_length=4,
            validation_sequences=4,
            test_sequences=4,
            long_sequence_length=8,
        ),
    )
    report = train(config, tmp_path / "three_states")
    assert report["test"]["learned"]["belief_kl"] >= 0


def test_cli_end_to_end(tmp_path):
    config = tmp_path / "tiny.toml"
    config.write_text(
        "[training]\nsteps=1\nbatch_size=2\nsequence_length=3\n"
        "validation_sequences=2\ntest_sequences=2\nlong_sequence_length=5\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vela_belief",
            "train",
            "--config",
            str(config),
            "--output",
            str(tmp_path / "cli"),
            "--seed",
            "123",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli/checkpoint.pt").exists()
    assert json.loads((tmp_path / "cli/config.json").read_text())["training"]["seed"] == 123
