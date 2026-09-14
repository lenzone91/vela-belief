"""Stage 1: supervised belief learning on independent HMM sequences.

    sample -> exact targets -> observation-only model -> loss -> validation

The trainer is intentionally specific to this benchmark. Models and memory do
not depend on the HMM, its targets, or the experiment configuration.
"""

import json
import math
import platform
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from vela_belief import __version__
from vela_belief.config import ExperimentConfig, config_from_dict
from vela_belief.environments.hmm import CategoricalHMM
from vela_belief.metrics import categorical_metrics
from vela_belief.models.memory import GRUMemory
from vela_belief.models.sequence import SequenceModel
from vela_belief.objectives import hmm_loss
from vela_belief.references.hmm import FilterResult, exact_filter, observation_only


@dataclass(frozen=True)
class BenchmarkBatch:
    observations: Tensor
    states: Tensor
    next_observations: Tensor
    reference: FilterResult


def make_batch(
    hmm: CategoricalHMM, size: int, length: int, generator: torch.Generator
) -> BenchmarkBatch:
    # One extra emission supplies a next-observation target for the last input.
    sample = hmm.sample(size, length + 1, generator=generator)
    inputs = sample.observations[:, :-1]
    return BenchmarkBatch(
        inputs, sample.states[:, :-1], sample.observations[:, 1:], exact_filter(hmm, inputs)
    )


def build_model(config: ExperimentConfig) -> SequenceModel:
    memory = GRUMemory(config.model.embedding_dim, config.model.hidden_dim)
    return SequenceModel(
        encoder=nn.Embedding(len(config.hmm.emission[0]), config.model.embedding_dim),
        memory=memory,
        heads={
            "belief": nn.Linear(memory.output_dim, len(config.hmm.initial)),
            "next_observation": nn.Linear(memory.output_dim, len(config.hmm.emission[0])),
        },
    )


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


@contextmanager
def _runtime(config: ExperimentConfig):
    """Reproducible CPU execution without changing the caller's RNG/settings."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(config.training.num_threads)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.training.seed)
            yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _scores(beliefs: Tensor, next_probs: Tensor, batch: BenchmarkBatch, start: int = 0) -> dict:
    return categorical_metrics(
        beliefs[:, start:],
        next_probs[:, start:],
        batch.reference.beliefs[:, start:],
        batch.reference.next_observation_probs[:, start:],
        batch.states[:, start:],
        batch.next_observations[:, start:],
    )


@torch.no_grad()
def _predict_batch(
    model: SequenceModel, hmm: CategoricalHMM, batch: BenchmarkBatch
) -> dict[str, tuple[Tensor, Tensor]]:
    model.eval()
    output = model(batch.observations)
    beliefs = output.predictions["belief"].softmax(-1)
    next_probs = output.predictions["next_observation"].softmax(-1)
    memoryless = observation_only(hmm, batch.observations)
    return {
        "learned": (beliefs, next_probs),
        "exact_filter": (batch.reference.beliefs, batch.reference.next_observation_probs),
        "observation_only": (memoryless.beliefs, memoryless.next_observation_probs),
    }


@torch.no_grad()
def _validate(model: SequenceModel, batch: BenchmarkBatch, config: ExperimentConfig) -> dict:
    model.eval()
    output = model(batch.observations)
    loss, components = hmm_loss(
        output.predictions,
        batch.reference.beliefs,
        batch.states,
        batch.next_observations,
        config.loss,
    )
    return {"loss": loss.item(), **{key: value.item() for key, value in components.items()}}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _test_report(model: SequenceModel, config: ExperimentConfig) -> tuple[dict, dict]:
    hmm = config.hmm.build()
    t = config.training
    test = make_batch(hmm, t.test_sequences, t.sequence_length, _generator(t.seed + 3))
    long_test = make_batch(hmm, t.test_sequences, t.long_sequence_length, _generator(t.seed + 4))
    report = {}
    for name, batch in (("test", test), ("long_test", long_test)):
        predictions = _predict_batch(model, hmm, batch)
        report[name] = {key: _scores(*probs, batch) for key, probs in predictions.items()}
        if name == "long_test":
            # Score the tail using the same predictions and full-prefix memory.
            report["long_test_tail"] = {
                key: _scores(*probs, batch, start=t.sequence_length)
                for key, probs in predictions.items()
            }
        else:
            trace = {
                "observations": batch.observations[0].tolist(),
                "states": batch.states[0].tolist(),
                "exact_beliefs": predictions["exact_filter"][0][0].tolist(),
                "learned_beliefs": predictions["learned"][0][0].tolist(),
                "observation_only_beliefs": predictions["observation_only"][0][0].tolist(),
            }
    return report, trace


def train(config: ExperimentConfig, output_dir: str | Path) -> dict:
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "config.json", asdict(config))
    started = time.perf_counter()
    with _runtime(config):
        hmm = config.hmm.build()
        t = config.training
        model = build_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=t.learning_rate)
        training_generator = _generator(t.seed + 1)
        validation = make_batch(
            hmm, t.validation_sequences, t.sequence_length, _generator(t.seed + 2)
        )
        initial_validation = _validate(model, validation, config)
        best_loss = float("inf")
        best_step = 0
        best_state = None
        with (output_dir / "history.jsonl").open("w", encoding="utf-8") as history:
            for step in range(1, t.steps + 1):
                model.train()
                batch = make_batch(hmm, t.batch_size, t.sequence_length, training_generator)
                output = model(batch.observations)  # Reset state for independent sequences.
                loss, _ = hmm_loss(
                    output.predictions,
                    batch.reference.beliefs,
                    batch.states,
                    batch.next_observations,
                    config.loss,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite training loss at step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), t.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                if step % t.eval_every == 0 or step == t.steps:
                    validation_scores = _validate(model, validation, config)
                    if not all(math.isfinite(v) for v in validation_scores.values()):
                        raise FloatingPointError(f"nonfinite validation score at step {step}")
                    record = {
                        "step": step,
                        "train_loss": loss.item(),
                        "validation": validation_scores,
                    }
                    history.write(json.dumps(record, allow_nan=False) + "\n")
                    history.flush()
                    print(
                        f"step {step:>5}/{t.steps}  val_loss={validation_scores['loss']:.5f}"
                        f"  belief_kl={validation_scores['belief']:.5f}",
                        flush=True,
                    )
                    if validation_scores["loss"] < best_loss:
                        best_loss = validation_scores["loss"]
                        best_step = step
                        best_state = {
                            key: value.detach().clone() for key, value in model.state_dict().items()
                        }
        assert best_state is not None
        model.load_state_dict(best_state)
        # Inference checkpoint: no optimizer state, so it is not a training-resume file.
        torch.save(
            {
                "format_version": 1,
                "config": asdict(config),
                "step": best_step,
                "model_state": best_state,
            },
            output_dir / "checkpoint.pt",
        )
        report, trace = _test_report(model, config)
        report.update(
            {
                "best_step": best_step,
                "best_validation_loss": best_loss,
                "initial_validation": initial_validation,
                "parameters": sum(p.numel() for p in model.parameters()),
                "elapsed_seconds": time.perf_counter() - started,
                "versions": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "vela_belief": __version__,
                },
                "device": "cpu",
                "seeds": {
                    "model": t.seed,
                    "train": t.seed + 1,
                    "validation": t.seed + 2,
                    "test": t.seed + 3,
                    "long_test": t.seed + 4,
                },
            }
        )
        _write_json(output_dir / "metrics.json", report)
        _write_json(output_dir / "trace.json", trace)
    return report


def load_checkpoint(path: str | Path) -> tuple[SequenceModel, ExperimentConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    config = config_from_dict(checkpoint["config"])
    with _runtime(config):
        model = build_model(config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), config


def evaluate_checkpoint(checkpoint: str | Path, output_dir: str | Path) -> dict:
    model, config = load_checkpoint(checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with _runtime(config):
        report, trace = _test_report(model, config)
    _write_json(output_dir / "config.json", asdict(config))
    _write_json(output_dir / "metrics.json", report)
    _write_json(output_dir / "trace.json", trace)
    return report
