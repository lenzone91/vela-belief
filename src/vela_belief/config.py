"""Small, strict TOML configuration; no framework or dynamic class loading."""

import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vela_belief.environments.hmm import CategoricalHMM


@dataclass(frozen=True)
class HMMConfig:
    initial: list[float] = field(default_factory=lambda: [0.5, 0.5])
    transition: list[list[float]] = field(default_factory=lambda: [[0.97, 0.03], [0.03, 0.97]])
    emission: list[list[float]] = field(default_factory=lambda: [[0.7, 0.3], [0.3, 0.7]])

    def build(self) -> CategoricalHMM:
        return CategoricalHMM(self.initial, self.transition, self.emission)


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 16
    hidden_dim: int = 32


@dataclass(frozen=True)
class LossConfig:
    belief: float = 1.0
    prediction: float = 1.0
    state: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    steps: int = 800
    batch_size: int = 64
    sequence_length: int = 64
    learning_rate: float = 0.003
    gradient_clip: float = 1.0
    eval_every: int = 100
    validation_sequences: int = 256
    test_sequences: int = 512
    long_sequence_length: int = 256
    num_threads: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    hmm: HMMConfig = field(default_factory=HMMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.hmm.build()
        for name, value in asdict(self.model).items():
            _positive_integer(name, value)
        t = self.training
        for name in (
            "steps",
            "batch_size",
            "sequence_length",
            "eval_every",
            "validation_sequences",
            "test_sequences",
            "long_sequence_length",
            "num_threads",
        ):
            _positive_integer(name, getattr(t, name))
        if type(t.seed) is not int or not 0 <= t.seed < 2**63 - 4:
            raise ValueError("seed must be an integer between 0 and 2**63 - 5")
        if t.long_sequence_length <= t.sequence_length:
            raise ValueError("long_sequence_length must exceed sequence_length")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(t, name)
            if not _finite_number(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        weights = asdict(self.loss)
        if any(not _finite_number(v) or v < 0 for v in weights.values()):
            raise ValueError("loss weights must be finite and nonnegative")
        if sum(weights.values()) == 0:
            raise ValueError("at least one loss weight must be positive")


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return config_from_dict(raw)


def config_from_dict(raw: dict) -> ExperimentConfig:
    sections = {
        "hmm": HMMConfig,
        "model": ModelConfig,
        "loss": LossConfig,
        "training": TrainingConfig,
    }
    unknown = raw.keys() - sections.keys()
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    try:
        config = ExperimentConfig(**{key: cls(**raw.get(key, {})) for key, cls in sections.items()})
        config.validate()
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid configuration: {error}") from error
    return config
