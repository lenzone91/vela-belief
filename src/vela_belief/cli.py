"""Command-line entry point for stage-one experiments."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from vela_belief.config import ExperimentConfig, load_config
from vela_belief.experiments.hmm import evaluate_checkpoint, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="vela-belief")
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser(
        "train", help="train and evaluate the categorical HMM benchmark"
    )
    training.add_argument("--config", type=Path, help="TOML file; omitted fields use defaults")
    training.add_argument("--seed", type=int, help="override the experiment seed")
    training.add_argument(
        "--output", type=Path, required=True, help="new directory for run artifacts"
    )
    evaluation = subparsers.add_parser("evaluate", help="reproduce held-out checkpoint evaluation")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument(
        "--output", type=Path, required=True, help="new directory for evaluation"
    )
    args = parser.parse_args()
    try:
        if args.command == "train":
            config = load_config(args.config) if args.config else ExperimentConfig()
            if args.seed is not None:
                config = replace(config, training=replace(config.training, seed=args.seed))
            report = train(config, args.output)
        else:
            report = evaluate_checkpoint(args.checkpoint, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(report["test"], indent=2, allow_nan=False))
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
