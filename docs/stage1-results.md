# Stage 1 results

The single-vector GRU learns a close approximation to the exact Bayesian belief on the default two-state categorical HMM. It substantially improves on the reference that only sees the current observation. These results validate the first benchmark and training pipeline; they do not establish advantages for a new memory architecture or transfer to language tasks.

## Protocol

- Configuration: [configs/hmm.toml](../configs/hmm.toml), with experiment seeds 42, 43, and 44.
- Model: 16-dimensional embeddings, a 32-dimensional GRU state, and two linear heads; 4,964 parameters.
- Training: 800 updates, batches of 64 independent sequences of length 64, exact-belief KL plus next-observation negative log-likelihood.
- Selection: lowest weighted loss on 256 held-out validation sequences.
- Test: 512 fresh sequences of length 64, plus 512 of length 256.
- Environment: Python 3.11.2, PyTorch 2.14.0+cpu, deterministic CPU execution with one thread.

Changing the experiment seed changes model initialization and each run's data streams. However, the offset scheme reuses some generator seeds across runs: a short test set can share prefixes with another run's longer test set. The HMM parameters remain fixed. These three dependent runs are a small reproducibility check, not independent statistical replications, a broad robustness study, or a confidence interval.

## Held-out results

KL and NLL are measured in nats and averaged over sequence positions. Lower is better. State accuracy is the fraction of correctly classified hidden states.

| Seed | Selected update | Belief KL | State accuracy | Next-observation NLL | Belief KL beyond the training horizon |
| --- | --- | --- | --- | --- | --- |
| 42 | 500 | 0.00009644 | 82.82% | 0.656118 | 0.00008374 |
| 43 | 200 | 0.00026388 | 82.52% | 0.659189 | 0.00027295 |
| 44 | 600 | 0.00017258 | 82.74% | 0.655555 | 0.00017903 |

The final column scores positions 64–255 of the longer test sequences, after processing their full prefix. Belief approximation remains close beyond the 64-step training horizon in this environment.

For seed 42, the references make the benefit of history visible:

| Model | Belief KL | State accuracy | Next-observation NLL |
| --- | --- | --- | --- |
| Learned GRU | 0.00009644 | 82.82% | 0.656118 |
| Exact Bayesian filter | ≈0 | 82.81% | 0.655768 |
| Observation-only oracle | 0.216326 | 70.18% | 0.680686 |

Both analytic references know the HMM parameters. The GRU receives observations and is supervised using exact posterior targets during training. Small reversals in finite-sample state accuracy do not imply that the learned model improves on Bayesian inference.

The [machine-readable results](stage1-results.json) include resolved configurations, software versions, and all three evaluation splits for each seed. They were obtained by reloading the selected checkpoints. Local checkpoints, logs, and sequence traces are under `runs/hmm-stage1-seed42`, `runs/hmm-stage1-seed43`, and `runs/hmm-stage1-seed44`; run directories are excluded from Git.

## Reproduce

After following the installation steps in the README:

```bash
vela-belief train --config configs/hmm.toml --seed 42 --output runs/reproduce-42
vela-belief train --config configs/hmm.toml --seed 43 --output runs/reproduce-43
vela-belief train --config configs/hmm.toml --seed 44 --output runs/reproduce-44
vela-belief evaluate --checkpoint runs/reproduce-42/checkpoint.pt --output runs/reproduce-42-eval
```

Use new output directories on subsequent runs. Numerical results may vary with the PyTorch version or platform; the configuration and runtime version are recorded with each training run.

## Verification and limits

The 38-test suite covers exact inference against exhaustive enumeration, impossible and low-probability observations, sampling distributions, causality, streaming equivalence, gradient flow, loss alignment, configuration validation, learning, reproducibility, checkpoint evaluation, and the CLI. Ruff checks and formatting checks also pass locally. A GitHub Actions workflow runs these checks on pushes and pull requests.

The current evidence concerns a fixed HMM with short discrete observations and privileged belief supervision. It does not yet measure transfer across HMM parameters, performance against trained fixed-window baselines, memory-slot benefits, unsupervised belief recovery, LLM compression, or action-conditioned world models. Those comparisons belong to subsequent experiments.
