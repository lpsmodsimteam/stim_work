# Fail-fast experiment framework

Config-driven runner for the three fail-fast techniques (arXiv:2511.15177) — Technique I
(importance-sampling ansatz), II (min-weight onset), III (splitting) — across bivariate-bicycle
codes and circuit families. One experiment = a YAML config; the engine is `src/experiment_runner.py`.

## One experiment

```bash
# from the repo root, in the project env
python -m experiment_runner --config experiments/configs/bb144_memory.yaml
```

- Writes per-technique JSON (`distance.json`, `spectrum.json`, `ansatz_fit.json`, `splitting.json`)
  + a combined `result.npz` + `config.json` to `runs/framework/<code>/<experiment>/`.
- **Resumable**: checkpointed after every IS weight; re-running resumes from the last completed
  weight (a killed/requeued job loses ≤ one weight).
- `--smoke` runs the same code paths at tiny budgets in <~1 min (a faithful dry run).
- `--cpus N` caps relay/BLAS thread pools (defaults to `$SLURM_CPUS_PER_TASK`).
- `--techniques II,IS,I,III`, `--code`, `--split-method {multiseed,replica}`, `--outdir` override the
  config from the CLI.

## Smoke-test before the cluster

```bash
python -m experiment_runner --config experiments/configs/bb6_memory.yaml --smoke
python -m experiment_runner --config experiments/configs/bb144_memory.yaml --smoke --techniques IS,I,III
pytest tests/test_experiment_runner.py
```

## Cluster (SLURM array)

The array index maps to a line of `experiments/manifest.yaml`. `submit.sh` submits each entry with its
own cpus/time:

```bash
bash experiments/slurm/submit.sh --dry-run   # preview the sbatch commands
bash experiments/slurm/submit.sh             # submit
```

Edit `experiments/slurm/run.sbatch` to activate your env (`conda activate qec`) and adjust
`--mem`/partition. Each job is independent and resumable, so a requeue just continues.

## Environment

```bash
pip install -e .                 # editable install (puts src/ modules on the path)
pip install -r experiments/requirements.txt
# relay_bp (Rust BP decoder) and ldpc (BP-OSD) come from the project's env setup, not PyPI defaults.
```

## Extending (the registries live in `src/experiment_runner.py`)

| Add a… | Do this |
|--------|---------|
| **code** | add a `BBCodeParams` to `src/bb_code_sim.py` + a `CODES` entry (the runner self-checks n/k) |
| **decoder** | add a `DECODERS` entry (factory taking `cfg`) |
| **circuit kind** (automorphism, joint-Pauli) | implement the builder in `src/gross_code_lpu_tdg.py` (stubs already there) + a `CIRCUIT_BUILDERS` entry |
| **experiment** | add a YAML config in `experiments/configs/` + a line in `manifest.yaml` |

## Status of the matrix

- **Active / P0:** `bb6_memory`, `bb144_memory` (memory circuits, all 3 techniques, relay decoder).
- **Seams (registered, configs committed, not in the default manifest):** `bb288` two-gross code,
  `lpu_x1`/`lpu_z1` circuits (builders exist; II/III need LPU adaptation), `automorphism`/`joint_pauli`
  circuits (`NotImplementedError` stubs), `bposd`/`pymatching` decoders.
