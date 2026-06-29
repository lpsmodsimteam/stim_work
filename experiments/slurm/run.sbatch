#!/bin/bash
# SLURM array task: index $SLURM_ARRAY_TASK_ID -> experiments/manifest.yaml configs[i] -> runner.
# Submit via experiments/slurm/submit.sh (it sets per-entry --cpus-per-task / --time / --array).
#SBATCH --job-name=qec-failfast
#SBATCH --output=runs/slurm/%x_%A_%a.out
#SBATCH --error=runs/slurm/%x_%A_%a.out
# Fallback resources (submit.sh overrides these per manifest entry):
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=16G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p runs/slurm

# --- activate the project env (EDIT for your cluster) -----------------------
# module load anaconda3
# conda activate qec
# (the env must have: stim, relay_bp, ldpc, numpy, scipy, pyyaml, matplotlib, and the editable
#  install: `pip install -e .` from the repo root — see experiments/README.md)

# Resolve this task's config path from the manifest.
CFG=$(python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys, yaml
print(yaml.safe_load(open("experiments/manifest.yaml"))["configs"][int(sys.argv[1])]["path"])
PY
)
echo "[task ${SLURM_ARRAY_TASK_ID}] config=${CFG} cpus=${SLURM_CPUS_PER_TASK:-1}"

# Resumable: re-running the same task picks up its checkpoint. --cpus caps relay/BLAS thread pools.
exec python -m experiment_runner --config "${CFG}" --cpus "${SLURM_CPUS_PER_TASK:-1}"
