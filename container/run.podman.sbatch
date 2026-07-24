#!/bin/bash
# SLURM array task that runs one experiment_runner config INSIDE the podman image.
# Drop-in replacement for experiments/slurm/run.sbatch on a podman-only cluster.
#
# Array index -> experiments/manifest.yaml configs[i].path (resolved inside the container, so the
# host needs neither python nor pyyaml — only podman + slurm). Submit via your existing
# experiments/slurm/submit.sh, pointing it at THIS template, or sbatch --array=... directly.
#
#SBATCH --job-name=qec-failfast
#SBATCH --output=runs/slurm/%x_%A_%a.out
#SBATCH --error=runs/slurm/%x_%A_%a.out
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=16G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p runs/slurm runs/framework

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"

# Bind-mount experiments/ (current configs + manifest -> edited shot counts take effect without a
# rebuild) and runs/ (checkpoints + results persist on the cluster filesystem). :Z relabels for
# SELinux hosts; drop the :Z if your cluster isn't SELinux and rejects it.
exec podman run --rm \
    --cpus "${CPUS}" \
    -e OMP_NUM_THREADS="${CPUS}" -e OPENBLAS_NUM_THREADS="${CPUS}" -e MKL_NUM_THREADS="${CPUS}" \
    -e TASK_IDX="${SLURM_ARRAY_TASK_ID:-0}" -e CPUS="${CPUS}" \
    -v "${PWD}/experiments:/opt/stim_work/experiments:Z" \
    -v "${PWD}/runs:/opt/stim_work/runs:Z" \
    -w /opt/stim_work \
    "${IMAGE}" \
    bash -lc '
        CFG=$(python -c "import os,yaml;print(yaml.safe_load(open(\"experiments/manifest.yaml\"))[\"configs\"][int(os.environ[\"TASK_IDX\"])][\"path\"])")
        echo "[task ${TASK_IDX}] config=${CFG} cpus=${CPUS}"
        exec python -m experiment_runner --config "${CFG}" --cpus "${CPUS}"
    '
