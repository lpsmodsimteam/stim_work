#!/bin/bash
# SMOKE TEST — is the container usable across the cluster's nodes?
# A SLURM array where each task lands on a (possibly different) node, runs the podman image, and:
#   (1) runs the preflight (imports + editable install + a real stim/relay_bp decode round-trip),
#   (2) builds the [[18,4,4]] BB code and Monte-Carlo-decodes a batch of idle-memory shots.
# Every task prints its hostname + a logical error rate, so N green lines == N working nodes.
#
# Submit (4 nodes; bump --array for more):
#     mkdir -p runs/slurm
#     sbatch --array=0-3 container/smoke.sbatch
# Then check: grep SMOKE runs/slurm/qec-smoke_*.out   (want one PASS line per array index)
#
#SBATCH --job-name=qec-smoke
#SBATCH --output=runs/slurm/%x_%A_%a.out
#SBATCH --error=runs/slurm/%x_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --mem=8G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p runs/slurm

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

echo "[smoke] host=$(hostname) task=${IDX} image=${IMAGE} cpus=${CPUS}"

exec podman run --rm \
    --cpus "${CPUS}" \
    -e OMP_NUM_THREADS="${CPUS}" -e OPENBLAS_NUM_THREADS="${CPUS}" -e RAYON_NUM_THREADS="${CPUS}" \
    -e IDX="${IDX}" \
    -w /opt/stim_work \
    "${IMAGE}" \
    bash -lc 'python test.py --no-slurm && python container/smoke_decode.py'
