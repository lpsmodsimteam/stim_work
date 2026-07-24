#!/bin/bash
# ONSET TOP-UP — resample the [[72,4,8]] failure-spectrum onsets (weights 2..10) at a 3e6-shot
# cap and merge into the cached spectra, tightening the onset floor from ~6e-5 toward ~1e-6.
# One SLURM array task per 72-code spectrum (17 total: 6 models, 5 ablations, 6 x5-asym mixes),
# run inside the podman image. Uses the SAME low-p-calibrated decoder as the campaign — the
# isolated/ablated spectra get per-model priors at DECODER_P; the x5-asym spectra get the x5-ray
# priors (make_*_asym at DECODER_P) — so the extra shots pool validly into the existing bins.
#
# Requires the calibrated campaign cache under runs/error_model_comparison_18_4_4/ (the script
# asserts each spectrum's decoder is calibrated_at=5e-4 before touching it). runs/ must be on
# shared storage. Merges are additive + resumable per weight, so a requeued task never double-counts.
#
# Submit (all 17 spectra; list the index->name map with --list):
#     mkdir -p runs/slurm
#     sbatch --array=0-16 container/onset_topup.sbatch
#     sbatch --array=0-16%6 container/onset_topup.sbatch      # cap to 6 nodes at a time
#     podman run --rm -v "$PWD/experiments:/opt/stim_work/experiments:Z" QEC_IMAGE \
#         python experiments/methods/onset_topup_72.py --list   # see which index is which spectrum
#
# Watch:  grep -h "onset top-up complete\|w=" runs/slurm/qec-onset_*.out
# Then re-execute notebooks/methods/error_model_comparison_18_4_4.ipynb to see the tightened onsets.
#
#SBATCH --job-name=qec-onset
#SBATCH --output=runs/slurm/%x_%A_%a.out
#SBATCH --error=runs/slurm/%x_%A_%a.out
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=16G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p runs/slurm

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

echo "[onset] host=$(hostname) task=${IDX} cpus=${CPUS}"

exec podman run --rm \
    --cpus "${CPUS}" \
    -e OMP_NUM_THREADS="${CPUS}" -e OPENBLAS_NUM_THREADS="${CPUS}" -e RAYON_NUM_THREADS="${CPUS}" \
    -v "${PWD}/experiments:/opt/stim_work/experiments:Z" \
    -v "${PWD}/runs:/opt/stim_work/runs:Z" \
    -w /opt/stim_work \
    "${IMAGE}" \
    python experiments/methods/onset_topup_72.py --index "${IDX}"
