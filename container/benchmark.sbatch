#!/bin/bash
# SCALING BENCHMARK — reproduce the full [[18,4,4]] + [[72,4,8]] error-model campaign
# (experiments/methods/run_error_model_comparison.py), split across nodes to see the speedup.
#
# The ~70 cache-keyed tasks are partitioned into 20 self-contained `--only` groups (below), each a
# SLURM array task = one podman container on one node writing to the SHARED runs/ cache. The 17
# expensive [[72,4,8]] failure-spectra are isolated one-per-node, so with a wide array the wall
# clock collapses from the ~3 h single-process run toward the slowest single 72-code spectrum.
#
# Dependencies stay within a group (tech3__M needs tech2__M — both are in group 0), so the groups
# have no cross-node ordering requirements. runs/ MUST be on shared storage (all nodes see it).
#
# Submit (all 19 groups; scheduler spreads them across nodes as slots free up):
#     mkdir -p runs/slurm
#     sbatch --array=0-18 container/benchmark.sbatch
#   Fewer nodes at once? Throttle concurrency, same total work:  sbatch --array=0-18%6 ...
#   Re-time from scratch (cache makes re-runs instant):          QEC_FORCE=1 sbatch --array=0-18 ...
#   Quick (no boost, 1x shots instead of 5x on the 72 code):     QEC_BOOST=0  sbatch --array=0-18 ...
#
# After it finishes:   grep -h "ran," runs/slurm/qec-bench_*.out      # per-node task counts + time
#   the benchmark wall clock = the SLOWEST task's time; compare to the ~3 h sequential baseline.
#   Everything then sits cached under runs/error_model_comparison_18_4_4/ — re-execute the report
#   notebook (notebooks/methods/error_model_comparison_18_4_4.ipynb) to view the reproduced models.
#
#SBATCH --job-name=qec-bench
#SBATCH --output=runs/slurm/%x_%A_%a.out
#SBATCH --error=runs/slurm/%x_%A_%a.out
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p runs/slurm runs

# ---- the partition: array index -> a self-contained group of run_error_model_comparison --only
# prefixes. Prefix matching is exact-startswith, so "tech2__" hits only the 18-code models (not
# tech2_abl__ / tech2_72__). NB "tech1_72_abl__no_meas" is itself a prefix of the *_no_meas_idle
# task, so those two 72-code ablations SHARE group 10 (they can't be split by prefix). Groups
# 0-2 = the fast 18-code work; 3-18 = the expensive [[72,4,8]] spectra. -----------------------
GROUPS=(
  "setup schedule tech2__ tech1__ tech3__ mc__ config__manifest"                                  #  0  all 18-code models (fast)
  "tech2_abl tech1_abl mc_abl"                                                                     #  1  18-code ablations (fast)
  "asym__full_18 asym__no_CZ_18 asym__no_meas_18 asym__no_prep_18 asym__no_gate_idle_18 asym__no_meas_idle_18"  # 2  asym 18 (fast)
  "tech2_72__full_symmetric tech1_72__full_symmetric mc72__full_symmetric"                        #  3  72-code: full
  "tech2_72__CZ_only tech1_72__CZ_only mc72__CZ_only"                                             #  4  72-code: CZ
  "tech2_72__meas_only tech1_72__meas_only mc72__meas_only"                                       #  5  72-code: meas
  "tech2_72__prep_only tech1_72__prep_only mc72__prep_only"                                       #  6  72-code: prep
  "tech2_72__gate_idle tech1_72__gate_idle mc72__gate_idle"                                       #  7  72-code: gate-idle
  "tech2_72__meas_idle tech1_72__meas_idle mc72__meas_idle"                                       #  8  72-code: meas-idle
  "tech1_72_abl__no_CZ"                                                                            #  9  72-code abl: no-CZ
  "tech1_72_abl__no_meas"                                                                          # 10  72-code abl: no-meas + no-meas-idle (prefix)
  "tech1_72_abl__no_prep"                                                                          # 11  72-code abl: no-prep
  "tech1_72_abl__no_gate_idle"                                                                     # 12  72-code abl: no-gate-idle
  "asym__full_72"                                                                                  # 13  asym 72: full
  "asym__no_CZ_72"                                                                                 # 14  asym 72: no-CZ
  "asym__no_meas_72"                                                                               # 15  asym 72: no-meas
  "asym__no_prep_72"                                                                               # 16  asym 72: no-prep
  "asym__no_gate_idle_72"                                                                          # 17  asym 72: no-gate-idle
  "asym__no_meas_idle_72"                                                                          # 18  asym 72: no-meas-idle
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
GROUP="${GROUPS[$IDX]:-}"
[[ -n "$GROUP" ]] || { echo "no group for array index $IDX (valid 0-$(( ${#GROUPS[@]} - 1 )))"; exit 1; }

IMAGE="${QEC_IMAGE:-localhost/stim-work-qec:latest}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
BOOST=$([[ "${QEC_BOOST:-1}" == "0" ]] && echo "" || echo "--boost72")
FORCE=$([[ "${QEC_FORCE:-0}" == "1" ]] && echo "--force ${GROUP}" || echo "")

echo "[bench] host=$(hostname) task=${IDX} cpus=${CPUS} boost='${BOOST}' force='${FORCE}'"
echo "[bench] group: ${GROUP}"

exec podman run --rm \
    --cpus "${CPUS}" \
    -e OMP_NUM_THREADS="${CPUS}" -e OPENBLAS_NUM_THREADS="${CPUS}" -e RAYON_NUM_THREADS="${CPUS}" \
    -v "${PWD}/experiments:/opt/stim_work/experiments:Z" \
    -v "${PWD}/runs:/opt/stim_work/runs:Z" \
    -w /opt/stim_work \
    "${IMAGE}" \
    python experiments/methods/run_error_model_comparison.py --only ${GROUP} ${BOOST} ${FORCE}
