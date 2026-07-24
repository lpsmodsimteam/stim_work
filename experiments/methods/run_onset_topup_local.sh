#!/bin/bash
# Run the [[72,4,8]] onset top-ups on ONE machine, no SLURM (e.g. the 96-core box). Work-queue:
# at most NPROC spectra run at once, each capped to THREADS relay/BLAS threads. By default it uses
# only FRACTION (0.5 = HALF) of the machine's cores, so it leaves room for other users. Each
# spectrum's stdout goes to runs/onset_logs/<name>.log.
#
# Native python: activate the env (stim/relay_bp/ldpc + `pip install -e .`) first, or set PYBIN.
# Runs all 17 spectra by default; pass names to run a subset. Resumable — rerun to continue
# (each spectrum skips its already-topped-up weights; merges never double-count).
#
#   bash experiments/methods/run_onset_topup_local.sh                     # half of the machine
#   FRACTION=0.25 bash experiments/methods/run_onset_topup_local.sh       # a quarter
#   NPROC=6 THREADS=8 bash experiments/methods/run_onset_topup_local.sh   # pin it exactly (48 cores)
#   bash experiments/methods/run_onset_topup_local.sh tech1_72__meas_only asym__full_72   # subset
#
# Tunables (env): FRACTION (0.5), CORES (auto = nproc), NPROC/THREADS (override the split),
#   ONSET_SHOTS_MAX (3000000), ONSET_TARGET (20), ONSET_CHUNK (50000).
set -euo pipefail
cd "$(dirname "$0")/../.."                    # repo root

FRACTION="${FRACTION:-0.5}"
CORES="${CORES:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
BUDGET=$(awk "BEGIN{b=int(${CORES}*${FRACTION}); if(b<1)b=1; print b}")   # cores we're allowed to use
THREADS="${THREADS:-8}"
(( THREADS > BUDGET )) && THREADS=$BUDGET                                 # never exceed the budget
NPROC="${NPROC:-$(( BUDGET/THREADS > 0 ? BUDGET/THREADS : 1 ))}"          # concurrent spectra
PYBIN="${PYBIN:-python}"
SCRIPT="experiments/methods/onset_topup_72.py"
mkdir -p runs/onset_logs
echo "[onset-local] machine has ${CORES} cores; using ~${BUDGET} (${FRACTION} cap) = "\
"NPROC=${NPROC} x THREADS=${THREADS}"

if [[ $# -gt 0 ]]; then
  NAMES=("$@")
else
  mapfile -t NAMES < <("$PYBIN" "$SCRIPT" --list | awk '{print $2}')
fi

echo "[onset-local] ${#NAMES[@]} spectra, NPROC=${NPROC} THREADS=${THREADS}, "\
"cap=${ONSET_SHOTS_MAX:-3000000} target=${ONSET_TARGET:-20}"

fail=0
for name in "${NAMES[@]}"; do
  while (( $(jobs -rp | wc -l) >= NPROC )); do wait -n || fail=1; done
  ( RAYON_NUM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
      "$PYBIN" "$SCRIPT" "$name" > "runs/onset_logs/${name}.log" 2>&1 \
      && echo "[done] ${name}" || { echo "[FAIL] ${name} (see runs/onset_logs/${name}.log)"; exit 1; } ) &
done
wait || fail=1

echo "[onset-local] all spectra finished (logs in runs/onset_logs/)"
exit "$fail"
