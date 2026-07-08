#!/bin/bash
# Weekend run: three fail-fast techniques on the gross [[144,12,12]] code, single-sector Z DEM,
# for all five noise channels. Technique II + I only (splitting is lowest priority -> --no-split);
# splitting can be added later with --split-only per model. Cheap/exact even-D channels first.
set -u
cd /c/Users/aksirot_local/Desktop/workspace/general/stim_work || exit 1
export PYTHONPATH=src
RUNNER=experiments/bravyi/bb6_fig10_sweep.py
BASE=notebooks/bb_code/bb144_z_noise
CORES=20
mkdir -p "$BASE"

echo "===== weekend run started $(date) ====="
for M in meas prep idle full cz; do
    echo ">>> $(date)  MODEL=$M  starting (Technique II + I, no splitting)"
    python "$RUNNER" --code bb144 --noise-model "$M" --no-split --onset-scan \
        --shots 8000 --shots-by-weight '6:40000,7:20000,8:16000' \
        --max-cores "$CORES" --mw-workers "$CORES" --plot \
        --outdir "$BASE/$M" > "$BASE/$M.log" 2>&1
    echo ">>> $(date)  MODEL=$M  finished (exit $?)"
done
echo "===== weekend run ALL MODELS DONE $(date) ====="
