#!/bin/bash
# Submit every experiments/manifest.yaml entry as a SLURM array task with its OWN cpus/time.
# Each entry becomes a single-index array (--array=i) sharing the run.sbatch template, so the array
# index -> manifest config mapping holds while resources still vary per experiment.
#
# Usage:
#   bash experiments/slurm/submit.sh            # submit all manifest entries
#   bash experiments/slurm/submit.sh --dry-run  # print the sbatch commands without submitting
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

mapfile -t ENTRIES < <(python - <<'PY'
import yaml
for i, c in enumerate(yaml.safe_load(open("experiments/manifest.yaml"))["configs"]):
    print(f"{i}\t{c.get('cpus',16)}\t{c.get('time','24:00:00')}\t{c['path']}")
PY
)

if [[ ${#ENTRIES[@]} -eq 0 ]]; then
  echo "no configs in experiments/manifest.yaml (all commented out?)"; exit 1
fi

for e in "${ENTRIES[@]}"; do
  IFS=$'\t' read -r idx cpus time path <<<"$e"
  name="qec-$(basename "$path" .yaml)"
  cmd=(sbatch --array="$idx" --cpus-per-task="$cpus" --time="$time" --job-name="$name"
       experiments/slurm/run.sbatch)
  echo "[$idx] $path  (cpus=$cpus time=$time)"
  if [[ $DRY -eq 1 ]]; then printf '    %q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done
