#!/bin/bash
# Submit experiments/manifest.yaml entries as SLURM array tasks with their OWN cpus/time/mem.
# Each entry becomes a single-index array (--array=i) sharing the run.sbatch template, so the array
# index -> manifest config mapping holds while resources still vary per experiment.
#
# Usage:
#   bash experiments/slurm/submit.sh                          # submit all manifest entries
#   bash experiments/slurm/submit.sh --dry-run                # print sbatch commands, submit nothing
#   bash experiments/slurm/submit.sh --only <substring> ...   # only entries whose path contains ANY
#                                                             # given substring (wave discipline:
#                                                             # never resubmit a live earlier wave)
#   Flags combine: --only memory --dry-run
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

DRY=0
ONLY=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --only)    [[ $# -ge 2 ]] || { echo "--only needs a substring"; exit 1; }
               ONLY+=("$2"); shift 2 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

mapfile -t ENTRIES < <(python - <<'PY'
import sys, yaml
sys.stdout.reconfigure(newline="\n")   # Windows python writes \r\n; keep local dry-runs faithful
for i, c in enumerate(yaml.safe_load(open("experiments/manifest.yaml"))["configs"]):
    print(f"{i}\t{c.get('cpus',16)}\t{c.get('time','24:00:00')}\t{c.get('mem','16G')}\t{c['path']}")
PY
)

if [[ ${#ENTRIES[@]} -eq 0 ]]; then
  echo "no configs in experiments/manifest.yaml (all commented out?)"; exit 1
fi

matches() {  # does $1 contain any of the --only substrings? (no --only = match everything)
  [[ ${#ONLY[@]} -eq 0 ]] && return 0
  local p="$1" s
  for s in "${ONLY[@]}"; do [[ "$p" == *"$s"* ]] && return 0; done
  return 1
}

N=0
for e in "${ENTRIES[@]}"; do
  IFS=$'\t' read -r idx cpus time mem path <<<"$e"
  matches "$path" || continue
  N=$((N+1))
  name="qec-$(basename "$path" .yaml)"
  cmd=(sbatch --array="$idx" --cpus-per-task="$cpus" --time="$time" --mem="$mem"
       --job-name="$name" experiments/slurm/run.sbatch)
  echo "[$idx] $path  (cpus=$cpus time=$time mem=$mem)"
  if [[ $DRY -eq 1 ]]; then printf '    %q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done

if [[ $N -eq 0 ]]; then
  echo "no manifest entries match --only ${ONLY[*]}"; exit 1
fi
