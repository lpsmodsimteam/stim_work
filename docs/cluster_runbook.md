# Cluster campaign runbook — K=12 memory ladder + LPU error budgets

Manual-submission runbook for the campaign plan of 2026-07-13 (plan:
`~/.claude/plans/delegated-conjuring-kite.md`; predecessor context in the repo memory).
Each wave is a copy-paste section: run top to bottom, check every box before `sbatch`.

**Global rules**
- Always submit through `bash experiments/slurm/submit.sh --only <substring> [--dry-run]` —
  never bare `submit.sh` (it would resubmit earlier waves against live outdirs).
- Dry-run first, every time. Verify the printed entries are EXACTLY the wave's jobs.
- Pin the cluster checkout per wave: record `git rev-parse HEAD` below at submit time and do
  not pull mid-wave — a requeued task must resume against the same code. Fast-forward only
  between waves, after the wave's gate passes.
- A timed-out/killed task is resumable: re-run the SAME sbatch line (per-weight checkpoint
  loses at most one weight). If it aborts with "weights_plan/seed mismatch": the config
  changed under a live run — do NOT edit configs of running jobs.
- Pull results home from the local box (any time; checkpoints are atomic):
  `rsync -av cluster:stim_work/runs/framework/ runs/cluster/framework/`
  `rsync -av cluster:stim_work/runs/slurm/     runs/cluster/slurm/`

---

## Wave 1 — memory full-noise ladder            submit day: 2026-07-14
Jobs: bb6_memory (Fig-10 control, 16c/8h), bb6_memory_m100 (ladder leg, 16c/12h),
bb144_memory (32c/72h), bb288_memory (48c/96h).
Pinned SHA: `____________________` (fill at submit: `git rev-parse HEAD`)

0. Preconditions (from local Day-0, all done 2026-07-13):
   - [x] all five original configs smoke locally; pytest green
   - [x] bb144 adaptive_failures=25 (+72h), bb288 failures=15 + explicit contiguous-head weights
   - [x] bb6_memory_m100 exists (decoder-unified num_sets=100; no f0 pin)
   - [x] submit.sh has --only + per-entry mem; manifest indices frozen (0-5)
1. One-time env build (login node), from the repo root at the pinned SHA:
   ```
   git clone <your-fork-url> stim_work && cd stim_work        # or git pull if cloned
   git checkout bb144-split-better && git rev-parse HEAD       # record above
   pip install -r requirements.txt && pip install -e .         # relay_bp needs the Rust toolchain
   python test.py                                              # preflight: packages + editable
                                                               # install + decode round-trip +
                                                               # SLURM reachable — must be 15/15
   python -m pytest -q                                         # must be green before anything
   ```
2. 15-minute compute-node smoke of the LARGEST config (catches env/arch issues at minute 15,
   not hour 90 — do not skip):
   ```
   srun --cpus-per-task=4 --mem=16G --time=00:20:00 \
     python -m experiment_runner --config experiments/configs/bb288_memory.yaml --smoke --cpus 4
   ```
   PASS = `runs/framework/bb288/memory_smoke/result.npz` exists.
3. Dry-run — verify EXACTLY four entries print (indices 0-3, cpus/time/mem as above):
   ```
   bash experiments/slurm/submit.sh --only memory --dry-run
   ```
4. Submit: `bash experiments/slurm/submit.sh --only memory`
5. Monitor: `squeue -u $USER`; logs `tail -f runs/slurm/qec-bb288_memory_*.out`.
   Expected completions: bb6 pair same day; bb144 ~07-17; bb288 ~07-18.
6. Pull results home (rsync lines above) — safe mid-run; bb6 lands within ~a day.
7. Close: on the local box, `python experiments/methods/qc_wave.py --wave 1` (available after
   LA lands, ~Day 4). PASS = G1: Fig-10 reproduction in tolerance; Λ(bb6_m100→bb144)(p_ref)
   finite ±σ and **Λ>1** (Λ<1 anywhere = decoder-degradation tell → STOP, no W2);
   bb144 onset-region zero-failure-bin fraction <20%.

## Wave 1b — LPU x1/z1 full-noise               submit day: 2026-07-14/15
Jobs: gross_lpu_x1, gross_lpu_z1 (16c/24h each). May share the submit day with Wave 1.
**Quarantine: no LPU-vs-memory comparison until the TdG-vs-Bravyi presentation test (G3a).**

1. Dry-run — exactly two entries (indices 4-5): `bash experiments/slurm/submit.sh --only lpu --dry-run`
2. Submit: `bash experiments/slurm/submit.sh --only lpu`
3. Monitor/pull as above. Results land in `runs/framework/bb144/lpu_x1|lpu_z1`.

## Wave 2 — bb6+bb144 memory channels           submit day: ~2026-07-16/17
20 jobs: `configs/channels/bb6_memory_m100__*` + `bb144_memory__*` (iso+abl × 5 channels).
Pinned SHA: `____________________`

0. Preconditions:
   - [ ] G1 passed (`qc_wave --wave 1`; bb144 may be partial — its checkpoint reweights)
   - [ ] G2 passed locally (channel tests + one iso/abl smoke per code; done Day-0/1)
   - [ ] Wave-2 manifest block uncommented (indices 6-25; do NOT reorder anything above it)
1. Fast-forward the cluster checkout to the recorded W2 SHA; `pytest -q`.
2. Dry-run: `bash experiments/slurm/submit.sh --only channels/bb6 --only channels/bb144 --dry-run`
   — exactly 20 entries, walltimes 4-72h per the manifest.
3. Submit: same line without `--dry-run`.
4. Close: `qc_wave --wave 2` — decomposition identity (Σ isolated + residual vs full within σ),
   verdict table at p_ref and 2×p_ref, no systematic Λ<1.

## Wave 3 — bb288 channels + LPU x1/z1 channels submit day: ~2026-07-20/21
30 jobs: `configs/channels/bb288_memory__*` + `gross_lpu_x1__*` + `gross_lpu_z1__*`.
Preconditions: G3 (LPU partition test green — already in tests/test_channel_filter.py;
G3a presentation test resolved either way; bb288 full-noise complete through the onset head;
W2 QC clean). Uncomment the Wave-3 block; submit
`bash experiments/slurm/submit.sh --only channels/bb288 --only channels/gross_lpu`.

## Wave 4 — targeted repair                     submit day: ~2026-07-24+ (data-driven)
Job list comes ONLY from `qc_wave` verdict tables: onset-boost configs (new config, explicit
onset weights stride 1, seed 43 — pooled downstream, never in-place) for "not robust" shares;
splitting anchors for bb288 full + iso_cz and one LPU circuit; Tech-II gap-fill for
bb288 iso_cz/abl_cz; p_hi-extension reruns for crossings that fell outside sampled mass.

## Wave 5 — new LPU operations                  after G5 (L3 battery) passes locally
In-module joint Pauli → shift automorphism → out-of-module joint Pauli. Per op: full-noise
first (sizing probe sets walltime), channels after that op's own partition test. Nothing
ships without: zero-noise determinism + TableauSimulator logical correctness + mini-scale
three-way (MC/IS/Tech-II) budget agreement.

## Wave 6 — double-gross LPU                    open-ended
Blocked on derive_lpu_layout at (12,12). First job = sizing probe, then mirror W1b→W3.
