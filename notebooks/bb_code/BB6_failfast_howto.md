# Running the BB(6) fail-fast pipeline (arXiv:2511.15177): Table 2, Figures 9 & 10

`bb6_fig10_sweep.py` reproduces the logical-error-rate-vs-physical-error-rate curve for the
distance-6 bivariate bicycle code **BB(6) = [[72,12,6]]** under circuit-level depolarizing
noise, decoded with **Relay-BP**, using all three "fail-fast" techniques:

| Technique | What it does | Module |
|-----------|--------------|--------|
| I — ansatz       | fit a low-parameter failure-spectrum ansatz `f(w)` and extrapolate `LER(p)` to low `p` | `src/importance_sampling.py` |
| II — min-weight  | exact circuit fault distance `D`, optimal onset `w0=D/2`, onset fraction `f0=f*(D/2)` (pins the ansatz) | `src/min_weight.py` |
| III — splitting  | multi-seeded Metropolis splitting cross-check of `LER(p)` | `src/splitting.py` |

## Environment (one-time)

The decoder stack needs `relay_bp` (Rust-built) and `ldpc`, installed into the **`qec`** conda env:

```powershell
# Rust toolchain (MSVC) — once
# (rustup-init.exe -y --default-toolchain stable --profile minimal)
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
$py = "C:\Users\aksirot_local\miniconda3\envs\qec\python.exe"
& $py -m pip install ldpc
# relay_bp from the repo's pinned commit (clone + local install avoids shell-quoting issues):
git clone https://github.com/trmue/relay.git $env:TEMP\relay
git -C $env:TEMP\relay checkout 6cab93095187012138a35c656fec71c5aad38eae
& $py -m pip install "$env:TEMP\relay[stim]"
```

## Smoke test FIRST (always, before the multi-hour run)

The production sweep runs for hours; validate every code path in ~3 min first (the full
pytest smoke suite runs in ~2.5 min, all 10 tests green):

```bash
# tiny end-to-end dry run (fast Relay settings, ~20 shots) — writes bb6_fig10_out_smoke/
python notebooks/bb_code/bb6_fig10_sweep.py --smoke --onset-scan --plot
# full pytest smoke suite (exercises resume + plan-mismatch guards too)
python -m pytest tests/test_bb6_fig10.py -v
```

Both must be green before launching the real job.

## Production run (multi-hour)

```bash
python notebooks/bb_code/bb6_fig10_sweep.py --onset-scan --plot
```

Paper-accurate Relay settings (§2.4, BB(6)) are the production defaults: `γ0=0.125`,
leg-1 = 80 iters, subsequent legs = 60 iters up to 600 legs (`num_sets=600`),
`γ ~ Unif[-0.24, 0.66]`, `stop_nconv=6`. These make each decode expensive — that's the
multi-hour cost. The IS sweep **checkpoints after every weight** to `bb6.spectrum.json`, so a
crash/restart resumes with at most one weight of lost work. Re-running with a different
plan/shots/seed refuses to silently resume (delete the out-dir to start fresh).

Outputs land in `bb6_fig10_out/`: `distance.json` (Tech II), `ansatz_fit.json` (Tech I),
`splitting.json` (Tech III), `bb6_fig10.npz` (all curves), `bb6_fig10.png` (the figure).

## Caveats / things to verify against the paper

- **Rounds.** The memory experiment uses `rounds = d = 6` (override with `--rounds`). Confirm
  against the Fig-10 caption/experiment description if exact reproduction matters.
- **p-grid / p_ref.** Defaults: circuit built at `p_ref=0.003`, reweighted/extrapolated over
  `p ∈ [1e-4, 1e-2]` (`--p-ref`, and `Config.p_lo/p_hi/n_p`). The paper targets LER down to
  ~1e-12 via the Technique-I extrapolation; widen the grid if you want the deep tail.
- **`stop_nconv` ↔ paper "S=6".** Mapped `S=6 → stop_nconv=6` by interpretation; verify against
  the relay_bp semantics if you need an exact match.
- **Technique III (splitting) is a cross-check, not a headline result.** It can overshoot
  (even return `P > 1`) when the Metropolis chains have not mixed — a documented limitation
  (see `notebooks/gross_code/SPLITTING.md`). Use it only where it *agrees* with the Technique-I
  ansatz over the overlapping p-range; trust the ansatz for the low-p tail. Use `--no-split`
  to skip it.

## Smoke-testing found two real Windows bugs (now fixed)

1. **cp1252 console crash** — progress lines print `μ/→/←`; the default Windows console
   encoding raised `UnicodeEncodeError` and killed the job. Fixed by forcing UTF-8 stdout.
2. **transient `os.replace` lock** — the after-every-weight checkpoint write hit a transient
   `PermissionError [WinError 5]` (AV/indexer on the Desktop path). Fixed with retry+backoff so
   a momentary lock can't kill a multi-hour run.
