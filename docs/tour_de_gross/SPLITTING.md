# Technique III — multi-seeded Metropolis splitting (arXiv:2511.15177 §5)

`splitting.py` implements the splitting / multi-seeded Metropolis rare-event
estimator and cross-checks it against the direct importance-sampling + ansatz
pipeline (`importance_sampling.py`, Techniques I/II) on the [[144,12,12]]
gross-code memory circuit.

This is a **complementary cross-check**, not a guaranteed win. The Metropolis
mixing time is unknown and can blow up, and the failing-configuration space is
generally disconnected under single-flip moves (the gross code has 12 logical
qubits). Multiple seeded chains mitigate but do not solve non-ergodicity. The
value is in whether the splitting curve *agrees* with the IS/ansatz curve over the
overlap p-range.

## Run the full memory-circuit validation

From the repo root, in the project venv (this is minutes-long; do not run in CI):

```bash
.venv/bin/python src/splitting.py \
    --p-ref 0.005 --p-high 0.006 --p-low 0.003 \
    --n-levels 6 --n-seeds 8 --chain-steps 4000 --burn-in 1000 \
    --anchor-shots 3000 --is-shots-per-weight 2000 --is-weights-hi 160 --plot
```

It prints the splitting ladder estimate, then the IS-raw and ansatz values at the
same p's, and with `--plot` writes `splitting_vs_is.png`.

**`--is-weights-hi`** controls the IS cross-check's contiguous fault-weight block
(`1..is_weights_hi`). It must bracket the failure onset (~47 for the gross memory
circuit; see the onset scan in `gross_code_sweep.py --onset-scan`) **and** the
dominant binomial mass `mu = N_exp*q` over the ladder (~88 at p=0.006). The default
160 covers both; setting it below the onset yields an all-zero IS spectrum and the
ansatz fit raises.

## Recommended parameters

| knob                  | quick check | production cross-check |
|-----------------------|-------------|------------------------|
| `--n-levels`          | 2–3         | 6–10                   |
| `--n-seeds`           | 2–4         | 8–16                   |
| `--chain-steps`       | 50–200      | 4000–20000             |
| `--burn-in`           | ~chain/4    | 1000–5000              |
| `--anchor-shots`      | 50          | 3000–10000             |
| `--p-high`/`--p-low`  | overlap with where IS sees failures (≈ 0.006 → 0.003) |

`--p-high` must be high enough that direct MC (the anchor) actually sees failures;
otherwise the run raises. Pin `--distance 12` (the code distance) so the min-weight
seed search skips the BP-OSD distance computation.

## How to read the output / caveats

- Each ratio `P(q_{i+1})/P(q_i)` is a Metropolis reweighting expectation; the
  across-seed spread (`log_ratios_se`) is the honest diagnostic. If seeds disagree
  by orders of magnitude, the chains have not mixed / are stuck in disconnected
  logical sectors — distrust the point and add seeds/steps.
- Each Metropolis-accepted candidate costs one decode, so cost ≈
  `n_levels * n_seeds * chain_steps * accept_rate` decodes plus the anchor and the
  IS cross-check. Keep `chain_steps` modest first and scale up.
- `P_logical_se` is a crude propagation (anchor binomial SE + per-level across-seed
  SE in quadrature); it does **not** certify Metropolis convergence.

### Known limitations / recommended next steps

Two structural issues mean the current estimator is a **validated scaffold**, not a
converged low-p extrapolator. Both are flagged in `splitting.py`'s module docstring:

1. **Single-flip mixing is slow.** With `N_exp ~ 2e5` and low `q`, an "add" proposal
   is accepted with probability `~q`, so the chain moves only ~`q` per step and needs
   `O(1/q)` steps to equilibrate. The illustrative `--chain-steps 4000` is likely
   orders of magnitude short at the lower ladder levels. Crank it up and watch the
   across-seed spread (`log_ratios_se`) shrink before trusting a point.
2. **Levels are not warm-started (deviation from §5).** The paper seeds each lower-q
   chain from the *final failing configs of the adjacent higher-q chain*. The
   `splitting_estimate`/`replica_exchange_estimate` code instead reuses one fixed seed
   pool for every level, so lower levels start cold from too-high-weight configs and
   **bias their ratios downward** (measured directly for bb144: see
   `docs/bb144_splitting_vs_is.md`).

   **Now addressed:** `splitting.multi_seeded_split_estimate` is a faithful
   implementation of the paper's Algorithm 2/3 + §5.3 — sequential warm-start from
   MC-sampled *typical* failing configs, the BAR ratio estimator, the adaptive σ+Δ
   precision controller, and the Eq.18 ladder. Prefer it over the older estimators when
   you want the paper's method; the older two remain for comparison.

Validate splitting over the p-range where IS+ansatz also have signal, and read
agreement there as the success criterion — not the low-p tail.
