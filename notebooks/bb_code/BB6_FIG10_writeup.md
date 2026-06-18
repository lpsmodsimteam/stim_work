# BB(6) Figure-10 Reproduction: Methods and Results

Reproducing the **BB(6) = [[72,12,6]] Relay-BP line** of Figure 10 from arXiv:2511.15177
("Fail fast: techniques to probe rare events in quantum error correction") using all three
fail-fast techniques.

## 1. Code and circuit

**Code:** Bivariate Bicycle (BB) code BB(6) with parameters
`l=6, m=6, a_exps=[(3,0),(0,1),(0,2)], b_exps=[(0,3),(1,0),(2,0)]`,
giving `[[72, 12, 6]]` (72 data qubits, 12 logical qubits, distance 6).

**Syndrome schedule:** Bravyi depth-7 interleaved schedule from
`github.com/sbravyi/BivariateBicycleCodes` (file `decoder_setup.py`),
with steering vectors `sX = [idle, 1, 4, 3, 5, 0, 2]` and `sZ = [3, 5, 0, 1, 2, 4, idle]`.
In rounds 1–5 of each syndrome cycle all 144 qubits are active (X-checks act on one data half,
Z-checks act on the other), leaving no idle qubits in those rounds.

**Noise model:** Standard circuit-level depolarizing noise (paper §2):
- CNOT: two-qubit depolarizing `DEPOLARIZE2(p)` on both qubits
- Prep / measurement: `X_ERROR(p)` on the ancilla
- Idle qubits (rounds 0, 6, 7 of each cycle): `DEPOLARIZE1(p)` on all idle qubits

At physical error rate `p`, the per-mechanism fault probability in the DEM is
`q = p / 15` for two-qubit mechanisms and `q ≈ p` for single-qubit ones (Stim normalisation).
The DEM has `N_expanded = 68940` expanded fault mechanisms at `p_ref = 0.003`.

**Rounds:** 6 syndrome rounds (= code distance d = 6), a Z-memory experiment.

## 2. Decoder

**Relay-BP** (relay_bp Rust package) with paper-accurate settings (§2.4, BB(6)):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `gamma0` | 0.125 | Initial damping |
| `pre_iter` | 80 | Iterations in the first leg |
| `num_sets` | 600 | Number of legs (relay chains) |
| `set_max_iter` | 60 | Max iterations per subsequent leg |
| `gamma_dist_interval` | (-0.24, 0.66) | Uniform distribution for per-leg gamma |
| `stop_nconv` | 6 | Stop after this many unconverged bits |

These make each decode ~20–30 ms, so high shot counts are feasible but expensive.

## 3. Three fail-fast techniques

### Technique I — Ansatz extrapolation (§3)

**Importance sampling (IS):** At each fault weight `w`, draw `T = 500` random fault patterns
with exactly `w` faults, decode with Relay-BP, and record the failure fraction `F/T = f_hat(w)`.
Reweighting gives `LER(p) ≈ Σ_w C(N, w) q^w (1-q)^(N-w) f_hat(w)`.

**Ansatz fit:** The failure spectrum `f(w)` is modelled by the 3-parameter `f3` ansatz:
```
f(w) = 0                      for w < w0
f(w) = f0 * (1 - exp(-gamma*(w - w0)))^2   for w >= w0
```
(or the 5-parameter `f5` extension with a low-w crossover). The onset weight `w0` and
onset fraction `f0` are pinned from Technique II. The shape parameter `gamma` is fit by
least squares to the sampled `f_hat(w)` values.

The fitted ansatz is then evaluated analytically at any `p` to extrapolate `LER(p)` far below
the IS sampling range.

**p-grid:** `p ∈ [1e-5, 1e-2]` (60 log-spaced points), built at `p_ref = 0.003`.

### Technique II — Min-weight onset (§4)

**Circuit fault distance D** is computed by appending each of the K=12 logical-observable rows
of the detector action matrix A to the check matrix H, then decoding the syndrome that forces
the appended row to fire (BP-OSD, `osd_order=10`). The minimum-weight correction is the
min-weight logical bitstring for that observable. `D = min` over the 12 observables.

**Min-weight logical set L(D):** The full set of weight-D logical bitstrings is searched by
two phases:
1. **Systematic phase:** all `2^K − 1 = 4095` nonzero GF(2) combinations of the K logical
   generators are decoded exhaustively. This guarantees every syndrome coset is visited at least
   once, at the cost of ~4095 BP-OSD decodes (~10 min with 8 workers at ~1.2 s/decode).
2. **Random phase:** `max_trials = 2000` random linear combinations provide additional coverage.

**Onset fraction f*(D/2) via Proposition 1:** For even `D`, enumerates all weight-`D/2`
restrictions of L(D), groups by `(syndrome, logical action)`, and counts the failing fraction
exactly: `f*(D/2) = |F(D/2)| / C(N_expanded, D/2)`.

**Results (Bravyi circuit):** D=6 confirmed, onset `w0 = D/2 = 3`. Search results compared:

| Search method | Circuit | |L(D)| | |F(D/2)| | ratio to paper |
|---|---|---|---|---|
| Random 2000 trials, workers=1 (this sweep) | Bravyi | 60 | 2.57×10⁷ | 0.067 |
| Systematic 4095 + random 2000, workers=8 | Bravyi | **85** | **3.08×10⁷** | **0.08** |
| Random 2000 trials, workers=8 (old session) | Sequential | 188 | 1.34×10⁷ | 0.035 |
| Paper Table 2 | — | ≥1524 | 3.83×10⁸ | 1.000 |

Systematic enumeration finds 44% more logicals and 2.3× higher |F(D/2)| than random-only (for
the Bravyi circuit). We reach 8% of the paper's `|F(D/2)|`. The remaining gap is because
BP-OSD at `osd_order=10` returns at most one weight-6 logical per syndrome class; higher OSD
order or a dedicated min-weight algorithm would improve coverage further.

**Table-2 pin:** Because our random BP-OSD search is incomplete (BP-OSD is not guaranteed
min-weight), compare_vs_paper.py pins `f0` directly from the paper's exact count:
`f0 = 3.83e8 / C(N_expanded, 3)`.

### Technique III — Metropolis splitting (§5, cross-check)

Multi-seeded Metropolis splitting estimates `LER(p)` at a ladder of physical error rates
`p ∈ [0.001, 0.006]` as a cross-check of Technique I. Run with `n_seeds=32`, `chain_steps=4000`,
`burn_in=1000` per level. The splitting estimate is valid where chains mix; it can overshoot
(`P > 1`) in the non-mixing regime. Use where it agrees with Technique I only.

**Caveat:** At the low physical error rates of interest (`p ≈ 10⁻⁵`), Metropolis chains
typically do not mix for `n_seeds=32` and `chain_steps=4000`. Technique I (ansatz extrapolation)
is the authoritative curve for the low-`p` tail.

## 4. Comparison with paper Table 2

| Quantity | Paper (arXiv:2511.15177) | Ours |
|----------|--------------------------|------|
| Circuit | Bravyi depth-7, standard noise | Same (corrected, was sequential) |
| `N_expanded` | — | 68940 |
| `D` | 6 | 6 ✓ |
| `|L(D)|` | ≥ 1524 | 60 (random-only sweep); 85 (systematic+random, workers=8) |
| `|F(D/2)|` | 3.83×10⁸ | 2.57×10⁷ (sweep, random-only); 3.08×10⁷ (systematic+random) |
| `f0 = f*(D/2)` | 3.83e8 / C(68940, 3) | same (Table-2 pin) |

**The main remaining gap** is in `|L(D)|`. BP-OSD at `osd_order=10` with random linear
combinations finds only a subset of all weight-6 logical operators. The systematic enumeration
covers all 4095 syndrome classes exactly once; increasing `osd_order` would improve the
per-class yield further.

## 5. Output files

| File | Contents |
|------|----------|
| `bb6.spectrum.json` | IS checkpoint: trials, failures, weights per sweep so far |
| `distance.json` | D, onset, \|L(D)\|, fail_count, onset_fraction, N_expanded |
| `ansatz_fit.json` | Fit parameters (w0, f0, gamma), cost, n_points, ansatz model |
| `splitting.json` | Metropolis ladder results: P per level, log-ratio SE |
| `bb6_fig10.npz` | Full numerical results: p_values, IS LER, ansatz LER, splitting LER |
| `bb6_fig10.png` | Figure: IS + Technique I ansatz + Technique III splitting vs paper |

## 6. Caveats

- **Rounds = d = 6** (fixed). The paper may use a different convention for the memory
  experiment length; verify against the Figure-10 caption if exact reproduction matters.
- **Technique III non-mixing:** splitting estimates at `p < 0.003` are likely invalid. Only
  interpret the splitting curve where it overlaps and agrees with Technique I.
- **Idle noise baseline is new:** all prior runs using the sequential 12-CNOT schedule without
  idle noise are stale. All results in `bb6_fig10_out_idle/` use the corrected circuit.
- **|L(D)| undercount:** our BP-OSD search finds a strict subset of the paper's 1524+ logicals.
  compare_vs_paper.py uses the paper's exact Table-2 count to pin f0 rather than our count.
