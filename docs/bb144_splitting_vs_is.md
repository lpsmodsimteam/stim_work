# Replica-exchange splitting vs importance sampling on bb144 — a direct-MC verdict

**TL;DR.** For the gross code `[[144,12,12]]` (single-sector), the importance-sampling **ansatz** is
the trustworthy logical-error-rate estimate (within ~1.5× of direct Monte-Carlo). The
replica-exchange **splitting** estimate **under-estimates** the LER — 2.5× at p=5e‑3, 13× at p=4e‑3,
worsening as p drops — even after the "bridge fix" that repaired its mixing. Trust the IS ansatz;
treat splitting as a sanity cross-check only.

## What the bridge fix did (and didn't) do

`experiments/bravyi/split_bb144_better.py` + the `gap_weights`/weight-matched-placement changes in
`src/splitting.py` fixed the original run's **mixing pathology**:

- swap-acceptance across all 44 rungs: **0.14 → 0.73–0.91** (the original had one choked 0.14 rung).
- mean-weight ladder: the original's cliff (42→15.6) became a **smooth** descent (83.6 → 10.5).
- the estimate rose ~10⁶ vs the original (the bottleneck was over-suppressing).

These are real improvements to the splitting *machinery* (and would help on codes where splitting
works). But they did **not** make the estimate accurate.

## Direct-MC ground truth (single-sector, same decoder)

| p | direct-MC (truth) | IS-ansatz | new splitting | IS-raw |
|---|---|---|---|---|
| 5e‑3 | **0.0643 ± 0.002** | 0.0424 (0.66×) | 0.0253 (0.39×) | 0.0082 (0.13×) |
| 4e‑3 | **0.0071 ± 0.0003** | 0.0060 (**0.84×**) | 0.00053 (**0.075×**) | 0.00098 (0.14×) |

The IS-ansatz is closest to truth (and is *better* deeper sub-threshold, its design regime). IS-raw's
~7–8× low is just the stride-4 weight under-count. Splitting is low and getting worse.

## Why splitting under-estimates (the mechanism)

The reweight ratio `P(q')/P(q) = E_{π_q}[(q'/q)^|x| ((1-q')/(1-q))^(N-|x|)]` depends on the **within-rung
weight distribution** matching the true failing distribution `π_q(w) ∝ f(w) C(N,w) q^w (1-q)^(N-w)`.
The swap-acceptance and smooth mean-weight are *cross-rung* diagnostics; they are necessary but **not
sufficient**. Comparing the chains' mean weight to the true failing mean weight (from the trusted IS
f(w)) shows the chains are systematically **miscalibrated**:

| p | chain mean-w | true failing mean-w | error |
|---|---|---|---|
| 6e‑3 | 83.6 | 78.7 | +5 |
| 2.9e‑3 | 58.7 | 44.1 | +15 |
| 1.4e‑3 | 43.8 | 26.2 | +18 |
| 4e‑4 | 10.5 | 17.9 | −7 |

The chains sit **+10 to +18 too high** through the mid-p region (then overshoot low at the bottom). A
too-high weight makes `(q'/q)^|x|` too small, so every per-rung ratio under-shoots, and the error
**compounds over 44 rungs** into the observed 2.5–13× underestimate. Root cause: single-flip local
moves cannot equilibrate the weight *within* a rung fast enough at N≈1.8×10⁵ in the available sweeps —
a slow within-rung relaxation that the cross-rung swap diagnostics don't reveal.

This quantitatively confirms the paper's caution (arXiv:2511.15177 §5) that replica-exchange splitting
is an unreliable absolute-LER estimator for codes with many inequivalent logicals / large expanded N.

## Reproduce

- Splitting (bridge-fix): `python experiments/bravyi/split_bb144_better.py` (writes
  `runs/bravyi/bb12/bb144_split_better/splitting.json`; `--pilot` for a quick check).
- IS reference: `runs/bravyi/bb12/bb144_adaptive_1e6/` (the adaptive 1e6-cap sweep).
- Ground-truth MC + the bias probe (per-rung weight vs truth): see the analysis in this directory's
  git history / the conversation that produced this note.

## Root cause confirmed against the paper, and the faithful method implemented

Reading §5 of arXiv:2511.15177 directly settled *why* `replica_exchange_estimate` under-estimates:
**it is not the paper's method.** The paper's actual recipe was never implemented; we built its
*future-work* idea (parallel tempering, named as future work in its limitations) on top of an
estimator and seeding the paper warns against. Specifically, the paper's method has three parts we
were missing:

- **Algorithm 2 — BAR ratio estimator + adaptive precision.** Bidirectional Bennett estimator
  `r_j = c·⟨g(cπ_{j-1}/π_j)⟩_{j-1}/⟨g(c⁻¹π_j/π_{j-1})⟩_j`, `g(x)=1/(1+x)`, optimal `c≈r_j`; grow each
  level's chain until `(σ+Δ) ≤ ε/√t` (relative SE σ + full-vs-first-half mixing discrepancy Δ,
  ε=0.25). We used a **one-sided** forward reweight (highest-variance, tail-dominated) with **fixed**
  chain length and **no** precision controller.
- **§5.3 multi-seeded warm-start.** Seed `p_0` with **MC-sampled *typical* failing configs**, and
  seed every lower-`p` chain from the **adjacent higher-`p` chain's final failing config**. The
  paper's BB footnote states their BB runs failed precisely when they seeded on **low-/min-weight
  circuit logicals** — which is exactly what our pool did, and exactly the too-high/too-low weight
  miscalibration measured above.
- **Eq. 18 ladder** `p_{j±1}=p_j·2^{∓1/√w_j}`, `w_j=max(D/2, p_jN)` (we used plain `geomspace`).

These are now implemented faithfully in `splitting.multi_seeded_split_estimate` (Algorithm 2/3 +
§5.3), kept separate from `replica_exchange_estimate` (the future-work variant). On the tiny
[[18,4,4]] code it **straddles** direct MC (0.99×, 0.72×, 0.85×, 1.37× across p, MC truth within the
instance-spread error bars) — i.e. noisy-but-unbiased, with **no** systematic downward collapse.

**Pending:** run `experiments/bravyi/split_bb144_multiseed.py` (use `--pilot` first) to test whether
the faithful method closes the gap to the IS-ansatz on bb144. The paper itself needed `T_init=10⁶`
and large compute for BB(12) (and dropped BB(18)), so this is a heavy run. Until that lands, the
verdict stands: **use IS for bb144; splitting is a cross-check.**
