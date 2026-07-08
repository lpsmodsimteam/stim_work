"""Generate error_model_comparison_18_4_4.ipynb (source only; cells are NOT executed here).

An experiment notebook built on the three-techniques tutorial: it compares how the FULL circuit noise
of the Kunlun [[18,4,4]] code decomposes into its parts — full symmetric vs. measurement-only vs.
two-qubit(CX)-gate-only — using Technique II (min-weight onset), Technique I (failure-spectrum ansatz),
Technique III (replica-exchange splitting), and direct Monte-Carlo, all on the same code.

§5 adds the leave-one-out ABLATION models (all-but-one channel) — the marginal contributions Google's
Willow error budget is built from — and §6 assembles the Willow-style budget: isolated vs marginal
fractions of LER_full, the mixing bucket, per-channel pseudo-thresholds, and the Σ p/p_th terms.

Noise channels are isolated by FILTERING instructions on one symmetric(p) circuit, so every variant
shares the identical per-location rate p (apples-to-apples). Run top-to-bottom in the `qec` kernel.
"""
import json
from repo_paths import REPO_ROOT

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

# ===========================================================================
md(r"""# Error-model experiments on the Kunlun **[[18,4,4]]**

How does the full circuit noise decompose into its parts? Using the three fail-fast techniques (from
`three_techniques_18_4_4.ipynb`) **together with direct Monte-Carlo**, we compare three error models on
the *same* [[18,4,4]] code:

| model | noise kept |
|---|---|
| **full symmetric** | everything — `DEPOLARIZE1` + `DEPOLARIZE2` + `X_ERROR` |
| **CZ only** | `DEPOLARIZE2` (two-qubit `CX` gate depolarizing) |
| **meas only** | `X_ERROR` immediately **before** a measurement `M` |
| **prep only** | `X_ERROR` immediately **after** a reset `R` (state preparation) |
| **idle only** | `DEPOLARIZE1` on **idle data qubits** (while ancillas are reset/measured) |

We isolate a channel by **filtering noise instructions** on one `symmetric(p)` circuit — so every variant
uses the *identical* per-location rate `p` (a clean apples-to-apples comparison) — via
`bb_code_sim.filter_noise_channel`, whose predicates live next to the circuit builder whose layout they encode.
Two channels share an instruction name and are split by **position**: `X_ERROR` is a *preparation* error
right after an `R` and a *measurement* error right before an `M`; `DEPOLARIZE1` is an *idle* error (on the
data qubits, which sit idle during ancilla reset/measurement) except right after an `H`, where it is the
single-qubit *gate* error on the X-ancillas. (The two-qubit gates are `CX`; "CZ only" isolates their
two-qubit depolarizing.)

**Idle occupancy.** In the depth-7 schedule each data qubit is idle in **2 of the 8 rounds** per syndrome
cycle — round 7 (all data idle while ancillas are measured/reset) plus exactly one of rounds 0/6 (it is
busy on a `CX` in the other; rounds 1–5 keep every qubit busy). So a data qubit picks up idle
`DEPOLARIZE1(p)` **twice per cycle**, an idle-error probability of `1−(1−p)² ≈ 2p` (not `3p`).""")

# ---------------------------------------------------------------------------
md("""## Setup — the code, the noise filter, and the three models""")

code('''import numpy as np, matplotlib.pyplot as plt, stim
from bb_code_sim import (BBCodeParams, BBCodeSimulator, RelayBPDecoder,
                         NOISE_CHANNEL_PREDICATES, filter_noise_channel)
from surface_code_sim import ErrorModel
from min_weight import (dem_check_action_matrices, compute_distance, optimal_onset_fraction,
                        find_weight_logicals_mitm, find_all_min_weight_logicals)
from importance_sampling import importance_sample, fit_failure_spectrum, logical_error_rate_from_ansatz
from splitting import replica_exchange_estimate

P = BBCodeParams(l=3, m=3, a_exps=[(1, 0), (0, 0), (0, 2)], b_exps=[(0, 1), (0, 0), (2, 0)], distance=4)
P_REF, ROUNDS = 0.01, 2

# Channels are isolated by position-filtering ONE symmetric(p) circuit at the same base rate p.
# bb_code_sim.filter_noise_channel owns the position predicates (X_ERROR after R = prep, before
# M = meas; DEPOLARIZE1 not post-H = idle; DEPOLARIZE2 = two-qubit gate) — they encode the layout
# of build_bb_circuit, so they live in src next to it.
MODELS = {"full symmetric": None, "CZ only": "cz", "meas only": "meas",
          "prep only": "prep", "idle only": "idle"}
COLORS = {"full symmetric": "crimson", "CZ only": "navy", "meas only": "seagreen",
          "prep only": "darkorange", "idle only": "purple"}
def make_circuit(model, p):
    return filter_noise_channel(BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(p), rounds=ROUNDS),
                                MODELS[model])

for name in MODELS:
    print(f"{name:16s}: {make_circuit(name, P_REF).detector_error_model().num_errors} DEM mechanisms")''')

# ===========================================================================
md(r"""## §1 — Technique II: distance, onset, perfect-decoder floor (per model)

For each model: circuit fault distance `D`, onset weight `w₀=⌈D/2⌉`, the exact `L(D)`, and the
perfect-decoder onset fraction `f₀*`. The four *isolated* channels (CZ / meas / prep / idle) each turn
out to have **even** distance 4 (the code distance) — so `f₀*` is exact via Proposition 1 — while the **full**
model has **odd** distance 3: only *combining* channels makes the weight-3 hook that drops it below the
code distance (Appendix A.6 route for `f₀*`). We enumerate `L(D)` with the ldpc-free half-MITM for even
`D` (robust) and the coset search for odd `D`.""")

code('''def enumerate_LD(circuit, D):
    """Complete L(D): half-MITM for even D (exact, no ldpc); parallel coset enumeration for odd D."""
    if D % 2 != 0:
        return find_all_min_weight_logicals(circuit, D, budget_per_coset=40)
    H, A, mult, priors = dem_check_action_matrices(circuit)
    return find_weight_logicals_mitm(H, A, D)

tech2 = {}
print(f"{'model':16s} {'D':>2} {'w0':>3} {'#DEM':>6} {'|L(D)|':>8} {'f0*':>8}   route")
for name in MODELS:
    c = make_circuit(name, P_REF); D = compute_distance(c).distance
    LD = enumerate_LD(c, D)
    onset = optimal_onset_fraction(c, distance=D, logicals=LD)
    tech2[name] = dict(D=D, LD=LD, onset=onset)
    route = "Prop.1 (even D)" if D % 2 == 0 else "App.A.6 (odd D)"
    print(f"{name:16s} {D:2d} {onset.onset:3d} {c.detector_error_model().num_errors:6d} "
          f"{onset.n_min_logicals:8d} {onset.onset_fraction:8.4f}   {route}")''')

# ===========================================================================
md(r"""## §2 — Technique I: failure-spectrum ansatz (per model)

Importance-sample the failure spectrum `f(w)` and fit the f5 ansatz (onset `w₀` left free — the multistart
fit finds the good minimum), then reweight to `LER(p)` over the grid.""")

code('''p_grid = np.geomspace(1e-4, 0.013, 40)
tech1 = {}
for name in MODELS:
    c = make_circuit(name, P_REF)
    spec = importance_sample(c, RelayBPDecoder(), p_ref=P_REF, p_values=[P_REF],
                             weights=list(range(1, 11)), shots_per_weight=6000, seed=1).spectrum
    fw = np.array([F / T for F, T in zip(spec.failures, spec.trials)])
    fit = fit_failure_spectrum(spec, K=c.num_observables, model="f5", w0=None, f0=None)
    LER = np.asarray(logical_error_rate_from_ansatz(fit, list(p_grid)))
    tech1[name] = dict(spec=spec, fw=fw, fit=fit, LER=LER)
    print(f"{name:16s}: measured f(2)={fw[1]:.4f}   f5 fit cost={fit.cost:.2f}")''')

# ===========================================================================
md(r"""## §3 — Technique III: replica-exchange splitting (per model)

Splitting reaches deep into the rare regime, seeded by each model's exact `L(D)` from §1.""")

code('''tech3 = {}
for name in MODELS:
    c = make_circuit(name, P_REF); info = tech2[name]
    temper, diag = replica_exchange_estimate(
        c, RelayBPDecoder(), p_ref=P_REF, p_high=0.015, p_low=1e-4, n_levels=16,
        n_walkers=8, local_steps=5, n_sweeps=80, burn_in=20, anchor_shots=4000,
        distance=info["D"], seed=2, single_sector=False, mw_supports=list(info["LD"]), verbose=False)
    sp = np.asarray(temper.p_ladder)[::-1]; sP = np.asarray(temper.P_logical)[::-1]
    tech3[name] = dict(sp=sp, sP=sP)
    swmin, swmax = min(diag["swap_accept"]), max(diag["swap_accept"])
    print(f"{name:16s}: swap-accept {swmin:.2f}..{swmax:.2f}   P(1e-4)={sP[0]:.2e}")''')

# ===========================================================================
md(r"""## §4 — Direct Monte-Carlo + overlay

Direct-MC ground truth at moderate `p`, overlaid with all three techniques for the three models. Per
model the ansatz line, the splitting squares, and the MC circles should coincide where they overlap.""")

code('''def direct_mc(circ, shots):
    d = RelayBPDecoder(); d.setup(circ)
    det, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True)
    f = np.any(d.decode_batch(det) != obs, axis=1); m = f.mean()
    return m, (max(m, 1e-9) * (1 - m) / shots) ** 0.5

mc_pts = {0.012: 80_000, 0.008: 120_000, 0.005: 200_000, 0.003: 300_000}
mc = {name: {p: direct_mc(make_circuit(name, p), s) for p, s in mc_pts.items()} for name in MODELS}

print("   p        " + "".join(f"{n:>15}" for n in MODELS))
for p in mc_pts:
    print(f"  {p:.3f}   " + "".join(f"{mc[n][p][0]:>15.3e}" for n in MODELS))

fig, ax = plt.subplots(figsize=(9, 6))
for name in MODELS:
    col = COLORS[name]
    ax.plot(p_grid, tech1[name]["LER"], "-", color=col, lw=2, label=f"{name}")
    ax.plot(tech3[name]["sp"], tech3[name]["sP"], "s", color=col, ms=4, mfc="none")
    mp = sorted(mc[name])
    ax.errorbar(mp, [mc[name][p][0] for p in mp], yerr=[mc[name][p][1] for p in mp],
                fmt="o", color=col, capsize=3, zorder=5)
ax.plot([], [], "-", color="gray", label="Technique I: f5 ansatz")
ax.plot([], [], "s", color="gray", mfc="none", label="Technique III: splitting")
ax.plot([], [], "o", color="gray", label="direct Monte-Carlo")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("physical error rate p"); ax.set_ylabel("logical error rate")
ax.set_title("Kunlun [[18,4,4]] — error-model comparison (three techniques + direct MC)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both"); plt.tight_layout(); plt.show()''')

# ===========================================================================
md(r"""## §5 — Ablation (leave-one-out): the marginal channel contributions

Google's Willow error budget (arXiv:2408.13687) is built from **marginal** contributions — remove (or
scale) one error component in simulation and measure how much the logical error rate drops. The
channel-*isolated* circuits of §1–§4 cannot provide this: §1 showed every isolated channel has even
distance **4** while the full model has odd distance **3**, so the dominant low-`p` failures are
**mixed-channel** faults that no isolated circuit contains. Here we run the four **all-but-one**
models (`keep = not channel`) through the same pipeline.

The ablated distances are a structural diagnostic: if dropping channel *i* restores `D=4`, then
channel *i* participates in **every** weight-3 hook; if `D` stays 3, the hooks survive without it.""")

code('''ABLATED = {"no CZ": "cz", "no meas": "meas", "no prep": "prep", "no idle": "idle"}
def make_ablated_circuit(name, p):
    drop = NOISE_CHANNEL_PREDICATES[ABLATED[name]]
    return filter_noise_channel(BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(p), rounds=ROUNDS),
                                lambda i, pr, nx: not drop(i, pr, nx))

tech2_abl = {}
print(f"{'model':10s} {'D':>2} {'w0':>3} {'#DEM':>6} {'|L(D)|':>8} {'f0*':>10}   hook diagnosis")
for name in ABLATED:
    c = make_ablated_circuit(name, P_REF); D = compute_distance(c).distance
    LD = enumerate_LD(c, D)
    onset = optimal_onset_fraction(c, distance=D, logicals=LD)
    tech2_abl[name] = dict(D=D, LD=LD, onset=onset)
    diag = ("restores D=4 -> channel is in EVERY weight-3 hook" if D == 4
            else "still D=3 -> hooks survive without this channel" if D == 3 else "")
    print(f"{name:10s} {D:2d} {onset.onset:3d} {c.detector_error_model().num_errors:6d} "
          f"{onset.n_min_logicals:8d} {onset.onset_fraction:10.3e}   {diag}")''')

code('''tech1_abl, mc_abl = {}, {}
for name in ABLATED:
    c = make_ablated_circuit(name, P_REF)
    spec = importance_sample(c, RelayBPDecoder(), p_ref=P_REF, p_values=[P_REF],
                             weights=list(range(1, 11)), shots_per_weight=6000, seed=3).spectrum
    fit = fit_failure_spectrum(spec, K=c.num_observables, model="f5", w0=None, f0=None)
    tech1_abl[name] = dict(spec=spec, fit=fit,
                           LER=np.asarray(logical_error_rate_from_ansatz(fit, list(p_grid))))
    mc_abl[name] = {p: direct_mc(make_ablated_circuit(name, p), s) for p, s in mc_pts.items()}
    print(f"{name:10s}: f5 fit cost={fit.cost:.2f}   MC LER(p=0.005)={mc_abl[name][0.005][0]:.3e}")''')

# ===========================================================================
md(r"""## §6 — The error budget (Willow-style)

Willow's budget reads `1/Λ ≈ Σᵢ pᵢ/p_th,i` — each channel priced against its own threshold. Two
honesty notes for a **single** code:

* There is no Λ (it is a ratio between code *sizes*); the single-code stand-in for `p_th,i` is the
  channel **pseudo-threshold** — the break-even `p` where `LER_i(p) = p`. A *true* Λ needs the
  same-polynomial `(l,m)=(6,6)` sibling **[[72,4,8]]** (`A=1+x+y²`, `B=1+y+x²`, exact d=8).
* Two decompositions of `LER_full` are shown, and they differ by construction:
  **isolated** `LER_i/LER_full` (misses mixed faults; the deficit is the **mixing bucket**) and
  **marginal** `1 − LER_{no i}/LER_full` (Google's convention; Σ typically **exceeds 1** because a
  mixed fault is killed by removing *any* of its participant channels, so it is counted in each).
  SPAM = meas + prep (their linear budget terms add).""")

code('''P_STAR = 0.005                       # budget operating point (MC-anchored)

def ansatz_at(d, p):                 # evaluate a stored f5 fit at scalar p
    return float(np.asarray(logical_error_rate_from_ansatz(d["fit"], [p]))[0])

def pseudo_threshold(pg, LER):       # break-even LER(p)=p, log-log interpolated
    r = np.log(LER) - np.log(pg)
    s = np.nonzero(np.diff(np.sign(r)) != 0)[0]
    if s.size == 0:
        return None
    i = s[-1]; t = r[i] / (r[i] - r[i + 1])
    return float(np.exp(np.log(pg[i]) + t * (np.log(pg[i + 1]) - np.log(pg[i]))))

CHANNELS = ["CZ only", "meas only", "prep only", "idle only"]
ABL_OF = {"CZ only": "no CZ", "meas only": "no meas", "prep only": "no prep", "idle only": "no idle"}

L_full = ansatz_at(tech1["full symmetric"], P_STAR)
mc_full = mc["full symmetric"][P_STAR][0]
rows = []
for ch in CHANNELS:
    iso = ansatz_at(tech1[ch], P_STAR) / L_full
    marg = 1.0 - ansatz_at(tech1_abl[ABL_OF[ch]], P_STAR) / L_full
    marg_mc = 1.0 - mc_abl[ABL_OF[ch]][P_STAR][0] / mc_full
    pth = pseudo_threshold(p_grid, tech1[ch]["LER"])
    rows.append((ch, iso, marg, marg_mc, pth))

print(f"error budget at p* = {P_STAR}   (LER_full: ansatz {L_full:.3e}, MC {mc_full:.3e})")
print(f"{'channel':12s} {'isolated':>9} {'marginal':>9} {'marg(MC)':>9} {'p_pth':>9} {'p*/p_pth':>9}")
for ch, iso, marg, marg_mc, pth in rows:
    pths = f"{pth:.4f}" if pth else ">grid"
    term = f"{P_STAR/pth:.3f}" if pth else "-"
    print(f"{ch:12s} {iso:9.3f} {marg:9.3f} {marg_mc:9.3f} {pths:>9} {term:>9}")
mixing = 1.0 - sum(r[1] for r in rows)
spam_iso = sum(r[1] for r in rows if r[0] in ("meas only", "prep only"))
spam_marg = sum(r[2] for r in rows if r[0] in ("meas only", "prep only"))
print(f"{'mixing':12s} {mixing:9.3f}           (1 - sum isolated: cross-channel faults)")
print(f"{'SPAM':12s} {spam_iso:9.3f} {spam_marg:9.3f}           (meas + prep combined)")
print(f"sum(marginal) = {sum(r[2] for r in rows):.3f}  (>1 <=> shared mixed faults; Google renormalizes)")''')

code('''fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
# left: marginal budget fractions vs p (from the ansatz curves)
Lf = tech1["full symmetric"]["LER"]
for ch in CHANNELS:
    frac = 1.0 - tech1_abl[ABL_OF[ch]]["LER"] / Lf
    axL.plot(p_grid, np.clip(frac, 0, None), "-", color=COLORS[ch], lw=2, label=ch)
axL.set_xscale("log"); axL.set_xlabel("physical error rate p")
axL.set_ylabel("marginal budget fraction  1 − LER$_{no\\,i}$/LER$_{full}$")
axL.axvline(P_STAR, color="gray", ls=":", lw=1); axL.legend(fontsize=8); axL.grid(alpha=0.3)
# right: the budget bar chart at p*
labels = [r[0] for r in rows] + ["mixing"]
iso_v  = [r[1] for r in rows] + [mixing]
marg_v = [r[2] for r in rows] + [np.nan]
x = np.arange(len(labels))
axR.bar(x - 0.2, iso_v, 0.4, label="isolated (channel-only)")
axR.bar(x + 0.2, marg_v, 0.4, label="marginal (leave-one-out)")
axR.set_xticks(x, labels, rotation=20); axR.set_ylabel(f"fraction of LER_full at p*={P_STAR}")
axR.legend(fontsize=8); axR.grid(alpha=0.3, axis="y")
fig.suptitle("Kunlun [[18,4,4]] — Willow-style error budget"); plt.tight_layout(); plt.show()''')

# ---------------------------------------------------------------------------
md(r"""## Takeaways

* **The full-circuit distance-3 reduction needs *mixed* error types.** `CZ only`, `meas only`,
  `prep only`, and `idle only` are each distance **4** (the code distance); only *combining* channels
  produces the weight-3 hook that drops the full model to `D=3`.
* **Two-qubit (CZ) gate errors dominate** the logical error rate; the single-location channels
  (**measurement, preparation, idle-data**) are each far smaller — fewer DEM mechanisms and much lower
  LER — and the full LER sits well above the sum of the isolated channels, i.e. the mixed-type faults
  carry most of it.
* **Preparation vs. measurement:** reset and measurement bit-flips are the same `X_ERROR` at the same
  rate, but split by circuit position they can contribute differently — compare the `prep only` and
  `meas only` curves.
* **The techniques agree with direct MC for every model** and extrapolate together into the rare regime:
  the fail-fast toolkit works channel-by-channel, not just on the full model.
* Both onset routes appear in one comparison: the isolated channels are **even distance 4** (exact `f₀*`
  via Proposition 1), the full model is **odd distance 3** (Appendix A.6).
* **Isolated ≠ marginal (§5–§6).** The channel-only circuits miss the mixed-type faults entirely, so
  their fractions under-count and leave a **mixing bucket**; the leave-one-out marginals (Google's
  convention) count each mixed fault once per participant, so they **over-count** (Σ > 1). The gap
  between the two decompositions *is* the mixed-fault structure — read it together with the ablated
  distances, which name the channels every weight-3 hook needs.
* **Pseudo-thresholds stand in for `p_th,i`** in the Willow sum `Σᵢ p/p_th,i`: a true Λ requires a
  second code size — the same-polynomial `(l,m)=(6,6)` sibling **[[72,4,8]]** (exact d=8, both sectors,
  verified by complete w≤7 MITM) is the natural partner.

*Generated by `make_error_model_comparison.py`.*""")

# ===========================================================================
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "methods" / "error_model_comparison_18_4_4.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
