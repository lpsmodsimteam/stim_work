"""Generate error_model_comparison_18_4_4.ipynb (source only; cells are NOT executed here).

An experiment notebook built on the three-techniques tutorial: it compares how the FULL circuit noise
of the Kunlun [[18,4,4]] code decomposes into its parts — full symmetric vs. measurement-only vs.
two-qubit(CX)-gate-only — using Technique II (min-weight onset), Technique I (failure-spectrum ansatz),
Technique III (replica-exchange splitting), and direct Monte-Carlo, all on the same code.

§5 adds the leave-one-out ABLATION models (all-but-one channel) — the marginal contributions Google's
Willow error budget is built from — and §6 assembles the Willow-style budget: isolated vs marginal
fractions of LER_full, the mixing bucket, per-channel pseudo-thresholds, and the Σ p/p_th terms.

§7 computes the TRUE Λ: the five channels rerun on the same-polynomial sibling BB_72_4_8 (d: 4→8,
rounds ∝ d, per-round ε), per-channel thresholds from the ε18=ε72 crossings, Λ(p) curves, the
per-(+2)-step λ = √Λ, and the Willow identity check 1/Λ vs Σ p/p_th.

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
| **gate idle** | `DEPOLARIZE1` on data during the one **CX layer** each data qubit sits out |
| **meas idle** | `DEPOLARIZE1` on data during the **ancilla measure+reset stage** |

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
`DEPOLARIZE1(p)` **twice per cycle**, an idle-error probability of `1−(1−p)² ≈ 2p` (not `3p`).
We treat the two slots as **separate channels** (`gate_idle` / `meas_idle`, split by schedule
position): on hardware the measure+reset dead time is much longer than a gate, so the two idles
have very different physical rates. The builder's single M/R-stage `DEPOLARIZE1` covers the
*combined* measure+reset dead time; reset is gate-duration-like by convention, so device-faithful
duration weighting (t_meas ≫ t_gate) belongs on the `meas_idle` rate.""")

# ---------------------------------------------------------------------------
md("""## Setup — the code, the noise filter, and the three models""")

code('''import numpy as np, matplotlib.pyplot as plt, stim
from bb_code_sim import (BBCodeParams, BBCodeSimulator, RelayBPDecoder,
                         NOISE_CHANNEL_PREDICATES, NOISE_INSTRUCTIONS, filter_noise_channel)
from surface_code_sim import ErrorModel
from min_weight import (dem_check_action_matrices, compute_distance, optimal_onset_fraction,
                        find_weight_logicals_mitm, find_all_min_weight_logicals)
from importance_sampling import importance_sample_adaptive, fit_failure_spectrum, logical_error_rate_from_ansatz
from splitting import replica_exchange_estimate

P = BBCodeParams(l=3, m=3, a_exps=[(1, 0), (0, 0), (0, 2)], b_exps=[(0, 1), (0, 0), (2, 0)], distance=4)
P_REF, ROUNDS = 0.01, 2

# Direct-MC budget knob: scales the §4/§5 MC shot counts. After the reweighting fix the MC points
# only VALIDATE the curves (the budget itself never reads them), so 0.15 (~seconds-minutes, wider
# error bars) is the iteration default; set 1.0 for full-fidelity anchors and the §4 overlay.
MC_SCALE = 0.15

# Channels are isolated by position-filtering ONE symmetric(p) circuit at the same base rate p.
# bb_code_sim.filter_noise_channel owns the position predicates (X_ERROR after R = prep, before
# M = meas; DEPOLARIZE1 not post-H = idle; DEPOLARIZE2 = two-qubit gate) — they encode the layout
# of build_bb_circuit, so they live in src next to it.
MODELS = {"full symmetric": None, "CZ only": "cz", "meas only": "meas",
          "prep only": "prep", "gate idle": "gate_idle", "meas idle": "meas_idle"}
COLORS = {"full symmetric": "crimson", "CZ only": "navy", "meas only": "seagreen",
          "prep only": "darkorange", "gate idle": "purple", "meas idle": "mediumvioletred"}
def make_circuit(model, p):
    return filter_noise_channel(BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(p), rounds=ROUNDS),
                                MODELS[model])

for name in MODELS:
    print(f"{name:16s}: {make_circuit(name, P_REF).detector_error_model().num_errors} DEM mechanisms")''')

# ===========================================================================
md(r"""## §0 — The syndrome-extraction schedule, up close

One cycle of the extraction circuit, layer by layer, for two data qubits (one per block) and one
ancilla of each type — derived from the built circuit itself, so it is exactly the layout the
noise-channel predicates key on. Inline `·channel` tags show how each noise instruction is
classified (`cz` / `meas` / `prep` / `gate_idle` / `meas_idle`): each data qubit is busy in six
of the seven CX layers, idles through the one it sits out (`·gate_idle`), and idles again while
the ancillas are measured and reset (`·meas_idle`). The second cell renders the full one-cycle
gate schedule as a stim `timeline-svg` diagram (noiseless build, all 36 qubits).""")

code('''watch = {0: "data q0 (L)", 9: "data q9 (R)", 18: "X-anc q18", 27: "Z-anc q27"}
c = make_circuit("full symmetric", P_REF)
insts = list(c.flattened())
CH = {k: v for k, v in NOISE_CHANNEL_PREDICATES.items() if k != "idle"}   # idle = union of the split
timeline = {q: [] for q in watch}
layer = 0
for i, inst in enumerate(insts):
    if inst.name == "TICK":
        layer += 1
        continue
    prev = insts[i - 1] if i > 0 else None
    nxt = insts[i + 1] if i + 1 < len(insts) else None
    if inst.name == "CX":
        t = [x.value for x in inst.targets_copy()]
        for a, b in zip(t[::2], t[1::2]):
            if a in watch: timeline[a].append((layer, f"CX→{b}"))
            if b in watch: timeline[b].append((layer, f"CX←{a}"))
    elif inst.name in ("H", "M", "R"):
        for x in inst.targets_copy():
            if x.value in watch: timeline[x.value].append((layer, inst.name))
    elif inst.name in NOISE_INSTRUCTIONS:
        ch = next((k for k, pred in CH.items() if pred(inst, prev, nxt)), None)
        tag = f"·{ch}" if ch else "·(1q gate)"          # post-H DEPOLARIZE1 = 1q-gate noise, no channel
        for x in inst.targets_copy():
            if x.value in watch: timeline[x.value].append((layer, tag))

n_show = 12                                             # ~one full cycle of layers
print(f"{'layer':>5} | " + " | ".join(f"{v:^24}" for v in watch.values()))
print("-" * (8 + 27 * len(watch)))
for L in range(n_show):
    row = [" ".join(s for l, s in timeline[q] if l == L) or "—" for q in watch]
    print(f"{L:>5} | " + " | ".join(f"{r:^24}" for r in row))''')

code('''# The same schedule as a stim timeline diagram — noiseless build, so it shows the pure gate
# structure (one cycle + final readout; 9 ticks, all 36 qubits). The annotated table above is
# the noise/channel legend; this is the geometry.
clean = BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(0.0), rounds=1)
clean.diagram("timeline-svg")''')

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

code('''# Grid top at 0.04 so the weak channels' break-even crossings (split idles ~0.012-0.02) land
# IN-grid instead of reporting a bound; the per-model weight windows auto-size from p_grid.max(),
# and the adaptive allocator makes the extra saturated-tail bins nearly free (~200 shots each).
p_grid = np.geomspace(1e-4, 0.04, 44)
# Adaptive 'hit N failures per weight' allocation: target 200 failures matches flat-6000's
# precision at the budget-critical onset bin (f(2)~0.03 -> ~160 failures) at ~4x fewer shots —
# the 200/f(w) schedule concentrates shots at the onset and skims the saturated high-f tail.
# The weight window is sized per model from the binomial mass at the TOP of the p grid
# (w_hi = μ + 4√μ, floored at 10), so the reweighted curves carry no truncation sag anywhere
# on the grid — the extra saturated-tail weights cost only ~200-400 shots each under adaptive.
def weight_window(c):
    mu_ref = sum(e.args_copy()[0] for e in c.detector_error_model().flattened() if e.type == "error")
    mu = mu_ref * p_grid.max() / P_REF
    return list(range(1, max(10, int(np.ceil(mu + 4 * np.sqrt(mu)))) + 1))

tech1 = {}
for name in MODELS:
    c = make_circuit(name, P_REF)
    W = weight_window(c)
    spec = importance_sample_adaptive(c, RelayBPDecoder(), p_ref=P_REF, p_values=[P_REF],
                                      weights=W, target_failures=200,
                                      shots_max=30_000, seed=1).spectrum
    fw = dict(zip(spec.weights, np.asarray(spec.failures) / np.asarray(spec.trials)))
    fit = fit_failure_spectrum(spec, K=c.num_observables, model="f5", w0=None, f0=None)
    LER = np.asarray(logical_error_rate_from_ansatz(fit, list(p_grid)))
    tech1[name] = dict(spec=spec, fw=fw, fit=fit, LER=LER)
    print(f"{name:16s}: w=1..{W[-1]}, measured f(2)={fw[2]:.4f} ({sum(spec.trials)} shots total)   "
          f"f5 fit cost={fit.cost:.2f}")''')

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

mc_pts = {p: int(s * MC_SCALE) for p, s in
          {0.03: 40_000, 0.012: 80_000, 0.008: 120_000, 0.005: 200_000, 0.003: 300_000}.items()}
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

code('''ABLATED = {"no CZ": "cz", "no meas": "meas", "no prep": "prep",
           "no gate idle": "gate_idle", "no meas idle": "meas_idle"}
def make_ablated_circuit(name, p):
    drop = NOISE_CHANNEL_PREDICATES[ABLATED[name]]
    return filter_noise_channel(BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(p), rounds=ROUNDS),
                                lambda i, pr, nx: not drop(i, pr, nx))

tech2_abl = {}
print(f"{'model':13s} {'D':>2} {'w0':>3} {'#DEM':>6} {'|L(D)|':>8} {'f0*':>10}   hook diagnosis")
for name in ABLATED:
    c = make_ablated_circuit(name, P_REF); D = compute_distance(c).distance
    LD = enumerate_LD(c, D)
    onset = optimal_onset_fraction(c, distance=D, logicals=LD)
    tech2_abl[name] = dict(D=D, LD=LD, onset=onset)
    diag = ("restores D=4 -> channel is in EVERY weight-3 hook" if D == 4
            else "still D=3 -> hooks survive without this channel" if D == 3 else "")
    print(f"{name:13s} {D:2d} {onset.onset:3d} {c.detector_error_model().num_errors:6d} "
          f"{onset.n_min_logicals:8d} {onset.onset_fraction:10.3e}   {diag}")''')

code('''tech1_abl, mc_abl = {}, {}
for name in ABLATED:
    c = make_ablated_circuit(name, P_REF)
    spec = importance_sample_adaptive(c, RelayBPDecoder(), p_ref=P_REF, p_values=[P_REF],
                                      weights=weight_window(c), target_failures=200,
                                      shots_max=30_000, seed=3).spectrum
    fit = fit_failure_spectrum(spec, K=c.num_observables, model="f5", w0=None, f0=None)
    tech1_abl[name] = dict(spec=spec, fit=fit,
                           LER=np.asarray(logical_error_rate_from_ansatz(fit, list(p_grid))))
    mc_abl[name] = {p: direct_mc(make_ablated_circuit(name, p), s) for p, s in mc_pts.items()}
    print(f"{name:13s}: f5 fit cost={fit.cost:.2f}   MC LER(p={min(mc_pts)}) = "
          f"{mc_abl[name][min(mc_pts)][0]:.3e}")''')

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

code('''# Budget operating point: DEEP in the suppression regime — below every channel's pseudo-threshold
# (the lowest, CZ, is ~1.5e-3, so 5e-4 sits ~3x under it and 12-25x under the rest). Direct MC is
# impractical down here (LERs ~1e-4..1e-6), so the cross-check column is Technique III SPLITTING
# (§3 reaches p=1e-4) instead of MC; §4 already validated ansatz-vs-MC at moderate p.
P_STAR = 5e-4

# Point values at p* come from the MEASURED spectra, binomially reweighted — NOT from the f5
# fits. The per-model weight windows cover every onset (w0=2, f(1)=0) AND the binomial mass over
# the whole grid, so the reweighting is exact-in-expectation everywhere; independently-fitted
# (w0, f0, shape) EXTRAPOLATIONS drift
# apart model-to-model down here and can even invert the full-vs-ablated ordering (negative
# "marginals"). The fits still supply the curve-wide pseudo-thresholds, where they are anchored.
from importance_sampling import reweight_spectrum

def ler_at(d, p):                    # reweighted measured spectrum at scalar p (no extrapolation)
    return float(reweight_spectrum(d["spec"], [p]).P_logical[0])

def split_at(name, p):               # Technique-III splitting estimate at p (log-log interpolated)
    sp, sP = tech3[name]["sp"], tech3[name]["sP"]
    return float(np.exp(np.interp(np.log(p), np.log(sp), np.log(sP))))

def crossing_p(pg, y1, y2):          # p where y1(p)=y2(p), log-log interpolated (None if never)
    r = np.log(y1) - np.log(y2)
    s = np.nonzero(np.diff(np.sign(r)) != 0)[0]
    if s.size == 0:
        return None
    i = s[-1]; t = r[i] / (r[i] - r[i + 1])
    return float(np.exp(np.log(pg[i]) + t * (np.log(pg[i + 1]) - np.log(pg[i]))))

def pseudo_threshold(pg, LER):       # break-even LER(p)=p (single-code threshold stand-in)
    return crossing_p(pg, LER, np.asarray(pg))

CHANNELS = ["CZ only", "meas only", "prep only", "gate idle", "meas idle"]
ABL_OF = {"CZ only": "no CZ", "meas only": "no meas", "prep only": "no prep",
          "gate idle": "no gate idle", "meas idle": "no meas idle"}

L_full = ler_at(tech1["full symmetric"], P_STAR)
S_full = split_at("full symmetric", P_STAR)
rows = []
for ch in CHANNELS:
    iso = ler_at(tech1[ch], P_STAR) / L_full
    marg = 1.0 - ler_at(tech1_abl[ABL_OF[ch]], P_STAR) / L_full   # reweighted spectra: no fit drift
    iso_split = split_at(ch, P_STAR) / S_full
    pth = pseudo_threshold(p_grid, tech1[ch]["LER"])
    rows.append((ch, iso, marg, iso_split, pth))

print(f"error budget at p* = {P_STAR}   (LER_full: reweighted {L_full:.3e}, splitting {S_full:.3e})")
print(f"{'channel':12s} {'isolated':>9} {'iso(split)':>10} {'marginal':>9} {'p_pth':>9} {'p*/p_pth':>9}")
for ch, iso, marg, iso_split, pth in rows:
    pths = f"{pth:.4f}" if pth else f">{p_grid.max():.3g}"      # no break-even crossing in-grid
    term = f"{P_STAR/pth:.3f}" if pth else f"<{P_STAR/p_grid.max():.3f}"   # bound, not a blank
    print(f"{ch:12s} {iso:9.3f} {iso_split:10.3f} {marg:9.3f} {pths:>9} {term:>9}")
mixing = 1.0 - sum(r[1] for r in rows)
spam_iso = sum(r[1] for r in rows if r[0] in ("meas only", "prep only"))
spam_marg = sum(r[2] for r in rows if r[0] in ("meas only", "prep only"))
idle_iso = sum(r[1] for r in rows if "idle" in r[0])
idle_marg = sum(r[2] for r in rows if "idle" in r[0])
print(f"{'mixing':12s} {mixing:9.3f}                      (1 - sum isolated: cross-channel faults)")
print(f"{'SPAM':12s} {spam_iso:9.3f} {'':10s} {spam_marg:9.3f}      (meas + prep combined)")
print(f"{'idle total':12s} {idle_iso:9.3f} {'':10s} {idle_marg:9.3f}      (gate + meas idle combined)")
print(f"sum(marginal) = {sum(r[2] for r in rows):.3f}  (>1 <=> shared mixed faults; Google renormalizes)")
print(f"Willow-form sum: {sum(P_STAR / r[4] for r in rows if r[4]):.3f} = "
      + " + ".join(f"{P_STAR/r[4]:.3f} ({r[0].replace(' only', '')})" for r in rows if r[4])
      + ("   [all terms < 1: genuinely sub-threshold]"
         if all(P_STAR / r[4] < 1 for r in rows if r[4])
         else "   [WARNING: a term >= 1 — p* is not below every pseudo-threshold]"))''')

code('''fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
# left: marginal budget fractions vs p — REWEIGHTED measured spectra (solid), the same estimator
# as the table/bars at p*, so the panels agree. The f5-fit version is kept as faint dashed lines
# (unclipped) to make the low-p extrapolation drift visible instead of hiding it. The per-model
# weight windows cover the binomial mass over the whole grid, so there is no truncation sag.
Lf_rw = reweight_spectrum(tech1["full symmetric"]["spec"], p_grid).P_logical
Lf_fit = tech1["full symmetric"]["LER"]
for ch in CHANNELS:
    frac_rw = 1.0 - reweight_spectrum(tech1_abl[ABL_OF[ch]]["spec"], p_grid).P_logical / Lf_rw
    axL.plot(p_grid, frac_rw, "-", color=COLORS[ch], lw=2, label=ch)
    axL.plot(p_grid, 1.0 - tech1_abl[ABL_OF[ch]]["LER"] / Lf_fit, "--", color=COLORS[ch], lw=1, alpha=0.4)
axL.plot([], [], "--", color="gray", lw=1, alpha=0.6, label="f5-fit version (drifts at low p)")
axL.axhline(0.0, color="gray", lw=0.8)
axL.set_xscale("log"); axL.set_xlabel("physical error rate p")
axL.set_ylabel("marginal fraction  1 − LER$_{no\\,i}$/LER$_{full}$")
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

# ===========================================================================
md(r"""## §7 — The true Λ: five channels on the [[72,4,8]] sibling

Everything so far used *pseudo*-thresholds because a real threshold is a crossing between code
sizes. The Kunlun polynomials give a distance-scaled partner: **[[72,4,8]]** at `(l,m)=(6,6)` —
same `A = 1+x+y²`, `B = 1+y+x²`, `k=4`, code distance **8 exact** (both sectors: `w≤7` eliminated by
complete split-MITM, weight-8 logicals exhibited). `d: 4 → 8` is two `+2` steps, the K=4 analog of
Google's `d=3,5,7` ladder.

**Conventions.** Rounds scale with distance (`d/2`: 2 rounds for d=4, 4 for d=8), and Λ compares
**per-round** logical error rates `ε(p) = 1 − (1−LER)^{1/rounds}`: `Λ_i(p) = ε₁₈,ᵢ(p)/ε₇₂,ᵢ(p)`.
The channel crossings `ε₁₈,ᵢ = ε₇₂,ᵢ` are the *true* per-channel thresholds `p_th,i` that the
Willow budget divides by. No exact `f₀*` at this size (the `L(D)`/`L(D+1)` enumerations are the
bb144-regime problem) — the Λ budget doesn't need it.

**Cost.** This section reruns the five channels on a 144-qubit circuit — expect **one to two
hours**; each cell checkpoints nothing, so run it when the kernel can sit.""")

code('''from bb_code_sim import BB_72_4_8

ROUNDS72 = BB_72_4_8.distance // 2       # rounds ∝ distance (the d=4 run above used d/2 = 2)
def make_circuit72(model, p):
    return filter_noise_channel(BBCodeSimulator(BB_72_4_8).build_circuit(ErrorModel.symmetric(p), rounds=ROUNDS72),
                                MODELS[model])

tech2_72 = {}
print(f"{'model':16s} {'D':>2}   (circuit fault distance, BP-OSD upper bound; f0 unpinned at this size)")
for name in MODELS:
    tech2_72[name] = compute_distance(make_circuit72(name, P_REF)).distance
    print(f"{name:16s} {tech2_72[name]:2d}")''')

code('''tech1_72, mc72 = {}, {}
mc72_pts = {0.008: 10_000}               # one slim anchor: MC only decorates the §7.4 plot now
for name in MODELS:
    c = make_circuit72(name, P_REF)
    # Weight window from the binomial mass: μ(p) = E[#faults] = Σ DEM error probs, rescaled to p.
    mu_ref = sum(e.args_copy()[0] for e in c.detector_error_model().flattened() if e.type == "error")
    w_hi = int(np.ceil((mu := mu_ref * p_grid.max() / P_REF) + 4 * np.sqrt(mu)))
    W = list(range(1, 16)) + list(range(16, w_hi + 1, 2))   # stride-2 tail: f5 pools weights, halves the cost
    # Adaptive allocation skims the saturated tail and pours shots into the near-onset bins that
    # decide the reweighted Λ(p*). stop_after_zero_bins caps the DEEP sub-onset cost: below the
    # observable onset every bin would otherwise burn the full shots_max chasing failures that
    # cannot be resolved at this budget anyway (they need ~1e6+ shots) — three consecutive empty
    # max-budget bins end the descent. Zero/skipped bins contribute exactly 0 to the reweighting
    # either way, so ε72 at very low p is a LOWER bound (Λ an upper bound) — compare Λ vs Λ(fit).
    spec = importance_sample_adaptive(c, RelayBPDecoder(), p_ref=P_REF, p_values=[P_REF],
                                      weights=W, target_failures=100, shots_max=10_000,
                                      stop_after_zero_bins=3, seed=4).spectrum
    fit = fit_failure_spectrum(spec, K=c.num_observables, model="f5", w0=None, f0=None)
    tech1_72[name] = dict(spec=spec, fit=fit,
                          LER=np.asarray(logical_error_rate_from_ansatz(fit, list(p_grid))))
    mc72[name] = {p: direct_mc(make_circuit72(name, p), s) for p, s in mc72_pts.items()}
    print(f"{name:16s}: weights 1..{w_hi} ({len(W)} sampled, {sum(spec.trials)} shots), "
          f"f5 fit cost={fit.cost:.2f}, MC LER(0.008)={mc72[name][0.008][0]:.3e}", flush=True)''')

code('''def per_round(LER, rounds):
    return 1.0 - (1.0 - np.clip(LER, 0.0, 1.0 - 1e-12)) ** (1.0 / rounds)

eps18 = {m: per_round(tech1[m]["LER"], ROUNDS) for m in MODELS}
eps72 = {m: per_round(tech1_72[m]["LER"], ROUNDS72) for m in MODELS}

# Λ evaluation point — matches the §6 budget's p*, deep in the suppression regime (well below every
# crossing). Point-Λ uses the REWEIGHTED measured spectra (both codes sampled from w=1 through their
# onsets), avoiding cross-fit extrapolation drift; the f5-based value is printed as a cross-check.
# Honesty note: if the [[72,4,8]] onset bins saw ZERO failures at these shots, the reweighted ε72 is
# a lower bound → Λ an upper bound (a large reweighted-vs-fit gap flags exactly this).
P_LAM = 5e-4
from importance_sampling import reweight_spectrum

def eps_at(d, p, rounds):            # per-round ε from the reweighted measured spectrum
    return float(per_round(reweight_spectrum(d["spec"], [p]).P_logical, rounds)[0])

i_lam = int(np.argmin(np.abs(p_grid - P_LAM)))
rows7 = []
print(f"{'channel':16s} {'p_th (ε18=ε72)':>15} {'Λ(p*)':>9} {'Λ(fit)':>9} {'p*/p_th':>9}     (p* = {P_LAM})")
for m in MODELS:
    pth = crossing_p(p_grid, eps18[m], eps72[m])
    lam = eps_at(tech1[m], P_LAM, ROUNDS) / eps_at(tech1_72[m], P_LAM, ROUNDS72)
    lam_fit = float(eps18[m][i_lam] / eps72[m][i_lam])
    rows7.append((m, pth, lam))
    pths = f"{pth:.4f}" if pth else f">{p_grid.max():.3g}"
    term = f"{P_LAM/pth:.3f}" if pth else f"<{P_LAM/p_grid.max():.3f}"
    print(f"{m:16s} {pths:>15} {lam:9.3g} {lam_fit:9.3g} {term:>9}")

lam_full = dict((m, l) for m, _, l in rows7)["full symmetric"]
print(f"\\nΛ_full(p*) = {lam_full:.3g}  →  per-(+2-distance)-step λ = √Λ = {np.sqrt(lam_full):.3g}  (d: 4→8 is two steps)")
inv_lam = 1.0 / lam_full
terms = {m: P_LAM / pth for m, pth, _ in rows7 if pth is not None and m != "full symmetric"}
spam_term = terms.get("meas only", 0) + terms.get("prep only", 0)
print(f"Willow identity check at p*: 1/Λ_full = {inv_lam:.3f}  vs  Σᵢ p*/p_th,i = {sum(terms.values()):.3f}"
      f"   (CZ {terms.get('CZ only', float('nan')):.3f}, SPAM {spam_term:.3f}, idle {terms.get('idle only', float('nan')):.3f})")
print("residual = mixed-channel faults (isolated channels cannot see them — see §1/§5); "
      "Σ < 1/Λ_full means the additive budget under-covers by that share.")''')

code('''fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
for m in MODELS:
    col = COLORS[m]
    axL.plot(p_grid, eps18[m], "-", color=col, lw=2, label=m)
    axL.plot(p_grid, eps72[m], "--", color=col, lw=1.5)
    mp = sorted(mc72[m])
    axL.plot(mp, [per_round(np.asarray([mc72[m][p][0]]), ROUNDS72)[0] for p in mp], "^", color=col, ms=6)
    axR.plot(p_grid, eps18[m] / eps72[m], "-", color=col, lw=2, label=m)
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel("physical error rate p"); axL.set_ylabel("per-round logical error rate ε")
axL.set_title("[[18,4,4]] (solid, 2 rounds) vs [[72,4,8]] (dashed, 4 rounds; ▲=MC)")
axL.legend(fontsize=8); axL.grid(alpha=0.3, which="both")
axR.set_xscale("log"); axR.set_yscale("log")
axR.axhline(1.0, color="gray", lw=1); axR.axvline(P_LAM, color="gray", ls=":", lw=1)
axR.set_xlabel("physical error rate p"); axR.set_ylabel(r"$\\Lambda(p) = \\varepsilon_{18}/\\varepsilon_{72}$")
axR.set_title("error suppression per channel (crossings = true $p_{th,i}$)")
axR.legend(fontsize=8); axR.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()''')

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
* **Pseudo-thresholds (§6) vs true thresholds (§7).** §6's break-even points need only one code; §7
  replaces them with the real thing — the `ε₁₈,ᵢ(p) = ε₇₂,ᵢ(p)` crossings against the same-polynomial
  sibling **[[72,4,8]]** — and reads off `Λ_i(p)` directly. With `d: 4→8` (two `+2` steps) the
  per-step suppression is `λ = √Λ_full`.
* **The Willow identity `1/Λ ≈ Σᵢ p/p_th,i` is the §7 punchline**: how far the sum falls short of
  `1/Λ_full` measures exactly how much the mixed-channel faults (the §1 `D=3` hook, invisible to
  every isolated channel) break the additive budget.

*Generated by `make_error_model_comparison.py`.*""")

# ===========================================================================
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "methods" / "error_model_comparison_18_4_4.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
