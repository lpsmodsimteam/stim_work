"""Generate error_model_comparison_18_4_4.ipynb — a REPORT over cached runner results.

The notebook no longer runs any simulation. All sampling lives in
``run_error_model_comparison.py``, which caches one JSON per task under
``runs/error_model_comparison_18_4_4/`` (rerunning only what is missing or whose config
changed, and recording per-task wall time). The notebook generated here LOADS those files
and renders the tables and plots: the only computation it does is cheap analysis —
binomial reweighting of the stored spectra, crossings, and error propagation — so it
re-executes in seconds and needs neither stim nor a decoder.

Contents (unchanged story): §0 schedule, §1 Technique II per channel, §2 Technique I
spectra, §3 splitting, §4 direct-MC overlay, §5 leave-one-out ablations, §6 the
Willow-style budget, §7 the true Λ against [[72,4,8]], §7.5 marginal Λ, §8 the
asymmetric operating point. New in the report: a per-section runner-time table, and the
Λ-share boxes (§7.5, §8) carry propagated standard errors plus a zero-failure-bin bound —
a NEGATIVE share is now printed with the evidence for whether it is real or an estimator
artifact of the under-resolved 72-code onset bins.
"""
import json
from repo_paths import REPO_ROOT

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

# ===========================================================================
md(r"""# Error-model experiments on the Kunlun **[[18,4,4]]** — report

How does the full circuit noise decompose into its parts? Using the three fail-fast techniques (from
`three_techniques_18_4_4.ipynb`) **together with direct Monte-Carlo**, we compare the error models on
the *same* [[18,4,4]] code:

| model | noise kept |
|---|---|
| **full symmetric** | everything — `DEPOLARIZE1` + `DEPOLARIZE2` + `X_ERROR` |
| **CZ only** | `DEPOLARIZE2` (two-qubit `CX` gate depolarizing) |
| **meas only** | `X_ERROR` immediately **before** a measurement `M` |
| **prep only** | `X_ERROR` immediately **after** a reset `R` (state preparation) |
| **gate idle** | `DEPOLARIZE1` on data during the one **CX layer** each data qubit sits out |
| **meas idle** | `DEPOLARIZE1` on data during the **ancilla measure+reset stage** |

Channels are isolated by **filtering noise instructions** on one `symmetric(p)` circuit — every variant
uses the *identical* per-location rate `p` — via `bb_code_sim.filter_noise_channel` (predicates live
next to the circuit builder whose layout they encode). `X_ERROR` splits by position into *prep* (after
`R`) and *meas* (before `M`); `DEPOLARIZE1` is *idle* noise except right after an `H` (the 1q-gate
error), split into `gate_idle` / `meas_idle` by schedule position.

**Idle occupancy.** In the depth-7 schedule each data qubit is idle in **2 of the 8 rounds** per
syndrome cycle — round 7 (all data idle while ancillas are measured/reset) plus exactly one of rounds
0/6. So a data qubit picks up idle `DEPOLARIZE1(p)` **twice per cycle** (`1−(1−p)² ≈ 2p`, not `3p`).
The two slots are **separate channels**: on hardware the measure+reset dead time is much longer than a
gate, so device-faithful duration weighting (t_meas ≫ t_gate) belongs on the `meas_idle` rate.

**This notebook is a report.** Every simulation result is loaded from
`runs/error_model_comparison_18_4_4/`, produced (and cached task-by-task) by
`experiments/methods/run_error_model_comparison.py`. To refresh the data, run that script — it
recomputes only tasks whose cache is missing or whose configuration changed — then re-execute this
notebook (seconds; it needs neither stim nor a decoder).""")

# ---------------------------------------------------------------------------
md("""## Setup — load the cached runner results""")

code(r'''import json, numpy as np, matplotlib.pyplot as plt
from IPython.display import SVG, display
from repo_paths import run_dir
from importance_sampling import FailureSpectrum, reweight_spectrum

RESULTS = run_dir("error_model_comparison_18_4_4")

def load_meta(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
def load(name):
    return load_meta(name)["result"]
def slug(name):
    return name.replace(" ", "_")
def spectrum_of(r):                       # rebuild the measured spectrum from its JSON fields
    return FailureSpectrum(**r["spectrum"])

MAN = load("config__manifest")
CFG = load_meta("config__manifest")["config"]
P_REF, ROUNDS, ROUNDS72 = CFG["p_ref"], CFG["rounds"], CFG["rounds72"]
p_grid = np.geomspace(CFG["p_grid"]["lo"], CFG["p_grid"]["hi"], CFG["p_grid"]["n"])
MODELS, ABLATED, CHANNELS = MAN["models"], MAN["ablated"], MAN["channels"]
ABL_OF = dict(zip(CHANNELS, ABLATED))     # "CZ only" -> "no CZ", ... (matching runner order)
P_STAR, P_LAM, R_OF = MAN["p_star"], MAN["p_lam"], MAN["r_of"]
COLORS = {"full symmetric": "crimson", "CZ only": "navy", "meas only": "seagreen",
          "prep only": "darkorange", "gate idle": "purple", "meas idle": "mediumvioletred"}

# --- runner timing: every task file records its wall time -------------------------------
TIMES = {f.stem: {k: json.loads(f.read_text(encoding="utf-8"))[k] for k in ("elapsed_s", "finished_at")}
         for f in sorted(RESULTS.glob("*.json"))}
def section_time(*groups):
    ms = [v for k, v in TIMES.items() if k.split("__")[0] in groups]
    if not ms:
        return "⏱ runner: no cached tasks found for this section"
    tot = sum(m["elapsed_s"] for m in ms)
    return f"⏱ runner: {tot:,.0f} s wall over {len(ms)} tasks (newest {max(m['finished_at'] for m in ms)})"

# --- estimators shared by the sections ---------------------------------------------------
def per_round(LER, rounds):
    return 1.0 - (1.0 - np.clip(LER, 0.0, 1.0 - 1e-12)) ** (1.0 / rounds)

def rw_stats(spec, p):
    """Reweighted-measured-spectrum LER at scalar p: (value, statistical SE, zero-bin headroom).

    SE propagates the per-bin binomial errors of the sampled f(w). `headroom` is how much the
    value could RISE if every sampled-but-zero-failure bin actually sat at its rule-of-three
    upper bound f(w) < 3/T(w). EVERY zero bin is priced, including those below tech2's D:
    D is the PERFECT-decoder floor, but the actual decoder miscorrects (returns a wrong-coset
    correction heavier than the fault) at rates the spectra measure directly — e.g. f(1) > 0 on
    the full 18-code mix — so an empty low-w bin is unresolved statistics, not a structural zero.
    """
    v = reweight_spectrum(spec, [p])
    up = FailureSpectrum(weights=spec.weights, trials=spec.trials,
                         failures=[f if f > 0 else min(3, t) for f, t in zip(spec.failures, spec.trials)],
                         n_expanded=spec.n_expanded, q_base=spec.q_base, p_ref=spec.p_ref)
    head = float(reweight_spectrum(up, [p]).P_logical[0]) - float(v.P_logical[0])
    return float(v.P_logical[0]), float(v.P_logical_se[0]), head

def eps_stats(spec, p, rounds):
    """Per-round ε at p with SE and zero-bin headroom (delta method through 1-(1-L)^{1/r})."""
    L, se, head = rw_stats(spec, p)
    g = (1.0 - min(L, 1.0 - 1e-12)) ** (1.0 / rounds - 1.0) / rounds
    return float(per_round(np.asarray([L]), rounds)[0]), g * se, g * head

def inv_lambda_stats(spec18, spec72, p):
    """1/Λ = ε72/ε18 at p: (value, SE, low, high) — low/high span the zero-bin truncation."""
    e18, s18, h18 = eps_stats(spec18, p, ROUNDS)
    e72, s72, h72 = eps_stats(spec72, p, ROUNDS72)
    inv = e72 / e18
    se = inv * float(np.hypot(s18 / e18, s72 / e72))
    return inv, se, e72 / (e18 + h18), (e72 + h72) / e18

def crossing_p(pg, y1, y2):          # p where y1(p)=y2(p), log-log interpolated (None if never)
    r = np.log(y1) - np.log(y2)
    s = np.nonzero(np.diff(np.sign(r)) != 0)[0]
    if s.size == 0:
        return None
    i = s[-1]; t = r[i] / (r[i] - r[i + 1])
    return float(np.exp(np.log(pg[i]) + t * (np.log(pg[i + 1]) - np.log(pg[i]))))

def pseudo_threshold(pg, LER):       # break-even LER(p)=p (single-code threshold stand-in)
    return crossing_p(pg, LER, np.asarray(pg))

dem = load("setup__dem_counts")
for name in MODELS:
    print(f"{name:16s}: {dem[name]} DEM mechanisms")''')

code(r'''# How long each section's simulations took in the runner (cached wall time, sequential).
SECTIONS = [("setup + §0 schedule",      ["setup", "schedule"]),
            ("§1 distances/onsets",      ["tech2"]),
            ("§2 spectra (Technique I)", ["tech1"]),
            ("§3 splitting",             ["tech3"]),
            ("§4 direct MC",             ["mc"]),
            ("§5 ablations (18)",        ["tech2_abl", "tech1_abl", "mc_abl"]),
            ("§7 [[72,4,8]] sweeps",     ["tech2_72", "tech1_72", "mc72"]),
            ("§7.5 72-code ablations",   ["tech1_72_abl"]),
            ("§8 asymmetric point",      ["asym"])]
print(f"{'section':28s} {'tasks':>5} {'wall time':>12}   newest result")
grand = 0.0
for label, groups in SECTIONS:
    ms = [v for k, v in TIMES.items() if k.split("__")[0] in groups]
    tot = sum(m["elapsed_s"] for m in ms); grand += tot
    newest = max((m["finished_at"] for m in ms), default="—")
    print(f"{label:28s} {len(ms):5d} {tot:11,.1f}s   {newest}")
print(f"{'TOTAL':28s} {'':5s} {grand:11,.1f}s   (§6 is pure analysis — no runner tasks)")''')

# ===========================================================================
md(r"""## §0 — The syndrome-extraction schedule, up close

One cycle of the extraction circuit, layer by layer, for two data qubits (one per block) and one
ancilla of each type — derived from the built circuit itself (by the runner), so it is exactly the
layout the noise-channel predicates key on. Inline `·channel` tags show how each noise instruction is
classified (`cz` / `meas` / `prep` / `gate_idle` / `meas_idle`): each data qubit is busy in six of
the seven CX layers, idles through the one it sits out (`·gate_idle`), and idles again while the
ancillas are measured and reset (`·meas_idle`). The second cell renders the NOISY one-cycle schedule
as a stim `timeline-svg` diagram, sliced to a closed 7-qubit star — one data qubit plus its six check
ancillas, gates kept only when both endpoints are inside — so every rail shown is fully involved and
the labelled noise boxes are readable.""")

code(r'''print(load("schedule__table")["table"])
print()
print(section_time("setup", "schedule"))''')

code(r'''# Every noise box is tagged and colored by CHANNEL (stim itself labels noise only with its
# probability — all three DEPOLARIZE1 channels would look identical): cz / meas / prep /
# g-idle / m-idle / 1q (post-H 1q-gate noise, not a budget channel). All boxes are p=0.01.
# TWO syndrome cycles are shown because the data m-idle slot only exists BETWEEN rounds; the
# g-idle box is the lone DEP box on the data rail during its sat-out CX layer, right after
# prep. Caveats: ancilla rails show only their coupling to the watched data qubit (other CXs
# are cropped), and rails are renumbered 0..6 — the mapping back is printed here.
star = load("schedule__star_svg")
print("rails: " + "   ".join(f"q{new} = {lbl}" for new, old, lbl in star["rails"]))
print("noise tags: " + "   ".join(f"{t} = {c}" for t, c in star["channel_colors"].items()))
display(SVG(star["svg"]))''')

# ===========================================================================
md(r"""## §1 — Technique II: distance, onset, perfect-decoder floor (per model)

For each model: circuit fault distance `D`, onset weight `w₀=⌈D/2⌉`, the exact `L(D)`, and the
perfect-decoder onset fraction `f₀*`. The four *isolated* channels (CZ / meas / prep / idle) each turn
out to have **even** distance 4 (the code distance) — so `f₀*` is exact via Proposition 1 — while the **full**
model has **odd** distance 3: only *combining* channels makes the weight-3 hook that drops it below the
code distance (Appendix A.6 route for `f₀*`). `L(D)` is enumerated with the ldpc-free half-MITM for even
`D` (robust) and the coset search for odd `D`.""")

code(r'''tech2 = {name: load(f"tech2__{slug(name)}") for name in MODELS}
print(f"{'model':16s} {'D':>2} {'w0':>3} {'#DEM':>6} {'|L(D)|':>8} {'f0*':>8}   route")
for name in MODELS:
    t = tech2[name]
    print(f"{name:16s} {t['D']:2d} {t['w0']:3d} {t['n_dem']:6d} {t['n_LD']:8d} {t['f0']:8.4f}   {t['route']}")
print()
print(section_time("tech2"))''')

# ===========================================================================
md(r"""## §2 — Technique I: failure-spectrum ansatz (per model)

The runner importance-samples the failure spectrum `f(w)` (adaptive 'hit N failures per weight'
allocation; the weight window is sized per model from the binomial mass at the top of the `p` grid,
so the reweighted curves carry no truncation sag anywhere on the grid) and fits the f5 ansatz (onset
`w₀` left free). The report loads both: the measured spectrum drives every point value below (via
binomial reweighting), the fit supplies the smooth `LER(p)` curves.

**Decoder convention (device calibration).** Every estimator in this report shares ONE decoder
(Relay-BP, `num_sets=20`) whose priors are frozen from the task's noise model built at the
**evaluation point** `p* = 5×10⁻⁴` (`DECODER_P` in the runner) — not at the sampling reference
`p_ref = 0.01`. This is the device convention: a real decoder is calibrated to device rates.
It matters because wrong-coset *pair* explanations scale ∼p² against a true single fault's ∼p —
at p_ref-priors, meas-pair explanations of the ×5 mix's weight-1 CZ hooks were up to 7.5× likelier
than the truth (Bayes-rational misdecodes); calibrated at `p*` the same decoder catches every
single fault (breakeven `p ≈ 1.3×10⁻³`). `f(w)` remains rate-independent because the decoder is
*fixed* — but curves above breakeven (the direct-MC anchors, the threshold crossings) describe a
decoder deliberately miscalibrated for that regime, so expect slightly elevated MC points and
slightly lower pseudo-thresholds than a per-point-calibrated decoder would give.""")

code(r'''tech1 = {}
for name in MODELS:
    r = load(f"tech1__{slug(name)}")
    tech1[name] = dict(spec=spectrum_of(r), LER=np.asarray(r["LER_fit"]), cost=r["fit"]["cost"], W=r["W"])
    print(f"{name:16s}: w=1..{r['W'][-1]}, measured f(2)={tech1[name]['spec'].f(2):.4f} "
          f"({r['shots']} shots total)   f5 fit cost={r['fit']['cost']:.2f}")
print()
print(section_time("tech1"))''')

# ===========================================================================
md(r"""## §3 — Technique III: replica-exchange splitting (per model)

Splitting reaches deep into the rare regime, seeded by each model's exact `L(D)` from §1.""")

code(r'''tech3 = {}
for name in MODELS:
    r = load(f"tech3__{slug(name)}")
    tech3[name] = dict(sp=np.asarray(r["sp"]), sP=np.asarray(r["sP"]))
    print(f"{name:16s}: swap-accept {r['swap_min']:.2f}..{r['swap_max']:.2f}   P(1e-4)={r['sP'][0]:.2e}")
print()
print(section_time("tech3"))''')

# ===========================================================================
md(r"""## §4 — Direct Monte-Carlo + overlay

Direct-MC ground truth at moderate `p`, overlaid with all three techniques. Per model the ansatz
line, the splitting squares, and the MC circles should coincide where they overlap. (MC only
*validates* the curves — the budget never reads it — so the runner's `MC_SCALE` trades error-bar
width for minutes.)""")

code(r'''mc = {name: {float(p): v for p, v in load(f"mc__{slug(name)}")["points"].items()} for name in MODELS}
mc_pts = sorted(next(iter(mc.values())))
print("   p        " + "".join(f"{n:>15}" for n in MODELS))
for p in sorted(mc_pts, reverse=True):
    print(f"  {p:.3f}   " + "".join(f"{mc[n][p][0]:>15.3e}" for n in MODELS))
print()
print(section_time("mc"))

fig, ax = plt.subplots(figsize=(9, 6))
for name in MODELS:
    col = COLORS[name]
    ax.plot(p_grid, tech1[name]["LER"], "-", color=col, lw=2, label=f"{name}")
    ax.plot(tech3[name]["sp"], tech3[name]["sP"], "s", color=col, ms=4, mfc="none")
    ax.errorbar(mc_pts, [mc[name][p][0] for p in mc_pts], yerr=[mc[name][p][1] for p in mc_pts],
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
**mixed-channel** faults that no isolated circuit contains. The runner puts the five **all-but-one**
models (`keep = not channel`) through the same pipeline.

The ablated distances are a structural diagnostic: if dropping channel *i* restores `D=4`, then
channel *i* participates in **every** weight-3 hook; if `D` stays 3, the hooks survive without it.""")

code(r'''tech2_abl = {name: load(f"tech2_abl__{slug(name)}") for name in ABLATED}
print(f"{'model':13s} {'D':>2} {'w0':>3} {'#DEM':>6} {'|L(D)|':>8} {'f0*':>10}   hook diagnosis")
for name in ABLATED:
    t = tech2_abl[name]
    diag = ("restores D=4 -> channel is in EVERY weight-3 hook" if t["D"] == 4
            else "still D=3 -> hooks survive without this channel" if t["D"] == 3 else "")
    print(f"{name:13s} {t['D']:2d} {t['w0']:3d} {t['n_dem']:6d} {t['n_LD']:8d} {t['f0']:10.3e}   {diag}")''')

code(r'''tech1_abl, mc_abl = {}, {}
for name in ABLATED:
    r = load(f"tech1_abl__{slug(name)}")
    tech1_abl[name] = dict(spec=spectrum_of(r), LER=np.asarray(r["LER_fit"]), cost=r["fit"]["cost"])
    mc_abl[name] = {float(p): v for p, v in load(f"mc_abl__{slug(name)}")["points"].items()}
    print(f"{name:13s}: f5 fit cost={r['fit']['cost']:.2f}   MC LER(p={min(mc_abl[name])}) = "
          f"{mc_abl[name][min(mc_abl[name])][0]:.3e}")
print()
print(section_time("tech2_abl", "tech1_abl", "mc_abl"))''')

# ===========================================================================
md(r"""## §6 — The error budget (Willow-style)

Willow's budget reads `1/Λ ≈ Σᵢ pᵢ/p_th,i` — each channel priced against its own threshold. Two
honesty notes for a **single** code:

* There is no Λ (it is a ratio between code *sizes*); the single-code stand-in for `p_th,i` is the
  channel **pseudo-threshold** — the break-even `p` where `LER_i(p) = p`. A *true* Λ needs the
  same-polynomial `(l,m)=(6,6)` sibling **[[72,4,8]]** (`A=1+x+y²`, `B=1+y+x²`, exact d=8) — §7.
* Two decompositions of `LER_full` are shown, and they differ by construction:
  **isolated** `LER_i/LER_full` (misses mixed faults; the deficit is the **mixing bucket**) and
  **marginal** `1 − LER_{no i}/LER_full` (Google's convention; Σ typically **exceeds 1** because a
  mixed fault is killed by removing *any* of its participant channels, so it is counted in each).
  SPAM = meas + prep (their linear budget terms add).

Point values at `p*` come from the MEASURED spectra, binomially reweighted — NOT from the f5 fits
(independently-fitted extrapolations drift apart model-to-model down here and can even invert the
full-vs-ablated ordering into negative "marginals"). The fits still supply the curve-wide
pseudo-thresholds, where they are anchored. The marginal column carries the propagated binomial
`±σ`. Cross-check column: Technique III splitting (§3 reaches p=1e-4; direct MC is impractical at
LERs of 1e-4..1e-6).""")

code(r'''def ler_at(d, p):                    # reweighted measured spectrum at scalar p (no extrapolation)
    return float(reweight_spectrum(d["spec"], [p]).P_logical[0])

def split_at(name, p):               # Technique-III splitting estimate at p (log-log interpolated)
    sp, sP = tech3[name]["sp"], tech3[name]["sP"]
    return float(np.exp(np.interp(np.log(p), np.log(sp), np.log(sP))))

L_full, L_full_se, _ = rw_stats(tech1["full symmetric"]["spec"], P_STAR)
S_full = split_at("full symmetric", P_STAR)
rows = []
for ch in CHANNELS:
    iso = ler_at(tech1[ch], P_STAR) / L_full
    La, La_se, _ = rw_stats(tech1_abl[ABL_OF[ch]]["spec"], P_STAR)
    marg = 1.0 - La / L_full
    marg_se = (La / L_full) * float(np.hypot(La_se / La, L_full_se / L_full))
    iso_split = split_at(ch, P_STAR) / S_full
    pth = pseudo_threshold(p_grid, tech1[ch]["LER"])
    rows.append((ch, iso, marg, iso_split, pth, marg_se))

print(f"error budget at p* = {P_STAR}   (LER_full: reweighted {L_full:.3e}, splitting {S_full:.3e})")
print(f"{'channel':12s} {'isolated':>9} {'iso(split)':>10} {'marginal':>9} {'±σ':>6} {'p_pth':>9} {'p*/p_pth':>9}")
for ch, iso, marg, iso_split, pth, mse in rows:
    pths = f"{pth:.4f}" if pth else f">{p_grid.max():.3g}"      # no break-even crossing in-grid
    term = f"{P_STAR/pth:.3f}" if pth else f"<{P_STAR/p_grid.max():.3f}"   # bound, not a blank
    print(f"{ch:12s} {iso:9.3f} {iso_split:10.3f} {marg:9.3f} {mse:6.3f} {pths:>9} {term:>9}")
mixing = 1.0 - sum(r[1] for r in rows)
spam_iso = sum(r[1] for r in rows if r[0] in ("meas only", "prep only"))
spam_marg = sum(r[2] for r in rows if r[0] in ("meas only", "prep only"))
idle_iso = sum(r[1] for r in rows if "idle" in r[0])
idle_marg = sum(r[2] for r in rows if "idle" in r[0])
print(f"{'mixing':12s} {mixing:9.3f}                             (1 - sum isolated: cross-channel faults)")
print(f"{'SPAM':12s} {spam_iso:9.3f} {'':10s} {spam_marg:9.3f}         (meas + prep combined)")
print(f"{'idle total':12s} {idle_iso:9.3f} {'':10s} {idle_marg:9.3f}         (gate + meas idle combined)")
print(f"sum(marginal) = {sum(r[2] for r in rows):.3f}  (>1 <=> shared mixed faults; Google renormalizes)")
print(f"Willow-form sum: {sum(P_STAR / r[4] for r in rows if r[4]):.3f} = "
      + " + ".join(f"{P_STAR/r[4]:.3f} ({r[0].replace(' only', '')})" for r in rows if r[4])
      + ("   [all terms < 1: genuinely sub-threshold]"
         if all(P_STAR / r[4] < 1 for r in rows if r[4])
         else "   [WARNING: a term >= 1 — p* is not below every pseudo-threshold]"))''')

code(r'''fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
# left: marginal budget fractions vs p — REWEIGHTED measured spectra (solid), the same estimator
# as the table/bars at p*, so the panels agree. The f5-fit version is kept as faint dashed lines
# (unclipped) to make the low-p extrapolation drift visible instead of hiding it.
Lf_rw = reweight_spectrum(tech1["full symmetric"]["spec"], p_grid).P_logical
Lf_fit = tech1["full symmetric"]["LER"]
for ch in CHANNELS:
    frac_rw = 1.0 - reweight_spectrum(tech1_abl[ABL_OF[ch]]["spec"], p_grid).P_logical / Lf_rw
    axL.plot(p_grid, frac_rw, "-", color=COLORS[ch], lw=2, label=ch)
    axL.plot(p_grid, 1.0 - tech1_abl[ABL_OF[ch]]["LER"] / Lf_fit, "--", color=COLORS[ch], lw=1, alpha=0.4)
axL.plot([], [], "--", color="gray", lw=1, alpha=0.6, label="f5-fit version (drifts at low p)")
axL.axhline(0.0, color="gray", lw=0.8)
axL.set_xscale("log"); axL.set_xlabel("physical error rate p")
axL.set_ylabel("marginal fraction  1 − LER$_{no\,i}$/LER$_{full}$")
axL.axvline(P_STAR, color="gray", ls=":", lw=1); axL.legend(fontsize=8); axL.grid(alpha=0.3)
# right: the budget bar chart at p*
labels = [r[0] for r in rows] + ["mixing"]
iso_v  = [r[1] for r in rows] + [mixing]
marg_v = [r[2] for r in rows] + [np.nan]
x = np.arange(len(labels))
axR.bar(x - 0.2, iso_v, 0.4, label="isolated (channel-only)")
axR.bar(x + 0.2, marg_v, 0.4, yerr=[r[5] for r in rows] + [np.nan], capsize=3,
        label="marginal (leave-one-out)")
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

**Cost.** The 144-qubit sweeps are the expensive part of the runner (order an hour+), which is
exactly why they are cached per task: a re-run of `run_error_model_comparison.py` touches them only
if their config changed. `--boost72` buys the 72-code spectra 2× target failures and 5× shots —
see the §7.5 box for when that is worth it.""")

code(r'''tech2_72 = {name: load(f"tech2_72__{slug(name)}")["D"] for name in MODELS}
print(f"{'model':16s} {'D':>2}   (circuit fault distance, BP-OSD upper bound; f0 unpinned at this size)")
for name in MODELS:
    print(f"{name:16s} {tech2_72[name]:2d}")''')

code(r'''tech1_72, mc72 = {}, {}
for name in MODELS:
    r = load(f"tech1_72__{slug(name)}")
    tech1_72[name] = dict(spec=spectrum_of(r), LER=np.asarray(r["LER_fit"]), cost=r["fit"]["cost"])
    mc72[name] = {float(p): v for p, v in load(f"mc72__{slug(name)}")["points"].items()}
    n_planned, n_sampled = len(r["W"]), len(r["spectrum"]["weights"])
    print(f"{name:16s}: weights 1..{r['W'][-1]} ({n_sampled}/{n_planned} sampled, {r['shots']} shots), "
          f"f5 fit cost={r['fit']['cost']:.2f}, MC LER(0.008)={mc72[name][0.008][0]:.3e}")
print()
print(section_time("tech2_72", "tech1_72", "mc72"))''')

md(r"""### The measured failure spectra — every one of them, both codes

Every point value, budget fraction, and Λ in §6–§8 is a binomial reweighting of a sampled
spectrum, and the spectra are where estimator pathologies hide — so *all* of them are plotted
here (6 models + 5 leave-one-out mixes + 6 asymmetric mixes, per code). Reading guide: the
dashed line is the full model's tech2 minimum failing weight `D` — the **perfect-decoder**
floor. Dots *below* it are **decoder-in-the-loop failures**, and their weight tells you which
kind: from `w₀ = ⌈D/2⌉` up they are irreducible coset ambiguity (any decoder can be forced to
misjudge them); *below* `w₀` they are decoder artifacts, and this report has eliminated both
known kinds — BP symmetry traps (fixed by `num_sets=20`) and prior-mismatch misdecodes of
trace-free hooks (fixed by calibrating the decoder at `p*` instead of `p_ref` — see the §2
decoder-convention note). A dot reappearing below `w₀` after a decoder or budget change is a
regression in one of those two categories. The thin dashed curves are the **f5 ansatz** (Eq. 10
of the paper; stored fits for the models, fitted here on the cached bins for the mixes): it is
identically zero below its fitted `w₀`, so a dot the dashed curve structurally cannot reach is
outside the ansatz family — the decoder-floor signature at a glance. **× marks** are sampled
bins with zero observed failures (drawn at `1/(2T)`): empty low-`w` bins are what the §7.5/§8
truncation interval prices at `3/T`, so watch them whenever budgets change. The alternating
high-`w` gaps are the stride-2 tail sampling (see §8.4's `fill_spectrum`).""")

code(r'''# ALL measured spectra in the report — every number in §6–§8 reweights one of these curves,
# so every one of them gets eyeballed here. Panels: rows = isolated+full models / leave-one-out
# mixes / asymmetric ×5 mixes; cols = [[18,4,4]] / [[72,4,8]]. Loaded straight from the cache
# (independent of the per-section cells below; curves whose task is not cached yet are listed
# as missing instead of crashing the figure). Dots: f(w)/T(w) where failures were observed;
# × = sampled zero bins (at 1/(2T)); thin dashed curve = f5 ansatz (stored fit where the runner
# fitted one, else fitted here on the cached bins); dashed vertical = the full model's tech2 D,
# the PERFECT-decoder floor. The ansatz is identically zero below its fitted w0 — a dot the
# dashed curve cannot reach is outside the ansatz family, the decoder-floor signature. Check
# that region whenever the decoder config or sampling budgets change.
from importance_sampling import failure_spectrum_ansatz, fit_failure_spectrum

def _plot_spec(ax, name, color, label):
    try:
        r = load(name)
    except FileNotFoundError:
        return label                               # not cached (e.g. stale-decoder quarantine)
    s = spectrum_of(r)
    w, f, t = (np.asarray(a) for a in (s.weights, s.failures, s.trials))
    ax.plot(w[f > 0], (f / t)[f > 0], ".-", ms=4, lw=0.8, color=color, label=label)
    z = f == 0
    if z.any():
        ax.plot(w[z], 0.5 / t[z], "x", color=color, alpha=0.4)
    try:
        if "fit" in r:
            fw = lambda wg, r=r: failure_spectrum_ansatz(
                wg, a=1.0 - 2.0 ** -r["K"], model=r["fit"]["model"], **r["fit"]["params"])
        else:
            fw = fit_failure_spectrum(s, K=r["K"], model="f5", w0=None, f0=None).f
        wg = np.geomspace(1.0, float(w.max()), 200)
        y = np.asarray(fw(wg))
        ax.plot(wg[y > 0], y[y > 0], "--", lw=1.0, alpha=0.55, color=color)
    except Exception:
        pass                                       # too-sparse spectrum, no converged fit — dots only
    return None

ABL_COLOR = {abl: COLORS[ch] for ch, abl in ABL_OF.items()}
FULLC = COLORS["full symmetric"]
ROWS = [
    ("isolated + full models",
     [(f"tech1__{slug(m)}", COLORS[m], m) for m in MODELS],
     [(f"tech1_72__{slug(m)}", COLORS[m], m) for m in MODELS]),
    ("leave-one-out mixes",
     [(f"tech1_abl__{slug(a)}", ABL_COLOR[a], a) for a in ABLATED],
     [(f"tech1_72_abl__{slug(a)}", ABL_COLOR[a], a) for a in ABLATED]),
    ("asymmetric ×5 mixes (meas, meas-idle ×5)",
     [("asym__full_18", FULLC, "full ×5")] + [(f"asym__{slug(a)}_18", ABL_COLOR[a], a) for a in ABLATED],
     [("asym__full_72", FULLC, "full ×5")] + [(f"asym__{slug(a)}_72", ABL_COLOR[a], a) for a in ABLATED]),
]
D_BY_COL = [load("tech2__full_symmetric")["D"], load("tech2_72__full_symmetric")["D"]]

fig, axes = plt.subplots(3, 2, figsize=(13, 13), sharex="col", sharey=True)
for r, (row_title, specs18, specs72) in enumerate(ROWS):
    for c, specs in enumerate((specs18, specs72)):
        ax = axes[r][c]
        missing = [m for name, col, lbl in specs if (m := _plot_spec(ax, name, col, lbl))]
        ax.axvline(D_BY_COL[c], color="k", ls="--", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{['[[18,4,4]]', '[[72,4,8]]'][c]} — {row_title}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        if missing:
            ax.text(0.03, 0.03, "missing: " + ", ".join(missing), transform=ax.transAxes,
                    fontsize=7, color="firebrick", va="bottom")
    axes[r][0].set_ylabel("per-bin failure fraction  f(w)/T(w)")
axes[0][0].plot([], [], "--", color="gray", lw=1, alpha=0.7, label="f5 ansatz")
axes[0][0].legend(fontsize=7)
axes[2][0].set_xlabel("fault weight w"); axes[2][1].set_xlabel("fault weight w")
plt.tight_layout(); plt.show()''')

code(r'''eps18 = {m: per_round(tech1[m]["LER"], ROUNDS) for m in MODELS}
eps72 = {m: per_round(tech1_72[m]["LER"], ROUNDS72) for m in MODELS}

# Λ evaluation point — matches the §6 budget's p*, deep in the suppression regime. Point-Λ uses
# the REWEIGHTED measured spectra (both codes sampled from w=1 through their onsets), avoiding
# cross-fit extrapolation drift; the f5-based value is printed as a cross-check. The ±σ column
# propagates the binomial errors of both spectra. Honesty note: zero-failure onset bins make the
# reweighted ε72 a lower bound → Λ an upper bound (a large reweighted-vs-fit gap flags this; the
# §7.5 box quantifies it per channel via the rule-of-three zero-bin headroom).
i_lam = int(np.argmin(np.abs(p_grid - P_LAM)))
rows7, inv7 = [], {}
print(f"{'channel':16s} {'p_th (ε18=ε72)':>15} {'Λ(p*)':>9} {'±σ':>8} {'Λ(fit)':>9} {'p*/p_th':>9}     (p* = {P_LAM})")
for m in MODELS:
    pth = crossing_p(p_grid, eps18[m], eps72[m])
    inv, inv_se, inv_lo, inv_hi = inv_lambda_stats(tech1[m]["spec"], tech1_72[m]["spec"], P_LAM)
    inv7[m] = (inv, inv_se, inv_lo, inv_hi)
    lam, lam_se = 1.0 / inv, inv_se / inv**2
    lam_fit = float(eps18[m][i_lam] / eps72[m][i_lam])
    rows7.append((m, pth, lam))
    pths = f"{pth:.4f}" if pth else f">{p_grid.max():.3g}"
    term = f"{P_LAM/pth:.3f}" if pth else f"<{P_LAM/p_grid.max():.3f}"
    print(f"{m:16s} {pths:>15} {lam:9.3g} {lam_se:8.2g} {lam_fit:9.3g} {term:>9}")

lam_full = dict((m, l) for m, _, l in rows7)["full symmetric"]
print(f"\nΛ_full(p*) = {lam_full:.3g}  →  per-(+2-distance)-step λ = √Λ = {np.sqrt(lam_full):.3g}  (d: 4→8 is two steps)")
inv_lam = 1.0 / lam_full
terms = {m: P_LAM / pth for m, pth, _ in rows7 if pth is not None and m != "full symmetric"}
spam_term = terms.get("meas only", 0) + terms.get("prep only", 0)
print(f"Willow identity check at p*: 1/Λ_full = {inv_lam:.3f}  vs  Σᵢ p*/p_th,i = {sum(terms.values()):.3f}"
      f"   (CZ {terms.get('CZ only', float('nan')):.3f}, SPAM {spam_term:.3f})")
print("residual = mixed-channel faults (isolated channels cannot see them — see §1/§5); "
      "Σ < 1/Λ_full means the additive budget under-covers by that share.")''')

code(r'''fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
for m in MODELS:
    col = COLORS[m]
    axL.plot(p_grid, eps18[m], "-", color=col, lw=2, label=m)
    axL.plot(p_grid, eps72[m], "--", color=col, lw=1.5)
    mp = sorted(mc72[m])
    axL.plot(mp, [per_round(np.asarray([mc72[m][p][0]]), ROUNDS72)[0] for p in mp], "^", color=col, ms=6)
    axR.plot(p_grid, np.sqrt(eps18[m] / eps72[m]), "-", color=col, lw=2, label=m)
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel("physical error rate p"); axL.set_ylabel("per-round logical error rate ε")
axL.set_title("[[18,4,4]] (solid, 2 rounds) vs [[72,4,8]] (dashed, 4 rounds; ▲=MC)")
axL.legend(fontsize=8); axL.grid(alpha=0.3, which="both")
axR.set_xscale("log"); axR.set_yscale("log")
axR.axhline(1.0, color="gray", lw=1); axR.axvline(P_LAM, color="gray", ls=":", lw=1)
# √Λ = per-(+2-distance)-step suppression (d: 4→8 is two steps) — directly comparable to the
# per-step λ Google quotes for the surface-code ladder (Willow: λ ≈ 2.14). λ=1 crossings = p_th.
axR.set_xlabel("physical error rate p")
axR.set_ylabel(r"$\lambda(p) = \sqrt{\varepsilon_{18}/\varepsilon_{72}}$  (per +2-distance step)")
axR.set_title("per-step error suppression (crossings at $\lambda=1$ = true $p_{th,i}$)")
axR.legend(fontsize=8); axR.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()''')

# ===========================================================================
md(r"""## §7.5 — Marginal Λ: the ablations on the larger code

§7.3's Λᵢ used channel-*isolated* circuits — the clean per-channel suppression capability, but not
a decomposition of Λ_full (the mixed-channel faults belong to no isolated circuit). Google's
budget convention is *marginal*: ablate one component from the FULL circuit and measure the
change. The runner runs the five leave-one-out circuits on **[[72,4,8]]** (the 18-code ablations
are §5's); we form `Λ_no-i(p*) = ε₁₈,no-i/ε₇₂,no-i` and read channel i's contribution to the full
suppression deficit as `1/Λ_full − 1/Λ_no-i`. The gap between Σ(contributions) and `1/Λ_full` is
the Λ-space mixing — the same story §6 tells in LER-space, now for error suppression.

**Reading the signs.** Each contribution is a *difference of two noisy ratios*, so this box now
carries the full uncertainty budget: `±σ` propagates the binomial errors of all four spectra, and
the `[lo, hi]` interval additionally spans the **zero-bin truncation** — every sampled-but-empty
weight bin priced at its rule-of-three upper bound `f(w) < 3/T`. That pricing deliberately
includes the bins *below* tech2's D: D is the perfect-decoder floor, but the measured spectra
show the actual decoder **miscorrects below it** (see the spectrum figure in §7 — `f(1) > 0` on
the full 18-code mix under a p_ref-calibrated decoder), so an empty low-`w` 72-code bin is
unresolved statistics, not a structural zero — and at low `p*` the reweighting is most sensitive
to exactly those bins. A negative share is only *real* (a channel whose faults the big code
handles better than the small one) if it stays negative within both; otherwise it is an estimator
artifact of the under-resolved [[72,4,8]] low-weight bins. `--boost72` tightens the onset
statistics; the truncation interval, however, shrinks only with more trials on the empty low-`w`
bins themselves (3/T), not with more onset failures.""")

code(r'''tech1_72_abl = {name: dict(spec=spectrum_of(load(f"tech1_72_abl__{slug(name)}"))) for name in ABLATED}
print(section_time("tech1_72_abl"))''')

code(r'''def lambda_decomposition(spec18_full, spec72_full, spec18_of, spec72_of, p):
    """The marginal-Λ box: contribution_i = 1/Λ_full − 1/Λ_no-i with ±σ and zero-bin interval."""
    inv_f, se_f, lo_f, hi_f = inv_lambda_stats(spec18_full, spec72_full, p)
    print(f"{'channel':12s} {'Λ_no-i(p*)':>11} {'1/Λ_no-i':>10} {'contribution':>13} {'±σ':>9} "
          f"{'share':>7}   verdict")
    tot = 0.0
    for ch in CHANNELS:
        inv_a, se_a, lo_a, hi_a = inv_lambda_stats(spec18_of(ch), spec72_of(ch), p)
        c = inv_f - inv_a
        sc = float(np.hypot(se_f, se_a))
        c_lo, c_hi = lo_f - hi_a, hi_f - lo_a      # zero-bin truncation interval around c
        tot += c
        if abs(c) < 2 * sc:
            verdict = "~0 within 2σ (noise)"
        elif c_lo < 0.0 < c_hi:
            verdict = "sign not robust to zero-bin truncation"
        else:
            verdict = "solid"
        print(f"{ch:12s} {1.0/inv_a:11.3g} {inv_a:10.3e} {c:13.3e} {sc:9.2e} {c/inv_f:7.2f}   {verdict}")
    print(f"\nsum of marginal contributions = {tot:.3e}  vs  1/Λ_full = {inv_f:.3e}"
          f"   (ratio {tot/inv_f:.2f})")
    print("ratio > 1: shared mixed faults counted once per participant (as in §6's Σ marginal > 1).")
    print("A NEGATIVE contribution is physical only when 'solid': removing that channel genuinely")
    print("HURT the suppression ratio (the big code handles its faults better than the small one).")
    print("'~0' / 'not robust' rows are sampling artifacts — tighten with run_error_model_comparison")
    print("--boost72 (2× failures, 5× shots on every 72-code spectrum feeding this box).")
    return inv_f

print(f"marginal Λ decomposition at p* = {P_LAM}   (Λ_full = {lam_full:.3g}, 1/Λ_full = {1/lam_full:.3e})")
_ = lambda_decomposition(tech1["full symmetric"]["spec"], tech1_72["full symmetric"]["spec"],
                         lambda ch: tech1_abl[ABL_OF[ch]]["spec"],
                         lambda ch: tech1_72_abl[ABL_OF[ch]]["spec"], P_LAM)''')

# ===========================================================================
md(r"""## §8 — A second operating point: meas & meas-idle ×5

Everything above is a **sensitivity analysis at one base point in rate-space** — the symmetric ray
where every channel runs at the same p. The budget fractions are components of a gradient of
`1/Λ` (and of LER), and gradients depend on where you evaluate them. Real devices don't sit on
the symmetric ray: measurement error and the measure/reset dead-time idle typically run several
times hotter than gates. Here we re-evaluate at a device-like point — **meas and meas_idle at
5×p, everything else at p** (`scale_noise_channels`) — and redo the marginal budget and Λ.

Two economies: (i) the **isolated** curves need no resampling — `f(w)` is rate-independent, so
channel i at rate `rᵢ·p` is just `reweight_spectrum(spec_i, [rᵢ·p])`; only the full and ablated
*mixes* (whose channel composition changes) are new sweeps. (ii) Ablating channel i from the
asymmetric mix composes the existing tools: `filter_noise_channel(scale_noise_channels(...))`.
Convention: `p` remains the base rate of the un-boosted channels, so `p*` comparisons across
§6/§8 are at equal gate noise. The Λ box carries the same `±σ` / zero-bin verdicts as §7.5.""")

code(r'''asym = {}
for label in ("18", "72"):
    asym[("full", label)] = spectrum_of(load(f"asym__full_{label}"))
    for abl_name in ABLATED:
        asym[(abl_name, label)] = spectrum_of(load(f"asym__{slug(abl_name)}_{label}"))
print(section_time("asym"))''')

code(r'''# The budget at the asymmetric point. Isolated fractions reuse the §2 spectra, reweighted at
# each channel's OWN rate (r_i * p*); marginals come from the new asymmetric ablated mixes.
L_asym, L_asym_se, _ = rw_stats(asym[("full", "18")], P_STAR)
rows8 = []
for ch in CHANNELS:
    iso = float(reweight_spectrum(tech1[ch]["spec"], [R_OF[ch] * P_STAR]).P_logical[0]) / L_asym
    La, La_se, _ = rw_stats(asym[(ABL_OF[ch], "18")], P_STAR)
    marg = 1.0 - La / L_asym
    marg_se = (La / L_asym) * float(np.hypot(La_se / La, L_asym_se / L_asym))
    rows8.append((ch, iso, marg, marg_se))
mix8 = 1.0 - sum(r[1] for r in rows8)
sym = {r[0]: (r[1], r[2]) for r in rows}                 # §6's symmetric-point rows for comparison
print(f"budget at the ASYMMETRIC point (meas, meas_idle ×5), p* = {P_STAR}:")
print(f"LER_full = {L_asym:.3e}  (symmetric point: {L_full:.3e} — the ×5 mix costs "
      f"{L_asym/L_full:.1f}× in error rate)")
print(f"{'channel':12s} {'isolated':>9} {'marginal':>9} {'±σ':>6}   vs symmetric {'iso':>6} {'marg':>6}")
for ch, iso, marg, mse in rows8:
    print(f"{ch:12s} {iso:9.3f} {marg:9.3f} {mse:6.3f}                {sym[ch][0]:6.3f} {sym[ch][1]:6.3f}")
print(f"{'mixing':12s} {mix8:9.3f}                               {mixing:6.3f}")
print(f"sum(marginal) = {sum(r[2] for r in rows8):.3f}   (symmetric: {sum(r[2] for r in rows):.3f})")''')

code(r'''# Λ at the asymmetric point + its marginal decomposition (both codes' asymmetric ablations),
# with the same ±σ and zero-bin verdicts as §7.5 — read the two boxes together to see how the
# gradient of 1/Λ rotates when the noise mix moves from the symmetric ray to the device-like ray.
inv8, inv8_se, _, _ = inv_lambda_stats(asym[("full", "18")], asym[("full", "72")], P_LAM)
lam8 = 1.0 / inv8
print(f"Λ_full(p*={P_LAM}) at the ×5 point: {lam8:.3g} ± {inv8_se/inv8**2:.2g}   (symmetric: {lam_full:.3g})"
      f"   per-step λ = {np.sqrt(lam8):.3g} (symmetric: {np.sqrt(lam_full):.3g})")
_ = lambda_decomposition(asym[("full", "18")], asym[("full", "72")],
                         lambda ch: asym[(ABL_OF[ch], "18")],
                         lambda ch: asym[(ABL_OF[ch], "72")], P_LAM)''')

# ---------------------------------------------------------------------------
md(r"""### §8.4 — the §7.4 panels on the ×5 ray

The point-decomposition table above is noise-limited at `p* = 5e-4`; the *curves* are not — away
from the deep-suppression regime the same cached spectra resolve the ×5 story cleanly. This is the
§7.4 figure remade at the device-like point, with **no new sampling**: the full mix reweights its
own asymmetric spectra, and each isolated channel reweights its §2/§7.2 spectrum at its **own**
rate `rᵢ·p` (the x-axis stays the base rate `p` of the un-boosted channels, the §6/§8 convention).
Boosted channels are drawn only while `rᵢ·p ≤` the grid top their weight windows were sized for —
beyond that the reweighted curve sags from window truncation, not physics — and every curve here
is the reweighted measured spectrum (lower-bound caveat at very low `p`), so the crossings table
recomputes the symmetric-ray values with the same estimator for an apples-to-apples comparison.""")

code(r'''# ε-curve pairs on the ×5 ray. Isolated channels: reweight at r_i·p (no new data). Mixes:
# reweight the asymmetric spectra along the ray. Mask boosted channels where r_i·p exceeds the
# sampled window coverage. TINY floors the logs in crossing_p (deep-sub-onset ε72 can underflow).
#
# Stride fix: the 72-code and asymmetric sweeps sample the weight tail at stride 2, and
# reweight_spectrum sums only sampled weights — at HIGH p (binomial mass inside the strided
# tail) the raw sum undercounts LER by ~2x. fill_spectrum inserts each missing tail weight
# with its neighbors' pooled counts (f varies slowly in w), which restores the full mass.
# Point values at p* are unaffected (the mass sits in the contiguous w<16 head down there).
def fill_spectrum(spec):
    W, T, F = list(spec.weights), list(spec.trials), list(spec.failures)
    w_out, t_out, f_out = [], [], []
    for i, (w, t, f) in enumerate(zip(W, T, F)):
        w_out.append(w); t_out.append(t); f_out.append(f)
        if i + 1 < len(W) and W[i + 1] == w + 2:
            w_out.append(w + 1); t_out.append(t + T[i + 1]); f_out.append(f + F[i + 1])
    return FailureSpectrum(weights=w_out, trials=t_out, failures=f_out,
                           n_expanded=spec.n_expanded, q_base=spec.q_base, p_ref=spec.p_ref)

TINY = 1e-300
X5 = "full ×5 mix"
R_OF_ALL = dict(R_OF, **{X5: 1.0})
eps18_x5 = {X5: per_round(reweight_spectrum(fill_spectrum(asym[("full", "18")]), p_grid).P_logical, ROUNDS)}
eps72_x5 = {X5: per_round(reweight_spectrum(fill_spectrum(asym[("full", "72")]), p_grid).P_logical, ROUNDS72)}
eps18_rw = {m: per_round(reweight_spectrum(tech1[m]["spec"], p_grid).P_logical, ROUNDS) for m in MODELS}
eps72_rw = {m: per_round(reweight_spectrum(fill_spectrum(tech1_72[m]["spec"]), p_grid).P_logical, ROUNDS72)
            for m in MODELS}
for ch in CHANNELS:
    eps18_x5[ch] = per_round(reweight_spectrum(tech1[ch]["spec"], R_OF[ch] * p_grid).P_logical, ROUNDS)
    eps72_x5[ch] = per_round(reweight_spectrum(fill_spectrum(tech1_72[ch]["spec"]),
                                               R_OF[ch] * p_grid).P_logical, ROUNDS72)
VALID = {m: R_OF_ALL[m] * p_grid <= p_grid.max() * (1 + 1e-9) for m in eps18_x5}

print(f"true per-channel thresholds on the ×5 ray (base-p convention) vs the symmetric ray")
print(f"{'channel':14s} {'p_th ×5':>10} {'p_th sym (rw)':>14}   note")
for m in [X5] + CHANNELS:
    v = VALID[m]
    pth = crossing_p(p_grid[v], np.maximum(eps18_x5[m][v], TINY), np.maximum(eps72_x5[m][v], TINY))
    sym_m = "full symmetric" if m == X5 else m
    pth_s = crossing_p(p_grid, np.maximum(eps18_rw[sym_m], TINY), np.maximum(eps72_rw[sym_m], TINY))
    top = p_grid[v].max()
    pths = f"{pth:.4f}" if pth else f">{top:.3g}"
    pss = f"{pth_s:.4f}" if pth_s else f">{p_grid.max():.3g}"
    note = ("×5 channel: threshold in its OWN rate = 5×p_th" if R_OF_ALL[m] > 1 else "")
    print(f"{m:14s} {pths:>10} {pss:>14}   {note}")
print("NB: these crossings sit near the K=4 saturation pinch (ε72 caps at 1−(2^-K)^{1/4}), so they")
print("are estimator-sensitive — §7's f5-fit versions (full 0.0227, CZ 0.0379) may differ; treat")
print("either as indicative. λ at very low p is an UPPER bound (zero-bin truncation of ε72).")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
CLR = dict(COLORS, **{X5: "crimson"})
for m in [X5] + CHANNELS:
    col, v = CLR[m], VALID[m]
    lw = 2.5 if m == X5 else 2
    axL.plot(p_grid[v], eps18_x5[m][v], "-", color=col, lw=lw, label=m)
    axL.plot(p_grid[v], eps72_x5[m][v], "--", color=col, lw=lw * 0.7)
    lam = np.sqrt(np.maximum(eps18_x5[m][v], TINY) / np.maximum(eps72_x5[m][v], TINY))
    axR.plot(p_grid[v], lam, "-", color=col, lw=lw, label=m)
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel("base physical error rate p (boosted channels at 5p)")
axL.set_ylabel("per-round logical error rate ε")
axL.set_title("×5 ray: [[18,4,4]] (solid) vs [[72,4,8]] (dashed) — reweighted spectra")
axL.legend(fontsize=8); axL.grid(alpha=0.3, which="both")
axR.set_xscale("log"); axR.set_yscale("log")
axR.axhline(1.0, color="gray", lw=1); axR.axvline(P_LAM, color="gray", ls=":", lw=1)
axR.set_xlabel("base physical error rate p")
axR.set_ylabel(r"$\lambda(p) = \sqrt{\varepsilon_{18}/\varepsilon_{72}}$  (per +2-distance step)")
axR.set_title("per-step suppression on the ×5 ray ($\lambda=1$ crossings = $p_{th,i}$)")
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
* **Λ shares are differences of noisy ratios — trust only flagged-solid signs (§7.5, §8).** Each
  marginal contribution now carries a propagated `±σ` and a zero-bin truncation interval; a negative
  share that is `~0 within 2σ` or `not robust to zero-bin truncation` is an artifact of the
  under-resolved 72-code onset bins, not physics. `--boost72` in the runner tightens exactly those
  spectra (and the cache means nothing else reruns).
* **The whole budget is a gradient at a base point (§8).** Re-evaluating at a device-like ray
  (meas + meas-idle ×5) shows how the decomposition rotates with the noise mix. The isolated
  basis reweights analytically to any rate vector (`f(w)` is rate-independent); only the full and
  ablated mixes need resampling — which is what makes multi-point sensitivity maps affordable.

*Report generated by `make_error_model_comparison.py`; data by `run_error_model_comparison.py`
(cached per task under `runs/error_model_comparison_18_4_4/`).*""")

# ===========================================================================
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT / "notebooks" / "methods" / "error_model_comparison_18_4_4.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
