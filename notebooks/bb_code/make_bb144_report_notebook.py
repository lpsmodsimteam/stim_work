"""Generate BB144_failfast_report.ipynb (source only; cells are NOT executed here).

A walkthrough of the three-technique fail-fast analysis for the gross code BB(12)=[[144,12,12]],
single-sector, produced by:

  bb6_fig10_sweep.py --code bb144 --plot --max-cores 22 --mw-workers 22 \
                     --split-p-low 0.0001 --split-n-levels 40 --outdir .../bb144_curve

It reuses the bb6_report helpers (compute + figures), which now handle this run's UNPINNED onset
(odd circuit distance => the even-D Proposition-1 onset fraction does not apply => f0 is fit, not
exact). The narrative foregrounds what is a BOUND vs exact, because — unlike BB(6) — exact min-weight
enumeration is infeasible at this scale (see the report's closing note).

Run order: the cells need the COMPLETED run's bb144_curve/ (bb6_fig10.npz is written last, after the
splitting ladder finishes).
"""
import json, pathlib

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

md(r"""# Fail-fast analysis — gross code **BB(12) = [[144,12,12]]** (single-sector)

Walkthrough of the three techniques (paper: *Fail fast: techniques to probe rare events in QEC*,
arXiv:2511.15177) applied to the **gross code** under circuit noise, **single-sector (Z-type)**
decoding, `rounds=12`, symmetric depolarizing (`p_meas=p_phys`), `p_ref=0.003`, 500 shots/weight.

> **Read this first — what is exact vs a bound.** For the small BB(6) code we could enumerate the
> minimum-weight logical operators *exactly* (a meet-in-the-middle). **For BB(12) we cannot** — the
> single-sector DEM has N≈8784 mechanisms with a uniform per-detector branching of ~33, so exact
> weight‑D enumeration is combinatorially out of reach (a birthday/MITM split gives no speedup here:
> measured collision factor ≈ 1.4 vs the ~10³ needed). This is exactly why the **paper itself reports
> bounds** for BB(12): distance **D ≤ 10**, **|L(D)| ≥ 3456**. So below, **D is an upper bound** (our
> symmetry-augmented BP‑OSD search), **|L(D)| is a lower bound** (search saturation), and because the
> circuit fault distance is **odd**, the onset fraction **f₀ is fit by the ansatz, not pinned**.

| technique | what it gives here | status |
|---|---|---|
| **II — min-weight onset** | D, onset w₀=⌈D/2⌉, \|L(D)\| | D≤ (bound), \|L(D)\|≥ (bound), f₀ **unpinned** |
| **I — failure-spectrum ansatz** | LER(p) extrapolation (f3/f5) | onset **fitted** from the IS spectrum |
| **III — replica-exchange splitting** | LER cross-check, ladder **to 1e-4** | poor mixing for 12 logicals → *sanity check* |
""")

md(r"""## Setup — locate the repo and load the run
The first cell only imports. The second runs `bb6_report.compute(bb144_curve)`, which loads the npz +
JSONs and fits the f3/f5 ansatze with bootstrap bands (~15-30 s). It needs the **completed** run.""")

code('''import sys, pathlib
print("kernel python:", sys.executable)   # expect the 'qec' env
_here = pathlib.Path.cwd()
_cands = [_here, *_here.parents] + [c / "notebooks" / "bb_code" for c in [_here, *_here.parents]]
_nbdir = next((c for c in _cands if (c / "bb6_report.py").exists()), None)
assert _nbdir is not None, "Could not locate bb6_report.py — run this notebook from inside the repo."
sys.path.insert(0, str(_nbdir)); sys.path.insert(0, str(_nbdir.parent.parent / "src"))
import numpy as np, matplotlib.pyplot as plt
import bb6_report
# Full-spectrum IS sweep (w->115, w0=6-pinned ansatz) + the replica-exchange splitting.json both live
# in bb144_curve_hi; the original bb144_curve was the under-sampled / wrong-splitter first pass.
CURVE = _nbdir / "bb144_curve_hi"
assert (CURVE / "bb6_fig10.npz").exists(), (
    f"{CURVE}/bb6_fig10.npz not found — the run writes it LAST (after the 1e-4 splitting ladder). "
    "Wait for the run to finish, or inspect distance.json / search_convergence.json directly.")
print("loading + fitting bb144_curve ...")''')

code('''import time; _t=time.time()
R = bb6_report.compute(CURVE)        # f0 unpinned here -> ansatz fits the onset fraction
m = R["meta"]
print(f"loaded in {time.time()-_t:.0f}s")
print(f"code={m['code_label']}  method={m['method']}  single_sector={m['single_sector']}")
print(f"D{'≤' if m.get('distance_is_bound') else '='}{m['D']}  onset w0={m['w0']}  "
      f"f0={'fitted' if not m['f0_pinned'] else 'exact'}={R['f0']:.3e}")
print(f"|L(D)|≥{m['n_min_logicals']} (compressed)   N(expanded)={m['N']}")
for mod in ("f3","f5"):
    print(f"  {mod}: LER(1e-4) = {R['LER'][mod][np.argmin(abs(R['p_grid']-1e-4))]:.3e}")''')

md(r"""## Technique II — minimum-weight onset (bounds)

Circuit fault distance, onset weight w₀=⌈D/2⌉, and the count of minimum-weight logicals. **All are
bounds** (see top): the BP-OSD search gives an upper bound on D and a lower bound on |L(D)|. The
single-sector circuit fault distance comes out **below the code distance 12** and **odd** — a real
property of the 12-round syndrome circuit, not the code.""")

code('''from IPython.display import Markdown, display
g = lambda k: ("—" if m.get(k) is None else f"{m[k]:.4g}" if isinstance(m[k],(int,float)) else str(m[k]))
rows = [("circuit fault distance D", f"≤ {m['D']}" if m.get('distance_is_bound') else g('D')),
        ("onset weight w0 = ceil(D/2)", g('mw_onset_weight')),
        ("onset fraction f0", f"{R['f0']:.3e} ({'fitted' if not m['f0_pinned'] else 'exact'})"),
        ("|L(D)| compressed", f"≥ {g('n_min_logicals')}"),
        ("|L(D)| expanded",   f"≥ {g('n_min_logicals_expanded')}"),
        ("N compressed (Ñ)",  g('n_compressed')),
        ("N expanded",        g('n_expanded'))]
display(Markdown("| quantity | value |\\n|---|---|\\n" + "\\n".join(f"| {a} | {b} |" for a,b in rows)))
print("\\nPaper BB(12) Table 2 (for reference): D ≤ 10,  |L(D)| ≥ 3456  — also bounds.")''')

md(r"""### Search saturation — the basis for the |L(D)| lower bound
|L(D)| found vs cumulative BP-OSD search trials. The **plateau** is the completeness signal the
search relies on (it stops finding new minimum-weight logicals). It is a lower bound, not a proof —
there is no exact MITM count to overlay here (unlike BB(6)).""")

code('''fig, ax = plt.subplots(figsize=(8.6, 5.0))
bb6_report.fig_search_convergence(R, ax)
ax.set_title("BB(12) — min-weight search saturation (|L(D)| lower bound)")
plt.show()''')

md(r"""## Technique I — failure-spectrum ansatz and LER(p)

The main figure: per-weight importance-sampling failure fractions reweighted to LER(p), plus the f3/f5
ansatz extrapolations and the Technique-III splitting points.

> **⚠ The ansatz is reliable only in the sampled (moderate-p) range — NOT at low p.** For bb144 the
> observable failures start at w≈46, ~40 weights above the true min-weight onset (w0=6), and f₀ is not
> pinnable (odd D, no exact MITM). The ansatz fits the observable spectrum well (the line tracks the IS
> points), but its low-p tail collapses to ~0 (it cannot see the w=6 onset) — a severe *under*estimate.
> Pinning w0=6 instead would *over*estimate (the fit breaks). **For the actual low-p LER, trust the
> Technique-III splitting points, not the ansatz extrapolation.**""")

code('''fig, ax = plt.subplots(figsize=(9, 6))
bb6_report.fig_ler_vs_p(R, ax)
ax.set_title("BB(12) [[144,12,12]] — logical vs physical error rate (single-sector)")
plt.show()''')

code('''fig, ax = plt.subplots(figsize=(9, 6))
bb6_report.fig_failure_spectrum(R, ax)
ax.set_title("BB(12) — failure spectrum f(w)  (onset fitted, not exact)")
plt.show()''')

md(r"""## Technique III — replica-exchange splitting (ladder to **1e-4**)

The splitting cross-check, run on a 40-level ladder all the way down to **p = 1e-4**. For BB(12) the
single-flip Metropolis chains mix **poorly** (12 inequivalent logical qubits fragment the failing-
configuration space), so this is a *sanity cross-check* — its agreement (or disagreement) with the
ansatz curve is the signal, not a precise number. The table shows per-rung splitting vs ansatz.""")

code('''if R["split"] is None:
    print("no splitting.json yet (Technique III runs last) — re-run after the full job completes.")
else:
    c = bb6_report.splitting_comparison(R)
    print(f"{'p':>10} {'split P':>12} {'± bar':>11} {'ansatz f3':>12} {'ratio':>7} {'consistent':>11}")
    for i in range(len(c["p"])):
        print(f"{c['p'][i]:>10.2e} {c['tP'][i]:>12.3e} {c['tSE'][i]:>11.2e} "
              f"{c['ansatz'][i]:>12.3e} {c['ratio'][i]:>7.2f} {str(bool(c['consistent'][i])):>11}")
    print(f"\\nladder reaches p_low = {c['p'].min():.1e} over {len(c['p'])} rungs")''')

md(r"""### Weight distribution of failing configurations
Median fault weight of the failing configs vs p (reweighted f3 ansatz), with the splitting chains'
mean weight overlaid where available. Collapses toward the onset w₀ at low p.""")

code('''fig, ax = plt.subplots(figsize=(9, 6))
bb6_report.fig_weight_vs_p(R, ax)
ax.set_title("BB(12) — failing-config weight vs p")
plt.show()''')

md(r"""## Takeaways

* **Single-sector circuit fault distance is below the code distance 12 and odd** (our search: D≤11;
  the paper: D≤10) — BP-OSD distance is an unstable upper bound, so the exact value is not pinned.
* **|L(D)| is a lower bound** from search saturation (≥ the plotted plateau); there is no exact count
  for BB(12) — exact min-weight enumeration is infeasible at this scale (the reason the paper, and
  this report, quote bounds rather than exact values).
* Because D is odd, the **onset fraction f₀ is fitted**, not pinned by an exact Proposition-1 count;
  the LER extrapolation is therefore ansatz-driven below the sampled regime.
* **Technique III (splitting) is a cross-check**, not a precise estimate, for this code — read its
  agreement with the ansatz, not its absolute value.

*Generated by `make_bb144_report_notebook.py`. To exceed the paper (exact |L(D)|/f₀) one would need a
generalized-birthday / SAT-or-ILP enumeration — noted as future work; the hand-rolled MITM does not
scale here.*""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent / "BB144_failfast_report.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
