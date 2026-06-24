"""Generate BB_failfast_fulldem_report.ipynb (source only; cells are not executed here).

Parallel to make_report_notebook.py (the single Z-sector report), but for the FULL both-sector DEM,
loaded from bb6_fulldem_curve/. Reuses the parameterized bb6_report module: same figure functions,
plus the two convergence plots (search saturation + decoder convergence) and a comparison section
overlaying the full-DEM vs single-sector LER curves and Table-2 numbers.
"""
import json, pathlib

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

md(r"""# *Fail Fast* on the **full both-sector DEM** for BB(6) = [[72,12,6]] (vs the single Z-sector)

Companion to `BB6_failfast_report.ipynb`. That report used the paper's **single CSS-sector (Z-type)**
decoding representation. Here we run the *same* three fail-fast techniques on the **full both-sector
detector error model** — the honest, un-projected representation — and **compare**.

| Technique | Single-sector report | This (full DEM) |
|---|---|---|
| **II** — min-weight onset (D, w0, f0, \|L(D)\|, \|F\|) | exact meet-in-the-middle | **exact MITM** (optimized pivot-chain) — the BP-OSD search *undercounts* \|L(D)\| ~7× |
| **I** — failure-spectrum ansatz + IS → LER(p) | f3/f5 pinned at the exact onset | f3/f5 pinned at the **exact** onset |
| **III** — replica-exchange splitting | cross-check to 1e-4 | cross-check to 1e-4 |
| **convergence** | — | **search-vs-exact** + **Relay-BP decoder convergence** |

**Why the full DEM differs:** the single-sector representation merges fault mechanisms that differ only
in their *other*-sector detectors; the full DEM does not. The fault distance D is representation-
independent (D=6, w0=3), but Ñ, N, \|L(D)\|, \|F\|, f0 and the LER differ. The full-DEM Table 2 is computed
by the **exact** detector-pivot MITM (the optimized `exact_min_weight_logicals_mitm`, ~46 min on the
full DEM). The BP-OSD *search* is run too — but it **saturates at only ~15% of the true \|L(D)\|** (a
false plateau, since BP-OSD is not a guaranteed min-weight decoder), which the convergence plot exposes.
So the exact MITM, not the search, pins the ansatz onset f0.

All heavy computation was done by `bb6_fig10_sweep.py --full-dem --decoder-conv`; this notebook loads
`bb6_fulldem_curve/` and re-fits/re-plots (fast).""")

md(r"""### Setup — imports, then load + fit (two cells)
The first cell only imports. The second calls `bb6_report.compute(bb6_fulldem_curve)` (fits f3/f5 +
bootstrap bands, ~10-15 s).""")

code('''import sys, pathlib
print("kernel python:", sys.executable)   # should be the 'qec' env (numpy/scipy/matplotlib)
_here = pathlib.Path.cwd()
_cands = [_here, *_here.parents] + [c / "notebooks" / "bb_code" for c in [_here, *_here.parents]]
_nbdir = next((c for c in _cands if (c / "bb6_report.py").exists()), None)
assert _nbdir is not None, "Could not locate bb6_report.py — run this notebook from inside the repo."
sys.path.insert(0, str(_nbdir)); sys.path.insert(0, str(_nbdir.parent.parent / "src"))
import numpy as np, matplotlib.pyplot as plt
import bb6_report
FULL_DIR = _nbdir / "bb6_fulldem_curve"      # full both-sector DEM results
SINGLE_DIR = _nbdir / "bb6_fig10_curve"      # single Z-sector results (for the comparison)
print("imports OK — run the next cell to load + fit the full-DEM results (~10-15s)")''')

code('''import time
print("loading + fitting full-DEM results ...", flush=True); _t = time.time()
R = bb6_report.compute(FULL_DIR)
m = R["meta"]
print(f"loaded {FULL_DIR.name} in {time.time()-_t:.0f}s — representation: "
      f"{'full both-sector DEM' if not m['single_sector'] else 'single-sector'}, method={m['method']}")''')

md(r"""## Technique II — exact min-weight onset on the full DEM (MITM)

The full-DEM Table-2 quantities are the **exact** detector-pivot MITM (`exact_min_weight_logicals_mitm`,
`distance_mitm.json`). The BP-OSD *search* is also run, but undercounts (next section), so the report
prefers the exact counts and pins the ansatz onset f0 to them.""")

code('''m = R["meta"]
def _f(x): return "—" if x is None else (f"{x:.4g}" if isinstance(x, (int, float)) else str(x))
print(f"method        : {m['method']}")
print(f"distance D    : {_f(m['D'])}     onset w0 : {_f(m['w0'])}")
print(f"Ñ (compressed): {_f(m['n_compressed'])}")
print(f"N (expanded)  : {_f(m['n_expanded'])}")
print(f"|L(D)| comp   : {_f(m['n_min_logicals'])}    exp : {_f(m['n_min_logicals_expanded'])}")
print(f"|F(D/2)|      : {_f(m['fail_count'])}")
print(f"f0 = f*(D/2)  : {_f(m['f0'])}")''')

md(r"""## Convergence diagnostics

**(left) Search vs exact** — `|L(D)|` found by the BP-OSD search vs cumulative trials. It *plateaus*,
which looks like completeness — but the exact MITM line shows the search saturates at only **~15%** of
the true `|L(D)|`. BP-OSD is not a guaranteed min-weight decoder, so the search-saturation heuristic is
**not** a reliable completeness check here; the exact MITM is authoritative (and is what pins f0).
**(right) Relay-BP decoder convergence** — logical error rate (and disagreement with the most-legs
decoder) vs the number of relay legs; the plateau / disagreement→0 shows the decoder has enough legs
to be reliable on the full DEM.""")

code('''fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.4))
bb6_report.fig_search_convergence(R, axL)
bb6_report.fig_decoder_convergence(R, axR)
plt.show()''')

md(r"""## Technique I — failure spectrum f(w) and the LER(p) curve (full DEM)

Same machinery as the single-sector report (f3 & f5 ansatze pinned at the onset, importance-sampled
spectrum, reweighted to LER(p)) — now on the full-DEM mechanism set.""")

code('''fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
bb6_report.fig_failure_spectrum(R, axL)   # weight-space view: f(w) + ansatz + exact onset star
bb6_report.fig_ler_vs_p(R, axR)           # the Figure-10 LER(p) curve
plt.show()
i = np.argmin(abs(R["p_grid"] - 1e-4))
print("full-DEM LER(1e-4): f3 = %.2e, f5 = %.2e" % (R["LER"]["f3"][i], R["LER"]["f5"][i]))''')

md(r"""## Technique III — replica-exchange splitting cross-check (full DEM)

If the full-DEM run included splitting (`splitting.json`), the replica-exchange estimate is compared
to the Technique-I ansatz across the ladder, exactly as in the single-sector report.""")

code('''if R["split"]:
    c = bb6_report.splitting_comparison(R)
    print(f"{'p':>10}{'mean P':>13}{'Tech-I':>11}{'ratio':>7}  1sig?")
    for k in range(0, c['p'].size, max(1, c['p'].size // 20)):
        ag = ' ok ' if c['consistent'][k] else 'off '
        print(f"{c['p'][k]:>10.2e}{c['tP'][k]:>13.2e}{c['ansatz'][k]:>11.2e}{c['ratio'][k]:>7.2f}  {ag}")
    fig, ax = plt.subplots(figsize=(9, 6)); bb6_report.fig_ler_vs_p(R, ax); plt.show()
else:
    print("no splitting.json in", FULL_DIR.name, "(run the sweep without --no-split to add it)")''')

md(r"""## Comparison — full DEM vs single Z-sector

Overlay the two LER(p) curves and put the Table-2 numbers side by side. The fault distance and onset
should match (D=6, w0=3); the counts and the LER differ because the full DEM does not merge the
other-sector detectors.""")

code('''R_single = bb6_report.compute(SINGLE_DIR)
from IPython.display import Markdown, display
display(Markdown(bb6_report.table2_compare(R_single, R)))
fig, ax = plt.subplots(figsize=(9, 6))
bb6_report.fig_compare_ler(R_single, R, ax)
plt.show()''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).resolve().parent / "BB_failfast_fulldem_report.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
