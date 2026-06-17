#!/usr/bin/env python3
"""Compare our BB(6) failure spectrum / LER to arXiv:2511.15177 Fig. 10 (BB(6)-relay).

Key correction over the first pass: the onset must be pinned with the paper's *complete*
min-weight count. Our Technique-II random BP-OSD search found only ~188 of the >=1524 min-weight
logicals (Table 2), so our |F(D/2)| (hence f0) is ~28x too low. Table 2 reports the BB(6)-circuit
value |F(D/2)| = 3.83e8 exactly, giving f0 = |F(D/2)| / C(N_expanded, D/2). We pin the ansatz
there and default to the 5-parameter (f5) ansatz the paper uses.

The plots show, against our own sampled relay f(w) (raw decoder output, no assumptions):
  * our f5/f3 ansatz pinned at the Table-2 onset,
  * the EXACT Table-2 bound (w=3, f0_paper),
  * our UNDER-counted Technique-II bound (w=3, f0_ours) for contrast,
  * optionally, paper relay points you read off the PDF (PAPER_RELAY_* below; empty by default,
    since by-eye digitization of the green *solid* relay curve proved unreliable — fill these in
    from the figure to overlay the paper's curve).

Usage:
    python notebooks/bb_code/compare_vs_paper.py --outdir notebooks/bb_code/bb6_fig10_out_500
    python notebooks/bb_code/compare_vs_paper.py --outdir ... --paper-fails 3.83e8
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.special import gammaln

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "src"))

import bb6_fig10_sweep as bb6  # noqa: E402
from importance_sampling import fit_failure_spectrum, logical_error_rate_from_ansatz  # noqa: E402

# arXiv:2511.15177 Table 2, BB(6)-circuit row (Z-type), EXACT paper numbers:
#   |L~(D)| >= 1524 (compressed min-weight logicals),  |F(D/2)| = 3.83e8 (failing weight-D/2 configs)
PAPER_BB6_FAILS_FD2 = 3.83e8

# Paper Fig-10 BB(6)-relay points you read off the PDF (the SOLID/triangle green curve, NOT the
# dashed/square bplsd one). Left empty: our earlier by-eye digitization picked the wrong curve and
# was unreliable. Fill from the figure to overlay the paper's actual relay curve.
PAPER_RELAY_P:  list = []   # physical error rates
PAPER_RELAY_LER: list = []  # logical error rates (right panel)
PAPER_RELAY_W:  list = []   # fault weights
PAPER_RELAY_FW: list = []   # failure fractions f(w) (left panel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=pathlib.Path, required=True)
    ap.add_argument("--paper-fails", type=float, default=PAPER_BB6_FAILS_FD2,
                    help="paper Table-2 |F(D/2)| for BB(6)-circuit (default 3.83e8)")
    args = ap.parse_args()

    ckpt = json.loads((args.outdir / "bb6.spectrum.json").read_text())
    cfg_d = json.loads((args.outdir / "config.json").read_text())
    dist = json.loads((args.outdir / "distance.json").read_text())

    spectrum = bb6._spectrum_from_checkpoint(ckpt)
    done = len(spectrum.weights)
    N = spectrum.n_expanded
    K = 12
    w0 = float(dist["onset"])                       # D/2 = 3
    half = int(round(w0))

    # Onset fraction f0 = |F(D/2)| / C(N, D/2). Ours (undercounted) vs the paper's Table-2 count.
    logC = gammaln(N + 1) - gammaln(half + 1) - gammaln(N - half + 1)
    f0_ours = float(dist["fail_count"]) / np.exp(logC)
    f0_paper = float(args.paper_fails) / np.exp(logC)
    print(f"N_exp={N}, C(N,{half})={np.exp(logC):.3e}")
    print(f"f0 ours  (|F|={dist['fail_count']:.3g}, |L(D)|={dist['n_min_logicals']}) = {f0_ours:.3e}")
    print(f"f0 paper (|F|={args.paper_fails:.3g}, Table 2)                = {f0_paper:.3e}"
          f"   ({f0_paper / f0_ours:.0f}x higher)")

    p = np.logspace(-5, -2, 60)
    fits = {}
    for model in ("f5", "f3"):
        fit = fit_failure_spectrum(spectrum, K=K, model=model, w0=w0, f0=f0_paper)
        fits[model] = (fit, logical_error_rate_from_ansatz(fit, list(p)))
        print(f"{model} (Table-2 pinned): "
              + ", ".join(f"{k}={v:.3g}" for k, v in fit.params.items())
              + f"  (cost={fit.cost:.3g})  LER(1e-4)={fits[model][1][np.searchsorted(p,1e-4)]:.2e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"f5": "darkorange", "f3": "crimson"}

    # ---------- LER(p) ----------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    raw = bb6.reweight_spectrum(spectrum, p)
    ax.plot(p, raw.P_logical, "o", color="steelblue", ms=3, alpha=0.45,
            label="ours: raw IS reweighted (partial)")
    for model, (fit, P) in fits.items():
        ax.plot(p, P, "-", color=colors[model], lw=2, label=f"ours: ansatz {model} (Table-2 pinned)")
    if PAPER_RELAY_P:
        ax.plot(PAPER_RELAY_P, PAPER_RELAY_LER, "k^", ms=9, label="paper Fig.10 BB(6)-relay (read off PDF)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-5, 1.2e-2); ax.set_ylim(1e-15, 2.0)
    ax.set_xlabel("Physical error rate $p$"); ax.set_ylabel("Logical error rate $P(p)$")
    ax.set_title(f"BB(6) Relay LER — Table-2-pinned ansatz ({done} weights, partial)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(args.outdir / "compare_vs_paper.png", dpi=150)
    print(f"wrote {args.outdir / 'compare_vs_paper.png'}")

    # ---------- failure spectrum f(w) ----------
    fig2, ax2 = plt.subplots(figsize=(8.5, 6))
    wq = np.asarray(spectrum.weights, float)
    T = np.asarray(spectrum.trials, float); Fc = np.asarray(spectrum.failures, float)
    fhat = Fc / T; se = np.sqrt(np.maximum(fhat * (1 - fhat), 1.0 / T) / T)
    msk = Fc > 0
    ax2.errorbar(wq[msk], fhat[msk], yerr=se[msk], fmt="o", color="steelblue", ms=4, capsize=2,
                 label=f"ours: sampled $f(w)$ relay ({int(msk.sum())} wts)")
    wgrid = np.arange(half, 64)
    for model, (fit, _P) in fits.items():
        ax2.plot(wgrid, fit.f(wgrid), "-", color=colors[model], lw=2,
                 label=f"ours: ansatz {model} (Table-2 pinned)")
    ax2.plot([w0], [f0_paper], "k*", ms=18, zorder=6,
             label=fr"paper Table-2 bound ($w_0$=3, $f_0$={f0_paper:.1e})")
    ax2.plot([w0], [f0_ours], "x", color="grey", ms=11, mew=2, zorder=6,
             label=fr"our undercounted bound ($f_0$={f0_ours:.1e}, |L(D)|={dist['n_min_logicals']})")
    if PAPER_RELAY_W:
        ax2.plot(PAPER_RELAY_W, PAPER_RELAY_FW, "k^", ms=9, label="paper BB(6)-relay $f(w)$ (read off PDF)")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(1, 1e3); ax2.set_ylim(1e-7, 2.0)
    ax2.set_xlabel("Fault weight $w$"); ax2.set_ylabel("Failure spectrum $f(w)$")
    ax2.set_title(f"BB(6) failure spectrum — Table-2-pinned ansatz ({done} weights, partial)")
    ax2.legend(fontsize=8, loc="lower right"); ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout(); fig2.savefig(args.outdir / "compare_spectrum_vs_paper.png", dpi=150)
    print(f"wrote {args.outdir / 'compare_spectrum_vs_paper.png'}")


if __name__ == "__main__":
    main()
