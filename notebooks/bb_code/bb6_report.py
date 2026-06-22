"""Analysis + figures for the BB(6) [[72,12,6]] Figure-10 reproduction report.

Loads the production single-sector IS sweep (bb6_fig10_curve/) and the splitting cross-check,
fits BOTH the f3 and f5 failure-spectrum ansatze (pinned by the exact Technique-II onset),
and builds the two report figures:

  fig_ler_vs_p          — logical error rate vs physical error rate (the Figure-10 curve)
  fig_failure_spectrum  — failure fraction f(w) vs fault weight (the weight-space view)

The Jupyter report (BB6_failfast_report.ipynb) imports these so its cells call tested code.
"""
import json, pathlib, warnings
import numpy as np
from scipy.special import gammaln

import sys
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))
from importance_sampling import (FailureSpectrum, fit_failure_spectrum,
                                 logical_error_rate_from_ansatz, failure_spectrum_ansatz)

DEFAULT_OUT = _HERE / "bb6_fig10_curve"
K = 12
# Exact Technique-II onset (bb6_exact_enum_mitm.py) — pins the ansatz; these are exact counts.
EXACT = dict(D=6, n_compressed=2233, n_expanded=46224, L_compressed=1524,
             L_expanded=6.0112e12, F=382503744, f0=2.3239e-05, w0=3)
PAPER = dict(n_expanded=46224, L_expanded=6.01e12, F=3.83e8, f0=2.33e-5, ler_1e4=1e-7)


def compute(outdir=DEFAULT_OUT, bootstrap=150, seed=0):
    """Load results, fit f3 & f5 ansatze (pinned w0/f0), compute LER(p) + bootstrap bands.

    `bootstrap` resampled fits per model set the 5/95 confidence bands; 150 keeps the call to
    ~12-15s (the f5 fits dominate). Pass a larger value for smoother bands, 0 to skip the bands.
    """
    outdir = pathlib.Path(outdir)
    npz = np.load(outdir / "bb6_fig10.npz")
    cfg = json.loads((outdir / "config.json").read_text()) if (outdir / "config.json").exists() else {}
    p_ref = float(cfg.get("p_ref", 0.003))
    w = np.asarray(npz["spectrum_weights"]); F = np.asarray(npz["spectrum_failures"]); T = np.asarray(npz["spectrum_trials"])
    N = int(npz["n_expanded"]); q_base = float(npz["q_base"])
    f0 = float(npz["onset_fraction"]); w0 = int(npz["onset"])
    p_grid = np.asarray(npz["ansatz_p"])
    isp, isL, isSE = np.asarray(npz["p_values"]), np.asarray(npz["is_P_logical"]), np.asarray(npz["is_P_logical_se"])

    def spectrum(failures):
        return FailureSpectrum(weights=list(w), trials=list(T), failures=list(failures),
                               n_expanded=N, q_base=q_base, p_ref=p_ref)

    fits, LER, band = {}, {}, {}
    # The f5 ansatz evaluates q**w for large w during fitting, which harmlessly overflows on some
    # bootstrap resamples (caught below); silence those RuntimeWarnings so they don't flood the
    # notebook (and so the warning machinery doesn't dominate the bootstrap runtime).
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for model in ("f3", "f5"):
            fit = fit_failure_spectrum(spectrum(F), K=K, model=model, w0=float(w0), f0=f0)
            fits[model] = fit
            LER[model] = np.asarray(logical_error_rate_from_ansatz(fit, list(p_grid)))
            # bootstrap band over resampled failures
            rng = np.random.default_rng(seed)
            bs = np.full((bootstrap, len(p_grid)), np.nan)
            for b in range(bootstrap):
                fr = rng.binomial(T.astype(int), np.clip(F / np.maximum(T, 1), 0, 1))
                try:
                    fb = fit_failure_spectrum(spectrum(fr), K=K, model=model, w0=float(w0), f0=f0)
                    bs[b] = logical_error_rate_from_ansatz(fb, list(p_grid))
                except Exception:
                    pass
            band[model] = (np.nanpercentile(bs, 5, axis=0), np.nanpercentile(bs, 95, axis=0))

    split = None
    sp = outdir / "splitting.json"
    if sp.exists():
        split = json.loads(sp.read_text())   # new schema: {tempered, bracket, diagnostics, compare}

    return dict(w=w, F=F, T=T, N=N, q_base=q_base, f0=f0, w0=w0, p_ref=p_ref,
                p_grid=p_grid, isp=isp, isL=isL, isSE=isSE, fits=fits, LER=LER, band=band,
                split=split)


def onset_LER(p, R):
    """Technique-II leading-order onset prediction: C(N,w0) q^w0 (1-q)^(N-w0) f0."""
    q = R["q_base"] * (np.asarray(p) / R["p_ref"]); N, w0, f0 = R["N"], R["w0"], R["f0"]
    logC = gammaln(N + 1) - gammaln(w0 + 1) - gammaln(N - w0 + 1)
    return np.exp(logC + w0 * np.log(q) + (N - w0) * np.log1p(-q)) * f0


def is_zero_count_upper(R, p_values=None):
    """Rule-of-three (3/T) upper-limit contribution to the IS LER from zero-count weight bins.

    The IS estimate is Σ_w C(N,w) q^w (1-q)^(N-w) f̂(w) over the *sampled* weights, with
    f̂(w)=F(w)/T(w). A bin with F(w)=0 contributes 0 to that sum AND 0 to its binomial SE
    (√(f̂(1-f̂)/T)=0 at f̂=0), so it is invisible to a count-based error bar even though its
    true failure fraction is only bounded by f ≤ 3/T (the 95% rule of three). At deep
    sub-threshold the dominant low-weight bins (w=3,4,5) are exactly these empty bins, which is
    why the plain IS bar cannot reach the onset-pinned ansatz. This returns, for each p, the
    extra LER those empty bins could add if each sat at its 3/T upper limit — a one-sided
    *upper* allowance (empty bins cannot lower the estimate)."""
    if p_values is None:
        p_values = R["isp"]
    w, F, T, N, w0 = R["w"].astype(float), np.asarray(R["F"]), R["T"].astype(float), R["N"], R["w0"]
    # Only w >= w0 get an allowance: Technique-II enumeration proves f(w)=0 exactly below the
    # onset (no logical fault of weight < w0 exists), so those empty bins are true zeros, not
    # undersampling — they must not contribute a 3/T upper limit.
    zero = (F == 0) & (T > 0) & (w >= w0)
    p_arr = np.asarray(p_values, dtype=float)
    if not zero.any():
        return np.zeros_like(p_arr)
    wz, f_upper = w[zero], 3.0 / T[zero]                       # rule of three at 0 counts
    logC = gammaln(N + 1) - gammaln(wz + 1) - gammaln(N - wz + 1)
    q = np.clip(R["q_base"] * (p_arr / R["p_ref"]), 1e-300, 1.0 - 1e-15)
    logterm = logC[:, None] + wz[:, None] * np.log(q)[None, :] + (N - wz)[:, None] * np.log1p(-q)[None, :]
    return (f_upper[:, None] * np.exp(logterm)).sum(axis=0)


def splitting_comparison(R):
    """Per-rung Technique-III (replica-exchange) vs Technique-I comparison, de-aliased.

    The ansatz is evaluated at each rung's EXACT p — the ratio stored in splitting.json uses the
    nearest *coarse* ansatz-grid point and so zig-zags. Returns arrays p, tP, tSE, ansatz, ratio,
    valid (within 2x of the ansatz) and inside_bracket (whether the sequential under/over bracket
    still brackets the estimate — False deep sub-threshold, where the sequential chains collapse)."""
    s = R["split"]
    tp = np.asarray(s["tempered"]["p_ladder"], float)
    tP = np.asarray(s["tempered"]["P_logical"], float)
    tSE = np.asarray(s["tempered"]["P_logical_se"], float)
    ans = np.asarray(logical_error_rate_from_ansatz(R["fits"]["f3"], list(tp)))
    ratio = tP / np.maximum(ans, 1e-300)
    inb = np.array([r["inside_bracket"] for r in s["compare"]], dtype=bool)
    return dict(p=tp, tP=tP, tSE=tSE, ansatz=ans, ratio=ratio,
                valid=(ratio >= 0.5) & (ratio <= 2.0), inside_bracket=inb)


def fig_ler_vs_p(R, ax=None):
    import matplotlib.pyplot as plt
    if ax is None: _, ax = plt.subplots(figsize=(8.4, 5.6))
    p = R["p_grid"]
    # IS error bars: binomial SE is ~equal to the value at low p (the estimate is dominated by a
    # few rare low-weight failures). A symmetric bar then reaches ~0 (misleading on a log axis), so
    # we draw it one-sided (upper only) wherever SE >= 0.5*value, keeping symmetric bars elsewhere.
    isL, isSE = R["isL"], R["isSE"]
    rel = isSE / np.maximum(isL, 1e-300)
    lo_err = np.where(rel < 0.5, isSE, 0.0)
    # Upper bar = statistical SE of the measured bins + rule-of-three (3/T) allowance for the
    # zero-count weight bins the estimator silently dropped. At low p this is dominated by the
    # empty w=3,4,5 bins and (correctly) reaches up past the ansatz — the data are consistent
    # with it; they just can't confirm it. At high p the dominant weights have counts, so the
    # allowance vanishes and the bar reverts to the plain SE.
    hi_err = isSE + is_zero_count_upper(R)
    ax.errorbar(R["isp"], isL, yerr=[lo_err, hi_err], fmt="o", color="steelblue", ms=4, capsize=2,
                lw=0.8, label="Technique I: IS reweighted (upper bar adds 3/T limit on zero-count weights)", zorder=4)
    for model, col in (("f3", "crimson"), ("f5", "purple")):
        lo, hi = R["band"][model]
        ax.fill_between(p, lo, hi, color=col, alpha=0.15)
        ax.plot(p, R["LER"][model], "-", color=col, lw=2,
                label=f"Technique I: {model} ansatz (LER(1e-4)={R['LER'][model][np.argmin(abs(p-1e-4))]:.2e})")
    # Technique II (the exact onset w0,f0) lives in weight x failure-fraction space and is shown on
    # the failure-spectrum / weight-distribution figures, not on this LER-vs-p plot.
    if R["split"]:
        s = R["split"]
        # The replica-exchange estimate is the result; the sequential under/over bracket is a
        # near-threshold mixing check that COLLAPSES deep sub-threshold (the trapping replica
        # exchange cures), so clip its band to the rungs where it actually brackets rather than
        # sweeping it down to ~1e-18. Validity is de-aliased agreement with the ansatz.
        c = splitting_comparison(R)
        tp, tP, tSE, valid = c["p"], c["tP"], c["tSE"], c["valid"]
        bp = np.array(s["bracket"]["p_ladder"]); blo = np.array(s["bracket"]["lo"]); bhi = np.array(s["bracket"]["hi"])
        inb = c["inside_bracket"]
        if inb.any():
            ax.fill_between(bp[inb], blo[inb], bhi[inb], color="seagreen", alpha=0.12,
                            label="Technique III: splitting bracket (near-threshold mixing check)")
        if valid.any():
            ax.errorbar(tp[valid], tP[valid], yerr=tSE[valid], fmt="s", color="seagreen", ms=6, capsize=3,
                        label="Technique III: replica-exchange splitting ± SE")
        if (~valid).any():
            ax.plot(tp[~valid], tP[~valid], "s", mfc="none", mec="seagreen", ms=6,
                    label="Technique III: replica-exchange (off ansatz >2x)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Physical error rate $p$"); ax.set_ylabel("Logical error rate")
    ax.set_title("BB(6) [[72,12,6]] — logical vs physical error rate (Figure 10)")
    ax.legend(fontsize=7.5, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
    return ax


def fig_failure_spectrum(R, ax=None):
    import matplotlib.pyplot as plt
    if ax is None: _, ax = plt.subplots(figsize=(8.4, 5.6))
    w = np.asarray(R["w"]); F = np.asarray(R["F"]); T = np.asarray(R["T"])
    f = F / np.maximum(T, 1); fse = np.sqrt(np.clip(f * (1 - f), 0, None) / np.maximum(T, 1))
    meas = F > 0                      # weights with >=1 observed failure: real measurements
    ax.errorbar(w[meas], f[meas], yerr=fse[meas], fmt="o", color="steelblue", ms=4, capsize=2,
                label="sampled $f(w)$ ± binomial SE", zorder=4)
    if (~meas).any():                 # 0 failures -> not a measurement; show 95% upper limit 3/T
        ul = 3.0 / np.maximum(T[~meas], 1)
        ax.plot(w[~meas], ul, "v", color="gray", ms=6, alpha=0.7,
                label="0 failures in $T$ shots: 95% upper limit ($3/T$)")
    ww = np.linspace(R["w0"], w.max(), 300)
    for model, col in (("f3", "crimson"), ("f5", "purple")):
        fw = failure_spectrum_ansatz(ww, a=R["fits"][model].a, model=model, **R["fits"][model].params)
        ax.plot(ww, fw, "-", color=col, lw=2, label=f"{model} ansatz fit")
    ax.axvline(R["w0"], color="darkorange", ls="--", lw=1.1)
    ax.plot([R["w0"]], [R["f0"]], "*", color="darkorange", ms=16, mec="k", mew=0.5, zorder=5,
            label=fr"Technique II onset $(w_0{{=}}{R['w0']},\ f_0{{=}}{R['f0']:.3g})$ exact")
    ax.set_yscale("log"); ax.set_xlabel("Fault weight $w$"); ax.set_ylabel(r"failure fraction $f(w)$")
    ax.set_title("BB(6) — failure spectrum $f(w)$ (weight-space view)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
    return ax


def _weight_percentiles(R, p, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """Percentiles of the failing-config weight distribution pi_q(w) ~ f(w) C(N,w) q^w (1-q)^(N-w)
    at physical rate p, computed analytically from the measured f(w). This is the distribution the
    splitting chains should visit (cf. paper Fig 9c)."""
    w = R["w"].astype(float); f = R["F"] / np.maximum(R["T"], 1)
    N = R["N"]; q = R["q_base"] * (p / R["p_ref"])
    logpi = (np.log(np.maximum(f, 1e-300)) + gammaln(N + 1) - gammaln(w + 1) - gammaln(N - w + 1)
             + w * np.log(q) + (N - w) * np.log1p(-q))
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
    csum = np.cumsum(pi)
    return [float(np.interp(qq, csum, w)) for qq in qs]


def fig_weight_vs_p(R, ax=None):
    """Fig-9(c)-style: failing-config weight vs physical error rate. Analytic pi_q(w) band (from
    measured f(w)) + the replica-exchange chains' mean weight per ladder rate."""
    import matplotlib.pyplot as plt
    if ax is None: _, ax = plt.subplots(figsize=(8.4, 5.6))
    pg = np.geomspace(1e-4, 1e-2, 60)
    P = np.array([_weight_percentiles(R, p) for p in pg])  # (60,5): 10,25,50,75,90
    ax.fill_between(pg, P[:, 0], P[:, 4], color="steelblue", alpha=0.18, label=r"analytic $\pi_q(w)$ 10–90%")
    ax.fill_between(pg, P[:, 1], P[:, 3], color="steelblue", alpha=0.30, label=r"analytic $\pi_q(w)$ 25–75%")
    ax.plot(pg, P[:, 2], "-", color="steelblue", lw=2, label=r"analytic median weight")
    if R["split"]:
        tp = np.array(R["split"]["tempered"]["p_ladder"]); mw = np.array(R["split"]["diagnostics"]["mean_weight"])
        ax.plot(tp, mw, "s", color="seagreen", ms=7, mec="k", mew=0.4, zorder=5,
                label="replica-exchange chain mean weight")
    ax.axhline(R["w0"], color="darkorange", ls="--", lw=1.1, label=fr"min-weight onset $w_0={R['w0']}$")
    ax.set_xscale("log"); ax.set_xlabel(r"Physical error rate $p$"); ax.set_ylabel("failing-config weight $w$")
    ax.set_title("BB(6) — weight distribution of failing configs (cf. paper Fig. 9c)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, which="both", alpha=0.3)
    return ax


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = compute()
    print("f3 params:", {k: round(v, 4) for k, v in R["fits"]["f3"].params.items()})
    print("f5 params:", {k: round(v, 4) for k, v in R["fits"]["f5"].params.items()})
    for m in ("f3", "f5"):
        print(f"  {m}: LER(1e-4) = {R['LER'][m][np.argmin(abs(R['p_grid']-1e-4))]:.3e}")
    fig, ax = plt.subplots(figsize=(8.4, 5.6)); fig_ler_vs_p(R, ax)
    fig.tight_layout(); fig.savefig(DEFAULT_OUT / "fig_ler_vs_p.png", dpi=150)
    fig, ax = plt.subplots(figsize=(8.4, 5.6)); fig_failure_spectrum(R, ax)
    fig.tight_layout(); fig.savefig(DEFAULT_OUT / "fig_failure_spectrum.png", dpi=150)
    print("wrote fig_ler_vs_p.png, fig_failure_spectrum.png")


if __name__ == "__main__":
    main()
