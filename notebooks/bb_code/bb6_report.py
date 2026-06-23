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
    # Prefer the exact-MITM Table 2 (distance_mitm.json) over the BP-OSD search (distance.json) for
    # BOTH the ansatz onset pin (f0/w0) and the Table-2 meta: the search can undercount |L(D)| even
    # when its saturation curve looks complete, so the exact enumeration wins and re-pins f0.
    _dm = outdir / "distance_mitm.json"; _ds = outdir / "distance.json"
    dist = json.loads((_dm if _dm.exists() else _ds).read_text()) if (_dm.exists() or _ds.exists()) else {}
    f0 = float(dist.get("onset_fraction", npz["onset_fraction"]))
    w0 = int(dist.get("onset", npz["onset"]))
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

    repro = None                              # multi-seed reproducibility of the replica-exchange run
    rp = outdir / "reproducibility.json"
    if rp.exists():
        repro = json.loads(rp.read_text())    # {seeds, p_ladder, mean, rel_spread, quoted_rel_se, ...}

    # Representation/Table-2 metadata: `dist` above already prefers the exact-MITM distance_mitm.json
    # over the search distance.json. The single-sector path is the exact-MITM-pinned EXACT constants.
    single_sector = bool(dist.get("single_sector", cfg.get("mw_single_sector", True)))
    meta = dict(single_sector=single_sector,
                code_label=cfg.get("code_label", "BB(6)=[[72,12,6]]"),
                method=dist.get("method", "mitm_exact" if dist.get("exact_pin") else "search"),
                D=int(dist.get("distance", 2 * w0)), w0=w0, N=N, f0=f0,
                n_compressed=dist.get("n_compressed"), n_expanded=dist.get("n_expanded", N),
                n_min_logicals=dist.get("n_min_logicals"),
                n_min_logicals_expanded=dist.get("n_min_logicals_expanded"),
                fail_count=dist.get("fail_count"))
    if single_sector and meta["n_compressed"] is None:   # exact-MITM-pinned single sector: use EXACT
        meta.update(method="mitm_exact", n_compressed=EXACT["n_compressed"],
                    n_expanded=EXACT["n_expanded"], n_min_logicals=EXACT["L_compressed"],
                    n_min_logicals_expanded=EXACT["L_expanded"], fail_count=EXACT["F"])

    search_conv = json.loads((outdir / "search_convergence.json").read_text()) \
        if (outdir / "search_convergence.json").exists() else None
    dconv = json.loads((outdir / "decoder_convergence.json").read_text()) \
        if (outdir / "decoder_convergence.json").exists() else None

    return dict(w=w, F=F, T=T, N=N, q_base=q_base, f0=f0, w0=w0, p_ref=p_ref,
                p_grid=p_grid, isp=isp, isL=isL, isSE=isSE, fits=fits, LER=LER, band=band,
                split=split, repro=repro, meta=meta, search_conv=search_conv, dconv=dconv)


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

    The point/error bar prefer the multi-seed reproducibility run when present: the plotted point is
    the across-seed MEAN and the bar is the empirical run-to-run spread (the *honest* uncertainty),
    which is ~2.8x the single-run walker-SE in splitting.json (that SE treats the swap-coupled
    walkers as independent and ignores decoder stochasticity, so it is optimistic). Falls back to
    the single committed run otherwise. The ansatz is evaluated at each rung's EXACT p (the ratio
    stored in splitting.json uses a coarse grid point and zig-zags). inside_bracket comes from the
    sequential under/over bracket (False deep sub-threshold, where those chains collapse)."""
    s = R["split"]
    if R.get("repro"):
        rp = R["repro"]
        tp = np.asarray(rp["p_ladder"], float)
        tP = np.asarray(rp["mean"], float)
        tSE = np.asarray(rp["rel_spread"], float) * tP          # run-to-run spread (absolute)
        bar = "spread"
    else:
        tp = np.asarray(s["tempered"]["p_ladder"], float)
        tP = np.asarray(s["tempered"]["P_logical"], float)
        tSE = np.asarray(s["tempered"]["P_logical_se"], float)
        bar = "se"
    ans = np.asarray(logical_error_rate_from_ansatz(R["fits"]["f3"], list(tp)))
    ratio = tP / np.maximum(ans, 1e-300)
    inb = np.array([r["inside_bracket"] for r in s["compare"]], dtype=bool)
    # "valid" = ansatz lies within the (honest) error bar, i.e. consistent within ~1 sigma.
    consistent = np.abs(tP - ans) <= np.maximum(tSE, 1e-300)
    return dict(p=tp, tP=tP, tSE=tSE, ansatz=ans, ratio=ratio, bar=bar,
                valid=(ratio >= 0.5) & (ratio <= 2.0), consistent=consistent, inside_bracket=inb)


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
        lbl = ("Technique III: replica-exchange (3-run mean ± run-to-run spread)"
               if c["bar"] == "spread" else "Technique III: replica-exchange splitting ± SE")
        if valid.any():
            # small marker + prominent caps so the (~4-16%) bars aren't hidden behind the squares
            # on the ~8-decade log axis; the bars are short because the estimate is that precise.
            ax.errorbar(tp[valid], tP[valid], yerr=tSE[valid], fmt="s", color="seagreen", ms=3.5,
                        capsize=4, elinewidth=1.3, capthick=1.3, ecolor="darkslategray", zorder=6,
                        label=lbl)
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
    at physical rate p. The reweighting (binomial factor) is analytic; the f(w) input is the f3
    ANSATZ (pinned nonzero at the onset w0), so pi_q(w) extends correctly to the onset and the
    median bends toward w0 at low p — unlike the raw *measured* f(w), which is truncated at the
    lowest sampled-with-failures weight (w=6) and would floor the median there. This is the
    distribution the splitting chains should visit (cf. paper Fig 9c)."""
    w = np.arange(int(R["w0"]), 60, dtype=float)              # ansatz is defined for all w >= w0
    f = R["fits"]["f3"].f(w)
    N = R["N"]; q = R["q_base"] * (p / R["p_ref"])
    logpi = (np.log(np.maximum(f, 1e-300)) + gammaln(N + 1) - gammaln(w + 1) - gammaln(N - w + 1)
             + w * np.log(q) + (N - w) * np.log1p(-q))
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
    csum = np.cumsum(pi)
    return [float(np.interp(qq, csum, w)) for qq in qs]


def fig_weight_vs_p(R, ax=None):
    """Fig-9(c)-style: failing-config weight vs physical error rate. pi_q(w) band = the f3 ANSATZ
    f(w) reweighted analytically by the binomial weight factor (semi-analytic: analytic reweighting,
    ansatz-fit input) + the replica-exchange chains' mean weight per ladder rate."""
    import matplotlib.pyplot as plt
    if ax is None: _, ax = plt.subplots(figsize=(8.4, 5.6))
    pg = np.geomspace(1e-4, 1e-2, 60)
    P = np.array([_weight_percentiles(R, p) for p in pg])  # (60,5): 10,25,50,75,90
    ax.fill_between(pg, P[:, 0], P[:, 4], color="steelblue", alpha=0.18, label=r"reweighted ansatz $\pi_q(w)$ 10–90%")
    ax.fill_between(pg, P[:, 1], P[:, 3], color="steelblue", alpha=0.30, label=r"reweighted ansatz $\pi_q(w)$ 25–75%")
    ax.plot(pg, P[:, 2], "-", color="steelblue", lw=2, label=r"reweighted ansatz median weight")
    if R["split"]:
        tp = np.array(R["split"]["tempered"]["p_ladder"]); mw = np.array(R["split"]["diagnostics"]["mean_weight"])
        ax.plot(tp, mw, "s", color="seagreen", ms=7, mec="k", mew=0.4, zorder=5,
                label="replica-exchange chain mean weight")
    ax.axhline(R["w0"], color="darkorange", ls="--", lw=1.1, label=fr"min-weight onset $w_0={R['w0']}$")
    ax.set_xscale("log"); ax.set_xlabel(r"Physical error rate $p$"); ax.set_ylabel("failing-config weight $w$")
    ax.set_title("BB(6) — weight distribution of failing configs (cf. paper Fig. 9c)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, which="both", alpha=0.3)
    return ax


def fig_search_convergence(R, ax=None):
    """Search saturation: |L(D)| found vs cumulative BP-OSD search trials. The plateau LOOKS like
    completeness — but where an exact-MITM count is available it is overlaid, and on the full DEM it
    exposes that the BP-OSD search saturates well *below* the true |L(D)| (BP-OSD is not a guaranteed
    min-weight decoder), so the exact enumeration — not the search plateau — is authoritative."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8.4, 5.2))
    sc = R.get("search_conv")
    if not sc or not sc.get("trace"):
        ax.text(0.5, 0.5, "no search_convergence.json\n(Table 2 was exact-MITM pinned)",
                ha="center", va="center", transform=ax.transAxes)
        return ax
    tr = np.asarray(sc["trace"], dtype=float)
    x, y = tr[:, 0], tr[:, 1]
    total = sc.get("n_systematic", 0) + sc.get("max_trials", 0)
    if total > x[-1]:                                   # extend to the full budget so the plateau shows
        x = np.append(x, total); y = np.append(y, y[-1])
    ax.step(x, y, where="post", color="teal", lw=2, label="|L(D)| found (BP-OSD search)")
    nsys = sc.get("n_systematic", 0)
    if nsys:
        ax.axvline(nsys, color="gray", ls=":", lw=1, label=f"systematic→random ({nsys})")
    ax.axhline(y[-1], color="teal", ls="--", lw=1, label=f"search plateau = {int(sc['final'])}")
    # Overlay the EXACT MITM |L(D)| when it is the authoritative count (full-DEM mitm_exact) and the
    # search undershot it — the gap is the search's incompleteness.
    meta = R.get("meta") or {}
    exact = meta.get("n_min_logicals")
    if meta.get("method") == "mitm_exact" and exact and abs(exact - y[-1]) > 0.01 * max(exact, 1):
        ax.axhline(exact, color="crimson", ls="-", lw=2,
                   label=f"exact MITM |L(D)| = {int(exact)}  ← search found only {int(y[-1])} ({100*y[-1]/exact:.0f}%)")
        ax.set_ylim(0, exact * 1.08)
    rep = "single-sector" if sc.get("single_sector") else "full DEM"
    ax.set_xlabel("cumulative search trials"); ax.set_ylabel("|L(D)| found")
    ax.set_title(f"Min-weight search vs exact MITM ({rep})")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
    return ax


def fig_decoder_convergence(R, ax=None):
    """Relay-BP convergence on the full DEM: logical error rate (left) and disagreement with the
    most-legs decoder (right) vs the number of relay legs (num_sets). LER plateaus and the
    disagreement → 0 as legs grow, showing the decoder has enough legs to be reliable."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8.4, 5.2))
    dc = R.get("dconv")
    if not dc or not dc.get("rows"):
        ax.text(0.5, 0.5, "no decoder_convergence.json", ha="center", va="center",
                transform=ax.transAxes)
        return ax
    rows = dc["rows"]
    ns = np.array([r["num_sets"] for r in rows], float)
    ler = np.array([r["ler"] for r in rows], float)
    se = np.array([r.get("ler_se", 0.0) for r in rows], float)
    ax.errorbar(ns, ler, yerr=se, fmt="o-", color="darkviolet", capsize=3, label="logical error rate")
    ax.set_xscale("log"); ax.set_xlabel("Relay-BP legs (num_sets)"); ax.set_ylabel("logical error rate")
    ax.set_title(f"Relay-BP decoder convergence (full DEM, p={dc.get('p')})")
    ax2 = ax.twinx()
    dis = np.array([r.get("disagree_with_best", np.nan) for r in rows], float)
    ax2.plot(ns, dis, "s--", color="darkorange", ms=4, alpha=0.8, label="disagreement vs most-legs")
    ax2.set_ylabel("fraction disagreeing with best", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right"); ax.grid(True, which="both", alpha=0.3)
    return ax


def fig_compare_ler(R_single, R_full, ax=None):
    """Overlay the LER(p) curves (f3 ansatz + IS points) for the single-sector and full-DEM runs."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8.6, 5.6))
    for R, col, tag in ((R_single, "crimson", "single-sector (Z)"), (R_full, "navy", "full DEM")):
        ax.plot(R["p_grid"], R["LER"]["f3"], "-", color=col, lw=2,
                label=f"{R['meta']['code_label']} · {tag} (f3)")
        ax.plot(R["isp"], R["isL"], "o", color=col, ms=3, alpha=0.45)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Physical error rate $p$"); ax.set_ylabel("Logical error rate")
    ax.set_title("LER(p): single-sector vs full DEM")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
    return ax


def table2_compare(R_single, R_full):
    """Markdown side-by-side of the Table-2 min-weight quantities (single-sector vs full DEM)."""
    def g(m, k):
        v = m.get(k)
        return "—" if v is None else (f"{v:.4g}" if isinstance(v, (int, float)) else str(v))
    ms, mf = R_single["meta"], R_full["meta"]
    rows = [("method", ms["method"], mf["method"]),
            ("distance D", g(ms, "D"), g(mf, "D")),
            ("onset w0", g(ms, "w0"), g(mf, "w0")),
            ("n_compressed", g(ms, "n_compressed"), g(mf, "n_compressed")),
            ("n_expanded", g(ms, "n_expanded"), g(mf, "n_expanded")),
            ("|L(D)| comp", g(ms, "n_min_logicals"), g(mf, "n_min_logicals")),
            ("|L(D)| exp", g(ms, "n_min_logicals_expanded"), g(mf, "n_min_logicals_expanded")),
            ("|F(D/2)|", g(ms, "fail_count"), g(mf, "fail_count")),
            ("f0", g(ms, "f0"), g(mf, "f0"))]
    head = f"| quantity | {ms['code_label']} single-sector | {mf['code_label']} full DEM |"
    return "\n".join([head, "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows])


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
