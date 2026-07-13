"""Cross-run Λ / error-budget analysis: ratios with honest uncertainties.

Extracted from the validated K=4 comparison notebook (the codegen in
``experiments/methods/make_error_model_comparison.py``) so the K=12 campaign's gate checks and
budget tables use ONE audited implementation. Everything here consumes measured
:class:`FailureSpectrum` objects (from ``importance_sampling`` or an ``experiment_runner``
checkpoint) and returns values with propagated uncertainty — the campaign's hard-won rules are
baked in rather than remembered:

* Every reweighted point value carries a binomial ±σ AND a zero-failure-bin "headroom" (the
  rule-of-three bound on how much the value could rise if every sampled-but-empty bin sat at
  its upper limit). Sub-onset zero bins contribute exactly 0, so reweighted values are lower
  bounds; headroom is the size of that exposure.
* Marginal-Λ contributions are differences of noisy ratios — a NEGATIVE share is physics only
  when it survives both the ±σ and the headroom interval (:func:`verdict`).
* Strided weight tails undercount curve-level reweighting by ~stride× at high p:
  :func:`fill_spectrum` (pooled-neighbor gap fill) is applied by :func:`reweight_filled` and
  everything built on it. Point values at low p (mass in a contiguous head) are unaffected.
* Reweighting is unbiased only while the binomial mass stays inside the sampled window:
  :func:`mass_window_p_max` gives the largest trustworthy p; curve/crossing helpers REFUSE to
  extrapolate beyond it (they return None / mask instead).
* Per-round(-cycle) ε and Λ conventions live in exactly one place: :func:`per_round` and
  :func:`cycles_of` (memory: QEC rounds; LPU ops: the repeated-measurement rounds ``lpu_C``).
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from importance_sampling import FailureSpectrum, reweight_spectrum

__all__ = [
    "fill_spectrum", "pool_spectra", "reweight_filled", "rw_stats", "eps_stats", "per_round",
    "cycles_of", "mass_window_p_max", "zero_bin_fraction", "InvLambda", "inv_lambda_stats",
    "crossing_p", "pseudo_threshold", "lambda_curve", "threshold_from_crossing", "verdict",
    "lambda_decomposition", "Run", "load_run",
]


# ================================ spectra =====================================
def fill_spectrum(spec: FailureSpectrum) -> FailureSpectrum:
    """Insert every missing tail weight with its neighbors' pooled counts.

    ``reweight_spectrum`` sums only sampled weights, so a stride-S tail undercounts LER by
    ~S× wherever the binomial mass sits inside the strided region. f(w) varies slowly in w,
    so nearest-neighbor pooling restores the mass to first order. Gaps of any stride are
    filled; already-contiguous spectra come back unchanged.
    """
    W, T, F = list(spec.weights), list(spec.trials), list(spec.failures)
    w_out: List[int] = []
    t_out: List[int] = []
    f_out: List[int] = []
    for i, (w, t, f) in enumerate(zip(W, T, F)):
        w_out.append(w); t_out.append(t); f_out.append(f)
        if i + 1 < len(W) and W[i + 1] > w + 1:
            t2, f2 = t + T[i + 1], f + F[i + 1]
            for wm in range(w + 1, W[i + 1]):
                w_out.append(wm); t_out.append(t2); f_out.append(f2)
    return FailureSpectrum(weights=w_out, trials=t_out, failures=f_out,
                           n_expanded=spec.n_expanded, q_base=spec.q_base, p_ref=spec.p_ref)


def pool_spectra(*specs: FailureSpectrum) -> FailureSpectrum:
    """Pool per-weight (trials, failures) across runs of the SAME circuit (boost passes).

    Valid only when every run sampled the same expanded representation (n_expanded, q_base,
    p_ref identical) with independent seeds; raises otherwise.
    """
    if not specs:
        raise ValueError("pool_spectra needs at least one spectrum")
    head = specs[0]
    for s in specs[1:]:
        if (s.n_expanded, s.q_base, s.p_ref) != (head.n_expanded, head.q_base, head.p_ref):
            raise ValueError("pool_spectra: spectra come from different expanded representations "
                             f"({(s.n_expanded, s.q_base, s.p_ref)} vs "
                             f"{(head.n_expanded, head.q_base, head.p_ref)})")
    T: Dict[int, int] = {}
    F: Dict[int, int] = {}
    for s in specs:
        for w, t, f in zip(s.weights, s.trials, s.failures):
            T[w] = T.get(w, 0) + int(t)
            F[w] = F.get(w, 0) + int(f)
    ws = sorted(T)
    return FailureSpectrum(weights=ws, trials=[T[w] for w in ws], failures=[F[w] for w in ws],
                           n_expanded=head.n_expanded, q_base=head.q_base, p_ref=head.p_ref)


def reweight_filled(spec: FailureSpectrum, p_values) -> np.ndarray:
    """Stride-safe reweighted LER over p_values (gap-filled first)."""
    return np.asarray(reweight_spectrum(fill_spectrum(spec), p_values).P_logical)


# ============================ point values + errors ===========================
def rw_stats(spec: FailureSpectrum, p: float) -> tuple:
    """Reweighted LER at scalar p: (value, statistical SE, zero-bin headroom).

    SE propagates the per-bin binomial errors of the sampled f(w). ``headroom`` is how much
    the value could RISE if every sampled-but-zero-failure bin sat at its rule-of-three upper
    bound f(w) < 3/T(w) — the truncation exposure of the lower-bound estimate at this p.
    """
    spec = fill_spectrum(spec)
    v = reweight_spectrum(spec, [p])
    up = FailureSpectrum(weights=spec.weights, trials=spec.trials,
                         failures=[f if f > 0 else min(3, t) for f, t in zip(spec.failures, spec.trials)],
                         n_expanded=spec.n_expanded, q_base=spec.q_base, p_ref=spec.p_ref)
    head = float(reweight_spectrum(up, [p]).P_logical[0]) - float(v.P_logical[0])
    return float(v.P_logical[0]), float(v.P_logical_se[0]), head


def per_round(LER, rounds: float):
    """Per-round(-cycle) logical error rate ε = 1 − (1−LER)^(1/rounds)."""
    return 1.0 - (1.0 - np.clip(np.asarray(LER, dtype=float), 0.0, 1.0 - 1e-12)) ** (1.0 / rounds)


def eps_stats(spec: FailureSpectrum, p: float, rounds: float) -> tuple:
    """Per-round ε at p with SE and headroom (delta method through the per-round transform)."""
    L, se, head = rw_stats(spec, p)
    g = (1.0 - min(L, 1.0 - 1e-12)) ** (1.0 / rounds - 1.0) / rounds
    return float(per_round(L, rounds)), g * se, g * head


def cycles_of(config: dict) -> int:
    """THE per-op ε convention: cycles that normalize ε for a run's experiment kind.

    memory → QEC rounds; lpu_* (and future LPU operations) → the repeated-measurement rounds
    ``lpu_C``. Keep every ladder/budget script on this one function so conventions can't fork.
    """
    if str(config.get("experiment", "memory")).startswith("lpu") or \
       str(config.get("experiment", "")) in ("automorphism", "joint_pauli"):
        return int(config["lpu_C"])
    return int(config["rounds"])


# ============================ validity windows ================================
def mass_window_p_max(spec: FailureSpectrum, n_sigma: float = 4.0) -> float:
    """Largest p at which reweighting the sampled window is trustworthy.

    Solves μ(p) + n_sigma·√μ(p) = w_max_sampled with μ(p) = N_expanded · q_base · p / p_ref.
    Beyond this p the binomial mass leaks above the sampled window and the reweighted value
    sags (window truncation, not physics).
    """
    w_max = max(spec.weights)
    # solve mu + n*sqrt(mu) = w_max  ->  sqrt(mu) = (-n + sqrt(n^2 + 4 w_max)) / 2
    mu_max = ((-n_sigma + math.sqrt(n_sigma * n_sigma + 4.0 * w_max)) / 2.0) ** 2
    return mu_max / (spec.n_expanded * spec.q_base) * spec.p_ref


def zero_bin_fraction(spec: FailureSpectrum) -> float:
    """Fraction of sampled bins with zero failures (G1's onset-resolution gate reads this)."""
    return sum(1 for f in spec.failures if f == 0) / len(spec.failures)


# ================================ ratios ======================================
@dataclass
class InvLambda:
    """1/Λ = ε_large/ε_small at one p, with its uncertainty budget."""
    value: float
    se: float          # propagated binomial SE
    lo: float          # zero-bin truncation interval (large-code headroom pushes value UP,
    hi: float          # small-code headroom pushes it DOWN — lo/hi span both)


def inv_lambda_stats(spec_small: FailureSpectrum, spec_large: FailureSpectrum, p: float,
                     rounds_small: float, rounds_large: float) -> InvLambda:
    e_s, se_s, h_s = eps_stats(spec_small, p, rounds_small)
    e_l, se_l, h_l = eps_stats(spec_large, p, rounds_large)
    inv = e_l / e_s
    se = inv * float(np.hypot(se_s / e_s, se_l / e_l))
    return InvLambda(value=inv, se=se, lo=e_l / (e_s + h_s), hi=(e_l + h_l) / e_s)


def crossing_p(pg, y1, y2) -> Optional[float]:
    """p where y1(p) = y2(p), log-log interpolated on the grid; None if no sign change."""
    pg = np.asarray(pg, dtype=float)
    r = np.log(np.maximum(np.asarray(y1, dtype=float), 1e-300)) \
        - np.log(np.maximum(np.asarray(y2, dtype=float), 1e-300))
    s = np.nonzero(np.diff(np.sign(r)) != 0)[0]
    if s.size == 0:
        return None
    i = s[-1]
    t = r[i] / (r[i] - r[i + 1])
    return float(np.exp(np.log(pg[i]) + t * (np.log(pg[i + 1]) - np.log(pg[i]))))


def pseudo_threshold(pg, LER) -> Optional[float]:
    """Break-even LER(p) = p (single-code threshold stand-in)."""
    return crossing_p(pg, LER, np.asarray(pg, dtype=float))


def lambda_curve(spec_small: FailureSpectrum, spec_large: FailureSpectrum, p_grid,
                 rounds_small: float, rounds_large: float):
    """(p_valid, Λ(p), ε_small, ε_large) over the grid, MASKED to both runs' mass windows.

    Curve-level values use gap-filled reweighting. The mask (not an extrapolation) enforces
    the trust window; callers that want the invalid region must compute it deliberately.
    """
    p_grid = np.asarray(p_grid, dtype=float)
    p_ok = min(mass_window_p_max(spec_small), mass_window_p_max(spec_large))
    m = p_grid <= p_ok * (1 + 1e-12)
    eps_s = per_round(reweight_filled(spec_small, p_grid[m]), rounds_small)
    eps_l = per_round(reweight_filled(spec_large, p_grid[m]), rounds_large)
    lam = np.maximum(eps_s, 1e-300) / np.maximum(eps_l, 1e-300)
    return p_grid[m], lam, eps_s, eps_l


def threshold_from_crossing(spec_small: FailureSpectrum, spec_large: FailureSpectrum, p_grid,
                            rounds_small: float, rounds_large: float) -> dict:
    """True threshold p_th from the ε_small = ε_large crossing inside the trust window.

    Returns {"p_th": float|None, "p_max_valid": float, "bounded": bool} — p_th None with
    bounded=True means "no crossing below p_max_valid" (report as > p_max_valid; NEVER
    extrapolate past the sampled mass to manufacture one).
    """
    p, lam, eps_s, eps_l = lambda_curve(spec_small, spec_large, p_grid,
                                        rounds_small, rounds_large)
    pth = crossing_p(p, eps_s, eps_l)
    return {"p_th": pth, "p_max_valid": float(p.max()) if p.size else float("nan"),
            "bounded": pth is None}


# ============================ decompositions ==================================
def verdict(contribution: float, sigma: float, lo: float, hi: float) -> str:
    """Classify a marginal-Λ contribution: trust the sign only when 'solid'."""
    if abs(contribution) < 2.0 * sigma:
        return "~0 within 2σ (noise)"
    if lo < 0.0 < hi:
        return "sign not robust to zero-bin truncation"
    return "solid"


def lambda_decomposition(spec_small_full: FailureSpectrum, spec_large_full: FailureSpectrum,
                         spec_small_of: Callable[[str], FailureSpectrum],
                         spec_large_of: Callable[[str], FailureSpectrum],
                         channels: Sequence[str], p: float,
                         rounds_small: float, rounds_large: float) -> dict:
    """Marginal Λ decomposition at p: contribution_i = 1/Λ_full − 1/Λ_no-i, with verdicts.

    ``spec_*_of(channel)`` return the leave-one-out (ablated) spectra. Returns a dict with
    the full-ratio stats and one row per channel (ready for printing or JSON) — the shares
    sum to > 1/Λ_full when mixed faults are shared (each counted once per participant).
    """
    inv_f = inv_lambda_stats(spec_small_full, spec_large_full, p, rounds_small, rounds_large)
    rows = []
    for ch in channels:
        inv_a = inv_lambda_stats(spec_small_of(ch), spec_large_of(ch), p,
                                 rounds_small, rounds_large)
        c = inv_f.value - inv_a.value
        sc = float(np.hypot(inv_f.se, inv_a.se))
        c_lo, c_hi = inv_f.lo - inv_a.hi, inv_f.hi - inv_a.lo
        rows.append({"channel": ch, "lambda_no_i": 1.0 / inv_a.value, "inv_no_i": inv_a.value,
                     "contribution": c, "sigma": sc, "lo": c_lo, "hi": c_hi,
                     "share": c / inv_f.value, "verdict": verdict(c, sc, c_lo, c_hi)})
    return {"p": p, "inv_full": inv_f.value, "inv_full_se": inv_f.se,
            "lambda_full": 1.0 / inv_f.value,
            "sum_contributions": float(sum(r["contribution"] for r in rows)), "rows": rows}


# ============================ run adapter =====================================
@dataclass
class Run:
    """One experiment_runner outdir, loaded: the spectrum + the sidecar artifacts."""
    outdir: pathlib.Path
    config: dict
    spectrum: FailureSpectrum
    distance: Optional[dict]      # distance.json (Technique II) or None
    ansatz: Optional[dict]        # ansatz_fit.json or None
    splitting: Optional[dict]     # splitting.json or None
    onset: Optional[dict]         # onset.json or None

    @property
    def cycles(self) -> int:
        return cycles_of(self.config)

    @property
    def p_ref(self) -> float:
        return float(self.config["p_ref"])

    def eps(self, p: float) -> tuple:
        """(ε, se, headroom) at p under THE per-op convention."""
        return eps_stats(self.spectrum, p, self.cycles)

    @property
    def done_fraction(self) -> float:
        planned = self.config.get("weights") or []
        return len(self.spectrum.weights) / len(planned) if planned else float("nan")


def _read_json(path: pathlib.Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_run(outdir) -> Run:
    """Load an experiment_runner run directory (works on a mid-run checkpoint too)."""
    outdir = pathlib.Path(outdir)
    config = _read_json(outdir / "config.json")
    ckpt = _read_json(outdir / "spectrum.json")
    if config is None or ckpt is None:
        raise FileNotFoundError(f"{outdir} is not a loadable run "
                                f"(config.json present: {config is not None}, "
                                f"spectrum.json present: {ckpt is not None})")
    done = {int(w): int(f) for w, f in ckpt["failures_by_weight"].items()}
    tbw = {int(w): int(t) for w, t in ckpt.get("trials_by_weight", {}).items()}
    shots = int(ckpt.get("shots_per_weight", 0))
    weights = [int(w) for w in ckpt["weights_plan"] if int(w) in done]
    spec = FailureSpectrum(weights=weights, trials=[tbw.get(w, shots) for w in weights],
                           failures=[done[w] for w in weights],
                           n_expanded=int(ckpt["n_expanded"]), q_base=float(ckpt["q_base"]),
                           p_ref=float(ckpt["p_ref"]))
    return Run(outdir=outdir, config=config, spectrum=spec,
               distance=_read_json(outdir / "distance.json"),
               ansatz=_read_json(outdir / "ansatz_fit.json"),
               splitting=_read_json(outdir / "splitting.json"),
               onset=_read_json(outdir / "onset.json"))
