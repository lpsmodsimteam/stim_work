"""
Weight-stratified importance sampling for QEC simulations.

Implements the core technique from arXiv:2511.15177 (Sections 2-3): estimate
the failure spectrum f(w) by sampling fault sets stratified by weight, then
reweight to obtain unbiased estimates of the logical error rate at any
target physical error rate p.

For circuit-level depolarizing noise with rate p, the fault mechanisms from
Stim's DEM each have probability proportional to p. We expand each mechanism
into multiple identical columns so all expanded columns share a common base
rate q. Sampling a weight-w fault set in the expanded representation is then
equivalent to drawing from the underlying non-uniform distribution.

Reweighting formula (Eq. 3 of the paper):

    P_logical(p) ≈ Σ_w f̂(w) · C(N_expanded, w) · q(p)^w · (1 - q(p))^(N_expanded - w)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln

if TYPE_CHECKING:                # stim is used only in `circuit: stim.Circuit` annotations, which
    import stim                  # `from __future__ import annotations` keeps as strings — so stim
                                 # is not needed at runtime. Keeping it lazy lets the failure-spectrum
                                 # ansatz tools (and the report notebook) run without stim installed.


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FailureSpectrum:
    """Estimated failure spectrum f(w) = F(w) / T(w) for a set of weights."""
    weights: List[int]
    trials: List[int]
    failures: List[int]
    n_expanded: int
    q_base: float
    p_ref: float

    def f(self, w: int) -> float:
        idx = self.weights.index(w)
        return 0.0 if self.trials[idx] == 0 else self.failures[idx] / self.trials[idx]


@dataclass
class ImportanceSamplingResult:
    p_values: np.ndarray
    P_logical: np.ndarray
    P_logical_se: np.ndarray
    spectrum: FailureSpectrum


# ---------------------------------------------------------------------------
# DEM parsing and expansion
# ---------------------------------------------------------------------------

def _parse_dem(circuit: stim.Circuit):
    """Extract (probs, detector-flip matrix, observable-flip matrix) from the circuit's DEM."""
    dem = circuit.detector_error_model(decompose_errors=False)
    M = dem.num_detectors
    K = dem.num_observables

    probs: List[float] = []
    det_rows: List[np.ndarray] = []
    obs_rows: List[np.ndarray] = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = float(inst.args_copy()[0])
        targets = inst.targets_copy()
        det_idx = [t.val for t in targets if t.is_relative_detector_id()]
        obs_idx = [t.val for t in targets if t.is_logical_observable_id()]
        d = np.zeros(M, dtype=bool)
        d[det_idx] = True
        o = np.zeros(K, dtype=bool)
        o[obs_idx] = True
        probs.append(p)
        det_rows.append(d)
        obs_rows.append(o)

    return np.array(probs), np.array(det_rows, dtype=bool), np.array(obs_rows, dtype=bool)


def _expand(probs: np.ndarray, q_base: Optional[float]):
    """Map expanded columns to source mechanisms so all share a common base rate q_base.

    Rather than materializing a dense (N_expanded x num_detectors) matrix, we only
    build an index array `col_to_mech` of length N_expanded that maps each expanded
    column back to its source mechanism. Sampling then indexes the original (small)
    mechanism matrices — identical results, but memory is O(N_expanded) ints instead
    of O(N_expanded * num_detectors) bools.
    """
    if q_base is None:
        q_base = float(probs.min())
    multipliers = np.maximum(np.round(probs / q_base).astype(int), 1)
    col_to_mech = np.repeat(np.arange(probs.shape[0], dtype=np.int32), multipliers)
    return col_to_mech, q_base, multipliers


# ---------------------------------------------------------------------------
# Sampling at fixed weight
# ---------------------------------------------------------------------------

def _sample_failures_at_weight(
    det_mat: np.ndarray,
    obs_mat: np.ndarray,
    col_to_mech: np.ndarray,
    w: int,
    T: int,
    decoder,
    rng: np.random.Generator,
) -> int:
    """Sample T weight-w fault configurations, decode, return number of decoding failures."""
    N_exp = col_to_mech.shape[0]
    M = det_mat.shape[1]
    K = obs_mat.shape[1]

    if w == 0:
        # zero faults → trivially correct (decoder predicts no flip from all-zero syndrome)
        return 0

    syndromes = np.zeros((T, M), dtype=bool)
    truths = np.zeros((T, K), dtype=bool)
    for t in range(T):
        # Sample expanded columns, then map back to source mechanisms. Two expanded
        # columns sharing a mechanism XOR to zero — identical to the dense expansion.
        mech_idxs = col_to_mech[rng.choice(N_exp, size=w, replace=False)]
        syndromes[t] = np.bitwise_xor.reduce(det_mat[mech_idxs], axis=0)
        truths[t] = np.bitwise_xor.reduce(obs_mat[mech_idxs], axis=0)

    predictions = decoder.decode_batch(syndromes)
    failures = np.any(predictions != truths, axis=1)
    return int(failures.sum())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def importance_sample(
    circuit: stim.Circuit,
    decoder,
    p_ref: float,
    p_values: Sequence[float],
    weights: Optional[Sequence[int]] = None,
    shots_per_weight: Union[int, Sequence[int], Dict[int, int]] = 1000,
    q_base: Optional[float] = None,
    seed: Optional[int] = None,
) -> ImportanceSamplingResult:
    """
    Run weight-stratified importance sampling on a noisy QEC circuit.

    Parameters
    ----------
    circuit : Noisy Stim circuit built at physical error rate p_ref.
    decoder : Decoder implementing setup(circuit) + decode_batch(events).
    p_ref   : Physical error rate the circuit was built with. The DEM's
              mechanism probabilities are interpreted as ∝ p_ref; targets in
              p_values are scaled relative to this.
    p_values : Target physical error rates at which to estimate P_logical.
    weights : Fault weights to sample at (default: 1..8).
    shots_per_weight : Trials per weight. Either an int (the same budget for every
        weight) or a per-weight allocation — a dict ``{w: T}`` keyed by weight, or a
        sequence aligned with ``weights``. Per-weight budgets let you pour shots into
        the rare near-onset bins (where f(w) is tiny and a flat budget yields 0
        failures → only a 3/T upper limit) and spend fewer on the high-weight bins
        where f(w)≈O(1) is already pinned. Downstream (per-weight binomial SE, the
        ansatz fit, the bootstrap) already handles heterogeneous trial counts.
    q_base : Base rate for the expanded representation. If None, inferred as
             min(DEM probabilities).
    seed : RNG seed.

    Returns
    -------
    ImportanceSamplingResult
    """
    rng = np.random.default_rng(seed)

    probs, det_mat, obs_mat = _parse_dem(circuit)
    col_to_mech, q_base, _ = _expand(probs, q_base)
    N_exp = col_to_mech.shape[0]

    decoder.setup(circuit)

    if weights is None:
        weights = list(range(1, min(9, N_exp + 1)))
    weights = list(weights)

    # Resolve shots_per_weight to a per-weight trial list T_w. int -> broadcast (the
    # historical behaviour); dict -> keyed by weight; sequence -> aligned with weights.
    if isinstance(shots_per_weight, int):
        T_w = [shots_per_weight] * len(weights)
    elif isinstance(shots_per_weight, dict):
        try:
            T_w = [int(shots_per_weight[w]) for w in weights]
        except KeyError as e:
            raise ValueError(f"shots_per_weight dict is missing weight {e.args[0]}") from None
    else:
        T_w = [int(t) for t in shots_per_weight]
        if len(T_w) != len(weights):
            raise ValueError(
                f"shots_per_weight sequence has {len(T_w)} entries but there are "
                f"{len(weights)} weights")
    if any(t < 0 for t in T_w):
        raise ValueError("shots_per_weight must be non-negative")

    failures = [
        _sample_failures_at_weight(det_mat, obs_mat, col_to_mech, w, T_w[j], decoder, rng)
        for j, w in enumerate(weights)
    ]

    spectrum = FailureSpectrum(
        weights=weights,
        trials=T_w,
        failures=failures,
        n_expanded=N_exp,
        q_base=q_base,
        p_ref=p_ref,
    )

    # Reweighting: assume q scales linearly with p (true for single-parameter depolarizing).
    p_arr = np.asarray(p_values, dtype=float)
    q_targets = np.clip(q_base * (p_arr / p_ref), 1e-300, 1.0 - 1e-15)
    log_q = np.log(q_targets)
    log_1mq = np.log(1.0 - q_targets)

    P_logical = np.zeros_like(p_arr)
    var = np.zeros_like(p_arr)

    for j, w in enumerate(weights):
        T = spectrum.trials[j]
        F = spectrum.failures[j]
        f = F / T if T > 0 else 0.0
        f_se = np.sqrt(f * (1.0 - f) / T) if T > 0 else 0.0
        log_binom = gammaln(N_exp + 1) - gammaln(w + 1) - gammaln(N_exp - w + 1)
        weight = np.exp(log_binom + w * log_q + (N_exp - w) * log_1mq)
        P_logical += f * weight
        var += (f_se * weight) ** 2

    return ImportanceSamplingResult(
        p_values=p_arr,
        P_logical=P_logical,
        P_logical_se=np.sqrt(var),
        spectrum=spectrum,
    )


# ===========================================================================
# Technique I — failure-spectrum ansatz and extrapolation
#
# arXiv:2511.15177 Section 3. Rather than reweighting the raw sampled f(w)
# (which needs a contiguous WEIGHTS block bracketing the dominant binomial
# mass), we fit a smooth low-parameter ansatz to the sampled spectrum and
# reweight the *fitted* f(w) over all weights 0..N. This removes the
# truncation requirement and lets us extrapolate to very low p.
# ===========================================================================

# Models and their free parameters beyond the always-present (w0, f0).
_ANSATZ_EXTRA: Dict[str, List[str]] = {
    "f2": [],
    "f3": ["gamma"],
    "f5": ["gamma1", "gamma2", "wc"],
}


def failure_spectrum_ansatz(
    w,
    w0: float,
    f0: float,
    a: float,
    *,
    model: str = "f3",
    gamma: Optional[float] = None,
    gamma1: Optional[float] = None,
    gamma2: Optional[float] = None,
    wc: Optional[float] = None,
    c: float = 2.0,
):
    """Failure-spectrum ansatz f(w) from Eq. (10) of arXiv:2511.15177.

    All three forms share the envelope ``a * (1 - exp(-(f0/a) * power))`` and are
    zero for ``w < w0``. ``a = 1 - 2^-K`` is the large-w saturation value (random
    bitstrings fail with probability 1 - 2^-K). At ``w = w0`` every form gives
    ``power = 1`` so ``f(w0) = a(1 - exp(-f0/a)) ≈ f0`` for small f0.

    model:
      "f2": power = (w/w0)**w0                              (params: w0, f0)
      "f3": power = (w/w0)**gamma                           (+ gamma)
      "f5": power = (w/w0)**gamma1 * S**((gamma2-gamma1)/c), (+ gamma1, gamma2, wc)
             S = (1 + (w/wc)**c) / (1 + (w0/wc)**c)         crossover, c fixed (=2)
    """
    w = np.asarray(w, dtype=float)
    ratio = np.where(w > 0, w / w0, 0.0)

    if model == "f2":
        power = ratio ** w0
    elif model == "f3":
        if gamma is None:
            raise ValueError("model 'f3' requires gamma")
        power = ratio ** gamma
    elif model == "f5":
        if gamma1 is None or gamma2 is None or wc is None:
            raise ValueError("model 'f5' requires gamma1, gamma2, wc")
        crossover = (1.0 + (w / wc) ** c) / (1.0 + (w0 / wc) ** c)
        power = ratio ** gamma1 * crossover ** ((gamma2 - gamma1) / c)
    else:
        raise ValueError(f"unknown ansatz model {model!r} (use f2/f3/f5)")

    val = a * (1.0 - np.exp(-(f0 / a) * power))
    return np.where(w < w0, 0.0, val)


@dataclass
class AnsatzFit:
    """A fitted failure-spectrum ansatz plus the metadata needed to reweight it."""
    model: str
    params: Dict[str, float]          # full param set incl. fixed w0/f0
    free_names: List[str]
    a: float                          # saturation 1 - 2^-K
    param_cov: Optional[np.ndarray]   # covariance over free_names (or None)
    n_expanded: int
    q_base: float
    p_ref: float
    n_points: int                     # number of (w, f) points used in the fit
    cost: float                       # final least-squares cost (0.5*||resid||^2)

    def f(self, w):
        return failure_spectrum_ansatz(w, a=self.a, model=self.model, **self.params)


def reweight_spectrum(spectrum: FailureSpectrum, p_values) -> ImportanceSamplingResult:
    """LER(p) by binomially reweighting a MEASURED failure spectrum — no ansatz, no extrapolation.

    LER(p) = Σ_w Binom(w; N_expanded, q(p)) · f̂(w) over the sampled weights. Unbiased at any p
    where the sampled weight range covers the support of both f(w)>0 and the binomial mass —
    in particular at LOW p whenever the spectrum was sampled down through the onset (then the
    truncation above the top sampled weight is negligible). Prefer this over evaluating an
    ansatz fit when comparing models at low p: fitted-(w0, f0, shape) differences between
    independently fitted models amplify exponentially under extrapolation, which can even
    invert physical orderings (e.g. an ablated circuit's fit crossing above the full model's).
    """
    N_exp = spectrum.n_expanded
    p_arr = np.asarray(p_values, dtype=float)
    q_targets = np.clip(spectrum.q_base * (p_arr / spectrum.p_ref), 1e-300, 1.0 - 1e-15)
    log_q = np.log(q_targets)
    log_1mq = np.log(1.0 - q_targets)

    P_logical = np.zeros_like(p_arr)
    var = np.zeros_like(p_arr)
    for w, T, F in zip(spectrum.weights, spectrum.trials, spectrum.failures):
        f = F / T if T > 0 else 0.0
        f_se = np.sqrt(f * (1.0 - f) / T) if T > 0 else 0.0
        log_binom = gammaln(N_exp + 1) - gammaln(w + 1) - gammaln(N_exp - w + 1)
        weight = np.exp(log_binom + w * log_q + (N_exp - w) * log_1mq)
        P_logical += f * weight
        var += (f_se * weight) ** 2

    return ImportanceSamplingResult(
        p_values=p_arr, P_logical=P_logical, P_logical_se=np.sqrt(var), spectrum=spectrum,
    )


def fit_failure_spectrum(
    spectrum: FailureSpectrum,
    K: int,
    *,
    model: str = "f3",
    w0: Optional[float] = None,
    f0: Optional[float] = None,
    init_params: Optional[Dict[str, float]] = None,
) -> AnsatzFit:
    """Fit an ansatz (f2/f3/f5) to a sampled failure spectrum.

    The fit is done on log f(w) (delta-method weights sigma_log = se/f) over the
    weights with at least one observed failure, per the paper's guidance to keep
    the wide dynamic range well conditioned. ``a = 1 - 2^-K`` with K the number of
    logical observables. ``w0`` and/or ``f0`` may be pinned (e.g. from Technique II,
    the min-weight analysis); otherwise they are fit.

    ``init_params`` warm-starts the solver from a previous fit's ``params`` and skips
    the multistart grid — use it for refits of a perturbed spectrum (e.g. bootstrap
    resamples), which land in the same basin at ~1/37th the f5 solver calls.
    """
    if model not in _ANSATZ_EXTRA:
        raise ValueError(f"unknown ansatz model {model!r} (use f2/f3/f5)")

    a = 1.0 - 2.0 ** (-K)
    w = np.asarray(spectrum.weights, dtype=float)
    T = np.asarray(spectrum.trials, dtype=float)
    F = np.asarray(spectrum.failures, dtype=float)

    mask = (F > 0) & (T > 0)
    if not mask.any():
        raise ValueError(
            "no sampled weight had any observed failures, so the spectrum is "
            "all-zero and the ansatz cannot be fit. Sample more shots per weight "
            "or include higher weights (closer to the dominant mass μ=N·q)."
        )
    wn, Fn, Tn = w[mask], F[mask], T[mask]
    fhat = Fn / Tn
    # Binomial SE, floored so saturated points (fhat≈1) keep finite weight.
    se = np.sqrt(np.maximum(fhat * (1.0 - fhat), 1.0 / Tn) / Tn)
    logy = np.log(fhat)
    sigma_log = se / fhat  # Var(log f) ≈ Var(f)/f^2

    # Assemble fixed vs free parameters.
    fixed: Dict[str, float] = {}
    if w0 is not None:
        fixed["w0"] = float(w0)
    if f0 is not None:
        fixed["f0"] = float(f0)
    free_names = [p for p in (["w0", "f0"] + _ANSATZ_EXTRA[model]) if p not in fixed]

    # Initial guesses / bounds.
    w0_init = fixed.get("w0", float(wn.min()))
    f0_init = fixed.get("f0", float(np.clip(fhat[np.argmin(wn)], 1e-12, 0.9 * a)))
    init = {
        "w0": w0_init,
        "f0": f0_init,
        "gamma": max(w0_init, 1.0),
        "gamma1": max(w0_init, 1.0),
        "gamma2": 1.0,
        "wc": max(2.0 * w0_init, float(np.median(wn))),
    }
    lo = {"w0": 1e-3, "f0": 1e-15, "gamma": 1e-3, "gamma1": 1e-3, "gamma2": 1e-3, "wc": 1e-3}
    hi = {"w0": float(w.max()) + 1, "f0": a, "gamma": 1e3, "gamma1": 1e3, "gamma2": 1e3, "wc": 1e6}

    if len(wn) < len(free_names) + 1:
        raise ValueError(
            f"too few non-zero failure points ({len(wn)}) to fit {len(free_names)} "
            f"parameters for model {model!r}; sample more weights or pin w0/f0"
        )

    def kwargs_from(theta) -> Dict[str, float]:
        kw = dict(fixed)
        for name, val in zip(free_names, theta):
            kw[name] = val
        return kw

    def residual(theta):
        kw = kwargs_from(theta)
        model_f = failure_spectrum_ansatz(wn, a=a, model=model, **kw)
        model_f = np.clip(model_f, 1e-300, None)
        return (np.log(model_f) - logy) / sigma_log

    param_cov: Optional[np.ndarray] = None
    if free_names:
        bounds = (
            np.array([lo[n] for n in free_names]),
            np.array([hi[n] for n in free_names]),
        )
        # Multistart: the (w0, f0, shape) least-squares is non-convex, so a single start can fall into
        # a bad local minimum (e.g. a low-gamma basin that matches only the onset and undershoots the
        # rest of the spectrum by several ×, especially when w0 is left free). Seed several shape starts
        # and keep the lowest-cost solution -- this is strictly >= the single-start quality.
        # A warm start (init_params) replaces the grid with that single start.
        if init_params is not None:
            starts = [{**init, **{k: v for k, v in init_params.items() if k in free_names}}]
        else:
            shape_grid = {"gamma": [2.0, 4.0, 6.0, 8.0], "gamma1": [2.0, 4.0, 6.0, 8.0],
                          "gamma2": [1.0, 1.5, 2.0], "wc": [2.0, 4.0, 8.0]}
            free_shape = [p for p in free_names if p in shape_grid]
            grid = product(*(shape_grid[p] for p in free_shape)) if free_shape else ()
            starts = [dict(init)] + [{**init, **dict(zip(free_shape, combo))} for combo in grid]
        sol = None
        for s in starts:
            x0 = np.clip(np.array([s[n] for n in free_names], dtype=float),
                         bounds[0] + 1e-9, bounds[1] - 1e-9)
            cand = least_squares(residual, x0, bounds=bounds, method="trf")
            if sol is None or cand.cost < sol.cost:
                sol = cand
        params = kwargs_from(sol.x)
        cost = float(sol.cost)
        # Covariance ≈ (JᵀJ)⁻¹ scaled by residual variance (Gauss-Newton approx).
        dof = max(len(wn) - len(free_names), 1)
        s_sq = 2.0 * sol.cost / dof
        try:
            param_cov = np.linalg.inv(sol.jac.T @ sol.jac) * s_sq
        except np.linalg.LinAlgError:
            param_cov = None
    else:
        params = dict(fixed)
        cost = float(0.5 * np.sum(residual(np.array([])) ** 2))

    return AnsatzFit(
        model=model,
        params=params,
        free_names=free_names,
        a=a,
        param_cov=param_cov,
        n_expanded=spectrum.n_expanded,
        q_base=spectrum.q_base,
        p_ref=spectrum.p_ref,
        n_points=len(wn),
        cost=cost,
    )


def logical_error_rate_from_ansatz(
    fit: AnsatzFit,
    p_values: Sequence[float],
    b: Optional[float] = None,
) -> np.ndarray:
    """Extrapolated LER P(p) = T{f_ansatz}(p) summed over all weights w0..N.

    Eq. (2)/(10) of arXiv:2511.15177. ``b`` is the noise scaling so q = p/b; it
    defaults to p_ref/q_base (i.e. q(p_ref) = q_base), which equals 15 for circuit
    noise and 1 for bit-flip noise in the expanded representation.
    """
    N = fit.n_expanded
    if b is None:
        b = fit.p_ref / fit.q_base

    p_arr = np.asarray(p_values, dtype=float)
    q = np.clip(p_arr / b, 1e-300, 1.0 - 1e-15)
    log_q = np.log(q)
    log_1mq = np.log(1.0 - q)

    w0 = fit.params["w0"]
    w = np.arange(int(np.ceil(w0)), N + 1, dtype=float)
    fw = failure_spectrum_ansatz(w, a=fit.a, model=fit.model, **fit.params)

    log_binom = gammaln(N + 1) - gammaln(w + 1) - gammaln(N - w + 1)
    # term[w, p] = C(N,w) q^w (1-q)^(N-w); sum_w f(w)*term over the weight axis.
    log_term = log_binom[:, None] + w[:, None] * log_q[None, :] + (N - w)[:, None] * log_1mq[None, :]
    P = (fw[:, None] * np.exp(log_term)).sum(axis=0)
    return P


@dataclass
class AnsatzSweepResult:
    """Combines the raw IS estimate with the ansatz-extrapolated LER curve."""
    p_values: np.ndarray
    P_logical_ansatz: np.ndarray
    fit: AnsatzFit
    raw: ImportanceSamplingResult


def importance_sample_with_ansatz(
    circuit: stim.Circuit,
    decoder,
    p_ref: float,
    p_values: Sequence[float],
    *,
    model: str = "f3",
    w0: Optional[float] = None,
    f0: Optional[float] = None,
    weights: Optional[Sequence[int]] = None,
    shots_per_weight: Union[int, Sequence[int], Dict[int, int]] = 1000,
    q_base: Optional[float] = None,
    seed: Optional[int] = None,
) -> AnsatzSweepResult:
    """Sample the failure spectrum, fit an ansatz, and return the extrapolated LER.

    Unlike :func:`importance_sample`, the sampled ``weights`` need not form a
    contiguous block up to the dominant binomial mass — they only need to
    constrain the ansatz fit. K (for the saturation value a = 1 - 2^-K) is read
    from the circuit's number of logical observables.
    """
    raw = importance_sample(
        circuit, decoder, p_ref=p_ref, p_values=p_values,
        weights=weights, shots_per_weight=shots_per_weight, q_base=q_base, seed=seed,
    )
    K = circuit.detector_error_model(decompose_errors=False).num_observables
    fit = fit_failure_spectrum(raw.spectrum, K, model=model, w0=w0, f0=f0)
    P = logical_error_rate_from_ansatz(fit, p_values)
    return AnsatzSweepResult(
        p_values=np.asarray(p_values, dtype=float),
        P_logical_ansatz=P,
        fit=fit,
        raw=raw,
    )


# ===========================================================================
# Adaptive per-weight shot allocation — "hit N failures per weight"
#
# Sweep weights HIGH -> LOW. Predict the next (lower) weight's failure fraction
# f(w) by log-linear extrapolation of the weights already measured, then give it
# just enough shots to expect ~target_failures failures:  T_w = N / f_pred
# (clamped). The top weights (f≈1) cost the fewest shots (≈N); the cost grows as
# f(w) falls toward the onset. Each weight is tuned from the results of the ones
# before it — no LER model, no binomial weighting.
# ===========================================================================

def predict_failure_fraction(measured: Dict[int, float], w: int, window: int = 3) -> float:
    """Predict f(w) for an unsampled weight from already-measured weights.

    `measured` maps weight -> measured failure fraction. Fits a line to log f vs
    weight over the `window` measured points nearest `w` (using only 0<f<1 points,
    where log f is informative) and extrapolates. Returns 1.0 when no informative
    point exists yet — the top of a high->low sweep, where f≈1 and a few shots
    suffice."""
    pts = sorted((int(ww), float(f)) for ww, f in measured.items() if 0.0 < f < 1.0)
    if not pts:
        if measured:                       # only saturated (f=1) or empty (f=0) points so far
            ww_near = min(measured, key=lambda x: abs(x - w))
            return float(np.clip(measured[ww_near], 1e-12, 1.0))
        return 1.0
    pts.sort(key=lambda t: abs(t[0] - w))   # the `window` points closest in weight to w
    use = sorted(pts[: max(2, window)])
    if len(use) < 2:
        return float(np.clip(use[0][1], 1e-12, 1.0))
    xs = np.array([t[0] for t in use], dtype=float)
    ys = np.log([t[1] for t in use])
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(np.clip(np.exp(intercept + slope * w), 1e-12, 1.0))


def shots_to_hit_failures(f_pred: float, target_failures: int, shots_min: int, shots_max: int) -> int:
    """Shots needed to expect ~target_failures failures at predicted fraction f_pred (clamped)."""
    need = target_failures / max(float(f_pred), 1e-300)
    return int(np.clip(np.ceil(need), shots_min, shots_max))


def importance_sample_adaptive(
    circuit: "stim.Circuit",
    decoder,
    p_ref: float,
    p_values: Sequence[float],
    weights: Optional[Sequence[int]] = None,
    *,
    target_failures: int = 50,
    shots_min: int = 4,
    shots_max: int = 20000,
    predict_window: int = 3,
    q_base: Optional[float] = None,
    seed: Optional[int] = None,
    stop_after_zero_bins: Optional[int] = None,
) -> ImportanceSamplingResult:
    """Weight-stratified IS with adaptive 'hit N failures per weight' shot allocation.

    Sweeps `weights` high->low; each weight's shots are sized from the extrapolated
    f(w) to expect ~`target_failures` failures (clamped to [shots_min, shots_max]).
    Returns the same ImportanceSamplingResult as :func:`importance_sample`, with
    per-weight `trials` reflecting the adaptive allocation (so the reweighted SE is
    per-weight correct).

    ``stop_after_zero_bins``: once that many CONSECUTIVE weights (descending) return
    zero failures at the full ``shots_max`` budget, the remaining (lower) weights are
    skipped and omitted from the spectrum. A zero-failure bin contributes exactly 0 to
    the reweighting — identical to not sampling it — and f(w) only shrinks with w, so
    below a run of empty max-budget bins the sweep is pure cost: sub-onset bins of a
    large code would otherwise each burn the full clamp (e.g. ~10 bins x shots_max
    below the observable onset)."""
    rng = np.random.default_rng(seed)
    probs, det_mat, obs_mat = _parse_dem(circuit)
    col_to_mech, q_base, _ = _expand(probs, q_base)
    N_exp = col_to_mech.shape[0]
    decoder.setup(circuit)
    if weights is None:
        weights = list(range(1, min(9, N_exp + 1)))
    weights = sorted({int(w) for w in weights})

    measured_f: Dict[int, float] = {}
    F_by: Dict[int, int] = {}
    T_by: Dict[int, int] = {}
    zero_run = 0
    for w in sorted(weights, reverse=True):
        f_pred = predict_failure_fraction(measured_f, w, predict_window)
        T_w = shots_to_hit_failures(f_pred, target_failures, shots_min, shots_max)
        F = _sample_failures_at_weight(det_mat, obs_mat, col_to_mech, w, T_w, decoder, rng)
        F_by[w], T_by[w] = F, T_w
        measured_f[w] = F / T_w if T_w else 0.0
        if stop_after_zero_bins is not None:
            zero_run = zero_run + 1 if (F == 0 and T_w >= shots_max) else 0
            if zero_run >= stop_after_zero_bins:
                break               # everything below is even deeper sub-onset: skip it

    sampled = [w for w in weights if w in T_by]
    spectrum = FailureSpectrum(
        weights=sampled,
        trials=[T_by[w] for w in sampled],
        failures=[F_by[w] for w in sampled],
        n_expanded=N_exp,
        q_base=q_base,
        p_ref=p_ref,
    )

    p_arr = np.asarray(p_values, dtype=float)
    q_targets = np.clip(q_base * (p_arr / p_ref), 1e-300, 1.0 - 1e-15)
    log_q = np.log(q_targets)
    log_1mq = np.log1p(-q_targets)
    P_logical = np.zeros_like(p_arr)
    var = np.zeros_like(p_arr)
    for w in sampled:
        T, F = T_by[w], F_by[w]
        f = F / T if T > 0 else 0.0
        f_se = np.sqrt(f * (1.0 - f) / T) if T > 0 else 0.0
        log_binom = gammaln(N_exp + 1) - gammaln(w + 1) - gammaln(N_exp - w + 1)
        weight = np.exp(log_binom + w * log_q + (N_exp - w) * log_1mq)
        P_logical += f * weight
        var += (f_se * weight) ** 2

    return ImportanceSamplingResult(
        p_values=p_arr, P_logical=P_logical, P_logical_se=np.sqrt(var), spectrum=spectrum,
    )
