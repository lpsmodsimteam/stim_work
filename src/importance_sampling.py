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

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import stim
from scipy.special import gammaln


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
    shots_per_weight: int = 1000,
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
    shots_per_weight : Trials per weight w.
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

    failures = [
        _sample_failures_at_weight(det_mat, obs_mat, col_to_mech, w, shots_per_weight, decoder, rng)
        for w in weights
    ]

    spectrum = FailureSpectrum(
        weights=weights,
        trials=[shots_per_weight] * len(weights),
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
