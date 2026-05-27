"""
Surface code simulation using Stim + PyMatching.

Supports:
  - Rotated and unrotated surface codes (distance d)
  - Isotropic Pauli (depolarizing) noise + independent measurement noise
  - PyMatching MWPM decoder
  - Logical idle experiments: logical error rate vs rounds / error rate
"""

from __future__ import annotations

import numpy as np
import stim
import pymatching
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Code type
# ---------------------------------------------------------------------------

class CodeType(str, Enum):
    ROTATED_Z   = "surface_code:rotated_memory_z"
    ROTATED_X   = "surface_code:rotated_memory_x"
    UNROTATED_Z = "surface_code:unrotated_memory_z"
    UNROTATED_X = "surface_code:unrotated_memory_x"


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

@dataclass
class ErrorModel:
    """
    Isotropic Pauli noise + independent measurement and reset noise.

    p_phys : Depolarizing rate applied after every Clifford gate.
             Single-qubit gates get 1Q depolarizing; 2-qubit gates get 2Q
             depolarizing — both controlled by this single parameter
             (the "isotropic" / symmetric-Pauli assumption).
    p_meas : Bit-flip probability applied before every measurement.
    p_reset: Bit-flip probability applied after every reset (state-prep noise).
             Defaults to None, which means "use the same value as p_meas".
    """
    p_phys: float
    p_meas: float
    p_reset: Optional[float] = None

    def __post_init__(self) -> None:
        for name, val in [("p_phys", self.p_phys), ("p_meas", self.p_meas)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if self.p_reset is not None and not (0.0 <= self.p_reset <= 1.0):
            raise ValueError(f"p_reset must be in [0, 1], got {self.p_reset}")

    @property
    def _p_reset(self) -> float:
        """Resolved reset error rate (falls back to p_meas if not set)."""
        return self.p_meas if self.p_reset is None else self.p_reset

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def symmetric(cls, p: float) -> "ErrorModel":
        """Equal gate and measurement error rates."""
        return cls(p_phys=p, p_meas=p)

    @classmethod
    def gate_dominated(cls, p_phys: float, meas_ratio: float = 0.1) -> "ErrorModel":
        """Measurement error is *meas_ratio* times the gate error rate."""
        return cls(p_phys=p_phys, p_meas=p_phys * meas_ratio)

    @classmethod
    def meas_dominated(cls, p_meas: float, gate_ratio: float = 0.1) -> "ErrorModel":
        """Gate error is *gate_ratio* times the measurement error rate."""
        return cls(p_phys=p_meas * gate_ratio, p_meas=p_meas)

    def __repr__(self) -> str:
        base = f"ErrorModel(p_phys={self.p_phys:.3g}, p_meas={self.p_meas:.3g}"
        if self.p_reset is not None:
            base += f", p_reset={self.p_reset:.3g}"
        return base + ")"


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

class Decoder:
    """Abstract decoder interface."""

    def setup(self, circuit: stim.Circuit) -> None:
        """Pre-compute matching graph from the circuit's detector error model."""
        raise NotImplementedError

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """
        Decode a batch of detection events.

        Parameters
        ----------
        detection_events : bool array of shape (shots, num_detectors)

        Returns
        -------
        predictions : bool array of shape (shots, num_observables)
        """
        raise NotImplementedError


class PyMatchingDecoder(Decoder):
    """
    Minimum-weight perfect matching via PyMatching.

    Uses the detector error model derived from the Stim circuit to build the
    matching graph, so no hand-crafted weights are needed.
    """

    def __init__(self) -> None:
        self._matching: Optional[pymatching.Matching] = None

    def setup(self, circuit: stim.Circuit) -> None:
        dem = circuit.detector_error_model(decompose_errors=True)
        self._matching = pymatching.Matching.from_detector_error_model(dem)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        if self._matching is None:
            raise RuntimeError("Call setup(circuit) before decode_batch.")
        return self._matching.decode_batch(detection_events)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Statistics from one logical idle experiment."""

    distance:              int
    rounds:                int
    error_model:           ErrorModel
    shots:                 int
    num_logical_errors:    int
    logical_error_rate:    float
    logical_error_rate_se: float  # binomial standard error

    def __repr__(self) -> str:
        return (
            f"SimulationResult("
            f"d={self.distance}, rounds={self.rounds}, "
            f"{self.error_model!r}, "
            f"LER={self.logical_error_rate:.4g} ± {self.logical_error_rate_se:.2g})"
        )


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

class SurfaceCodeSimulator:
    """
    Build Stim circuits, inject noise, and decode to estimate logical error rates
    for a surface code logical idle experiment.

    Parameters
    ----------
    distance  : Code distance d (number of physical qubits per row/column).
    code_type : Which surface code variant to use (default: rotated Z-memory).
    """

    def __init__(
        self,
        distance: int,
        code_type: CodeType = CodeType.ROTATED_Z,
    ) -> None:
        if distance < 2:
            raise ValueError("distance must be >= 2")
        self.distance  = distance
        self.code_type = code_type

    # ------------------------------------------------------------------
    # Circuit builder
    # ------------------------------------------------------------------

    def build_circuit(
        self,
        error_model: ErrorModel,
        rounds: int,
    ) -> stim.Circuit:
        """
        Return a noisy Stim circuit for the logical idle experiment.

        Noise layers
        ------------
        after_clifford_depolarization   → isotropic Pauli noise on gates
        before_measure_flip_probability → measurement bit-flip noise
        after_reset_flip_probability    → reset / state-prep noise
        """
        return stim.Circuit.generated(
            self.code_type.value,
            distance=self.distance,
            rounds=rounds,
            after_clifford_depolarization=error_model.p_phys,
            before_measure_flip_probability=error_model.p_meas,
            after_reset_flip_probability=error_model._p_reset,
        )

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def run(
        self,
        error_model: ErrorModel,
        rounds: int,
        shots: int,
        decoder: Optional[Decoder] = None,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run a logical idle experiment and return logical error statistics.

        Parameters
        ----------
        error_model : Noise model.
        rounds      : Syndrome extraction rounds (≥ 1).
        shots       : Number of Monte Carlo samples.
        decoder     : Decoder instance (default: PyMatchingDecoder).
        seed        : RNG seed for reproducibility.

        Returns
        -------
        SimulationResult
        """
        circuit = self.build_circuit(error_model, rounds)

        if decoder is None:
            decoder = PyMatchingDecoder()
        decoder.setup(circuit)

        sampler = circuit.compile_detector_sampler(seed=seed)
        detection_events, observable_flips = sampler.sample(
            shots, separate_observables=True
        )

        predictions = decoder.decode_batch(detection_events)

        # A logical error occurs when ANY observable is mispredicted
        logical_errors_per_shot = np.any(predictions != observable_flips, axis=1)
        n_err = int(np.sum(logical_errors_per_shot))
        ler   = n_err / shots
        from scipy.stats import beta as _beta
        lo, hi = _beta.interval(0.95, n_err + 0.5, shots - n_err + 0.5)
        ler_se = float((hi - lo) / 2)

        return SimulationResult(
            distance=self.distance,
            rounds=rounds,
            error_model=error_model,
            shots=shots,
            num_logical_errors=n_err,
            logical_error_rate=ler,
            logical_error_rate_se=ler_se,
        )

    # ------------------------------------------------------------------
    # Convenience sweeps
    # ------------------------------------------------------------------

    def sweep_p(
        self,
        p_values: List[float],
        rounds: Optional[int] = None,
        shots: int = 10_000,
        p_meas_factor: float = 1.0,
        decoder: Optional[Decoder] = None,
        seed: Optional[int] = None,
    ) -> List[SimulationResult]:
        """
        Sweep physical error rates.  p_meas = p_phys * p_meas_factor.

        Parameters
        ----------
        p_values     : Physical error rates to iterate over.
        rounds       : Syndrome rounds (default: distance).
        shots        : Shots per data point.
        p_meas_factor: Ratio of measurement to gate error rate.
        decoder      : Decoder (default: PyMatchingDecoder).
        seed         : RNG seed.
        """
        if rounds is None:
            rounds = self.distance
        results = []
        for p in p_values:
            em = ErrorModel(p_phys=p, p_meas=p * p_meas_factor)
            results.append(
                self.run(em, rounds=rounds, shots=shots, decoder=decoder, seed=seed)
            )
        return results

    def sweep_rounds(
        self,
        round_values: List[int],
        error_model: ErrorModel,
        shots: int = 10_000,
        decoder: Optional[Decoder] = None,
        seed: Optional[int] = None,
    ) -> List[SimulationResult]:
        """
        Sweep syndrome extraction rounds at a fixed error model.
        """
        results = []
        for r in round_values:
            results.append(
                self.run(error_model, rounds=r, shots=shots, decoder=decoder, seed=seed)
            )
        return results


# ---------------------------------------------------------------------------
# Multi-distance threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(
    distances: List[int],
    p_values: List[float],
    rounds_per_distance: Optional[Dict[int, int]] = None,
    shots: int = 10_000,
    p_meas_factor: float = 1.0,
    code_type: CodeType = CodeType.ROTATED_Z,
    decoder_cls=PyMatchingDecoder,
    seed: Optional[int] = None,
) -> Dict[int, List[SimulationResult]]:
    """
    Sweep distances × error rates to locate the error threshold.

    Parameters
    ----------
    distances           : List of code distances.
    p_values            : Physical error rates to sweep.
    rounds_per_distance : Optional dict mapping d → rounds (default: d).
    shots               : Shots per data point.
    p_meas_factor       : p_meas = p_phys * p_meas_factor.
    code_type           : Surface code variant.
    decoder_cls         : Decoder class (must implement Decoder interface).
    seed                : RNG seed.

    Returns
    -------
    Dict mapping distance → List[SimulationResult]
    """
    all_results: Dict[int, List[SimulationResult]] = {}
    for d in distances:
        sim    = SurfaceCodeSimulator(distance=d, code_type=code_type)
        rounds = (rounds_per_distance or {}).get(d, d)
        all_results[d] = sim.sweep_p(
            p_values,
            rounds=rounds,
            shots=shots,
            p_meas_factor=p_meas_factor,
            decoder=decoder_cls(),
            seed=seed,
        )
    return all_results
