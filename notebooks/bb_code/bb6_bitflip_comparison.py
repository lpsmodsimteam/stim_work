#!/usr/bin/env python3
"""Match paper Table 2 BB(6)-bitflip results, exploit toric symmetry, and plot Figure-6-style convergence.

Paper Table 2 (BB(6)-bitflip, Z-type noise, N=72):
  |ℒ(D)|       = 84     weight-6 logical X operators
  |ℒ̃(D)|      = 84     (= |ℒ(D)| since N=72, no expansion)
  |ℒ(D)|_{D/2} = 1392   distinct weight-3 sub-patterns of weight-6 logicals

Bypasses the full circuit DEM: works directly with H_Z (36×72) and log_Z (12×72).
With N=72, each BP-OSD decode takes ~0.1–1 ms, so 4095 systematic + 50k random
trials completes in 2–5 minutes.

Also plots a Figure-6-style convergence curve: |L(D)| found vs cumulative trial
count, for both the bitflip (N=72, fast) and circuit-level (N=68940, from
mw_improved.log) cases.  The bitflip panel shows both with and without Z_6×Z_6
shift-automorphism acceleration (Fig. 6b of the paper).

The circuit-level symmetry builder (build_circuit_translation_perms) is also
provided for use in bb6_fig10_sweep.py; it requires a stim.Circuit built by
build_bb_circuit and the H/A matrices from dem_check_action_matrices.

Usage:
    python notebooks/bb_code/bb6_bitflip_comparison.py
    python notebooks/bb_code/bb6_bitflip_comparison.py --random-trials 100000
    python notebooks/bb_code/bb6_bitflip_comparison.py --no-systematic --random-trials 200000
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))

# Force UTF-8 stdout/stderr on Windows (cp1252 can't encode script special chars).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from bb_code_sim import BB_72_12_6, build_parity_checks, find_logical_ops  # noqa: E402
from ldpc.bposd_decoder import BpOsdDecoder  # noqa: E402
from min_weight import build_circuit_translation_perms  # noqa: E402  (re-exported for callers)


# ---------------------------------------------------------------------------
# Core BP-OSD helpers
# ---------------------------------------------------------------------------

def _bposd(H: np.ndarray, channel_prob: float, osd_order: int = 10, max_iter: int = 200):
    return BpOsdDecoder(
        H.astype(np.uint8),
        error_channel=[channel_prob] * H.shape[1],
        max_iter=max_iter,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_cs",
        osd_order=osd_order,
    )


def _forcing_correction(H: np.ndarray, g: np.ndarray, channel_prob: float,
                        osd_order: int, max_iter: int) -> np.ndarray:
    """One BP-OSD decode: augment H with g, fire the forcing row, return correction."""
    M = H.shape[0]
    H_aug = np.vstack([H, g[None, :]])
    dec = _bposd(H_aug, channel_prob, osd_order, max_iter)
    syndrome = np.zeros(M + 1, dtype=np.uint8)
    syndrome[M] = 1
    dec.decode(syndrome)
    return np.asarray(dec.osdw_decoding, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Toric translation symmetry — bitflip model (data qubits only)
# ---------------------------------------------------------------------------

def _qubit_torus_perm(a: int, b: int, l: int = 6, m: int = 6) -> np.ndarray:
    """Permutation on 2*l*m data qubits for toric translation T_(a,b).

    BB code layout:
      A-type qubit at torus position (i,j): index = i*m + j         (in [0, l*m))
      B-type qubit at torus position (i,j): index = l*m + i*m + j   (in [l*m, 2*l*m))
    """
    n = l * m
    perm = np.empty(2 * n, dtype=np.int32)
    for q in range(2 * n):
        qt, s = divmod(q, n)
        i, j = divmod(s, m)
        perm[q] = qt * n + ((i + a) % l) * m + ((j + b) % m)
    return perm


def _all_torus_perms(l: int = 6, m: int = 6) -> List[np.ndarray]:
    """All l*m toric translation permutations on data qubits (index 0 = identity)."""
    return [_qubit_torus_perm(a, b, l, m) for a in range(l) for b in range(m)]


# ---------------------------------------------------------------------------
# Toric translation symmetry — circuit-level DEM (mechanism index permutations)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Logical operator search — bitflip model
# ---------------------------------------------------------------------------

def find_bitflip_logicals(
    H: np.ndarray,
    A: np.ndarray,
    D: int,
    *,
    systematic: bool = True,
    max_trials: int = 50000,
    osd_order: int = 10,
    max_iter: int = 200,
    channel_prob: float = 0.01,
    seed: int = 42,
    progress_every: int = 500,
    torus_perms: Optional[List[np.ndarray]] = None,
) -> Tuple[Set[FrozenSet[int]], List[Tuple[int, int]]]:
    """Systematic + random search for weight-D logicals, returning the found set
    and a convergence trace [(trial_idx, cumulative_count), ...].

    For the bitflip model H=H_Z (36x72), A=log_Z (12x72), D=6.
    Each decode is ~0.1-1 ms so 4095+50k trials takes ~2-5 min.

    If torus_perms is provided (list of 36 qubit permutation arrays from
    _all_torus_perms()), each newly found logical is immediately expanded by all
    36 toric translations (Z_6 x Z_6 shift automorphisms).  This can reduce the
    number of trials needed from ~2500 to ~a handful.
    """
    K = A.shape[0]
    syndrome = np.zeros(H.shape[0] + 1, dtype=np.uint8)
    syndrome[-1] = 1

    found: Set[FrozenSet[int]] = set()
    trace: List[Tuple[int, int]] = []
    trial = 0

    def _check_and_add(corr: np.ndarray) -> bool:
        """Add a valid weight-D logical to found, expanding by symmetry if enabled.
        Returns True iff corr is a genuinely new (not previously seen) logical."""
        if int(corr.sum()) != D:
            return False
        if (H @ corr % 2).any() or not (A @ corr % 2).any():
            return False
        supp = frozenset(np.flatnonzero(corr).tolist())
        if supp in found:
            return False
        found.add(supp)
        if torus_perms is not None:
            # Expand by all l*m toric translations; each is guaranteed valid
            # by the exact Z_6xZ_6 symmetry of the BB code.
            for perm in torus_perms:
                ts = frozenset(int(perm[q]) for q in supp)
                found.add(ts)
        return True

    # --- Systematic phase: all 2^K - 1 bitmasks ---
    if systematic:
        n_sys = (1 << K) - 1
        print(f"  [systematic] {n_sys} syndrome classes ...", flush=True)
        for mask in range(1, n_sys + 1):
            coeffs = np.array([(mask >> i) & 1 for i in range(K)], dtype=np.uint8)
            g = (coeffs @ A) % 2
            corr = _forcing_correction(H, g, channel_prob, osd_order, max_iter)
            _check_and_add(corr)
            trial += 1
            trace.append((trial, len(found)))
            if progress_every and mask % progress_every == 0:
                print(f"  [systematic] {mask}/{n_sys}, |L(D)|={len(found)}", flush=True)
        print(f"  [systematic] done: |L(D)|={len(found)}", flush=True)

    # --- Random phase ---
    rng = np.random.default_rng(seed)
    print(f"  [random] {max_trials} trials ...", flush=True)
    prev = len(found)
    for t in range(max_trials):
        coeffs = rng.integers(0, 2, size=K)
        if not coeffs.any():
            coeffs[rng.integers(K)] = 1
        g = (coeffs @ A) % 2
        corr = _forcing_correction(H, g, channel_prob, osd_order, max_iter)
        _check_and_add(corr)
        trial += 1
        trace.append((trial, len(found)))
        if progress_every and (t + 1) % progress_every == 0:
            delta = len(found) - prev
            print(f"  [random] {t+1}/{max_trials}, |L(D)|={len(found)} (+{delta})", flush=True)
            prev = len(found)

    return found, trace


def count_restrictions(logicals: Set[FrozenSet[int]], half: int) -> int:
    """Count distinct weight-half subsets of all logicals (= |ℒ(D)|_{D/2}|)."""
    restrictions: Set[FrozenSet[int]] = set()
    for support in logicals:
        supp_list = sorted(support)
        for sub in combinations(supp_list, half):
            restrictions.add(frozenset(sub))
    return len(restrictions)


# ---------------------------------------------------------------------------
# Parse circuit-level convergence from mw_improved.log
# ---------------------------------------------------------------------------

def parse_mw_log(log_path: pathlib.Path) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Extract systematic and random convergence traces from mw_improved.log.

    Returns (sys_trials, sys_counts, rand_trials, rand_counts).
    """
    sys_trials, sys_counts = [], []
    rand_trials, rand_counts = [], []

    if not log_path.exists():
        return sys_trials, sys_counts, rand_trials, rand_counts

    text = log_path.read_text(errors="replace")
    # Systematic lines: "L(D) systematic: 152/4095, |L(D)|=31"
    for m in re.finditer(r"L\(D\) systematic[^\n]*?(\d+)/(\d+), \|L\(D\)\|=(\d+)", text):
        done, total, cnt = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sys_trials.append(done)
        sys_counts.append(cnt)
    # Final systematic done: only if not captured above
    for m in re.finditer(r"L\(D\) systematic done: \|L\(D\)\|=(\d+)", text):
        sys_trials.append(sys_trials[-1] if sys_trials else 4095)
        sys_counts.append(int(m.group(1)))
    # Random lines: "L(D) random [parallel x8]: 152/2000, |L(D)|=85"
    base = sys_trials[-1] if sys_trials else 0
    for m in re.finditer(r"L\(D\) random[^\n]*?(\d+)/(\d+), \|L\(D\)\|=(\d+)", text):
        done, cnt = int(m.group(1)), int(m.group(3))
        rand_trials.append(base + done)
        rand_counts.append(cnt)
    return sys_trials, sys_counts, rand_trials, rand_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random-trials", type=int, default=50000,
                    help="random trials after systematic sweep (default 50000)")
    ap.add_argument("--no-systematic", action="store_true",
                    help="skip systematic sweep (faster but may miss some logicals)")
    ap.add_argument("--osd-order", type=int, default=10)
    ap.add_argument("--outdir", type=pathlib.Path,
                    default=_HERE / "bb6_bitflip_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plot", action="store_true", help="skip figure output")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Build H_Z and log_Z from BB(6) code structure
    H_X, H_Z = build_parity_checks(BB_72_12_6)
    log_Z, log_X = find_logical_ops(H_X, H_Z)
    N = H_Z.shape[1]  # 72 data qubits
    K = log_Z.shape[0]  # 12 logicals
    D = 6
    half = D // 2  # = 3

    print(f"BB(6) bitflip model: N={N}, K={K}, D={D}, half={half}")
    print(f"H_Z shape: {H_Z.shape}, log_Z shape: {log_Z.shape}")
    print(f"Paper targets: |ℒ(D)|=84, |ℒ(D)|_{{D/2}}=1392")
    print()

    # Toric permutations for shift-automorphism acceleration
    all_perms = _all_torus_perms()  # 36 permutations on 72 data qubits

    # --- Run WITHOUT symmetry ---
    print("=== Search WITHOUT symmetry ===")
    t0 = time.perf_counter()
    logicals_nosym, trace_nosym = find_bitflip_logicals(
        H_Z, log_Z, D,
        systematic=not args.no_systematic,
        max_trials=args.random_trials,
        osd_order=args.osd_order,
        seed=args.seed,
        progress_every=500,
        torus_perms=None,
    )
    elapsed_nosym = time.perf_counter() - t0
    n_nosym = len(logicals_nosym)
    print(f"  |L(D)| = {n_nosym}  (elapsed {elapsed_nosym:.1f}s)\n")

    # --- Run WITH symmetry (fewer random trials needed — use 5000) ---
    print("=== Search WITH Z6xZ6 shift-automorphism symmetry ===")
    sym_random = min(args.random_trials, 5000)
    t0 = time.perf_counter()
    logicals_sym, trace_sym = find_bitflip_logicals(
        H_Z, log_Z, D,
        systematic=not args.no_systematic,
        max_trials=sym_random,
        osd_order=args.osd_order,
        seed=args.seed,
        progress_every=500,
        torus_perms=all_perms,
    )
    elapsed_sym = time.perf_counter() - t0
    n_sym = len(logicals_sym)
    print(f"  |L(D)| = {n_sym}  (elapsed {elapsed_sym:.1f}s)\n")

    # Use the larger set for restriction counting
    logicals = logicals_sym if n_sym >= n_nosym else logicals_nosym
    n_logicals = len(logicals)
    n_restrictions = count_restrictions(logicals, half)

    print(f"Results summary:")
    print(f"  WITHOUT symmetry: |L(D)| = {n_nosym}   ({elapsed_nosym:.1f}s)")
    print(f"  WITH symmetry:    |L(D)| = {n_sym}   ({elapsed_sym:.1f}s)")
    print(f"  |ℒ(D)|         = {n_logicals:6d}   (paper: 84)")
    print(f"  |ℒ(D)|_{{D/2}} = {n_restrictions:6d}   (paper: 1392)")
    print(f"  N               = {N:6d}   (paper: 72)")
    print(f"  C(N, D/2)       = {int(N*(N-1)*(N-2)//6):6d}")
    f0 = n_restrictions / (N*(N-1)*(N-2)//6)
    print(f"  f0 = |F|/C(N,3) = {f0:.6f}")

    # Save results
    import json
    result = {
        "n_logicals_nosym": n_nosym, "n_logicals_sym": n_sym,
        "n_logicals": n_logicals, "n_restrictions": n_restrictions,
        "N": N, "D": D, "K": K,
        "elapsed_nosym_s": elapsed_nosym, "elapsed_sym_s": elapsed_sym,
        "paper_n_logicals": 84, "paper_n_restrictions": 1392,
        "systematic": not args.no_systematic, "random_trials": args.random_trials,
        "osd_order": args.osd_order,
    }
    (args.outdir / "bitflip_results.json").write_text(json.dumps(result, indent=2))

    if args.no_plot:
        return

    # ------------------------------------------------------------------
    # Figure-6-style convergence plot
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel (a): bitflip convergence — with vs without symmetry ---
    ax = axes[0]
    n_sys = (1 << K) - 1 if not args.no_systematic else 0

    # Without symmetry (blue)
    ta = np.array([t for t, _ in trace_nosym])
    ca = np.array([c for _, c in trace_nosym])
    ax.plot(ta, ca, color="steelblue", lw=1.5, label=f"no symmetry")

    # With symmetry (orange)
    tb = np.array([t for t, _ in trace_sym])
    cb = np.array([c for _, c in trace_sym])
    ax.plot(tb, cb, color="darkorange", lw=1.8, label=f"Z₆×Z₆ shift-automorphisms")

    ax.axhline(84, color="black", ls="--", lw=1.2, label="paper target: 84")
    ax.set_xlabel("Cumulative trials")
    ax.set_ylabel("|ℒ(D)| found")
    ax.set_title(
        "BB(6) bitflip — convergence with vs without toric symmetry\n"
        f"N=72, D=6, K=12, osd_order={args.osd_order}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if n_sys > 0:
        ax.axvline(n_sys, color="gray", ls=":", alpha=0.5, lw=0.8)
        ax.text(n_sys + 20, 2, "sys→rnd", fontsize=7, color="gray")

    # Annotate trials-to-saturation
    def _trials_to_target(ta, ca, target):
        for t, c in zip(ta, ca):
            if c >= target:
                return t
        return None

    t_sat_nosym = _trials_to_target(ta, ca, 84)
    t_sat_sym = _trials_to_target(tb, cb, 84)
    if t_sat_nosym is not None:
        ax.annotate(f"sat@{t_sat_nosym}", xy=(t_sat_nosym, 84),
                    xytext=(t_sat_nosym + 50, 75), fontsize=7, color="steelblue",
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8))
    if t_sat_sym is not None:
        ax.annotate(f"sat@{t_sat_sym}", xy=(t_sat_sym, 84),
                    xytext=(t_sat_sym + 50, 65), fontsize=7, color="darkorange",
                    arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8))

    # --- Panel (b): circuit-level convergence from mw_improved.log ---
    ax2 = axes[1]
    log_path = _HERE / "mw_improved.log"
    sys_t, sys_c, rand_t, rand_c = parse_mw_log(log_path)

    if sys_t:
        ax2.plot(sys_t, sys_c, color="steelblue", lw=1.5,
                 label=f"circuit systematic ({sys_t[-1]} classes)")
    if rand_t:
        ax2.plot(rand_t, rand_c, color="darkorange", lw=1.5,
                 label=f"circuit random ({rand_t[-1] - (sys_t[-1] if sys_t else 0)} trials)")

    ax2.axhline(85, color="green", ls=":", lw=1.2, label="our result: 85")
    ax2.set_xlabel("Cumulative trials")
    ax2.set_ylabel("|L(D)| found")
    ax2.set_title("BB(6) circuit — logical operator search convergence\n"
                  "N=68940, D=6, osd_order=10  (from mw_improved.log)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    if sys_t:
        ax2.axvline(sys_t[-1], color="steelblue", ls=":", alpha=0.5, lw=0.8)

    fig.tight_layout()
    out = args.outdir / "bb6_convergence.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
