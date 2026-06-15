"""
Technique III — multi-seeded Metropolis splitting estimator (arXiv:2511.15177 §5).

This is the "splitting method" / "multi-seeded Metropolis sampling" rare-event
estimator. It estimates the logical failure probability P(q) of a QEC memory
circuit as a function of the per-mechanism fault rate q, deep in the low-q regime
where direct Monte Carlo never observes a failure.

It is a COMPLEMENTARY cross-check to the direct importance-sampling + ansatz
pipeline in :mod:`importance_sampling` (Techniques I/II), NOT a guaranteed
improvement. The paper is explicit that the Metropolis mixing time is unknown and
can blow up; the failing-configuration space is also generally disconnected under
single-flip moves (badly so for codes with many inequivalent logicals — the gross
[[144,12,12]] code has 12 logical qubits). We mitigate, but do not solve, that with
multiple independent seeded chains. Treat its output as a sanity cross-check whose
agreement (or disagreement) with the IS/ansatz curve is itself the signal.

Expanded representation (shared with :mod:`importance_sampling`)
---------------------------------------------------------------
A fault configuration is a SUBSET ``x`` of expanded-column indices ``{0..N_exp-1}``.
Its weight ``|x|`` is the number of chosen columns. Its syndrome and logical action
are the XOR over the *source mechanisms* of the chosen columns:

    mechs    = col_to_mech[sorted(x)]
    syndrome = XOR_reduce(det_mat[mechs])
    truth    = XOR_reduce(obs_mat[mechs])

(Two expanded columns sharing a mechanism cancel — identical to
``importance_sampling._sample_failures_at_weight``.) A config "fails" iff the
decoder's prediction from ``syndrome`` differs from ``truth``.

Algorithm (§5)
--------------
1. Splitting in error rate. Pick a decreasing ladder q_0 > q_1 > ... > q_n, with q_0
   high enough that direct MC gives a measurable P(q_0). Then

       P(q_n) = P(q_0) · Π_{i=0}^{n-1} [ P(q_{i+1}) / P(q_i) ].

2. Each ratio via a failure-restricted Metropolis chain. Target distribution

       π_q(x) ∝ q^|x| (1-q)^(N_exp-|x|) · 1[x fails].

   Single-column flip proposals (add or remove one expanded column), accepted with
   the standard Metropolis ratio of q^|x|(1-q)^(N_exp-|x|), and any proposal that
   does NOT fail is rejected (the chain stays inside the failing set; one decode per
   Metropolis-accepted candidate). The ratio estimate is the reweighting expectation

       P(q_{i+1})/P(q_i) = E_{π_{q_i}}[ (q_{i+1}/q_i)^|x| ·
                                        ((1-q_{i+1})/(1-q_i))^(N_exp-|x|) ].

   With N_exp ~ 2e5 the (1-q) factors underflow, so all ratio accumulation is done
   in log-space (log-sum-exp).

3. Multi-seeded. Run several independent chains from different seeds and combine.
   Seeds come from two sources:
     (a) typical failing configs found by direct MC at the high rate q_0;
     (b) the distinct min-weight logical operators from Technique II
         (:func:`min_weight.find_min_weight_logicals`), which live in *mechanism*
         index space; we lift each to a weight-|S| failing expanded-column seed by
         picking, for each mechanism, one expanded column mapping to it.

Recommended full memory-circuit run and validation: see
``notebooks/gross_code/SPLITTING.md``, or run this module directly:

    .venv/bin/python src/splitting.py --p-high 0.006 --p-low 0.003 \
        --n-levels 6 --n-seeds 8 --chain-steps 4000 --burn-in 1000

(Those are *illustrative*; convergence is not guaranteed — increase chain-steps and
n-seeds and watch the per-seed spread.)

KNOWN LIMITATION (single-flip mixing). The proposal flips one of N_exp~2e5 expanded
columns per step, so at low q an "add" is accepted with prob ~q/(1-q)~q. The chain
moves only ~q per step and needs O(1/q) steps to equilibrate — the illustrative
chain-steps above are likely orders of magnitude too short at the lower ladder
levels. Compounding this: every level is currently seeded from the SAME fixed seed
pool, NOT warm-started from the adjacent higher-q level's final configs as the
paper's splitting prescribes, so lower levels start cold from too-high-weight
configs and bias their ratios downward. Treat this as a scaffold to validate
against IS where they overlap, not yet a converged low-p extrapolator. See
SPLITTING.md "caveats" for recommended next steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import stim
from scipy.special import logsumexp

from importance_sampling import _expand, _parse_dem


# ---------------------------------------------------------------------------
# Config -> syndrome / truth / failure  (mirrors _sample_failures_at_weight)
# ---------------------------------------------------------------------------

def _config_syndrome_truth(
    det_mat: np.ndarray,
    obs_mat: np.ndarray,
    col_to_mech: np.ndarray,
    cols: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Syndrome (M,) and truth (K,) for a single expanded-column config.

    XOR-reduces the source-mechanism rows of the chosen columns, exactly as
    ``importance_sampling._sample_failures_at_weight`` does. An empty config maps
    to the all-zero syndrome/truth.
    """
    M = det_mat.shape[1]
    K = obs_mat.shape[1]
    if len(cols) == 0:
        return np.zeros(M, dtype=bool), np.zeros(K, dtype=bool)
    mech_idxs = col_to_mech[np.fromiter(cols, dtype=np.int64, count=len(cols))]
    syndrome = np.bitwise_xor.reduce(det_mat[mech_idxs], axis=0)
    truth = np.bitwise_xor.reduce(obs_mat[mech_idxs], axis=0)
    return syndrome, truth


def _config_fails(
    det_mat: np.ndarray,
    obs_mat: np.ndarray,
    col_to_mech: np.ndarray,
    cols: Sequence[int],
    decoder,
) -> bool:
    """True iff this single config is a decoding failure (one decode call)."""
    syndrome, truth = _config_syndrome_truth(det_mat, obs_mat, col_to_mech, cols)
    pred = decoder.decode_batch(syndrome[None, :])  # (1, K)
    return bool(np.any(pred[0] != truth))


# ---------------------------------------------------------------------------
# Seed acquisition
# ---------------------------------------------------------------------------

def _mech_to_cols(col_to_mech: np.ndarray) -> Dict[int, List[int]]:
    """Map each mechanism index -> list of expanded-column indices mapping to it."""
    out: Dict[int, List[int]] = {}
    for c, j in enumerate(col_to_mech.tolist()):
        out.setdefault(j, []).append(c)
    return out


def min_weight_logical_seeds(
    circuit: stim.Circuit,
    col_to_mech: np.ndarray,
    det_mat: np.ndarray,
    obs_mat: np.ndarray,
    decoder,
    *,
    distance: Optional[int] = None,
    max_trials: int = 400,
    seed: Optional[int] = None,
) -> List[FrozenSet[int]]:
    """Technique-II min-weight logicals lifted to failing expanded-column seeds.

    :func:`min_weight.find_min_weight_logicals` returns frozensets of *mechanism*
    indices (columns of H). For each such support S we pick, for every mechanism
    j in S, the first expanded column c with ``col_to_mech[c] == j`` (a precomputed
    mechanism->columns map), giving a weight-|S| expanded-column config. We keep
    only configs that actually fail under ``decoder`` (they should, being logicals,
    but BP-OSD's min-weight search is heuristic, so we verify each).
    """
    from min_weight import find_min_weight_logicals

    supports = find_min_weight_logicals(
        circuit, distance, max_trials=max_trials, seed=seed
    )
    m2c = _mech_to_cols(col_to_mech)
    seeds: List[FrozenSet[int]] = []
    for support in supports:
        cols = []
        ok = True
        for j in support:
            cs = m2c.get(int(j))
            if not cs:
                ok = False
                break
            cols.append(cs[0])
        if not ok:
            continue
        cfg = frozenset(cols)
        if _config_fails(det_mat, obs_mat, col_to_mech, cfg, decoder):
            seeds.append(cfg)
    return seeds


def high_rate_mc_seeds(
    det_mat: np.ndarray,
    obs_mat: np.ndarray,
    col_to_mech: np.ndarray,
    decoder,
    q_high: float,
    *,
    n_seeds: int,
    max_shots: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> List[FrozenSet[int]]:
    """Find up to ``n_seeds`` typical failing configs by direct Bernoulli(q_high) MC.

    Draws batches of i.i.d. Bernoulli(q_high) expanded-column configs, decodes them,
    and keeps the ones that fail. These are "typical" high-rate failing seeds (source
    (a) in §5), complementing the rare low-weight min-weight seeds (source (b)).
    """
    if rng is None:
        rng = np.random.default_rng()
    N_exp = col_to_mech.shape[0]
    M = det_mat.shape[1]
    K = obs_mat.shape[1]
    found: List[FrozenSet[int]] = []
    shots_done = 0
    batch = max(64, n_seeds * 8)
    while len(found) < n_seeds and shots_done < max_shots:
        b = min(batch, max_shots - shots_done)
        configs: List[np.ndarray] = []
        syndromes = np.zeros((b, M), dtype=bool)
        truths = np.zeros((b, K), dtype=bool)
        for t in range(b):
            cols = np.flatnonzero(rng.random(N_exp) < q_high)
            configs.append(cols)
            syndromes[t], truths[t] = _config_syndrome_truth(
                det_mat, obs_mat, col_to_mech, cols
            )
        preds = decoder.decode_batch(syndromes)
        fails = np.any(preds != truths, axis=1)
        for t in np.flatnonzero(fails):
            found.append(frozenset(configs[t].tolist()))
            if len(found) >= n_seeds:
                break
        shots_done += b
    return found


# ---------------------------------------------------------------------------
# Failure-restricted Metropolis chain
# ---------------------------------------------------------------------------

@dataclass
class ChainResult:
    """Diagnostics for one Metropolis chain run at a single level q_i.

    log_ratio_terms : log of each per-sample reweight term
        |x| log(q_next/q_cur) + (N_exp-|x|) log((1-q_next)/(1-q_cur))
    """
    accept_rate: float
    mean_weight: float
    n_samples: int
    log_ratio_mean: float           # log of the Monte-Carlo mean of the reweight term
    log_ratio_terms: np.ndarray     # per-sample log terms (for combining/variance)


def _metropolis_chain(
    det_mat: np.ndarray,
    obs_mat: np.ndarray,
    col_to_mech: np.ndarray,
    decoder,
    q_cur: float,
    q_next: float,
    seed_config: FrozenSet[int],
    *,
    chain_steps: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
) -> ChainResult:
    """Run one failure-restricted Metropolis chain targeting π_{q_cur}, accumulating
    the reweighting term toward q_next in log-space.

    Proposal: pick a uniform expanded column c in {0..N_exp-1}; if c is in the
    current set, propose removing it (|x| -> |x|-1), else propose adding it
    (|x| -> |x|+1). Metropolis acceptance for the unnormalised target
    q_cur^|x| (1-q_cur)^(N_exp-|x|): adding multiplies the density by
    q_cur/(1-q_cur), removing by (1-q_cur)/q_cur. The 1[x fails] indicator is
    enforced by hard-rejecting non-failing proposals (revert the flip). The seed
    config must fail.
    """
    N_exp = col_to_mech.shape[0]
    log_odds = np.log(q_cur) - np.log1p(-q_cur)  # log(q/(1-q)); add => +, remove => -

    log_r_next = np.log(q_next) - np.log(q_cur)
    log_r_1m = np.log1p(-q_next) - np.log1p(-q_cur)

    cur = set(int(c) for c in seed_config)
    if not _config_fails(det_mat, obs_mat, col_to_mech, cur, decoder):
        raise ValueError("seed_config does not fail under the decoder")

    accepts = 0
    proposals = 0
    log_terms: List[float] = []
    weights_seen: List[int] = []

    total = burn_in + chain_steps
    for step in range(total):
        proposals += 1
        c = int(rng.integers(N_exp))
        adding = c not in cur
        # Metropolis log-acceptance from the q^|x|(1-q)^(N-|x|) ratio.
        log_alpha = log_odds if adding else -log_odds
        if np.log(rng.random()) < min(0.0, log_alpha):
            # Tentatively flip, then enforce the failing-set indicator with a decode.
            if adding:
                cur.add(c)
            else:
                cur.discard(c)
            if _config_fails(det_mat, obs_mat, col_to_mech, cur, decoder):
                accepts += 1
            else:
                # Reject: revert the flip (stay in the failing set).
                if adding:
                    cur.discard(c)
                else:
                    cur.add(c)
        # else: Metropolis rejection, no decode needed.

        if step >= burn_in and ((step - burn_in) % thin == 0):
            w = len(cur)
            weights_seen.append(w)
            log_terms.append(w * log_r_next + (N_exp - w) * log_r_1m)

    log_terms_arr = np.asarray(log_terms, dtype=float)
    if log_terms_arr.size == 0:
        log_ratio_mean = -np.inf
    else:
        log_ratio_mean = float(logsumexp(log_terms_arr) - np.log(log_terms_arr.size))

    return ChainResult(
        accept_rate=accepts / proposals if proposals else 0.0,
        mean_weight=float(np.mean(weights_seen)) if weights_seen else float("nan"),
        n_samples=int(log_terms_arr.size),
        log_ratio_mean=log_ratio_mean,
        log_ratio_terms=log_terms_arr,
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SplittingResult:
    """Result of a multi-seeded splitting run.

    q_ladder / p_ladder : the rate ladder, length n_levels+1 (q_0 .. q_n).
    P_logical    : estimate of P at each ladder rate (length n_levels+1).
    P_logical_se : crude SE from the across-seed spread of the cumulative product.
    log_ratios   : per-level combined log[P(q_{i+1})/P(q_i)] (length n_levels).
    log_ratios_se: across-seed SE of each per-level log-ratio (length n_levels).
    P_high       : direct-MC anchor P(q_0).
    P_high_se    : binomial SE of the anchor.
    per_level_chains : list (len n_levels) of lists of ChainResult (one per seed).
    n_seeds_used : number of seed chains actually run.
    """
    q_ladder: np.ndarray
    p_ladder: np.ndarray
    P_logical: np.ndarray
    P_logical_se: np.ndarray
    log_ratios: np.ndarray
    log_ratios_se: np.ndarray
    P_high: float
    P_high_se: float
    per_level_chains: List[List[ChainResult]]
    n_seeds_used: int
    q_base: float
    p_ref: float
    n_expanded: int


# ---------------------------------------------------------------------------
# Direct-MC anchor at q_0
# ---------------------------------------------------------------------------

def _direct_mc_failure_prob(
    det_mat: np.ndarray,
    obs_mat: np.ndarray,
    col_to_mech: np.ndarray,
    decoder,
    q_high: float,
    shots: int,
    rng: np.random.Generator,
) -> Tuple[float, float, List[FrozenSet[int]]]:
    """Direct Bernoulli(q_high) MC: returns (P_hat, binomial_SE, failing_configs)."""
    N_exp = col_to_mech.shape[0]
    M = det_mat.shape[1]
    K = obs_mat.shape[1]
    syndromes = np.zeros((shots, M), dtype=bool)
    truths = np.zeros((shots, K), dtype=bool)
    configs: List[np.ndarray] = []
    for t in range(shots):
        cols = np.flatnonzero(rng.random(N_exp) < q_high)
        configs.append(cols)
        syndromes[t], truths[t] = _config_syndrome_truth(
            det_mat, obs_mat, col_to_mech, cols
        )
    preds = decoder.decode_batch(syndromes)
    fails = np.any(preds != truths, axis=1)
    n_fail = int(fails.sum())
    P_hat = n_fail / shots
    se = float(np.sqrt(max(P_hat * (1.0 - P_hat), 1e-12) / shots))
    failing = [frozenset(configs[t].tolist()) for t in np.flatnonzero(fails)]
    return P_hat, se, failing


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def splitting_estimate(
    circuit: stim.Circuit,
    decoder,
    *,
    p_ref: float,
    p_high: float,
    p_low: float,
    n_levels: int,
    n_seeds: int,
    chain_steps: int,
    burn_in: int,
    thin: int = 1,
    anchor_shots: int = 2000,
    q_base: Optional[float] = None,
    use_min_weight_seeds: bool = True,
    min_weight_max_trials: int = 400,
    distance: Optional[int] = None,
    seed: Optional[int] = None,
) -> SplittingResult:
    """Multi-seeded Metropolis splitting estimate of P_logical from p_high to p_low.

    Parameters
    ----------
    circuit, decoder : noisy Stim circuit (built at ``p_ref``) and a decoder
        implementing ``setup(circuit)`` + ``decode_batch(events)``.
    p_ref : physical error rate the circuit was built at; q = q_base*(p/p_ref).
    p_high, p_low : top/bottom of the physical-rate ladder. p_high should be high
        enough that direct MC sees failures; p_low is the target rare rate.
    n_levels : number of ladder *steps* (the ladder has n_levels+1 rates).
    n_seeds : number of independent seed chains per level.
    chain_steps, burn_in, thin : Metropolis chain length, warm-up, and thinning.
    anchor_shots : direct-MC shots for the P(q_0) anchor.
    q_base : expanded-rep base rate (default min DEM probability, as in IS).
    use_min_weight_seeds : also seed with Technique-II min-weight logicals.
    distance : pass the known code distance to skip the min-weight distance search.
    seed : RNG seed.

    Returns
    -------
    SplittingResult

    Notes
    -----
    Combining across seeds: per level i we pool the per-sample reweight terms from
    all seed chains (a pooled Monte-Carlo mean of the ratio, in log-space), giving
    one combined log-ratio per level. The product down the ladder gives P(q_n). The
    SE is a crude across-seed propagation and is descriptive only — it does NOT
    certify Metropolis convergence.
    """
    rng = np.random.default_rng(seed)

    probs, det_mat, obs_mat = _parse_dem(circuit)
    col_to_mech, q_base, _ = _expand(probs, q_base)
    N_exp = col_to_mech.shape[0]

    decoder.setup(circuit)

    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")

    # Rate ladder (geometric in p for an even spread across orders of magnitude).
    p_ladder = np.geomspace(p_high, p_low, n_levels + 1)
    q_ladder = np.clip(q_base * (p_ladder / p_ref), 1e-300, 1.0 - 1e-15)
    q_high = float(q_ladder[0])

    # Anchor P(q_0) via direct MC; also harvest typical failing seeds (source a).
    P_high, P_high_se, mc_seeds = _direct_mc_failure_prob(
        det_mat, obs_mat, col_to_mech, decoder, q_high, anchor_shots, rng
    )
    if P_high <= 0.0:
        raise ValueError(
            f"direct MC at p_high={p_high:g} (q={q_high:g}) saw no failures over "
            f"{anchor_shots} shots; raise p_high or anchor_shots so the ladder has "
            "a measurable anchor."
        )

    # Source (b): min-weight-logical seeds (Technique II), if requested.
    mw_seeds: List[FrozenSet[int]] = []
    if use_min_weight_seeds:
        mw_seeds = min_weight_logical_seeds(
            circuit, col_to_mech, det_mat, obs_mat, decoder,
            distance=distance, max_trials=min_weight_max_trials,
            seed=int(rng.integers(2**31)),
        )

    # Assemble the seed pool: a mix of (b) min-weight and (a) typical seeds.
    seed_pool: List[FrozenSet[int]] = []
    seed_pool.extend(mw_seeds)
    seed_pool.extend(mc_seeds)
    if not seed_pool:
        raise ValueError("no failing seed configurations could be found")
    # Top up with extra high-rate MC seeds if the pool is too small.
    if len(seed_pool) < n_seeds:
        extra = high_rate_mc_seeds(
            det_mat, obs_mat, col_to_mech, decoder, q_high,
            n_seeds=n_seeds - len(seed_pool), rng=rng,
        )
        seed_pool.extend(extra)
    # Choose n_seeds seeds (with replacement only if the pool is still too small).
    replace = len(seed_pool) < n_seeds
    idx = rng.choice(len(seed_pool), size=n_seeds, replace=replace)
    chosen_seeds = [seed_pool[int(i)] for i in idx]
    n_seeds_used = len(chosen_seeds)

    # Run a chain per (level, seed). Each level's chain targets π_{q_i} and
    # measures the reweight term toward q_{i+1}.
    per_level_chains: List[List[ChainResult]] = []
    combined_log_ratio = np.zeros(n_levels)
    combined_log_ratio_se = np.zeros(n_levels)

    for i in range(n_levels):
        q_cur = float(q_ladder[i])
        q_next = float(q_ladder[i + 1])
        chains: List[ChainResult] = []
        per_seed_log_ratio: List[float] = []
        for sc in chosen_seeds:
            cr = _metropolis_chain(
                det_mat, obs_mat, col_to_mech, decoder,
                q_cur, q_next, sc,
                chain_steps=chain_steps, burn_in=burn_in, thin=thin,
                rng=np.random.default_rng(int(rng.integers(2**31))),
            )
            chains.append(cr)
            if np.isfinite(cr.log_ratio_mean):
                per_seed_log_ratio.append(cr.log_ratio_mean)
        per_level_chains.append(chains)

        # Pool all per-sample terms across seeds for the combined ratio estimate.
        nonempty = [c.log_ratio_terms for c in chains if c.log_ratio_terms.size]
        if nonempty:
            all_terms = np.concatenate(nonempty)
            combined_log_ratio[i] = float(logsumexp(all_terms) - np.log(all_terms.size))
        else:
            combined_log_ratio[i] = -np.inf
        # Across-seed spread of the per-seed log-ratio as a crude SE.
        if len(per_seed_log_ratio) > 1:
            arr = np.asarray(per_seed_log_ratio)
            combined_log_ratio_se[i] = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        else:
            combined_log_ratio_se[i] = float("nan")

    # Cumulative products down the ladder: P(q_k) = P(q_0) * prod_{i<k} ratio_i.
    log_P_high = np.log(max(P_high, 1e-300))
    cum_log_ratio = np.concatenate([[0.0], np.cumsum(combined_log_ratio)])
    log_P = log_P_high + cum_log_ratio
    P_logical = np.exp(log_P)

    # Crude SE propagation: anchor relative SE + per-level log-ratio SEs in quadrature.
    rel_anchor = (P_high_se / P_high) if P_high > 0 else 0.0
    cum_log_var = np.concatenate(
        [[0.0], np.cumsum(np.nan_to_num(combined_log_ratio_se, nan=0.0) ** 2)]
    )
    rel_se = np.sqrt(rel_anchor**2 + cum_log_var)
    P_logical_se = P_logical * rel_se

    return SplittingResult(
        q_ladder=q_ladder,
        p_ladder=p_ladder,
        P_logical=P_logical,
        P_logical_se=P_logical_se,
        log_ratios=combined_log_ratio,
        log_ratios_se=combined_log_ratio_se,
        P_high=P_high,
        P_high_se=P_high_se,
        per_level_chains=per_level_chains,
        n_seeds_used=n_seeds_used,
        q_base=q_base,
        p_ref=p_ref,
        n_expanded=N_exp,
    )


# ---------------------------------------------------------------------------
# CLI / validation entry point (does NOT run on import)
# ---------------------------------------------------------------------------

def _build_gross_memory_circuit(p_ref: float = 0.005):
    """Build the [[144,12,12]] gross-code memory circuit exactly as the sweep does."""
    from bb_code_sim import BBCodeSimulator, BB_144_12_12
    from surface_code_sim import ErrorModel

    em = ErrorModel.symmetric(p_ref)
    circuit = BBCodeSimulator(BB_144_12_12).build_circuit(em, rounds=BB_144_12_12.distance)
    return circuit, p_ref


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Technique III splitting estimate on the [[144,12,12]] memory "
                    "circuit, cross-checked against importance-sampling + ansatz."
    )
    parser.add_argument("--p-ref", type=float, default=0.005)
    parser.add_argument("--p-high", type=float, default=0.006)
    parser.add_argument("--p-low", type=float, default=0.003)
    parser.add_argument("--n-levels", type=int, default=6)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--chain-steps", type=int, default=4000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--anchor-shots", type=int, default=3000)
    parser.add_argument("--is-shots-per-weight", type=int, default=2000)
    parser.add_argument(
        "--is-weights-hi", type=int, default=160,
        help="IS cross-check samples contiguous fault weights 1..is_weights_hi. Must "
             "bracket the failure onset (~47 for the gross memory circuit) AND the "
             "dominant binomial mass mu=N_exp*q over [p_low,p_high] (~88 at p=0.006). "
             "Too low (e.g. <onset) gives an all-zero spectrum and the ansatz fit raises.",
    )
    parser.add_argument("--no-min-weight-seeds", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true",
                        help="save a comparison plot to splitting_vs_is.png")
    args = parser.parse_args(argv)

    from bb_code_sim import RelayBPDecoder
    from importance_sampling import importance_sample_with_ansatz

    circuit, p_ref = _build_gross_memory_circuit(args.p_ref)

    print(f"[splitting] building gross memory circuit at p_ref={p_ref}")
    res = splitting_estimate(
        circuit, RelayBPDecoder(),
        p_ref=p_ref, p_high=args.p_high, p_low=args.p_low,
        n_levels=args.n_levels, n_seeds=args.n_seeds,
        chain_steps=args.chain_steps, burn_in=args.burn_in, thin=args.thin,
        anchor_shots=args.anchor_shots,
        use_min_weight_seeds=not args.no_min_weight_seeds,
        distance=12, seed=args.seed,
    )

    print(f"[splitting] anchor P(p_high={args.p_high:g}) = "
          f"{res.P_high:.3e} +/- {res.P_high_se:.1e}  (seeds used: {res.n_seeds_used})")
    print("[splitting] ladder estimate:")
    for p, P, se in zip(res.p_ladder, res.P_logical, res.P_logical_se):
        print(f"    p={p:.4g}   P_split={P:.3e} +/- {se:.1e}")

    # Cross-check with direct IS + ansatz over the overlapping p-range.
    print("[is] running importance_sample_with_ansatz for cross-check ...")
    is_res = importance_sample_with_ansatz(
        circuit, RelayBPDecoder(), p_ref=p_ref,
        p_values=res.p_ladder, model="f3",
        weights=list(range(1, 13)),
        shots_per_weight=args.is_shots_per_weight, seed=args.seed,
    )
    print("[compare]   p        P_split       P_is_raw      P_ansatz")
    for k, p in enumerate(res.p_ladder):
        print(f"    {p:.4g}   {res.P_logical[k]:.3e}   "
              f"{is_res.raw.P_logical[k]:.3e}   {is_res.P_logical_ansatz[k]:.3e}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(res.p_ladder, res.P_logical, yerr=res.P_logical_se,
                    marker="o", label="splitting (Technique III)")
        ax.plot(res.p_ladder, is_res.raw.P_logical, "s--", label="IS raw")
        ax.plot(res.p_ladder, is_res.P_logical_ansatz, "x-", label="ansatz (Tech I)")
        ax.set_xlabel("physical error rate p")
        ax.set_ylabel("logical failure probability")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("[[144,12,12]] memory: splitting vs IS+ansatz")
        fig.tight_layout()
        fig.savefig("splitting_vs_is.png", dpi=120)
        print("[plot] wrote splitting_vs_is.png")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
