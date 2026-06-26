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
moves only ~q per step and needs O(1/q) steps to equilibrate. At p=0.001,
q~6.7e-5, so chain_steps=4000 gives ~0.3 expected moves — essentially frozen.
Use chain_steps >= 50_000 and keep p_low >= 0.003 for meaningful mixing.
Warm-starting between levels (paper §5 prescription) is now implemented: each
level's chains start from the final states of the previous level's chains.
"""

from __future__ import annotations

import itertools
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
    sector: Optional[int] = None,
    supports=None,
) -> List[FrozenSet[int]]:
    """Technique-II min-weight logicals lifted to failing expanded-column seeds.

    :func:`min_weight.find_min_weight_logicals` returns frozensets of *mechanism*
    indices (columns of H). For each such support S we pick, for every mechanism
    j in S, the first expanded column c with ``col_to_mech[c] == j`` (a precomputed
    mechanism->columns map), giving a weight-|S| expanded-column config. We keep
    only configs that actually fail under ``decoder`` (they should, being logicals,
    but BP-OSD's min-weight search is heuristic, so we verify each).
    """
    if supports is None:
        from min_weight import find_min_weight_logicals
        supports = find_min_weight_logicals(
            circuit, distance, max_trials=max_trials, seed=seed, sector=sector
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
) -> Tuple[ChainResult, FrozenSet[int]]:
    """Run one failure-restricted Metropolis chain targeting π_{q_cur}, accumulating
    the reweighting term toward q_next in log-space.

    Returns (ChainResult, final_config) where final_config is the chain's last state
    (always a failing config — the chain never leaves the failing set). Pass it as
    seed_config for the next ladder level to warm-start the chain.

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
    ), frozenset(cur)


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
    single_sector: bool = False,
    sector: int = 0,
    mw_supports=None,
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

    if single_sector:
        # Stim circuit -> single Z-sector projection (the paper's representation), decoded by
        # Relay-BP on the projected matrices — matching the IS sweep's representation so the
        # splitting cross-check is on the SAME N/q_base, not the full both-sector DEM.
        from min_weight import single_sector_dem
        H, A, mult, probs_merged, _ = single_sector_dem(circuit, detector_type=sector)
        det_mat = H.T.astype(bool); obs_mat = A.T.astype(bool)
        if q_base is None:
            q_base = float(_parse_dem(circuit)[0].min())   # expansion base rate (~p/15)
        col_to_mech = np.repeat(np.arange(H.shape[1], dtype=np.int64), mult)
        decoder.setup_from_matrices(H, probs_merged, A)
    else:
        probs, det_mat, obs_mat = _parse_dem(circuit)
        col_to_mech, q_base, _ = _expand(probs, q_base)
        decoder.setup(circuit)
    N_exp = col_to_mech.shape[0]

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
            sector=(sector if single_sector else None),
            supports=mw_supports,
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
    # Warm-starting: after each level the final chain states (always failing configs)
    # become the seeds for the next level, matching the paper's prescription (§5).
    per_level_chains: List[List[ChainResult]] = []
    combined_log_ratio = np.zeros(n_levels)
    combined_log_ratio_se = np.zeros(n_levels)

    level_seeds = list(chosen_seeds)  # updated after each level (warm-start)

    for i in range(n_levels):
        q_cur = float(q_ladder[i])
        q_next = float(q_ladder[i + 1])
        chains: List[ChainResult] = []
        next_level_seeds: List[FrozenSet[int]] = []
        per_seed_log_ratio: List[float] = []
        for sc in level_seeds:
            cr, final_cfg = _metropolis_chain(
                det_mat, obs_mat, col_to_mech, decoder,
                q_cur, q_next, sc,
                chain_steps=chain_steps, burn_in=burn_in, thin=thin,
                rng=np.random.default_rng(int(rng.integers(2**31))),
            )
            chains.append(cr)
            next_level_seeds.append(final_cfg)
            if np.isfinite(cr.log_ratio_mean):
                per_seed_log_ratio.append(cr.log_ratio_mean)
        level_seeds = next_level_seeds  # warm-start: carry final states to next level
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


def _local_metropolis(cur, q, n_steps, det_mat, obs_mat, col_to_mech, decoder, rng):
    """In-place Metropolis-Hastings on the failing-set-restricted target q^|x|(1-q)^(N-|x|),
    using a **balanced** add/remove proposal so weight mixes in O(steps), not O(N).

    Each step proposes (50/50) adding a uniform non-member or removing a uniform member; the
    Hastings factor (N-|x|)/(|x|+1) [add] or |x|/(N-|x|+1) [remove] corrects the asymmetric
    proposal. `cur` is a set of expanded columns (must already fail); returns the accept count.
    """
    N_exp = col_to_mech.shape[0]
    log_odds = np.log(q) - np.log1p(-q)
    logN = np.log(N_exp)
    accepts = 0
    for _ in range(n_steps):
        nx = len(cur)
        if nx == 0 or rng.random() < 0.5:
            # propose ADD a uniform non-member; accept min(1, q/(1-q) * (N-nx)/(nx+1))
            c = int(rng.integers(N_exp))
            while c in cur:
                c = int(rng.integers(N_exp))
            log_acc = log_odds + np.log(N_exp - nx) - np.log(nx + 1)
            if np.log(rng.random()) < min(0.0, log_acc):
                cur.add(c)
                if _config_fails(det_mat, obs_mat, col_to_mech, cur, decoder):
                    accepts += 1
                else:
                    cur.discard(c)
        else:
            # propose REMOVE a uniform member; accept min(1, (1-q)/q * nx/(N-nx+1))
            c = next(itertools.islice(cur, int(rng.integers(nx)), None))
            log_acc = -log_odds + np.log(nx) - np.log(N_exp - nx + 1)
            if np.log(rng.random()) < min(0.0, log_acc):
                cur.discard(c)
                if _config_fails(det_mat, obs_mat, col_to_mech, cur, decoder):
                    accepts += 1
                else:
                    cur.add(c)
    return accepts


def replica_exchange_estimate(
    circuit: stim.Circuit,
    decoder,
    *,
    p_ref: float,
    p_high: float,
    p_low: float,
    n_levels: int,
    n_walkers: int = 12,
    local_steps: int = 25,
    n_sweeps: int = 400,
    burn_in: int = 100,
    thin: int = 1,
    anchor_shots: int = 5000,
    q_base: Optional[float] = None,
    distance: Optional[int] = None,
    seed: Optional[int] = None,
    single_sector: bool = False,
    sector: int = 0,
    mw_supports=None,
    verbose: bool = True,
) -> Tuple[SplittingResult, dict]:
    """Replica-exchange (parallel-tempering) splitting estimate of P_logical(p_low).

    Fixes the weight-space non-mixing of :func:`splitting_estimate`: instead of independent
    warm-started chains (which stay trapped near their seed weight), it runs ``n_walkers``
    walkers, each holding one failing replica per ladder rate, and alternates local single-flip
    Metropolis moves with **swaps between adjacent rates**. Swaps need no decode (both configs
    already fail) and accept with log A = (|x_{i+1}|-|x_i|)(logodds_i - logodds_{i+1}), so
    high-weight configs migrate up-ladder and low-weight down — equilibrating each level's weight
    distribution `pi_{q_i}(w) ~ f(w) C(N,w) q^w (1-q)^(N-w)`, which is what the reweight ratio needs.

    Returns (SplittingResult, diagnostics) where diagnostics has per-pair swap-accept rates and
    per-level mean weights (mean weight should rise with q when mixed).
    """
    rng = np.random.default_rng(seed)

    if single_sector:
        from min_weight import single_sector_dem
        H, A, mult, probs_merged, _ = single_sector_dem(circuit, detector_type=sector)
        det_mat = H.T.astype(bool); obs_mat = A.T.astype(bool)
        if q_base is None:
            q_base = float(_parse_dem(circuit)[0].min())
        col_to_mech = np.repeat(np.arange(H.shape[1], dtype=np.int64), mult)
        decoder.setup_from_matrices(H, probs_merged, A)
    else:
        probs, det_mat, obs_mat = _parse_dem(circuit)
        col_to_mech, q_base, _ = _expand(probs, q_base)
        decoder.setup(circuit)
    N_exp = col_to_mech.shape[0]
    L = n_levels
    if L < 1:
        raise ValueError("n_levels must be >= 1")

    p_ladder = np.geomspace(p_high, p_low, L + 1)
    q_ladder = np.clip(q_base * (p_ladder / p_ref), 1e-300, 1.0 - 1e-15)
    logodds = np.log(q_ladder) - np.log1p(-q_ladder)               # per level
    log_r_next = np.log(q_ladder[1:]) - np.log(q_ladder[:-1])      # toward q_{i+1}
    log_r_1m = np.log1p(-q_ladder[1:]) - np.log1p(-q_ladder[:-1])

    # Anchor P(q_0) by direct MC (also harvests typical high-rate failing seeds).
    P_high, P_high_se, mc_seeds = _direct_mc_failure_prob(
        det_mat, obs_mat, col_to_mech, decoder, float(q_ladder[0]), anchor_shots, rng)
    if P_high <= 0.0:
        raise ValueError(f"direct MC at p_high={p_high:g} saw no failures; raise p_high/anchor_shots")

    # Failing-config seed pool: min-weight logicals (low weight) + MC typical (high weight).
    pool: List[FrozenSet[int]] = []
    pool += min_weight_logical_seeds(circuit, col_to_mech, det_mat, obs_mat, decoder,
                                     distance=distance, seed=int(rng.integers(2**31)),
                                     sector=(sector if single_sector else None), supports=mw_supports)
    pool += mc_seeds
    if not pool:
        raise ValueError("no failing seed configs found")

    # Initialise walkers: each is a list of L+1 failing configs (one per rate). Seed lower rates
    # with low-weight (min-weight) configs and higher rates with high-weight (MC) configs when
    # available, to give tempering a head start; swaps fix any mismatch.
    lo_pool = [s for s in pool if len(s) <= distance + 4] or pool   # low-weight (min-weight logicals)
    hi_pool = mc_seeds or pool                                       # high-weight (typical at q_high)
    def pick(p): return set(p[int(rng.integers(len(p)))])
    # level 0 = highest q (seed high weight) ... level L = lowest q (seed low weight)
    replicas = [[pick(hi_pool if i <= L // 2 else lo_pool) for i in range(L + 1)]
                for _ in range(n_walkers)]

    swap_attempts = np.zeros(L); swap_accepts = np.zeros(L)
    walker_terms = [[[] for _ in range(L)] for _ in range(n_walkers)]
    wsum = np.zeros(L + 1); wcount = 0

    M = det_mat.shape[1]; K = obs_mat.shape[1]
    import time as _time; _t0 = _time.time()
    if verbose:
        print(f"  [tempering] setup done (anchor+seeds); running {n_sweeps} sweeps "
              f"x {n_walkers} walkers x {L+1} rungs ...", flush=True)
    for sweep in range(n_sweeps):
        if verbose and (sweep % 10 == 0 or sweep == n_sweeps - 1):
            _el = _time.time() - _t0
            _eta = _el / max(sweep, 1) * (n_sweeps - sweep)
            print(f"  [tempering] sweep {sweep}/{n_sweeps}  ({_el:.0f}s, ETA {_eta:.0f}s)", flush=True)
        # (a) local moves — balanced add/remove proposal, with the failing-set indicator decoded
        #     in ONE batch across all replicas per sub-step (failing is q-independent).
        for _sub in range(local_steps):
            cands = []  # (w, i, candidate_set)
            for w in range(n_walkers):
                for i in range(L + 1):
                    cur = replicas[w][i]; nx = len(cur); lo = float(logodds[i])
                    if nx == 0 or rng.random() < 0.5:
                        c = int(rng.integers(N_exp))
                        while c in cur:
                            c = int(rng.integers(N_exp))
                        log_acc = lo + np.log(N_exp - nx) - np.log(nx + 1)
                        adding = True
                    else:
                        c = next(itertools.islice(cur, int(rng.integers(nx)), None))
                        log_acc = -lo + np.log(nx) - np.log(N_exp - nx + 1)
                        adding = False
                    if np.log(rng.random()) < min(0.0, log_acc):
                        cand = set(cur)
                        cand.add(c) if adding else cand.discard(c)
                        cands.append((w, i, cand))
            if cands:
                synd = np.zeros((len(cands), M), dtype=bool); truth = np.zeros((len(cands), K), dtype=bool)
                for k, (w, i, cand) in enumerate(cands):
                    synd[k], truth[k] = _config_syndrome_truth(det_mat, obs_mat, col_to_mech, cand)
                preds = decoder.decode_batch(synd)
                fails = np.any(preds != truth, axis=1)
                for k, (w, i, cand) in enumerate(cands):
                    if fails[k]:
                        replicas[w][i] = cand           # accept (still failing); else keep old
        # (b) swaps on adjacent pairs (even then odd) — no decode needed
        for parity in (0, 1):
            for i in range(parity, L, 2):
                d_logodds = float(logodds[i] - logodds[i + 1])
                for w in range(n_walkers):
                    xi, xj = replicas[w][i], replicas[w][i + 1]
                    swap_attempts[i] += 1
                    logA = (len(xj) - len(xi)) * d_logodds
                    if np.log(rng.random()) < min(0.0, logA):
                        replicas[w][i], replicas[w][i + 1] = xj, xi
                        swap_accepts[i] += 1
        # (c) collect reweight terms + weights (post burn-in)
        if sweep >= burn_in and ((sweep - burn_in) % thin == 0):
            wcount += 1
            for w in range(n_walkers):
                for i in range(L + 1):
                    wsum[i] += len(replicas[w][i])
                for i in range(L):
                    wt = len(replicas[w][i])
                    walker_terms[w][i].append(wt * log_r_next[i] + (N_exp - wt) * log_r_1m[i])

    # Combine: pooled logsumexp per level; SE from per-walker spread.
    combined = np.zeros(L); combined_se = np.zeros(L)
    for i in range(L):
        all_terms = np.concatenate([np.asarray(walker_terms[w][i]) for w in range(n_walkers)
                                    if walker_terms[w][i]])
        combined[i] = float(logsumexp(all_terms) - np.log(all_terms.size)) if all_terms.size else -np.inf
        per_w = [float(logsumexp(np.asarray(walker_terms[w][i])) - np.log(len(walker_terms[w][i])))
                 for w in range(n_walkers) if walker_terms[w][i]]
        combined_se[i] = float(np.std(per_w, ddof=1) / np.sqrt(len(per_w))) if len(per_w) > 1 else np.nan

    log_P = np.log(max(P_high, 1e-300)) + np.concatenate([[0.0], np.cumsum(combined)])
    P_logical = np.exp(log_P)
    rel = np.sqrt((P_high_se / P_high if P_high > 0 else 0.0) ** 2
                  + np.concatenate([[0.0], np.cumsum(np.nan_to_num(combined_se) ** 2)]))
    diag = {"swap_accept": (swap_accepts / np.maximum(swap_attempts, 1)).tolist(),
            "mean_weight": (wsum / max(wcount * n_walkers, 1)).tolist(),
            "P_high": float(P_high), "n_pool": len(pool), "n_collect": int(wcount)}
    if verbose:
        print(f"  [tempering] anchor P(q0)={P_high:.3e}, pool={len(pool)}, collected {wcount} sweeps", flush=True)
        print(f"  [tempering] swap-accept (adj pairs): "
              f"{', '.join(f'{x:.2f}' for x in diag['swap_accept'])}", flush=True)
        print(f"  [tempering] mean weight by level (hi-q..lo-q): "
              f"{', '.join(f'{x:.1f}' for x in diag['mean_weight'])}", flush=True)
    return SplittingResult(
        q_ladder=q_ladder, p_ladder=p_ladder, P_logical=P_logical, P_logical_se=P_logical * rel,
        log_ratios=combined, log_ratios_se=combined_se, P_high=P_high, P_high_se=P_high_se,
        per_level_chains=[], n_seeds_used=n_walkers, q_base=q_base, p_ref=p_ref, n_expanded=N_exp,
    ), diag


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
