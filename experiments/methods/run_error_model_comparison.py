"""Run (and cache) every simulation behind error_model_comparison_18_4_4.ipynb.

The notebook is a pure REPORT: it loads the JSON results this script writes under
``runs/error_model_comparison_18_4_4/`` and renders tables/plots (regenerate it with
``make_error_model_comparison.py``). This script owns ALL sampling: spectra, splitting,
direct MC, distances/onsets, the [[72,4,8]] sweeps, and the asymmetric-point sweeps.

Caching: one JSON file per task, holding {config, elapsed_s, finished_at, result}. A task
reruns only if its file is missing or its stored config differs from the current one — so
editing e.g. the decoder or a shot budget invalidates exactly the tasks it affects, and an
interrupted run resumes where it left off. Timing is recorded per task; the report notebook
aggregates it per section.

Usage:
    python run_error_model_comparison.py            # run everything not yet cached
    python run_error_model_comparison.py --list     # show cache status, run nothing
    python run_error_model_comparison.py --only tech1_72 asym   # name-prefix filter
    python run_error_model_comparison.py --force tech1_72__full_symmetric  # rerun these
    python run_error_model_comparison.py --boost72  # 2x failures / 5x shots on the 72-code
                                                    # spectra (tightens the Λ shares; new
                                                    # config -> those caches rerun)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np

from repo_paths import run_dir
from bb_code_sim import (BBCodeParams, BBCodeSimulator, RelayBPDecoder, NOISE_CHANNEL_PREDICATES,
                         NOISE_INSTRUCTIONS, BB_72_4_8, filter_noise_channel, scale_noise_channels)
from surface_code_sim import ErrorModel
from min_weight import (dem_check_action_matrices, compute_distance, optimal_onset_fraction,
                        find_weight_logicals_mitm, find_all_min_weight_logicals)
from importance_sampling import (importance_sample_adaptive, fit_failure_spectrum,
                                 logical_error_rate_from_ansatz)
from splitting import replica_exchange_estimate

RESULTS = run_dir("error_model_comparison_18_4_4")

# ---------------------------------------------------------------------------
# Experiment configuration (mirrors the former in-notebook constants)
# ---------------------------------------------------------------------------
P = BBCodeParams(l=3, m=3, a_exps=[(1, 0), (0, 0), (0, 2)], b_exps=[(0, 1), (0, 0), (2, 0)], distance=4)
P_REF, ROUNDS = 0.01, 2
ROUNDS72 = BB_72_4_8.distance // 2            # rounds ∝ distance (d/2)
P_GRID = dict(lo=1e-4, hi=0.04, n=44)         # §2 grid; the top sizes the weight windows
p_grid = np.geomspace(P_GRID["lo"], P_GRID["hi"], P_GRID["n"])

# Direct-MC budget knob: MC only VALIDATES the curves (the budget never reads it), so the
# 0.15 iteration default trades error-bar width for minutes; 1.0 = full-fidelity anchors.
MC_SCALE = 0.15
MC_BASE_POINTS = {0.03: 40_000, 0.012: 80_000, 0.008: 120_000, 0.005: 200_000, 0.003: 300_000}
MC72_POINTS = {0.008: 10_000}                 # one slim anchor for the §7.4 plot

# The ONE decoder for every sampling task (paper-style Relay legs at num_sets=5 — see the
# notebook's setup note: the speed-tuned default misconverges below the perfect-decoder
# onset on [[72,4,8]] and fakes Λ<1). Λ compares codes, so both MUST use the same decoder.
DEC_CFG = dict(gamma0=0.125, pre_iter=80, num_sets=5, set_max_iter=60,
               gamma_dist_interval=(-0.24, 0.66), stop_nconv=3)

def DEC():
    return RelayBPDecoder(**DEC_CFG)

MODELS = {"full symmetric": None, "CZ only": "cz", "meas only": "meas",
          "prep only": "prep", "gate idle": "gate_idle", "meas idle": "meas_idle"}
ABLATED = {"no CZ": "cz", "no meas": "meas", "no prep": "prep",
           "no gate idle": "gate_idle", "no meas idle": "meas_idle"}
SCALE = {"meas": 5.0, "meas_idle": 5.0}       # §8 device-like ray: meas-type channels x5

# Importance-sampling budgets. The 72-code onset bins decide the §7.5/§8 Λ shares; when they
# see ZERO failures the reweighted ε72 is a lower bound (Λ an upper bound) and the shares can
# go spuriously negative — --boost72 buys them 2x target failures and 5x shots.
IS18 = dict(target_failures=200, shots_max=30_000)
IS72 = dict(target_failures=100, shots_max=10_000, stop_after_zero_bins=3)
IS72_BOOST = dict(target_failures=200, shots_max=50_000, stop_after_zero_bins=3)

WINDOW_RULE = "w_hi=ceil(mu+4*sqrt(mu)) at p_grid.max(); 18: 1..max(10,w_hi); 72: 1..15 + 16..w_hi step 2"

CODE18 = repr(P)
CODE72 = repr(BB_72_4_8)


# ---------------------------------------------------------------------------
# Circuit builders (identical to the former notebook cells)
# ---------------------------------------------------------------------------
def make_circuit(model, p):
    return filter_noise_channel(BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(p), rounds=ROUNDS),
                                MODELS[model])

def make_ablated_circuit(name, p):
    drop = NOISE_CHANNEL_PREDICATES[ABLATED[name]]
    return filter_noise_channel(BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(p), rounds=ROUNDS),
                                lambda i, pr, nx: not drop(i, pr, nx))

def make_circuit72(model, p):
    return filter_noise_channel(
        BBCodeSimulator(BB_72_4_8).build_circuit(ErrorModel.symmetric(p), rounds=ROUNDS72), MODELS[model])

def make_ablated_circuit72(name, p):
    drop = NOISE_CHANNEL_PREDICATES[ABLATED[name]]
    return filter_noise_channel(
        BBCodeSimulator(BB_72_4_8).build_circuit(ErrorModel.symmetric(p), rounds=ROUNDS72),
        lambda i, pr, nx: not drop(i, pr, nx))

def make_full_asym(code_params, rounds, p):
    return scale_noise_channels(
        BBCodeSimulator(code_params).build_circuit(ErrorModel.symmetric(p), rounds=rounds), SCALE)

def make_abl_asym(code_params, rounds, ch, p):
    drop = NOISE_CHANNEL_PREDICATES[ch]
    return filter_noise_channel(make_full_asym(code_params, rounds, p),
                                lambda i, pr, nx: not drop(i, pr, nx))


# ---------------------------------------------------------------------------
# Task framework
# ---------------------------------------------------------------------------
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_jsonable(v) for v in x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


class Runner:
    def __init__(self, only=None, force=None, list_only=False):
        self.only = only or []
        self.force = force or []
        self.list_only = list_only
        self.n_ran = self.n_cached = self.n_skipped = 0

    def _selected(self, name):
        return not self.only or any(name.startswith(pfx) for pfx in self.only)

    def _forced(self, name):
        return any(name.startswith(pfx) for pfx in self.force)

    def task(self, name, config, fn):
        """Return the (JSON-normalized) result of `name`, computing it only if needed.

        Cache rule: reuse iff the file exists AND its stored config equals `config`.
        Skipped-by---only tasks still return their cached result when present (dependencies
        like tech3 <- tech2 keep working), and return None if there is nothing cached.
        """
        path = RESULTS / f"{name}.json"
        config = _jsonable(config)
        cached = None
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cached = None
        fresh = cached is not None and cached.get("config") == config

        if not self._selected(name) or self.list_only:
            status = "cached" if fresh else ("STALE " if cached else "missing")
            if self.list_only:
                el = f"{cached['elapsed_s']:8.1f}s  {cached['finished_at']}" if cached else ""
                print(f"  [{status:7s}] {name:34s} {el}")
            self.n_skipped += 1
            return cached["result"] if cached else None

        if fresh and not self._forced(name):
            print(f"[cached] {name:34s} ({cached['elapsed_s']:.1f}s on {cached['finished_at']})")
            self.n_cached += 1
            return cached["result"]

        why = "forced" if fresh else ("stale config" if cached else "new")
        print(f"[run   ] {name:34s} ({why}) ...", flush=True)
        t0 = time.perf_counter()
        result = _jsonable(fn())
        dt = time.perf_counter() - t0
        payload = {"config": config, "elapsed_s": dt,
                   "finished_at": datetime.now().isoformat(timespec="seconds"),
                   "result": result}
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"[done  ] {name:34s} {dt:.1f}s", flush=True)
        self.n_ran += 1
        return result


def slug(name):
    return name.replace(" ", "_")


# ---------------------------------------------------------------------------
# Task bodies
# ---------------------------------------------------------------------------
def spectrum_payload(spec):
    return dict(weights=list(spec.weights), trials=list(spec.trials), failures=list(spec.failures),
                n_expanded=spec.n_expanded, q_base=spec.q_base, p_ref=spec.p_ref)


def weight_window_18(c):
    mu_ref = sum(e.args_copy()[0] for e in c.detector_error_model().flattened() if e.type == "error")
    mu = mu_ref * p_grid.max() / P_REF
    return list(range(1, max(10, int(np.ceil(mu + 4 * np.sqrt(mu)))) + 1))


def weight_window_72(c):
    mu_ref = sum(e.args_copy()[0] for e in c.detector_error_model().flattened() if e.type == "error")
    mu = mu_ref * p_grid.max() / P_REF
    w_hi = int(np.ceil(mu + 4 * np.sqrt(mu)))
    return list(range(1, 16)) + list(range(16, w_hi + 1, 2))   # stride-2 tail: f5 pools weights


def enumerate_LD(circuit, D):
    """Complete L(D): half-MITM for even D (exact, no ldpc); parallel coset search for odd D."""
    if D % 2 != 0:
        return find_all_min_weight_logicals(circuit, D, budget_per_coset=40)
    H, A, mult, priors = dem_check_action_matrices(circuit)
    return find_weight_logicals_mitm(H, A, D)


def tech2_body(circ):
    D = compute_distance(circ).distance
    LD = enumerate_LD(circ, D)
    onset = optimal_onset_fraction(circ, distance=D, logicals=LD)
    return dict(D=D, w0=onset.onset, n_dem=circ.detector_error_model().num_errors,
                n_LD=onset.n_min_logicals, n_LDp1=onset.n_min_logicals_Dp1,
                f0=onset.onset_fraction, LD=[sorted(s) for s in LD],
                route="Prop.1 (even D)" if D % 2 == 0 else "App.A.6 (odd D)")


def spectrum_body(circ, window, is_cfg, seed, with_fit=True):
    W = window(circ)
    spec = importance_sample_adaptive(circ, DEC(), p_ref=P_REF, p_values=[P_REF],
                                      weights=W, seed=seed, **is_cfg).spectrum
    out = dict(spectrum=spectrum_payload(spec), W=W, K=circ.num_observables,
               n_dem=circ.detector_error_model().num_errors, shots=int(sum(spec.trials)))
    if with_fit:
        fit = fit_failure_spectrum(spec, K=circ.num_observables, model="f5", w0=None, f0=None)
        out["fit"] = dict(model=fit.model, params=fit.params, cost=fit.cost)
        out["LER_fit"] = list(logical_error_rate_from_ansatz(fit, list(p_grid)))
    return out


def tech3_body(circ, D, LD):
    temper, diag = replica_exchange_estimate(
        circ, DEC(), p_ref=P_REF, p_high=0.015, p_low=1e-4, n_levels=16,
        n_walkers=8, local_steps=5, n_sweeps=80, burn_in=20, anchor_shots=4000,
        distance=D, seed=2, single_sector=False,
        mw_supports=[frozenset(s) for s in LD], verbose=False)
    return dict(sp=list(np.asarray(temper.p_ladder)[::-1]), sP=list(np.asarray(temper.P_logical)[::-1]),
                sP_se=list(np.asarray(temper.P_logical_se)[::-1]),
                swap_min=float(min(diag["swap_accept"])), swap_max=float(max(diag["swap_accept"])))


def direct_mc(circ, shots):
    d = DEC(); d.setup(circ)
    det, obs = circ.compile_detector_sampler().sample(shots, separate_observables=True)
    f = np.any(d.decode_batch(det) != obs, axis=1); m = float(f.mean())
    return m, float((max(m, 1e-9) * (1 - m) / shots) ** 0.5)

def mc_body(make, points):
    out = {}
    for pp, shots in points.items():
        m, se = direct_mc(make(pp), shots)
        out[str(pp)] = [m, se, shots]
    return dict(points=out)


def schedule_table_body():
    watch = {0: "data q0 (L)", 9: "data q9 (R)", 18: "X-anc q18", 27: "Z-anc q27"}
    c = make_circuit("full symmetric", P_REF)
    insts = list(c.flattened())
    CH = {k: v for k, v in NOISE_CHANNEL_PREDICATES.items() if k != "idle"}   # idle = union of the split
    timeline = {q: [] for q in watch}
    layer = 0
    for i, inst in enumerate(insts):
        if inst.name == "TICK":
            layer += 1
            continue
        prev = insts[i - 1] if i > 0 else None
        nxt = insts[i + 1] if i + 1 < len(insts) else None
        if inst.name == "CX":
            t = [x.value for x in inst.targets_copy()]
            for a, b in zip(t[::2], t[1::2]):
                if a in watch: timeline[a].append((layer, f"CX→{b}"))
                if b in watch: timeline[b].append((layer, f"CX←{a}"))
        elif inst.name in ("H", "M", "R"):
            for x in inst.targets_copy():
                if x.value in watch: timeline[x.value].append((layer, inst.name))
        elif inst.name in NOISE_INSTRUCTIONS:
            ch = next((k for k, pred in CH.items() if pred(inst, prev, nxt)), None)
            tag = f"·{ch}" if ch else "·(1q gate)"      # post-H DEPOLARIZE1 = 1q-gate noise, no channel
            for x in inst.targets_copy():
                if x.value in watch: timeline[x.value].append((layer, tag))
    n_show = 12                                         # ~one full cycle of layers
    lines = [f"{'layer':>5} | " + " | ".join(f"{v:^24}" for v in watch.values()),
             "-" * (8 + 27 * len(watch))]
    for L in range(n_show):
        row = [" ".join(s for l, s in timeline[q] if l == L) or "—" for q in watch]
        lines.append(f"{L:>5} | " + " | ".join(f"{r:^24}" for r in row))
    return dict(table="\n".join(lines))


def schedule_svg_body():
    import stim
    # A closed SLICE of the NOISY circuit: ONE data qubit + its six check ancillas, keeping only
    # instructions whose endpoints are ALL inside the slice (ancilla rails show only their
    # coupling to the watched data qubit; their other five CXs are cropped).
    noisy = BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(P_REF), rounds=1)
    DATA_Q = 0
    star = {DATA_Q}
    for inst in noisy.flattened():
        if inst.name == "CX":
            t = [x.value for x in inst.targets_copy()]
            for a, b in zip(t[::2], t[1::2]):
                if DATA_Q in (a, b):
                    star |= {a, b}
    sl = stim.Circuit()
    for inst in noisy.flattened():
        nm = inst.name
        if nm in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        if nm == "TICK":
            sl.append("TICK")
            continue
        t = [x.value for x in inst.targets_copy()]
        args = inst.gate_args_copy()
        if nm in ("CX", "DEPOLARIZE2"):
            pairs = [(a, b) for a, b in zip(t[::2], t[1::2]) if a in star and b in star]
            if pairs:
                sl.append(nm, [q for ab in pairs for q in ab], args)
        else:
            keep = [q for q in t if q in star]
            if keep:
                sl.append(nm, keep, args)
    return dict(data_q=DATA_Q, star=sorted(star - {DATA_Q}), svg=str(sl.diagram("timeline-svg")))


# ---------------------------------------------------------------------------
# The task list
# ---------------------------------------------------------------------------
def run_all(r: Runner, boost72=False):
    is72 = IS72_BOOST if boost72 else IS72
    base18 = dict(code=CODE18, p_ref=P_REF, rounds=ROUNDS, decoder=DEC_CFG)
    base72 = dict(code=CODE72, p_ref=P_REF, rounds=ROUNDS72, decoder=DEC_CFG)
    grid = dict(p_grid=P_GRID, window=WINDOW_RULE)
    mc_pts = {p: int(s * MC_SCALE) for p, s in MC_BASE_POINTS.items()}

    r.task("setup__dem_counts", dict(code=CODE18, p_ref=P_REF, rounds=ROUNDS, models=list(MODELS)),
           lambda: {name: make_circuit(name, P_REF).detector_error_model().num_errors for name in MODELS})
    r.task("schedule__table", dict(code=CODE18, p_ref=P_REF, rounds=ROUNDS, n_show=12), schedule_table_body)
    r.task("schedule__star_svg", dict(code=CODE18, p_ref=P_REF, rounds=1, data_q=0), schedule_svg_body)

    # §1 + §3 (tech3 needs tech2's D and L(D)); §2; §4
    for name in MODELS:
        t2 = r.task(f"tech2__{slug(name)}", dict(**base18, model=name, budget_per_coset=40),
                    lambda name=name: tech2_body(make_circuit(name, P_REF)))
        r.task(f"tech1__{slug(name)}", dict(**base18, **grid, model=name, **IS18, seed=1),
               lambda name=name: spectrum_body(make_circuit(name, P_REF), weight_window_18, IS18, seed=1))
        if t2 is not None:
            r.task(f"tech3__{slug(name)}",
                   dict(**base18, model=name, p_high=0.015, p_low=1e-4, n_levels=16, n_walkers=8,
                        local_steps=5, n_sweeps=80, burn_in=20, anchor_shots=4000, seed=2, D=t2["D"]),
                   lambda name=name, t2=t2: tech3_body(make_circuit(name, P_REF), t2["D"], t2["LD"]))
        r.task(f"mc__{slug(name)}", dict(**base18, model=name, points=mc_pts, mc_scale=MC_SCALE),
               lambda name=name: mc_body(lambda pp: make_circuit(name, pp), mc_pts))

    # §5 ablations on the 18-code
    for name in ABLATED:
        r.task(f"tech2_abl__{slug(name)}", dict(**base18, ablate=ABLATED[name], budget_per_coset=40),
               lambda name=name: tech2_body(make_ablated_circuit(name, P_REF)))
        r.task(f"tech1_abl__{slug(name)}", dict(**base18, **grid, ablate=ABLATED[name], **IS18, seed=3),
               lambda name=name: spectrum_body(make_ablated_circuit(name, P_REF), weight_window_18,
                                               IS18, seed=3))
        r.task(f"mc_abl__{slug(name)}", dict(**base18, ablate=ABLATED[name], points=mc_pts, mc_scale=MC_SCALE),
               lambda name=name: mc_body(lambda pp: make_ablated_circuit(name, pp), mc_pts))

    # §7 the [[72,4,8]] sibling
    for name in MODELS:
        r.task(f"tech2_72__{slug(name)}", dict(**base72, model=name),
               lambda name=name: dict(D=compute_distance(make_circuit72(name, P_REF)).distance))
        r.task(f"tech1_72__{slug(name)}", dict(**base72, **grid, model=name, **is72, seed=4),
               lambda name=name: spectrum_body(make_circuit72(name, P_REF), weight_window_72, is72, seed=4))
        r.task(f"mc72__{slug(name)}", dict(**base72, model=name, points=MC72_POINTS),
               lambda name=name: mc_body(lambda pp: make_circuit72(name, pp), MC72_POINTS))

    # §7.5 leave-one-out on the 72-code (spectra only; the Λ shares read these)
    for name in ABLATED:
        r.task(f"tech1_72_abl__{slug(name)}", dict(**base72, **grid, ablate=ABLATED[name], **is72, seed=5),
               lambda name=name: spectrum_body(make_ablated_circuit72(name, P_REF), weight_window_72,
                                               is72, seed=5, with_fit=False))

    # §8 the asymmetric operating point (meas, meas_idle x5), full + ablated, both codes.
    # NB: the stride-2 window + the 72-code budgets apply to BOTH codes here (as in the
    # original in-notebook sweep) — the asymmetric mixes are §8-only inputs.
    for label, (cp, rr, window, cfg) in {"18": (P, ROUNDS, weight_window_72, is72),
                                         "72": (BB_72_4_8, ROUNDS72, weight_window_72, is72)}.items():
        base = dict(code=repr(cp), p_ref=P_REF, rounds=rr, decoder=DEC_CFG, scale=SCALE)
        r.task(f"asym__full_{label}", dict(**base, **grid, **cfg, seed=6),
               lambda cp=cp, rr=rr, window=window, cfg=cfg:
                   spectrum_body(make_full_asym(cp, rr, P_REF), window, cfg, seed=6, with_fit=False))
        for abl_name, ch in ABLATED.items():
            r.task(f"asym__{slug(abl_name)}_{label}", dict(**base, **grid, **cfg, ablate=ch, seed=6),
                   lambda cp=cp, rr=rr, window=window, cfg=cfg, ch=ch:
                       spectrum_body(make_abl_asym(cp, rr, ch, P_REF), window, cfg, seed=6, with_fit=False))

    # manifest: everything the report needs to interpret the files (written last = run complete)
    r.task("config__manifest",
           dict(code18=CODE18, code72=CODE72, p_ref=P_REF, rounds=ROUNDS, rounds72=ROUNDS72,
                p_grid=P_GRID, mc_scale=MC_SCALE, decoder=DEC_CFG, scale=SCALE, boost72=boost72,
                is18=IS18, is72=is72, window=WINDOW_RULE),
           lambda: dict(models=list(MODELS), ablated=list(ABLATED), channels=list(MODELS)[1:],
                        p_star=5e-4, p_lam=5e-4,
                        r_of={"CZ only": 1.0, "meas only": SCALE["meas"], "prep only": 1.0,
                              "gate idle": 1.0, "meas idle": SCALE["meas_idle"]}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="show cache status, run nothing")
    ap.add_argument("--only", nargs="+", default=None, metavar="PREFIX",
                    help="run only tasks whose name starts with one of these prefixes")
    ap.add_argument("--force", nargs="+", default=None, metavar="PREFIX",
                    help="rerun these tasks even if cached and config-fresh")
    ap.add_argument("--boost72", action="store_true",
                    help="2x target failures / 5x shots for the 72-code spectra (invalidates them)")
    args = ap.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    r = Runner(only=args.only, force=args.force, list_only=args.list)
    t0 = time.perf_counter()
    run_all(r, boost72=args.boost72)
    if not args.list:
        print(f"\n{r.n_ran} ran, {r.n_cached} cached, {r.n_skipped} skipped "
              f"in {time.perf_counter() - t0:.1f}s -> {RESULTS}")
        print("report: re-execute notebooks/methods/error_model_comparison_18_4_4.ipynb "
              "(regenerate it first via make_error_model_comparison.py if it changed)")


if __name__ == "__main__":
    sys.exit(main())
