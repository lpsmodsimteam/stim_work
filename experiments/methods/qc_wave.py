"""Per-wave gate tables for the K=12 + LPU cluster campaign (runbook "close the wave" step).

    python experiments/methods/qc_wave.py --wave 1 [--root runs/cluster/framework] [--p 0.003]
    python experiments/methods/qc_wave.py --wave 2

Reads experiment_runner outdirs (mid-run checkpoints are fine — pull with rsync any time) and
prints the gate table for the wave; exit code 0 = every hard gate passed, so the runbook's
"preconditions" checkboxes for the NEXT wave are just this program's exit status.

Wave 1 (G1): run health for the four memory full-noise jobs; bb6 Fig-10 onset-pin proxy;
Λ(bb6_m100 → bb144) at p (and 2p) with ±σ + zero-bin interval — **Λ < 1 anywhere = STOP**
(decoder-degradation tell); bb144 zero-failure-bin fraction < 20%.

Wave 2 (G-close): per-channel marginal-Λ decomposition (bb6_m100 → bb144) with verdicts at p
and 2p, plus the LER-space decomposition identity (Σ isolated + mixing vs full).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from lambda_analysis import (Run, load_run, inv_lambda_stats, lambda_decomposition,
                             mass_window_p_max, rw_stats, zero_bin_fraction)

CHANNELS = ["cz", "meas", "prep", "gate_idle", "meas_idle"]
GATES: list = []          # (ok, hard, label)


def gate(ok: bool, label: str, hard: bool = True) -> None:
    GATES.append((ok, hard, label))
    print(f"  [{'PASS' if ok else ('FAIL' if hard else 'warn')}] {label}")


def try_load(root: pathlib.Path, rel: str, suffix: str) -> Run | None:
    d = root / (rel + suffix)
    try:
        return load_run(d)
    except FileNotFoundError:
        print(f"  [....] {rel}{suffix}: not present (yet)")
        return None


def health(run: Run, name: str) -> None:
    s = run.spectrum
    zf = zero_bin_fraction(s)
    df = run.done_fraction
    done = f"{len(s.weights)} weights" + ("" if df != df else f" ({df:.0%} of plan)")
    print(f"  {name:28s} {done}, {sum(s.trials):,} shots, zero-bin {zf:.0%}, "
          f"mass-window p<= {mass_window_p_max(s):.3g}")


def wave1(root: pathlib.Path, p: float, suffix: str) -> None:
    print(f"== Wave 1 gate (G1) at p={p} ==")
    ctrl = try_load(root, "bb6/memory", suffix)
    small = try_load(root, "bb6/memory_m100", suffix)
    large = try_load(root, "bb144/memory", suffix)
    big = try_load(root, "bb288/memory", suffix)

    for r, n in ((ctrl, "bb6/memory (control)"), (small, "bb6/memory_m100"),
                 (large, "bb144/memory"), (big, "bb288/memory")):
        if r:
            health(r, n)

    if ctrl and ctrl.distance:
        d = ctrl.distance
        f0 = d.get("onset_fraction")
        ok = (d.get("distance") == 6 and d.get("onset") == 3
              and f0 is not None and abs(f0 / 2.3239e-5 - 1) < 0.05)
        gate(ok, f"bb6 control reproduces the Fig-10 onset pin "
                 f"(D={d.get('distance')}, w0={d.get('onset')}, f0={f0})")
    if small and small.distance:
        d = small.distance
        gate(d.get("distance") == 6, f"bb6_m100 Technique II re-derived D=6 under the m100 "
                                     f"decoder (D={d.get('distance')}, f0={d.get('onset_fraction')})")

    if small and large:
        for pp in (p, 2 * p):
            try:
                inv = inv_lambda_stats(small.spectrum, large.spectrum, pp, small.cycles, large.cycles)
                lam, lam_se = 1.0 / inv.value, inv.se / inv.value ** 2
            except (ZeroDivisionError, FloatingPointError) as e:
                gate(False, f"Λ({pp:g}) computable (ε=0 in a spectrum? {e})")
                continue
            print(f"  Λ(bb6_m100→bb144)({pp:g}) = {lam:.3g} ± {lam_se:.2g}   "
                  f"[zero-bin interval 1/hi..1/lo = {1/inv.hi:.3g}..{1/inv.lo:.3g}]")
            gate(inv.value > 0 and lam_se == lam_se, f"Λ({pp:g}) finite with propagated ±σ")
            gate(lam > 1.0, f"Λ({pp:g}) > 1  (Λ<1 = decoder-degradation tell -> STOP, no W2)")
    else:
        gate(False, "Λ(bb6_m100→bb144): both runs present", hard=True)

    if large:
        zf = zero_bin_fraction(large.spectrum)
        gate(zf < 0.20, f"bb144 zero-failure-bin fraction {zf:.0%} < 20% "
                        "(else raise channel failure targets in gen_channel_configs first)")


def wave2(root: pathlib.Path, p: float, suffix: str) -> None:
    print(f"== Wave 2 close at p={p} ==")
    small = try_load(root, "bb6/memory_m100", suffix)
    large = try_load(root, "bb144/memory", suffix)
    if not (small and large):
        gate(False, "full-noise parents present")
        return

    def spec_of(root_rel):
        def _f(ch, kind):
            r = try_load(root, f"{root_rel}__{kind}_{ch}", suffix)
            return r.spectrum if r else None
        return _f

    s_of, l_of = spec_of("bb6/memory_m100"), spec_of("bb144/memory")

    # LER-space decomposition identity per code: sum(isolated) + mixing == 1 by construction;
    # the informative check is that no isolated share is negative and full > every isolated.
    for name, run, of in (("bb6_m100", small, s_of), ("bb144", large, l_of)):
        L, L_se, _ = rw_stats(run.spectrum, p)
        got = {ch: of(ch, "iso") for ch in CHANNELS}
        missing = [ch for ch, s in got.items() if s is None]
        if missing:
            gate(False, f"{name}: isolated runs present (missing {missing})")
            continue
        iso = {ch: rw_stats(s, p)[0] / L for ch, s in got.items()}
        mixing = 1.0 - sum(iso.values())
        print(f"  {name}: LER_full({p:g}) = {L:.3e} ± {L_se:.1e};  isolated shares "
              + ", ".join(f"{ch} {v:.3f}" for ch, v in iso.items()) + f";  mixing {mixing:.3f}")
        gate(all(v >= -0.01 for v in iso.values()) and mixing > -0.05,
             f"{name}: isolated shares physical (>=0) and mixing bucket sane")

    # Marginal-Λ decomposition with verdicts, at p and 2p
    for pp in (p, 2 * p):
        abls_s = {ch: s_of(ch, "abl") for ch in CHANNELS}
        abls_l = {ch: l_of(ch, "abl") for ch in CHANNELS}
        missing = [ch for ch in CHANNELS if abls_s[ch] is None or abls_l[ch] is None]
        if missing:
            gate(False, f"ablated runs present on both codes (missing {missing})")
            return
        d = lambda_decomposition(small.spectrum, large.spectrum,
                                 lambda ch: abls_s[ch], lambda ch: abls_l[ch],
                                 CHANNELS, pp, small.cycles, large.cycles)
        print(f"  marginal Λ decomposition at p={pp:g}: Λ_full={d['lambda_full']:.3g}, "
              f"1/Λ={d['inv_full']:.3e} ± {d['inv_full_se']:.1e}, "
              f"Σ contributions / (1/Λ) = {d['sum_contributions']/d['inv_full']:.2f}")
        for r in d["rows"]:
            print(f"    {r['channel']:10s} Λ_no-i={r['lambda_no_i']:9.3g}  share={r['share']:6.2f}"
                  f"  ±{r['sigma']:.1e}  -> {r['verdict']}")
        n_noise = sum(1 for r in d["rows"] if r["verdict"] != "solid")
        gate(True, f"verdict table at p={pp:g} ({n_noise}/{len(CHANNELS)} rows not solid -> "
                   "W4 boost candidates)", hard=False)
        neg_solid = [r["channel"] for r in d["rows"]
                     if r["contribution"] < 0 and r["verdict"] == "solid"]
        gate(not neg_solid or pp != p,
             f"no SOLID negative share at p={pp:g} (found: {neg_solid or 'none'}) — a solid "
             "negative is physics, review before W3", hard=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wave", type=int, required=True, choices=[1, 2])
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("runs/cluster/framework"),
                    help="rsync'd results root (default runs/cluster/framework)")
    ap.add_argument("--p", type=float, default=0.003, help="evaluation point (default p_ref=0.003)")
    ap.add_argument("--suffix", default="", help="outdir suffix, e.g. _smoke for plumbing tests")
    args = ap.parse_args(argv)

    {1: wave1, 2: wave2}[args.wave](args.root, args.p, args.suffix)

    hard = [(ok, lbl) for ok, h, lbl in GATES if h]
    n_bad = sum(1 for ok, _ in hard if not ok)
    print(f"\n{len(hard) - n_bad}/{len(hard)} hard gates passed"
          + ("" if n_bad == 0 else " — DO NOT submit the next wave"))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
