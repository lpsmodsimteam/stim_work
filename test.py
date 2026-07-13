#!/usr/bin/env python
"""Cluster/environment preflight: is everything installed, and is SLURM reachable?

Run on the cluster login node BEFORE any sbatch (runbook Wave-1 step 1):

    python test.py              # all checks incl. SLURM
    python test.py --no-slurm   # local/dev box: skip the SLURM checks

Exit code 0 = every check passed. Each check prints one PASS/FAIL line; failures end with a
hint. This is deliberately NOT a pytest file — it must run before pytest is even known to work.
"""
from __future__ import annotations

import importlib
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent
RESULTS: list[tuple[bool, str]] = []


def check(name: str, fn):
    try:
        detail = fn() or ""
        RESULTS.append((True, name))
        print(f"  PASS  {name}{'  (' + str(detail) + ')' if detail else ''}")
    except Exception as e:
        RESULTS.append((False, name))
        print(f"  FAIL  {name}: {e}")


# --------------------------- packages ---------------------------
def has(mod, attr="__version__"):
    def _fn():
        m = importlib.import_module(mod)
        return getattr(m, attr, "ok")
    return _fn


def check_editable_install():
    # the project modules must import from anywhere, not just with cwd=src (pip install -e .)
    mods = ["experiment_runner", "bb_code_sim", "min_weight", "importance_sampling",
            "splitting", "repo_paths"]
    for m in mods:
        importlib.import_module(m)
    import repo_paths
    if pathlib.Path(repo_paths.REPO_ROOT).resolve() != REPO:
        raise RuntimeError(f"repo_paths.REPO_ROOT={repo_paths.REPO_ROOT} != {REPO} "
                           "(stale editable install from another checkout?)")
    return f"{len(mods)} project modules"


def check_decoder_roundtrip():
    # tiny build+decode proves stim AND the relay_bp Rust extension work on this arch
    import numpy as np
    from bb_code_sim import BB_18_4_4, BBCodeSimulator, RelayBPDecoder
    from surface_code_sim import ErrorModel
    circ = BBCodeSimulator(BB_18_4_4).build_circuit(ErrorModel.symmetric(0.005), rounds=2)
    dec = RelayBPDecoder(num_sets=5)
    dec.setup(circ)
    det, obs = circ.compile_detector_sampler().sample(50, separate_observables=True)
    pred = dec.decode_batch(det)
    ler = float(np.any(pred != obs, axis=1).mean())
    return f"decoded 50 shots, LER={ler:.2f}"


def check_registries():
    from experiment_runner import CODES, CIRCUIT_BUILDERS
    for k in ("bb6", "bb144", "bb288"):
        if k not in CODES:
            raise RuntimeError(f"code {k!r} missing from CODES")
    for k in ("memory", "lpu_x1", "lpu_z1"):
        if k not in CIRCUIT_BUILDERS:
            raise RuntimeError(f"experiment {k!r} missing from CIRCUIT_BUILDERS")
    return f"{len(CODES)} codes, {len(CIRCUIT_BUILDERS)} experiments"


def check_manifest_and_configs():
    import yaml
    from experiment_runner import load_config
    man = yaml.safe_load((REPO / "experiments" / "manifest.yaml").read_text())
    entries = man["configs"]
    if not entries:
        raise RuntimeError("manifest has no active entries")
    for e in entries:
        p = REPO / e["path"]
        if not p.exists():
            raise RuntimeError(f"manifest points at missing config {e['path']}")
        load_config(p)              # full Config validation (channel keys, unknown fields, ...)
    return f"{len(entries)} active manifest entries load"


def check_runs_writable():
    d = REPO / "runs" / "slurm"
    d.mkdir(parents=True, exist_ok=True)
    probe = d / ".preflight_probe"
    probe.write_text("ok")
    probe.unlink()
    return str(d)


# --------------------------- SLURM ---------------------------
def _run(cmd, timeout=15):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> rc={r.returncode}: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def check_slurm_binaries():
    missing = [b for b in ("sbatch", "squeue", "sinfo") if shutil.which(b) is None]
    if missing:
        raise RuntimeError(f"not on PATH: {missing} (are you on a login node?)")
    return _run(["sbatch", "--version"], timeout=10)


def check_slurm_responds():
    # sinfo talks to the controller — proves SLURM is reachable, not just installed
    out = _run(["sinfo", "-h", "-o", "%P %a %D %l"])
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError("sinfo returned no partitions")
    return f"{len(lines)} partitions, e.g. '{lines[0]}'"


def check_slurm_queue():
    _run(["squeue", "-h", "--me", "-o", "%i"])
    return "controller answered"


def main(argv=None) -> int:
    args = set(argv if argv is not None else sys.argv[1:])
    no_slurm = "--no-slurm" in args

    print(f"preflight @ {REPO}  (python {sys.version.split()[0]})")
    print("\n-- packages --")
    if sys.version_info < (3, 10):
        RESULTS.append((False, "python >= 3.10"))
        print(f"  FAIL  python >= 3.10: found {sys.version.split()[0]}")
    for mod in ("numpy", "scipy", "yaml", "matplotlib", "stim", "ldpc", "relay_bp"):
        check(f"import {mod}", has(mod))
    print("\n-- project --")
    check("editable install (pip install -e .)", check_editable_install)
    check("stim + relay_bp decode round-trip", check_decoder_roundtrip)
    check("runner registries (bb6/bb144/bb288, memory/lpu)", check_registries)
    check("manifest + every active config loads", check_manifest_and_configs)
    check("runs/ writable", check_runs_writable)
    if no_slurm:
        print("\n-- slurm -- skipped (--no-slurm)")
    else:
        print("\n-- slurm --")
        check("slurm binaries on PATH", check_slurm_binaries)
        check("controller reachable (sinfo)", check_slurm_responds)
        check("queue queryable (squeue --me)", check_slurm_queue)

    n_fail = sum(1 for ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed"
          + ("" if n_fail == 0 else f"  ({n_fail} FAILED)"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
