"""Generate the five-channel campaign configs (waves W2/W3) from their full-noise parents.

One YAML per (parent, model) with model = iso_<ch> (noise_channel) or abl_<ch> (ablate_channel),
written to experiments/configs/channels/. The files are COMMITTED artifacts: regeneration is
idempotent and diff-reviewable. Never hand-edit a clone — edit OVERLAYS/CLASS_TABLE here and
regenerate:

    python experiments/configs/gen_channel_configs.py

Design rules (campaign plan 2026-07-13):
- Channel configs never inherit the parent weight plan (different N_expanded/onset): every clone
  runs `onset` to size a contiguous window; per-class p_hi is widened so the code-pair crossing
  lands inside the sampled binomial mass (isolated meas/prep thresholds sit at several percent).
- Technique III dropped everywhere (W4 adds targeted splitting anchors); Technique II kept for
  bb6/bb144 memory channels, dropped for bb288 channels (W4 back-fills) and LPU channels (Tech II
  LPU plumbing is Track-L2).
- Decoder: inherited from the parent — all parents already carry the unified num_sets=100 block.
- Explicit outdir on every clone (parent dir + __iso_/__abl_ suffix) so nothing collides.
"""
from __future__ import annotations

import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "channels"

CHANNELS = ["cz", "meas", "prep", "gate_idle", "meas_idle"]

# Parents: (config file, tier, outdir base, manifest cpus/mem)
PARENTS = {
    "bb6_memory_m100": dict(tier="bb6", outbase="runs/framework/bb6/memory_m100",
                            cpus=16, mem="8G"),
    "bb144_memory":    dict(tier="bb144", outbase="runs/framework/bb144/memory",
                            cpus=32, mem="32G"),
    "bb288_memory":    dict(tier="bb288", outbase="runs/framework/bb288/memory",
                            cpus=48, mem="64G"),
    "gross_lpu_x1":    dict(tier="lpu", outbase="runs/framework/bb144/lpu_x1",
                            cpus=16, mem="16G"),
    "gross_lpu_z1":    dict(tier="lpu", outbase="runs/framework/bb144/lpu_z1",
                            cpus=16, mem="16G"),
}

# Per-model-class overlay: p-window, weight stride, failures by tier, walltime by tier.
# stride>1 => curve-level reweighting downstream must gap-fill (lambda_analysis enforces).
CLASS_TABLE = {
    "iso_cz":        dict(p_lo=3e-4, p_hi=1.5e-2, stride={"bb6": 2, "bb144": 2, "bb288": 4, "lpu": 2},
                          failures=dict(bb6=100, bb144=25, bb288=15, lpu=20),
                          time=dict(bb6="12:00:00", bb144="72:00:00", bb288="96:00:00", lpu="24:00:00")),
    "iso_meas":      dict(p_lo=1e-3, p_hi=5e-2, stride=1,
                          failures=dict(bb6=100, bb144=100, bb288=50, lpu=50),
                          time=dict(bb6="04:00:00", bb144="08:00:00", bb288="12:00:00", lpu="04:00:00")),
    "iso_prep":      "iso_meas",
    "iso_gate_idle": dict(p_lo=5e-4, p_hi=3e-2, stride=1,
                          failures=dict(bb6=100, bb144=50, bb288=25, lpu=25),
                          time=dict(bb6="06:00:00", bb144="16:00:00", bb288="24:00:00", lpu="08:00:00")),
    "iso_meas_idle": "iso_gate_idle",
    "abl_cz":        dict(p_lo=3e-4, p_hi=1.2e-2, stride={"bb6": 2, "bb144": 2, "bb288": 4, "lpu": 2},
                          failures=dict(bb6=100, bb144=25, bb288=15, lpu=20),
                          time=dict(bb6="06:00:00", bb144="36:00:00", bb288="48:00:00", lpu="12:00:00")),
    "abl_meas":      dict(p_lo=3e-4, p_hi=1.2e-2, stride={"bb6": 2, "bb144": 2, "bb288": 4, "lpu": 2},
                          failures=dict(bb6=100, bb144=25, bb288=15, lpu=20),
                          time=dict(bb6="12:00:00", bb144="72:00:00", bb288="96:00:00", lpu="24:00:00")),
    "abl_prep":      "abl_meas",
    "abl_gate_idle": "abl_meas",
    "abl_meas_idle": "abl_meas",
}

# Keys a clone must NOT inherit (weight plans, pins and splitting are parent-run specific).
DROP_KEYS = ["weights", "weights_range", "weights_explicit", "weight_stride",
             "mw_f0_override", "mw_w0_override",
             "split_p_high", "split_p_low", "split_L", "split_M", "split_T_init",
             "split_ladder", "split_method", "outdir"]


def class_of(model: str) -> dict:
    c = CLASS_TABLE[model]
    return CLASS_TABLE[c] if isinstance(c, str) else c


def techniques_for(tier: str) -> list:
    return ["onset", "II", "IS", "I"] if tier in ("bb6", "bb144") else ["onset", "IS", "I"]


def make_clone(parent_name: str, parent: dict, model: str) -> dict:
    meta = PARENTS[parent_name]
    tier = meta["tier"]
    cls = class_of(model)
    kind, ch = model.split("_", 1)          # iso/abl, channel key
    cfg = {k: v for k, v in parent.items() if k not in DROP_KEYS}
    stride = cls["stride"] if isinstance(cls["stride"], int) else cls["stride"][tier]
    cfg.update({
        "label": f"{parent_name}__{model}",
        "techniques": techniques_for(tier),
        "outdir": f"{meta['outbase']}__{model}",
        "p_lo": cls["p_lo"], "p_hi": cls["p_hi"],
        "weight_stride": stride,
        "adaptive": True,
        "adaptive_failures": cls["failures"][tier],
        # campaign convention: FREE ansatz onset — even-D channels would otherwise pin
        # f(w0) to the perfect-decoder floor, hiding any decoder misconvergence there
        # (the [[72,4,8]] gate-idle lesson). Reweighted point values are unaffected.
        "pin_onset": False,
        ("noise_channel" if kind == "iso" else "ablate_channel"): ch,
    })
    return cfg


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest_rows = []
    for parent_name, meta in PARENTS.items():
        parent = yaml.safe_load((HERE / f"{parent_name}.yaml").read_text())
        for ch in CHANNELS:
            for kind in ("iso", "abl"):
                model = f"{kind}_{ch}"
                cfg = make_clone(parent_name, parent, model)
                path = OUT / f"{parent_name}__{model}.yaml"
                header = (f"# GENERATED by gen_channel_configs.py - edit the generator, not this file.\n"
                          f"# {kind}: {'keep ONLY' if kind == 'iso' else 'leave out'} the "
                          f"'{ch}' channel of {parent_name}.\n")
                path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                manifest_rows.append((f"experiments/configs/channels/{path.name}",
                                      meta["cpus"], class_of(model)["time"][meta["tier"]], meta["mem"]))
    print(f"wrote {len(manifest_rows)} configs -> {OUT}")
    print("\n# manifest block (paste under the wave header, uncomment on submit day):")
    for p, cpus, time, mem in manifest_rows:
        print(f"  # - path: {p}\n  #   cpus: {cpus}\n  #   time: \"{time}\"\n  #   mem: \"{mem}\"")


if __name__ == "__main__":
    main()
