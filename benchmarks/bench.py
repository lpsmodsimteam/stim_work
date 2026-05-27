import time
import json
import datetime
import numpy as np
from bb_code_sim import BBCodeSimulator, BB_72_12_6, ErrorModel, RelayBPDecoder, BPOSDDecoder

sim = BBCodeSimulator(BB_72_12_6)
em = ErrorModel.symmetric(0.003)
rounds = 6
shots = 50

circuit = sim.build_circuit(em, rounds)
sampler = circuit.compile_detector_sampler(seed=42)
det, obs = sampler.sample(shots, separate_observables=True)

def bench(decoder_cls, name, **kwargs):
    d = decoder_cls(**kwargs)
    d.setup(circuit)
    t0 = time.perf_counter()
    preds = d.decode_batch(det)
    elapsed = time.perf_counter() - t0
    n_err = int(np.sum(np.any(preds != obs, axis=1)))
    ler = n_err / shots
    ler_se = float(np.sqrt(ler * (1 - ler) / shots))
    return {"decoder": name, "elapsed_s": round(elapsed, 3), "ler": round(ler, 4),
            "ler_se": round(ler_se, 4), "n_errors": n_err}

results = []
results.append(bench(RelayBPDecoder, "RelayBPDecoder", parallel=True))
results.append(bench(BPOSDDecoder,   "BPOSDDecoder"))

metadata = {
    "timestamp": datetime.datetime.now().isoformat(),
    "code": "BB_72_12_6",
    "code_params": {"l": 6, "m": 6, "distance": 6},
    "error_model": {"p_phys": em.p_phys, "p_meas": em.p_meas},
    "rounds": rounds,
    "shots": shots,
    "seed": 42,
    "results": results,
    "speedup": round(
        next(r["elapsed_s"] for r in results if r["decoder"] == "BPOSDDecoder") /
        next(r["elapsed_s"] for r in results if r["decoder"] == "RelayBPDecoder"), 2
    ),
}

outfile = f"bench_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(outfile, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Code: BB_72_12_6  p={em.p_phys}  rounds={rounds}  shots={shots}\n")
for r in results:
    print(f"  {r['decoder']:20s}  {r['elapsed_s']:7.2f}s  LER={r['ler']:.4f} ± {r['ler_se']:.4f}")
print(f"\nSpeedup: {metadata['speedup']}x faster")
print(f"Results written to {outfile}")
