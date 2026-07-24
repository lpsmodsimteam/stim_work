#!/usr/bin/env python
"""Container smoke: build the [[18,4,4]] BB code and Monte-Carlo-decode a batch of idle-memory
shots with Relay-BP. Prints one `SMOKE PASS ...` line with the node hostname, so N array tasks
that each print it prove the image runs (and decodes) on N cluster nodes.

Seed comes from $IDX (the SLURM array index) so parallel tasks sample independently.
"""
import os
import socket
import time

import numpy as np

from bb_code_sim import BB_18_4_4, BBCodeSimulator, RelayBPDecoder
from surface_code_sim import ErrorModel

seed = int(os.environ.get("IDX", "0"))
shots = int(os.environ.get("SMOKE_SHOTS", "2000"))

p = float(os.environ.get("SMOKE_P", "0.005"))
circuit = BBCodeSimulator(BB_18_4_4).build_circuit(ErrorModel.symmetric(p), rounds=4)
decoder = RelayBPDecoder()
decoder.setup(circuit)

t0 = time.perf_counter()
det, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
ler = float(np.any(decoder.decode_batch(det) != obs, axis=1).mean())
dt = time.perf_counter() - t0

print(f"SMOKE PASS host={socket.gethostname()} task={seed} [[18,4,4]] d=4 "
      f"{shots}-shot LER={ler:.4f} decode+sample {dt:.1f}s", flush=True)
