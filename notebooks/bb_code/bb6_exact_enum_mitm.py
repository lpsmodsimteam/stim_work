"""Exact complete weight-6 logical enumeration via anchored MITM 2+2+2.

Canonicalize each logical so its global-minimum detector has check-index 0 (8 detectors).
At that detector d0 the logical has an even subset S0 (>=2) of the min-det==d0 columns; the
remaining 6-|S0| columns have min-det>d0. For |S0|=2 the remaining 4 split into two pairs,
matched by a GF(2)-linear 64-bit hash of their detector-XOR (vectorized searchsorted).
|S0|=4 and 6 handled directly. Expand canonical solutions by the 36 Z6xZ6 shifts.

Loads bb6_ref_data.pkl (H, A, mult, col_h, col_a, dc).
"""
import sys, pickle, time, itertools
import numpy as np
sys.path.insert(0, r"C:\Users\aksirot_local\Desktop\workspace\general\stim_work\src")
from min_weight import (min_weight_fail_count, expanded_logical_count,
                        build_circuit_translation_perms)
from math import comb

d = pickle.load(open(r"C:\Users\aksirot_local\AppData\Local\Temp\bbref\bb6_ref_data.pkl","rb"))
H, A, mult = d["H"], d["A"], d["mult"]
col_h, col_a = d["col_h"], d["col_a"]          # python-int bitmasks
N = d["N"]; n_det = H.shape[0]
perms = build_circuit_translation_perms(None, H, det_coords=d["dc"], verbose=False)

mindet = [ (col_h[j] & -col_h[j]).bit_length()-1 if col_h[j] else -1 for j in range(N) ]
cols_mindet = [[] for _ in range(n_det)]
for j in range(N):
    if mindet[j] >= 0: cols_mindet[mindet[j]].append(j)

# GF(2)-linear 64-bit hash of detector bitmask -> hash(x^y)=hash(x)^hash(y)
rng = np.random.default_rng(12345)
masks = [int(rng.integers(0, 1<<32)) | (int(rng.integers(0, 1<<32))<<32) |
         (int(rng.integers(0,1<<32))<<64) | (int(rng.integers(0,1<<32))<<96) |
         (int(rng.integers(0,1<<32))<<128) | (int(rng.integers(0,1<<32))<<160) |
         (int(rng.integers(0,1<<32))<<192) | (int(rng.integers(0,1<<32))<<224) |
         (int(rng.integers(0,1<<32))<<256) for _ in range(64)]
col_hash = np.zeros(N, dtype=np.uint64)
for j in range(N):
    h = 0
    x = col_h[j]
    for b in range(64):
        if bin(x & masks[b]).count("1") & 1: h |= (1 << b)
    col_hash[j] = np.uint64(h)

results = set()
def Axor(cols):
    a = 0
    for c in cols: a ^= col_a[c]
    return a

t0 = time.time()
canon = [c*36 for c in range(n_det//36)]
for d0 in canon:
    cols0 = cols_mindet[d0]
    cand = np.array([c for c in range(N) if mindet[c] > d0], dtype=np.int64)
    nc = len(cand)
    # all rem pairs and their linear hashes
    pi, pj = np.triu_indices(nc, 1)
    P1 = cand[pi]; P2 = cand[pj]
    phash = col_hash[P1] ^ col_hash[P2]
    order = np.argsort(phash, kind="stable")
    phash_s = phash[order]
    # |S0| = 2  (main case): anchor pair (a,b) from cols0; rem 4 = pair1 + pair2
    for a, b in itertools.combinations(cols0, 2):
        Tfull = col_h[a] ^ col_h[b]
        Th = np.uint64(col_hash[a] ^ col_hash[b])
        q = phash ^ Th                              # hash each pair1 -> needed pair2 hash
        pos = np.searchsorted(phash_s, q)
        ok = (pos < len(phash_s))
        pos2 = np.where(ok, np.minimum(pos, len(phash_s)-1), 0)
        match = ok & (phash_s[pos2] == q)
        for m in np.flatnonzero(match):
            k2 = order[pos2[m]]
            i1, j1 = int(P1[m]), int(P2[m])
            i2, j2 = int(P1[k2]), int(P2[k2])
            six = {a, b, i1, j1, i2, j2}
            if len(six) != 6:                       # columns must be distinct
                continue
            if (col_h[i1]^col_h[j1]^col_h[i2]^col_h[j2]) != Tfull:   # hash collision guard
                continue
            if Axor(six) == 0:
                continue
            results.add(frozenset(six))
    # |S0| = 4: anchor 4 cols from cols0; rem pair
    pset = {int(h): [] for h in np.unique(phash)}
    for idx in range(len(phash)):
        pset[int(phash[idx])].append(idx)
    for S0 in itertools.combinations(cols0, 4):
        T = 0
        for c in S0: T ^= col_h[c]
        Th = 0
        for c in S0: Th ^= int(col_hash[c])
        for idx in pset.get(Th, []):
            i1, j1 = int(P1[idx]), int(P2[idx])
            if (col_h[i1]^col_h[j1]) != T: continue
            six = set(S0) | {i1, j1}
            if len(six) != 6: continue
            if Axor(six) == 0: continue
            results.add(frozenset(six))
    # |S0| = 6: all six at d0
    for S0 in itertools.combinations(cols0, 6):
        x = 0
        for c in S0: x ^= col_h[c]
        if x == 0 and Axor(S0) != 0:
            results.add(frozenset(S0))

print(f"canonical weight-6 logicals: {len(results)}  ({time.time()-t0:.0f}s)", flush=True)
full = set()
for L in results:
    for p in perms:
        full.add(frozenset(int(p[c]) for c in L))
print(f"after 36-shift expansion: |L(D)| = {len(full)}", flush=True)
fails, n_exp = min_weight_fail_count(H, A, full, mult)
ld_exp = expanded_logical_count(full, mult)
f0 = fails / comb(n_exp, 3)
print("\n===== EXACT (MITM 2+2+2) TECHNIQUE II vs PAPER =====")
print(f"  |L(D)| comp = {len(full):>14}")
print(f"  |L(D)| exp  = {ld_exp:>14}   (paper 6.01e12)")
print(f"  |F(D/2)|    = {fails:>14}   (paper 383000000)")
print(f"  f0=f*(D/2)  = {f0:.6e}   (paper 2.33e-5)")
