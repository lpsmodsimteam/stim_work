"""Port decoder_setup.py's single-sector construction to BB(6) and check Ñ / N.

Faithful to sbravyi/BivariateBicycleCodes decoder_setup.py, but:
  - BB(6): ell,m=6,6; a=(3,1,2), b=(3,1,2)
  - logical basis from our find_logical_ops (partition into (syndrome,action) classes
    is basis-independent, so Ñ/N are unaffected)
Goal: confirm num_cycles=6 (+2 noiseless) reproduces paper Ñ=2233, N=46224.
"""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\aksirot_local\Desktop\workspace\general\stim_work\src")
from bb_code_sim import BB_72_12_6, build_parity_checks, find_logical_ops

ell, m = 6, 6
a1,a2,a3 = 3,1,2
b1,b2,b3 = 3,1,2
n = 2*m*ell          # 72 data
n2 = m*ell           # 36 checks each type
p = 0.003

I_ell = np.identity(ell,dtype=int); I_m = np.identity(m,dtype=int)
x = {i: np.kron(np.roll(I_ell,i,axis=1),I_m) for i in range(ell)}
y = {i: np.kron(I_ell,np.roll(I_m,i,axis=1)) for i in range(m)}
A1,A2,A3 = x[a1],y[a2],y[a3]
B1,B2,B3 = y[b1],x[b2],x[b3]

# linear order: Xchecks, data_left, data_right, Zchecks  (total 2n=144)
lin_order = {}; data_qubits=[]; Xchecks=[]; Zchecks=[]; cnt=0
for i in range(n2): Xchecks.append(('Xcheck',i)); lin_order[('Xcheck',i)]=cnt; cnt+=1
for i in range(n2): data_qubits.append(('data_left',i)); lin_order[('data_left',i)]=cnt; cnt+=1
for i in range(n2): data_qubits.append(('data_right',i)); lin_order[('data_right',i)]=cnt; cnt+=1
for i in range(n2): Zchecks.append(('Zcheck',i)); lin_order[('Zcheck',i)]=cnt; cnt+=1

nbs={}
for i in range(n2):
    cn=('Xcheck',i)
    nbs[(cn,0)]=('data_left',np.nonzero(A1[i,:])[0][0])
    nbs[(cn,1)]=('data_left',np.nonzero(A2[i,:])[0][0])
    nbs[(cn,2)]=('data_left',np.nonzero(A3[i,:])[0][0])
    nbs[(cn,3)]=('data_right',np.nonzero(B1[i,:])[0][0])
    nbs[(cn,4)]=('data_right',np.nonzero(B2[i,:])[0][0])
    nbs[(cn,5)]=('data_right',np.nonzero(B3[i,:])[0][0])
for i in range(n2):
    cn=('Zcheck',i)
    nbs[(cn,0)]=('data_left',np.nonzero(B1[:,i])[0][0])
    nbs[(cn,1)]=('data_left',np.nonzero(B2[:,i])[0][0])
    nbs[(cn,2)]=('data_left',np.nonzero(B3[:,i])[0][0])
    nbs[(cn,3)]=('data_right',np.nonzero(A1[:,i])[0][0])
    nbs[(cn,4)]=('data_right',np.nonzero(A2[:,i])[0][0])
    nbs[(cn,5)]=('data_right',np.nonzero(A3[:,i])[0][0])

sX=['idle',1,4,3,5,0,2]; sZ=[3,5,0,1,2,4,'idle']

def build_cycle():
    cycle=[]
    # round 0
    for q in Xchecks: cycle.append(('PrepX',q))
    done=[]
    for tgt in Zchecks:
        ctrl=nbs[(tgt,sZ[0])]; cycle.append(('CNOT',ctrl,tgt)); done.append(ctrl)
    for q in data_qubits:
        if q not in done: cycle.append(('IDLE',q))
    # rounds 1-5
    for t in range(1,6):
        for ctrl in Xchecks:
            tgt=nbs[(ctrl,sX[t])]; cycle.append(('CNOT',ctrl,tgt))
        for tgt in Zchecks:
            ctrl=nbs[(tgt,sZ[t])]; cycle.append(('CNOT',ctrl,tgt))
    # round 6
    for q in Zchecks: cycle.append(('MeasZ',q))
    done=[]
    for ctrl in Xchecks:
        tgt=nbs[(ctrl,sX[6])]; cycle.append(('CNOT',ctrl,tgt)); done.append(tgt)
    for q in data_qubits:
        if q not in done: cycle.append(('IDLE',q))
    # round 7
    for q in data_qubits: cycle.append(('IDLE',q))
    for q in Xchecks: cycle.append(('MeasX',q))
    for q in Zchecks: cycle.append(('PrepZ',q))
    return cycle

def simulate_Z(C):
    state=np.zeros(2*n,dtype=int); sh=[]; smap={}; sc=0; ec=0
    for g in C:
        t=g[0]
        if t=='CNOT':
            c=lin_order[g[1]]; tt=lin_order[g[2]]
            state[c]=(state[tt]+state[c])%2
        elif t=='PrepX':
            state[lin_order[g[1]]]=0
        elif t=='MeasX':
            q=lin_order[g[1]]; sh.append(state[q]); smap.setdefault(g[1],[]).append(sc); sc+=1
        elif t in ('Z','Y'):
            ec+=1; state[lin_order[g[1]]]^=1
        elif t in ('ZZ','YY','YZ','ZY'):
            ec+=1; state[lin_order[g[1]]]^=1; state[lin_order[g[2]]]^=1
    return np.array(sh,dtype=int),state,smap,ec

def build_matrices(num_cycles):
    """Return (H, A, probs, mult, det_coords, n_det) for the single-sector reference DEM."""
    cycle=build_cycle()
    cycle_repeated=num_cycles*cycle
    H_X,H_Z=build_parity_checks(BB_72_12_6)
    log_Z,log_X=find_logical_ops(H_X,H_Z)
    L = log_X
    k = L.shape[0]
    n_det = n2*(num_cycles+2)          # sparsified detector rows
    circuits=[]; probs=[]
    head=[]; tail=cycle_repeated.copy()
    for g in cycle_repeated:
        if g[0]=='MeasX': circuits.append(head+[('Z',g[1])]+tail); probs.append(p)
        head=head+[g]; tail=tail[1:]
        if g[0]=='PrepX': circuits.append(head+[('Z',g[1])]+tail); probs.append(p)
        if g[0]=='IDLE': circuits.append(head+[('Z',g[1])]+tail); probs.append(p*2/3)
        if g[0]=='CNOT':
            circuits.append(head+[('Z',g[1])]+tail); probs.append(p*4/15)
            circuits.append(head+[('Z',g[2])]+tail); probs.append(p*4/15)
            circuits.append(head+[('ZZ',g[1],g[2])]+tail); probs.append(p*4/15)
    # merge by (sparsified detector syndrome, logical) support
    Hdict={}
    for ci,circ in enumerate(circuits):
        sh,state,smap,ec=simulate_Z(circ+cycle+cycle)
        assert ec==1
        sdata=np.array([state[lin_order[q]] for q in data_qubits])
        logical=(L@sdata)%2
        shc=sh.copy()
        for c in Xchecks:
            pos=smap[c]
            for r in range(1,len(pos)): sh[pos[r]]+=shc[pos[r-1]]
        sh%=2
        aug=np.hstack([sh,logical])
        supp=tuple(np.nonzero(aug)[0])
        Hdict.setdefault(supp,[]).append(ci)
    cols=list(Hdict.keys())
    Ntilde=len(cols)
    qbase=p/15
    H=np.zeros((n_det,Ntilde),dtype=np.uint8)
    A=np.zeros((k,Ntilde),dtype=np.uint8)
    probs_col=np.empty(Ntilde); mult=np.empty(Ntilde,dtype=np.int64)
    for j,supp in enumerate(cols):
        pc=sum(probs[i] for i in Hdict[supp])
        probs_col[j]=pc; mult[j]=max(round(pc/qbase),1)
        for idx in supp:
            if idx < n_det: H[idx,j]=1
            else: A[idx-n_det,j]=1
    # detector-row coordinates for toric perms: row -> (type0, check_index, cycle)
    det_coords={r:(0, r % n2, r // n2) for r in range(n_det)}
    return H,A,probs_col,mult,det_coords,n_det

def _check_sigma(H, sigma, n2=36):
    """Test whether a spatial check-index permutation sigma (len-n2 array) is a valid
    circuit automorphism of the single-sector DEM. Returns col_perm or None.

    Builds det_perm (row=(s,c) -> (sigma[s],c)), matches columns by permuted detector
    support, and verifies H[ix_(det_perm, col_perm)] == H (same condition as the toric
    builder). Assumes no H-column collisions (confirmed for BB(6))."""
    n_det, N = H.shape
    det_perm = np.empty(n_det, dtype=np.int64)
    for r in range(n_det):
        det_perm[r] = sigma[r % n2] + n2 * (r // n2)
    cols = [frozenset(np.flatnonzero(H[:, j]).tolist()) for j in range(N)]
    h_sig = {c: j for j, c in enumerate(cols)}
    if len(h_sig) != N:
        return None
    col_perm = np.empty(N, dtype=np.int64)
    for j in range(N):
        img = frozenset(int(det_perm[d]) for d in cols[j])
        if img not in h_sig:
            return None
        col_perm[j] = h_sig[img]
    if not np.array_equal(H[np.ix_(det_perm, col_perm)], H):
        return None
    return col_perm


if __name__=="__main__":
    import sys as _s
    if "--sym" in _s.argv:
        H,A,pc,mult,dc,nd=build_matrices(6)
        def transpose(s): i,j=divmod(s,6); return j*6+i
        def invert(s):    i,j=divmod(s,6); return ((-i)%6)*6 + ((-j)%6)
        def tinv(s):      return invert(transpose(s))
        for name,f in [("transpose (x<->y)",transpose),("inversion (i,j)->(-i,-j)",invert),
                       ("transpose+inversion",tinv)]:
            sigma=np.array([f(s) for s in range(36)],dtype=np.int64)
            cp=_check_sigma(H,sigma)
            print(f"  {name:<28}: {'VALID automorphism' if cp is not None else 'not a symmetry'}")
        _s.exit()
    if "--symsearch" in _s.argv:
        # Enumerate all linear maps (i,j) -> M@(i,j) mod 6 with M invertible mod 6,
        # test each as a single-sector automorphism. Combined with the 36 translations
        # this gives the full affine automorphism group of the sector.
        H,A,pc,mult,dc,nd=build_matrices(6)
        valid=[]
        for a in range(6):
         for b in range(6):
          for cc in range(6):
           for d in range(6):
            det=(a*d-b*cc)%6
            if det not in (1,5):   # invertible mod 6
                continue
            sigma=np.array([((a*(s//6)+b*(s%6))%6)*6 + ((cc*(s//6)+d*(s%6))%6)
                            for s in range(36)],dtype=np.int64)
            if _check_sigma(H,sigma) is not None:
                valid.append((a,b,cc,d))
        print(f"valid linear automorphisms (M mod 6): {len(valid)}")
        for M in valid: print("   ",M)
        # how many are non-translation point symmetries (M != identity)?
        nid=[M for M in valid if M!=(1,0,0,1)]
        print(f"non-identity linear maps: {len(nid)}")
        _s.exit()
    if "--counts" in _s.argv:
        for nc in (5,6,7,12):
            H,A,pc,mult,dc,nd=build_matrices(nc)
            print(f"num_cycles={nc:>2}:  N~={H.shape[1]:>5}  N_exp={int(mult.sum()):>7}   (paper 2233/46224)")
        _s.exit()
    if "--mw" in _s.argv:
        from min_weight import (compute_distance, find_min_weight_logicals,
                                min_weight_fail_count, expanded_logical_count,
                                build_circuit_translation_perms)
        import time
        H,A,pc,mult,dc,nd=build_matrices(6)
        print(f"Reference single-sector DEM: Ñ={H.shape[1]} (paper 2233), "
              f"N_exp={int(mult.sum())} (paper 46224)", flush=True)
        t0=time.time()
        D=compute_distance(matrices=(H,A,pc), workers=20, progress=True).distance
        print(f"D={D}  ({time.time()-t0:.0f}s)", flush=True)
        perms=build_circuit_translation_perms(None, H, det_coords=dc, verbose=True)
        K=A.shape[0]; n_sys=(1<<K)-1
        logicals=find_min_weight_logicals(None, D, matrices=(H,A,pc),
            systematic=True, max_trials=2000, symmetry_perms=perms, workers=20,
            progress_every=max(n_sys//30,1), seed=42)
        ld_comp=len(logicals); ld_exp=expanded_logical_count(logicals, mult)
        fails,n_exp=min_weight_fail_count(H,A,logicals,mult)
        from math import comb
        f0=fails/comb(n_exp,D//2)
        print("\n===== REFERENCE-CIRCUIT TECHNIQUE II vs PAPER TABLE 2 =====")
        print(f"  Ñ           = {H.shape[1]:>14}   (paper 2233)")
        print(f"  N_expanded  = {n_exp:>14}   (paper 46224)")
        print(f"  D           = {D:>14}   (paper 6)")
        print(f"  |L(D)| comp = {ld_comp:>14}")
        print(f"  |L(D)| exp  = {ld_exp:>14}   (paper 6.01e12 = {6.01e12:.0f})")
        print(f"  |F(D/2)|    = {fails:>14}   (paper 3.83e8 = {3.83e8:.0f})")
        print(f"  f0=f*(D/2)  = {f0:.6e}   (paper 2.33e-5)")
        _s.exit()
    if "--mwfull" in _s.argv:
        from min_weight import (find_all_min_weight_logicals, min_weight_fail_count,
                                expanded_logical_count, build_circuit_translation_perms)
        from math import comb
        import time
        budget = 40
        for a in _s.argv:
            if a.startswith("--budget="): budget = int(a.split("=")[1])
        H,A,pc,mult,dc,nd=build_matrices(6)
        perms=build_circuit_translation_perms(None, H, det_coords=dc, verbose=False)
        # Run the coset enumeration under several prior schemes and UNION the results:
        # BP-OSD's column ordering is prior-driven, so each scheme reaches a different
        # subset of weight-D logicals (natural priors favour high-mult idle/meas columns;
        # uniform/inverted surface the low-mult CNOT-column logicals).
        schemes=[("natural", np.asarray(pc,float))]
        if "--union" in _s.argv or "--uniform" in _s.argv:
            schemes.append(("uniform", np.ones(H.shape[1])))
        if "--union" in _s.argv:
            schemes.append(("inverted", 1.0/np.asarray(pc,float)))
        print(f"Exhaustive coset enumeration (budget/coset={budget}, schemes={[s for s,_ in schemes]}) ...", flush=True)
        t0=time.time()
        logicals=set()
        for name,pri in schemes:
            ls=find_all_min_weight_logicals(matrices=(H,A,pc), D=6, budget_per_coset=budget,
                symmetry_perms=perms, workers=20, progress_every=400, priors=pri)
            logicals|=ls
            print(f"  [{name}] found {len(ls)}, running union {len(logicals)}", flush=True)
        ld_comp=len(logicals); ld_exp=expanded_logical_count(logicals, mult)
        fails,n_exp=min_weight_fail_count(H,A,logicals,mult)
        f0=fails/comb(n_exp,3)
        print(f"\n===== EXHAUSTIVE TECHNIQUE II vs PAPER (budget={budget}, {time.time()-t0:.0f}s) =====")
        print(f"  |L(D)| comp = {ld_comp:>14}")
        print(f"  |L(D)| exp  = {ld_exp:>14}   (paper 6.01e12)")
        print(f"  |F(D/2)|    = {fails:>14}   (paper 383000000)")
        print(f"  f0=f*(D/2)  = {f0:.6e}   (paper 2.33e-5)")
        _s.exit()
    if "--exact" in _s.argv:
        # Exact, complete weight-6 logical enumeration by canonical-detector DFS
        # (= the paper's symmetry + fault-restriction method). Low memory.
        from min_weight import (min_weight_fail_count, expanded_logical_count,
                                build_circuit_translation_perms)
        from math import comb
        import time, sys as _ss
        _ss.setrecursionlimit(10000)
        H,A,pc,mult,dc,nd=build_matrices(6)
        N=H.shape[1]; n_det=H.shape[0]
        perms=build_circuit_translation_perms(None,H,det_coords=dc,verbose=False)
        # column detector-support (int bitmask) and A-action (int)
        col_h=[0]*N; col_a=[0]*N; mindet=[0]*N
        for j in range(N):
            for d in np.flatnonzero(H[:,j]): col_h[j]|=(1<<int(d))
            for r in np.flatnonzero(A[:,j]): col_a[j]|=(1<<int(r))
            mindet[j]=(col_h[j]&-col_h[j]).bit_length()-1 if col_h[j] else -1
        cols_mindet=[[] for _ in range(n_det)]
        for j in range(N):
            if mindet[j]>=0: cols_mindet[mindet[j]].append(j)
        for d in range(n_det): cols_mindet[d].sort()
        results=set(); nodes=[0]; t0=time.time(); nextp=[50_000_000]
        # Precompute, per column, the set of detectors it touches that are >= its min-det
        # (all of them) — used implicitly via col_h. Complete subset-per-detector DFS:
        def dfs(chosen, xor, amask, d):
            nodes[0]+=1
            if nodes[0]>=nextp[0]:
                print(f"   ... nodes={nodes[0]:,}  |L_canon|={len(results)}  {time.time()-t0:.0f}s",flush=True)
                nextp[0]+=50_000_000
            if xor==0:                                   # irreducible logical fully cancelled
                if len(chosen)==6 and amask!=0:
                    results.add(frozenset(chosen))
                return
            if len(chosen)>=6: return
            while d<n_det and ((xor>>d)&1)==0 and not cols_mindet[d]:
                d+=1
            if d==n_det: return
            if ((xor>>d)&1)==1 and not cols_mindet[d]:
                return                                   # bit d on, no column can fix it
            cm=cols_mindet[d]; need=(xor>>d)&1
            def pick(i, ch, xr, am, cnt):
                if len(ch)>6: return
                if i==len(cm):
                    if (cnt&1)==need:                    # subset makes detector d even
                        dfs(ch, xr, am, d+1)
                    return
                pick(i+1, ch, xr, am, cnt)                                          # exclude
                c=cm[i]
                pick(i+1, ch+[c], xr^col_h[c], am^col_a[c], cnt+1)                  # include
            pick(0, chosen, xor, amask, 0)
        # Symmetry anchor: global-minimum detector forced to check-index 0 (8 detectors, one per
        # cycle), covered by an even (>=2) subset of its min-det columns. Expand by 36 shifts after.
        canon=[c*36 for c in range(n_det//36)]
        print(f"exact anchored complete DFS: {len(canon)} canonical min-detectors ...",flush=True)
        for d0 in canon:
            cm0=cols_mindet[d0]
            def pick0(i,ch,xr,am,cnt):
                if len(ch)>6: return
                if i==len(cm0):
                    if cnt>=2 and (cnt&1)==0:
                        dfs(ch,xr,am,d0+1)
                    return
                pick0(i+1,ch,xr,am,cnt)
                c=cm0[i]; pick0(i+1,ch+[c],xr^col_h[c],am^col_a[c],cnt+1)
            pick0(0,[],0,0,0)
        print(f"canonical (min-det check-idx 0) logicals: {len(results)}  "
              f"(nodes={nodes[0]:,}, {time.time()-t0:.0f}s)",flush=True)
        full=set()
        for L in results:
            for p in perms: full.add(frozenset(int(p[c]) for c in L))
        print(f"after 36-shift expansion: |L(D)|={len(full)}",flush=True)
        fails,n_exp=min_weight_fail_count(H,A,full,mult)
        ld_exp=expanded_logical_count(full,mult)
        f0=fails/comb(n_exp,3)
        print("\n===== EXACT (canonical-DFS) TECHNIQUE II vs PAPER =====")
        print(f"  |L(D)| comp = {len(full):>14}")
        print(f"  |L(D)| exp  = {ld_exp:>14}   (paper 6.01e12)")
        print(f"  |F(D/2)|    = {fails:>14}   (paper 383000000)")
        print(f"  f0=f*(D/2)  = {f0:.6e}   (paper 2.33e-5)")
        _s.exit()
    if "--swap" in _s.argv:
        from min_weight import (find_min_weight_logicals, find_all_min_weight_logicals,
                                min_weight_fail_count, expanded_logical_count,
                                build_circuit_translation_perms)
        from collections import defaultdict
        from math import comb
        import time
        seedbudget=40; osd=10
        for a in _s.argv:
            if a.startswith("--seedbudget="): seedbudget=int(a.split("=")[1])
            if a.startswith("--osd="): osd=int(a.split("=")[1])
        H,A,pc,mult,dc,nd=build_matrices(6)
        N=H.shape[1]
        perms=build_circuit_translation_perms(None,H,det_coords=dc,verbose=False)
        print(f"seeding with BP-OSD coset enumeration (budget={seedbudget}, osd_order={osd}) ...",flush=True)
        seed=find_all_min_weight_logicals(matrices=(H,A,pc),D=6,budget_per_coset=seedbudget,
            symmetry_perms=perms,workers=20,progress_every=0,osd_order=osd)
        print(f"seed |L|={len(seed)}",flush=True)
        # column detector-support and A-action as Python int bitmasks
        col_h=[0]*N; col_a=[0]*N
        for j in range(N):
            for d in np.flatnonzero(H[:,j]): col_h[j]|=(1<<int(d))
            for r in np.flatnonzero(A[:,j]): col_a[j]|=(1<<int(r))
        t0=time.time()
        pair_hash=defaultdict(list)
        for i in range(N):
            hi=col_h[i]
            for j in range(i+1,N):
                pair_hash[hi^col_h[j]].append((i,j))
        print(f"pair_hash: {len(pair_hash)} keys ({time.time()-t0:.0f}s)",flush=True)
        # BFS expand by 2-swaps (purely combinatorial, no BP-OSD)
        found=set(seed); queue=list(seed); t1=time.time()
        while queue:
            L=queue.pop(); cols=sorted(L)
            for a_i in range(6):
                for b_i in range(a_i+1,6):
                    ci,cj=cols[a_i],cols[b_i]
                    rest=[c for c in cols if c!=ci and c!=cj]; rest_set=set(rest)
                    for (ck,cl) in pair_hash[col_h[ci]^col_h[cj]]:
                        if ck in rest_set or cl in rest_set: continue
                        newL=frozenset(rest+[ck,cl])
                        if len(newL)!=6 or newL in found: continue
                        am=0
                        for c in newL: am^=col_a[c]
                        if am==0: continue          # trivial (stabilizer), not a logical
                        found.add(newL); queue.append(newL)
        print(f"2-swap closure: |L|={len(found)} ({time.time()-t1:.0f}s)",flush=True)
        active=sorted(set().union(*found)) if found else []
        print(f"active columns (appear in some logical): {len(active)} / {N}",flush=True)
        import pickle
        with open("bb6_ref_data.pkl","wb") as fp:
            pickle.dump(dict(H=H,A=A,mult=mult,pc=pc,dc=dc,N=N,
                             found=[tuple(sorted(f)) for f in found],
                             active=active,col_h=col_h,col_a=col_a), fp)
        print("saved bb6_ref_data.pkl",flush=True)
        ld_comp=len(found); ld_exp=expanded_logical_count(found,mult)
        fails,n_exp=min_weight_fail_count(H,A,found,mult)
        f0=fails/comb(n_exp,3)
        print("\n===== 2-SWAP COMPLETE TECHNIQUE II vs PAPER =====")
        print(f"  |L(D)| comp = {ld_comp:>14}")
        print(f"  |L(D)| exp  = {ld_exp:>14}   (paper 6.01e12)")
        print(f"  |F(D/2)|    = {fails:>14}   (paper 383000000)")
        print(f"  f0=f*(D/2)  = {f0:.6e}   (paper 2.33e-5)")
        _s.exit()
    # diagnostic for num_cycles=6
    H,A,pc,mult,dc,nd=build_matrices(6)
    Ntilde=H.shape[1]
    detsupps=[frozenset(np.flatnonzero(H[:,j]).tolist()) for j in range(Ntilde)]
    n_distinct_det=len(set(detsupps))
    n_empty_det=sum(1 for s in detsupps if len(s)==0)
    print(f"Ñ (cols)={Ntilde}  distinct detector-supports={n_distinct_det}  "
          f"empty-detector cols={n_empty_det}  H-collisions={Ntilde-n_distinct_det}")
    print(f"N_expanded={int(mult.sum())}, n_det rows={nd}, A shape={A.shape}")
