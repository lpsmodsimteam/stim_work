# Podman container for the fail-fast QEC framework

Runs `experiment_runner` (importance sampling + Technique II/I on BB/gross codes) inside a
self-contained OCI image, so the cluster needs only `podman` — no conda env, no Rust toolchain,
no `pip install` on the login node.

## 1. Build (once, from the REPO ROOT)

```bash
podman build -f container/Containerfile -t stim-work-qec:latest .
```

The build compiles `relay_bp` from source (Rust), so the first build takes a few minutes; after
that the dependency layer is cached and only rebuilds when `container/requirements.txt` changes.
The image bakes the code in, so it also runs standalone.

## 2. Verify

```bash
# preflight: packages + editable install + a real stim/relay_bp decode round-trip (skip SLURM)
podman run --rm stim-work-qec:latest python test.py --no-slurm      # expect all PASS

# optional: the unit tests
podman run --rm stim-work-qec:latest python -m pytest -q tests/test_min_weight.py
```

## 3. Run the three LPU configs (with your own shot counts)

Edit the shot budget in each config first — the relevant keys in
`experiments/configs/gross_lpu_idle.yaml`, `gross_lpu_y1.yaml`, `gross_automorphism.yaml`:

```yaml
adaptive_failures: 20        # target failures per weight (raise for tighter error bars)
adaptive_shots_max: 3000     # hard per-weight cap (raise to sample deeper into the tail)
adaptive_stop_zero_bins: 3   # stop after N full-budget zero bins (raise to sample more tail)
```

Because `experiments/` is bind-mounted, edited configs take effect with **no rebuild**. Launch
each (bind-mount `runs/` so checkpoints + results land on the cluster filesystem):

```bash
for cfg in gross_lpu_idle gross_lpu_y1 gross_automorphism; do
  podman run --rm --cpus 16 \
    -e OMP_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 \
    -v "$PWD/experiments:/opt/stim_work/experiments:Z" \
    -v "$PWD/runs:/opt/stim_work/runs:Z" \
    stim-work-qec:latest \
    python -m experiment_runner --config experiments/configs/$cfg.yaml --cpus 16
done
```

Runs are **resumable**: re-running the same config picks up its per-weight checkpoint under
`runs/framework/bb144/<experiment>/` (loses at most the in-progress weight). Changing
`weights_range`/`seed` invalidates the checkpoint — delete `runs/framework/.../spectrum.json` to
start fresh.

## 4. Via SLURM + your submit.sh

`run.podman.sbatch` is a drop-in replacement for `experiments/slurm/run.sbatch` that wraps the
runner in `podman run`. It resolves the array-index → config from `experiments/manifest.yaml`
*inside* the container, so the host needs no python/pyyaml. Point your `submit.sh` at it (or
`sbatch --array=<i> container/run.podman.sbatch`). Override the image with `QEC_IMAGE=...`.

## Notes / gotchas

- **`:Z`** on the bind mounts relabels for SELinux hosts. If your cluster isn't SELinux and
  rejects `:Z`, drop it (or use `:z` for a shared label).
- **Threads:** `--cpus N` caps the Relay/BLAS pools (the runner also sets `RAYON_NUM_THREADS`
  from it); pass `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` too so BLAS doesn't grab all host cores
  before the runner starts. Match `N` to `--cpus-per-task`.
- **Rootless podman:** files written to bind-mounted `runs/` are owned by your user via the user
  namespace — no `USER` directive needed. If your site requires `--userns=keep-id`, add it.
- **Reproducibility:** the image bakes a code snapshot; `relay_bp` is pinned to an exact commit.
  For a frozen run, use only baked configs (skip the `experiments/` bind-mount) and record the
  image digest (`podman inspect --format '{{.Digest}}' stim-work-qec:latest`).
- **Rust version:** built against the stable toolchain. If `relay_bp` ever fails to compile, pin a
  known-good toolchain in the builder stage (`--default-toolchain 1.XX.0`).
