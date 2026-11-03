# ladmm-tv-ct

Total-variation (TV) regularised reconstruction for **sparse-view cone-beam CT**.
It uses Linearised ADMM (L-ADMM), with [TIGRE](https://github.com/CERN/TIGRE)
GPU projectors and [SCICO](https://github.com/lanl/scico) for the TV operators.

## Problem

$$\min_{x\ge 0}\;\tfrac12\|Ax-b\|_2^2 + \lambda\,\|Dx\|_{2,1}$$

- $A$ is the TIGRE cone-beam forward projector (`tigre.Ax`). $A^\top$ is `tigre.Atb` (matched).
- $b$ holds the flat/dark-corrected, $-\log$-transformed projections.
- $D$ is the 3-D forward-difference operator (`scico.linop.FiniteDifference`).
- $\|\cdot\|_{2,1}$ is isotropic TV (`scico.functional.L21Norm`).

## Algorithm

The split $z = Dx$ gives the scaled augmented Lagrangian
$L_\rho(x,z,u) = \tfrac12\|Ax-b\|^2 + \lambda\|z\|_{2,1} + \tfrac\rho2\|Dx - z + u\|^2$.
Each iteration does three steps:

1. **x-update.** One projected gradient step on $L_\rho$, not an exact solve:
   $x \leftarrow \max\bigl(0,\; x - \tau\,[A^\top(Ax-b) + \rho\,D^\top(Dx - z + u)]\bigr)$
2. **z-update.** The exact TV prox (block soft-threshold):
   $z \leftarrow \mathrm{prox}_{(\lambda/\rho)\|\cdot\|_{2,1}}(Dx + u)$
3. **u-update.** Dual ascent: $u \leftarrow u + Dx - z$

**Why linearised.** Standard ADMM solves the x-subproblem with CG on
$A^\top A + \rho D^\top D$, and that needs $A^\top$ to be the exact adjoint of $A$.
GPU back-projectors such as TIGRE's `Atb` are only approximately matched.
A single gradient step tolerates that mismatch and costs one `Ax` and one `Atb` per iteration.

**Step size.** $\tau = 1/(L + \rho\|D\|^2)$, where:

- $L = \|A^\top A\|_2$ is estimated by power iteration (`power_iteration`).
- $\|D^\top D\|_2 \le 12$ for 3-D forward differences.

**Initialisation.** By default the solver starts from a non-negative FDK reconstruction.

**Monitoring.** The solver logs the objective, the primal residual $\|Dx - z\|$
and the dual residual $\rho\|D^\top(z - z_{\text{old}})\|$ every `log_every` iterations.
They are returned in a `LADMMHistory` object.

Projections and $A$/$A^\top$ run on the GPU (TIGRE). The TV operators run in JAX.
Each iteration moves the volume between NumPy and JAX.

## Installation

TIGRE is **not on PyPI**. The PyPI package called `tigre` is an unrelated
bioinformatics tool. Build TIGRE from source with CUDA:

```bash
git clone --depth=1 https://github.com/CERN/TIGRE.git
pip install ./TIGRE          # set CUDA_HOME / CUDA_PATH if nvcc isn't found
```

Then install this package:

```bash
pip install .                # or: pip install .[examples]  (matplotlib + Jupyter)
```

Tested with Python 3.12, TIGRE 3.1.3, SCICO 0.0.7 and JAX 0.8.1 (see `requirements.txt`).

## Usage

### Command line

```bash
ladmm-tv-ct calibrated_radiographs.npz -o recon.nii \
    --sparse-step 10 --sdd 557 --sod 530 --pixel-size 0.117 \
    --n-xy 400 --n-z 700 --lam 0.005 --rho 1.0 --n-iter 150
```

Run `ladmm-tv-ct -h` for all options.

### Python

```python
from ladmm_tv_ct import load_projections, make_cone_geometry, ladmm_tv, save_nifti

proj, angles = load_projections('calibrated_radiographs.npz', sparse_step=10)
geo, dv = make_cone_geometry(*proj.shape[1:], sdd=557, sod=530,
                             pixel_size=0.117, n_xy=400, n_z=700)
recon, hist = ladmm_tv(proj, geo, angles, lam=0.005, rho=1.0, n_iter=150)
save_nifti(recon, 'recon.nii', dv)
```

### Input format

The input is an `.npz` file. Its `data` key holds an array of shape
`(n_angles, nDetV, nDetU)`. The views are assumed to be evenly spaced over 360°.

### Output format

The volume uses TIGRE's axis order `(nX, nZ, nY)`, where Z is the rotation axis.
`save_nifti` reorders it to `(nZ, nX, nY)` and writes isotropic voxel spacing.

`examples/ladmm_tv_reconstruction.ipynb` walks through the same pipeline step by step.
It also shows FDK vs. L-ADMM slices and convergence plots.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `lam` | 0.005 | TV weight λ. Higher values give smoother images. |
| `rho` | 1.0 | ADMM penalty ρ |
| `n_iter` | 150 | Outer iterations |
| `n_power` | 8 | Power iterations used to estimate $L$ |
| `sparse_step` | 10 | Keeps every N-th view. 360 → 36 views. |

## References

- E. Y. Sidky, X. Pan, "Image reconstruction in circular cone-beam CT by constrained, total-variation minimization", *Phys. Med. Biol.* 53 (2008).
- S. Boyd et al., "Distributed optimization and statistical learning via ADMM", *Found. Trends Mach. Learn.* 3 (2011).
- S. Ramani, J. A. Fessler, "A splitting-based iterative algorithm for accelerated statistical X-ray CT reconstruction", *IEEE TMI* 31 (2012).
- A. Biguri et al., "TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction", *Biomed. Phys. Eng. Express* 2 (2016).

## License

MIT
