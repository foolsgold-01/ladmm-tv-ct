"""Linearised ADMM for TV-regularised cone-beam CT.

Solves  min_{x >= 0}  1/2 ||Ax - b||^2 + lam * ||Dx||_{2,1}

with TIGRE GPU projectors for A / A^T and SCICO for the finite-difference
operator D and the L21 proximal map.
"""

from dataclasses import dataclass, field
from time import time

import jax.numpy as jnp
import numpy as np
import tigre
import tigre.algorithms as algs
from scico import functional, linop

# Upper bound on ||D^T D||_2 for 3-D forward differences (4 per axis).
D_NORM_SQ_3D = 12.0


def power_iteration(geo, angles, n_iter=8, seed=0, verbose=True):
    """Estimate L = ||A^T A||_2 with the TIGRE projector pair."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(tuple(geo.nVoxel)).astype(np.float32)
    x /= np.linalg.norm(x)

    t0 = time()
    L = 0.0
    for k in range(n_iter):
        x = tigre.Atb(tigre.Ax(x, geo, angles), geo, angles, 'matched').astype(np.float32)
        L = float(np.linalg.norm(x))
        x /= L
        if verbose:
            print(f'  power iter [{k+1:2d}/{n_iter}]  L ≈ {L:.4e}  ({time()-t0:.0f}s)')
    return L


def fdk_warm_start(proj, geo, angles):
    """FDK reconstruction clipped to be non-negative."""
    x = algs.fdk(proj, geo, angles).astype(np.float32)
    return np.clip(x, 0.0, None)


@dataclass
class LADMMHistory:
    iters: list = field(default_factory=list)
    objective: list = field(default_factory=list)
    tv: list = field(default_factory=list)
    res_primal: list = field(default_factory=list)
    res_dual: list = field(default_factory=list)


def ladmm_tv(proj, geo, angles, lam=0.005, rho=1.0, n_iter=150, x0=None,
             L=None, n_power=8, d_norm_sq=D_NORM_SQ_3D, log_every=10, verbose=True):
    """Run L-ADMM TV reconstruction.

    Parameters
    ----------
    proj : ndarray, shape (n_angles, nDetV, nDetU)
        -log transformed projections.
    geo : tigre.geometry
    angles : ndarray, shape (n_angles,)
    lam : float
        TV weight.
    rho : float
        Augmented-Lagrangian penalty.
    n_iter : int
        Number of outer iterations.
    x0 : ndarray, optional
        Initial volume (nX, nZ, nY). Defaults to an FDK reconstruction.
    L : float, optional
        ||A^T A||_2. Estimated by power iteration if not given.
    n_power : int
        Power iterations used when ``L`` is None.
    d_norm_sq : float
        Upper bound on ||D^T D||_2.
    log_every : int
        Logging period (iterations); 0 disables monitoring.

    Returns
    -------
    recon : ndarray, shape (nX, nZ, nY), float32
    history : LADMMHistory
    """
    proj = np.asarray(proj, dtype=np.float32)

    if L is None:
        L = power_iteration(geo, angles, n_power, verbose=verbose)
    tau = 1.0 / (L + rho * d_norm_sq)

    if x0 is None:
        x0 = fdk_warm_start(proj, geo, angles)

    vol_shape = tuple(int(n) for n in geo.nVoxel)
    D = linop.FiniteDifference(input_shape=vol_shape, append=0, circular=False)
    g = functional.L21Norm()

    x = jnp.array(x0, dtype=jnp.float32)
    z = D(x)
    u = jnp.zeros_like(z)

    hist = LADMMHistory()
    if verbose:
        print(f'L-ADMM: {n_iter} iterations  τ={tau:.4e}  λ={lam}  ρ={rho}  vol={vol_shape}')
    t_total = time()

    for k in range(n_iter):
        # (1) x-update: projected gradient step on the augmented Lagrangian
        Ax = tigre.Ax(np.asarray(x, dtype=np.float32), geo, angles)
        grad_data = jnp.array(tigre.Atb(Ax - proj, geo, angles, 'matched'))
        grad_reg = D.T(D(x) - z + u)
        x = jnp.clip(x - tau * (grad_data + rho * grad_reg), 0.0, None)

        # (2) z-update: block soft-threshold (TV prox)
        z_old = z
        Dx = D(x)
        z = g.prox(Dx + u, lam / rho)

        # (3) u-update: scaled dual ascent
        u = u + Dx - z

        if log_every and k % log_every == 0:
            tv_val = float(g(Dx))
            obj = 0.5 * float(np.sum((Ax - proj) ** 2)) + lam * tv_val
            r_p = float(jnp.linalg.norm(Dx - z))
            r_d = float(jnp.linalg.norm(rho * D.T(z - z_old)))
            hist.iters.append(k)
            hist.objective.append(obj)
            hist.tv.append(tv_val)
            hist.res_primal.append(r_p)
            hist.res_dual.append(r_d)
            if verbose:
                print(f'[{k:4d}/{n_iter}]  obj={obj:.4e}  TV={tv_val:.4e}  '
                      f'|r_p|={r_p:.3e}  |r_d|={r_d:.3e}  t={time()-t_total:.0f}s')

    if verbose:
        print(f'L-ADMM done in {time()-t_total:.0f}s')

    return np.asarray(x, dtype=np.float32), hist
