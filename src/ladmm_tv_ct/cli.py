"""Command-line entry point: ``ladmm-tv-ct projections.npz -o recon.nii``."""

import argparse

from .geometry import make_cone_geometry
from .io import load_projections, save_nifti
from .solver import ladmm_tv


def main(argv=None):
    p = argparse.ArgumentParser(description='L-ADMM TV cone-beam CT reconstruction (TIGRE + SCICO).')
    p.add_argument('input', help='.npz file with -log projections (n_angles, nDetV, nDetU)')
    p.add_argument('-o', '--output', default='recon_ladmm_tv.nii', help='output NIfTI path')
    p.add_argument('--key', default='data', help='array key inside the .npz')
    p.add_argument('--sparse-step', type=int, default=10, help='keep every N-th view')
    p.add_argument('--sdd', type=float, default=557.0, help='source-detector distance [mm]')
    p.add_argument('--sod', type=float, default=530.0, help='source-object distance [mm]')
    p.add_argument('--pixel-size', type=float, default=0.117, help='detector pixel size [mm]')
    p.add_argument('--n-xy', type=int, default=400, help='transaxial grid size [voxels]')
    p.add_argument('--n-z', type=int, default=700, help='axial grid size [voxels]')
    p.add_argument('--lam', type=float, default=0.005, help='TV weight λ')
    p.add_argument('--rho', type=float, default=1.0, help='ADMM penalty ρ')
    p.add_argument('--n-iter', type=int, default=150, help='L-ADMM iterations')
    p.add_argument('--n-power', type=int, default=8, help='power iterations for ||A||²')
    args = p.parse_args(argv)

    proj, angles = load_projections(args.input, args.key, args.sparse_step)
    print(f'Projections: {proj.shape}  ({len(angles)} views)')

    geo, dv = make_cone_geometry(proj.shape[1], proj.shape[2], args.sdd, args.sod,
                                 args.pixel_size, args.n_xy, args.n_z)
    recon, _ = ladmm_tv(proj, geo, angles, lam=args.lam, rho=args.rho,
                        n_iter=args.n_iter, n_power=args.n_power)

    save_nifti(recon, args.output, dv)
    print(f'Saved: {args.output}  voxel={dv:.4f} mm')


if __name__ == '__main__':
    main()
