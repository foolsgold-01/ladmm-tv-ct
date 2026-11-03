"""Loading projections and saving reconstructions."""

import os

import nibabel as nib
import numpy as np


def load_projections(path, key='data', sparse_step=1):
    """Load projections (n_angles, nDetV, nDetU) from an .npz file.

    Every ``sparse_step``-th view is kept; angles are assumed to be evenly
    spaced over a full 2*pi rotation.
    """
    data = np.load(os.path.expanduser(path))
    proj = data[key][::sparse_step].astype(np.float32)
    angles = np.linspace(0, 2 * np.pi, proj.shape[0], endpoint=False).astype(np.float32)
    return proj, angles


def save_nifti(recon, path, voxel_size):
    """Save a TIGRE volume (nX, nZ, nY) as NIfTI with axes (nZ, nX, nY)."""
    vol = np.ascontiguousarray(np.transpose(recon, (1, 0, 2)), dtype=np.float32)
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    nib.save(nib.Nifti1Image(vol, affine), path)
