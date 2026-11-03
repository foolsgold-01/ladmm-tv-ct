"""TIGRE cone-beam geometry helpers."""

import numpy as np
import tigre


def make_cone_geometry(n_det_v, n_det_u, sdd, sod, pixel_size, n_xy, n_z):
    """Build a circular cone-beam TIGRE geometry.

    The voxel size is the detector pixel size demagnified to the iso-centre
    (``pixel_size * sod / sdd``), rescaled so that ``n_xy`` voxels span the
    same field of view as ``n_det_u`` detector columns.

    TIGRE convention: ``nVoxel = [nX, nZ, nY]`` (Z = axial / rotation axis),
    so reconstructed NumPy volumes have shape ``(nX, nZ, nY)``.

    Returns
    -------
    geo : tigre.geometry
    dv : float
        Effective isotropic voxel size [mm].
    """
    voxel_size_full = pixel_size * sod / sdd
    dv = voxel_size_full * (n_det_u / n_xy)

    geo = tigre.geometry()
    geo.mode = 'cone'
    geo.DSD = sdd
    geo.DSO = sod

    geo.nDetector = np.array([n_det_v, n_det_u])
    geo.dDetector = np.array([pixel_size, pixel_size])
    geo.sDetector = geo.nDetector * geo.dDetector

    geo.nVoxel = np.array([n_xy, n_xy, n_z])
    geo.dVoxel = np.array([dv, dv, dv])
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    geo.offOrigin = np.zeros(3)
    geo.offDetector = np.zeros(2)
    geo.COR = 0.0
    geo.rotDetector = np.zeros(3)
    geo.accuracy = 0.5

    return geo, dv
