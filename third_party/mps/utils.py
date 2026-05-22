import numpy as np
import pyvista as pv

def _signed_pow(x, p):
    return np.sign(x) * (np.abs(x) ** p)


def sample_superquadric(sq_obj, resolution=80):
    """
    Sample a superquadric surface using the parameters stored in sq_obj.
    sq_obj is the original class from superquadrics.py.
    Returns X, Y, Z grids.
    """

    eps1, eps2 = sq_obj.shape
    a, b, c = sq_obj.scale
    R = sq_obj.RotM                 # rotation matrix
    t = sq_obj.translation          # translation

    # Create parameter grid
    u = np.linspace(-np.pi/2, np.pi/2, resolution)
    v = np.linspace(-np.pi, np.pi, resolution)
    u, v = np.meshgrid(u, v)

    # Parametric equations
    x = a * _signed_pow(np.cos(u), eps1) * _signed_pow(np.cos(v), eps2)
    y = b * _signed_pow(np.cos(u), eps1) * _signed_pow(np.sin(v), eps2)
    z = c * _signed_pow(np.sin(u), eps1)

    # Apply rotation + translation
    P = np.stack([x, y, z], axis=-1).reshape(-1, 3).T   # (3, N)
    P = (R @ P).T + t                                   # (N, 3)

    X = P[:, 0].reshape(x.shape)
    Y = P[:, 1].reshape(x.shape)
    Z = P[:, 2].reshape(x.shape)

    return X, Y, Z


def superquadric_to_mesh(sq_obj, resolution=50):
    X, Y, Z = sample_superquadric(sq_obj, resolution)
    return pv.StructuredGrid(X, Y, Z)

