import numpy as np
from pathlib import Path

def quat_to_rot(q, order="xyzw"):
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError("Quaternion must have 4 elements")
    if order == "wxyz":
        w, x, y, z = q
    else:
        x, y, z, w = q
    n = np.linalg.norm([w, x, y, z])
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project_to_rotation_matrix(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt
    return R_proj


def pose_from_npz_sam3d(path: Path, quat_order: str = "xyzw") -> np.ndarray:
    data = np.load(str(path))
    t = data["translation"].reshape(-1)
    # SAM3D convention fix: flip Y translation.
    t[1] *= -1
    r = data["rotation"]
    if r.shape[-1] == 4:
        R = quat_to_rot(r.reshape(-1), order=quat_order)
    else:
        R = r.reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t[:3]
    return T


def remap_sam3d_object_pose(T_CO: np.ndarray) -> np.ndarray:
    T = np.asarray(T_CO, dtype=float).copy()
    R = T[:3, :3]
    t = T[:3, 3]

    T_remap = np.eye(4, dtype=float)
    T_remap[:3, :] = np.array(
        [
            [ R[2, 2],  R[2, 0], -R[2, 1], -t[0]],
            [-R[1, 2], -R[1, 0], -R[1, 1], -t[1]],
            [ R[0, 2],  R[0, 0],  R[0, 1],  t[2]],
        ],
        dtype=float,
    )
    T_remap[:3, :3] = project_to_rotation_matrix(T_remap[:3, :3])
    return T_remap


def compose_world_object_from_sam3d(T_WC: np.ndarray, T_CO: np.ndarray) -> np.ndarray:
    return T_WC @ remap_sam3d_object_pose(T_CO)
