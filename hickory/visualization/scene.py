import argparse
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
import trimesh

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

from hickory.utils.third_party import add_third_party_paths

add_third_party_paths()

from mps.superquadrics import superquadric
from hickory.utils.frame_conventions import compose_world_object_from_sam3d

def load_camera_pose(path: Path) -> np.ndarray:
    mat = np.loadtxt(str(path))
    if mat.size == 12:
        mat = mat.reshape(3, 4)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
    if mat.shape != (4, 4):
        raise ValueError(f"Unexpected camera pose shape in {path}: {mat.shape}")
    return mat


def load_camera_trajectory(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.load(str(path))
    else:
        data = None
        # First try numeric-only loads (csv or whitespace).
        try:
            data = np.loadtxt(str(path), delimiter=",")
        except Exception:
            try:
                data = np.loadtxt(str(path), delimiter=None)
            except Exception:
                data = None
        # If still not numeric-only, parse lines and extract the last 16 floats.
        if data is None:
            rows = []
            for line in Path(path).read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 16:
                    continue
                try:
                    vals = [float(x) for x in parts[-16:]]
                except ValueError:
                    continue
                rows.append(vals)
            if not rows:
                raise ValueError(f"No valid pose rows found in {path}")
            data = np.asarray(rows, dtype=float)
    data = np.asarray(data)
    if data.ndim == 2 and data.shape[1] > 16:
        data = data[:, -16:]
    if data.ndim == 2 and data.shape in [(3, 4), (4, 4)]:
        data = data.reshape(1, *data.shape)
    if data.ndim == 2 and data.shape[1] in [12, 16]:
        mats = []
        for row in data:
            mat = row.reshape(3, 4) if row.size == 12 else row.reshape(4, 4)
            if mat.shape == (3, 4):
                mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
            mats.append(mat)
        return np.stack(mats, axis=0)
    if data.ndim == 3 and data.shape[1:] in [(3, 4), (4, 4)]:
        if data.shape[1:] == (3, 4):
            data = np.concatenate(
                [data, np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (data.shape[0], 1, 1))],
                axis=1,
            )
        return data
    raise ValueError(f"Unexpected trajectory shape in {path}: {data.shape}")


def load_camera_trajectory_from_pattern(pattern: str) -> np.ndarray:
    if "{id}" in pattern:
        glob_pattern = pattern.replace("{id}", "*")
    else:
        glob_pattern = pattern
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No pose files matched pattern: {pattern}")

    def frame_key(p: Path) -> int:
        stem = p.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else 0

    paths = sorted(paths, key=frame_key)
    mats = [load_camera_pose(p) for p in paths]
    return np.stack(mats, axis=0)


def load_transform(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        mat = np.load(str(path))
    else:
        mat = np.loadtxt(str(path))
    if mat.size == 12:
        mat = mat.reshape(3, 4)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
    if mat.shape != (4, 4):
        raise ValueError(f"Unexpected transform shape in {path}: {mat.shape}")
    return mat


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh | None:
    mesh = None
    if path.suffix.lower() not in {".glb", ".gltf"}:
        mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh is None or len(mesh.vertices) == 0:
        try:
            tri = trimesh.load(str(path), force="mesh")
        except Exception:
            return None
        if tri is None or not isinstance(tri, trimesh.Trimesh) or len(tri.vertices) == 0:
            return None
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tri.vertices, dtype=np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tri.faces, dtype=np.int32))
        visual = getattr(tri, "visual", None)
        if visual is not None:
            if getattr(visual, "kind", None) == "texture":
                try:
                    visual = visual.to_color()
                except Exception:
                    visual = None
            vertex_colors = getattr(visual, "vertex_colors", None) if visual is not None else None
            if vertex_colors is not None and len(vertex_colors) == len(tri.vertices):
                vertex_colors = np.asarray(vertex_colors, dtype=np.float64)
                if vertex_colors.shape[1] >= 3:
                    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[:, :3] / 255.0)
    mesh.compute_vertex_normals()
    return mesh


def load_geometry(path: Path) -> o3d.geometry.Geometry | None:
    if path.suffix == ".vtk":
        return None
    mesh = load_mesh(path)
    if mesh is not None:
        return mesh
    if path.suffix.lower() in {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}:
        return None
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd is None or len(pcd.points) == 0:
        return None
    return pcd


def load_object_pointcloud(obj_dir: Path) -> o3d.geometry.PointCloud | None:
    for path in object_pointcloud_candidates(obj_dir):
        if not path.exists():
            continue
        if path.suffix == ".npy":
            try:
                pts = np.asarray(np.load(str(path)), dtype=np.float64)
            except Exception:
                continue
            if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            return pcd

        pcd = o3d.io.read_point_cloud(str(path))
        if pcd is not None and len(pcd.points) > 0:
            return pcd
    return None


def object_pointcloud_candidates(obj_dir: Path):
    return [
        obj_dir / "point_cloud_full.ply",
        obj_dir / "point_cloud_full.npy",
        obj_dir / "point_cloud_world.ply",
        obj_dir / "point_cloud_world.npy",
        obj_dir / "point_cloud.ply",
        obj_dir / "point_cloud.npy",
    ]


def find_object_pointcloud_path(obj_dir: Path) -> Path | None:
    for path in object_pointcloud_candidates(obj_dir):
        if not path.exists():
            continue
        if path.suffix == ".npy":
            try:
                pts = np.load(str(path), mmap_mode="r")
            except Exception:
                continue
            if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
                continue
            return path

        pcd = o3d.io.read_point_cloud(str(path))
        if pcd is not None and len(pcd.points) > 0:
            return path
    return None


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0


def npy_payload_bytes(path: Path) -> int | None:
    if path.suffix != ".npy":
        return None
    try:
        return int(np.load(str(path), mmap_mode="r").nbytes)
    except Exception:
        return None


def add_layer_storage(storage, layer: str, path: Path | None):
    if path is None or not path.exists():
        return
    entry = storage[layer]
    entry["files"] += 1
    entry["disk_bytes"] += path.stat().st_size
    payload_bytes = npy_payload_bytes(path)
    if payload_bytes is not None:
        entry["npy_payload_bytes"] += payload_bytes
        entry["npy_files"] += 1


def collect_representation_storage(
    objects_dir: Path,
    show_object_pointclouds: bool,
    show_object_meshes: bool,
    load_sq: bool,
    load_expanded_sq: bool,
    expanded_sq_params_name: str,
):
    storage = {
        "object point clouds": {"files": 0, "disk_bytes": 0, "npy_files": 0, "npy_payload_bytes": 0},
        "object meshes": {"files": 0, "disk_bytes": 0, "npy_files": 0, "npy_payload_bytes": 0},
        "superquadrics": {"files": 0, "disk_bytes": 0, "npy_files": 0, "npy_payload_bytes": 0},
        "expanded superquadrics": {"files": 0, "disk_bytes": 0, "npy_files": 0, "npy_payload_bytes": 0},
    }
    obj_dirs = sorted([p for p in objects_dir.iterdir() if p.is_dir() and p.name.startswith("obj_")])
    for obj_dir in obj_dirs:
        if show_object_pointclouds:
            add_layer_storage(storage, "object point clouds", find_object_pointcloud_path(obj_dir))
        if show_object_meshes:
            add_layer_storage(storage, "object meshes", obj_dir / "obj_mesh.glb")
        if load_sq:
            add_layer_storage(storage, "superquadrics", obj_dir / "obj_sq_params.npy")
        if load_expanded_sq:
            add_layer_storage(storage, "expanded superquadrics", obj_dir / expanded_sq_params_name)
    return storage


def print_representation_storage(title: str, storage):
    print(f"\nRepresentation storage: {title}")
    print(f"{'Layer':<24} {'Files':>7} {'Disk':>12} {'NumPy payload':>16}")
    print("-" * 63)
    total_disk = 0
    total_payload = 0
    for layer, entry in storage.items():
        disk_bytes = entry["disk_bytes"]
        payload_bytes = entry["npy_payload_bytes"]
        total_disk += disk_bytes
        total_payload += payload_bytes
        payload_text = format_bytes(payload_bytes) if entry["npy_files"] else "-"
        print(f"{layer:<24} {entry['files']:>7} {format_bytes(disk_bytes):>12} {payload_text:>16}")
    print("-" * 63)
    print(f"{'Total':<24} {'':>7} {format_bytes(total_disk):>12} {format_bytes(total_payload):>16}")


def apply_color(geom: o3d.geometry.Geometry, color):
    if isinstance(geom, o3d.geometry.TriangleMesh):
        geom.compute_vertex_normals()
        geom.paint_uniform_color(color)
    elif isinstance(geom, o3d.geometry.PointCloud):
        geom.paint_uniform_color(color)
    return geom


def stable_color_from_name(name: str):
    seed = sum(ord(c) for c in name) % 9973
    rng = np.random.default_rng(seed)
    color = rng.uniform(0.2, 0.95, size=3)
    return color.tolist()


def parse_obj_id(obj_dir: Path, fallback: int):
    suffix = obj_dir.name.split("_")[-1]
    return int(suffix) if suffix.isdigit() else fallback


def parse_object_id_specs(specs):
    ids = set()
    for spec in specs or []:
        text = str(spec).strip()
        if not text:
            continue
        if text.startswith("obj_"):
            text = text.split("_")[-1]
        ids.add(int(text))
    return ids


def find_default_camera_trajectory(objects_dir: Path) -> Path | None:
    candidates = [
        "camera_poses_world.txt",
        "camera_poses_world.npy",
        "camera_trajectory.txt",
        "camera_trajectory.npy",
        "camera_poses.txt",
        "camera_poses.npy",
        "trajectory.txt",
        "trajectory.npy",
        "traj.txt",
        "traj.npy",
        "poses.txt",
        "poses.npy",
    ]
    for name in candidates:
        cand = objects_dir / name
        if cand.exists():
            return cand
    return None


def is_dataset_world_trajectory(path: Path | None) -> bool:
    return path is not None and path.stem == "camera_poses_world"


def find_object_pose_path(obj_dir: Path, pose_source: str) -> Path:
    if pose_source == "sam3d":
        return obj_dir / "pose_sam3d.txt"
    if pose_source == "fp":
        return obj_dir / "pose_foundation.txt"
    raise ValueError(f"Unsupported pose_source: {pose_source}")


def _signed_pow(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(values) * (np.abs(values) ** exponent)


def superquadrics_to_mesh(params: np.ndarray, resolution: int = 40) -> o3d.geometry.TriangleMesh | None:
    if params is None or len(params) == 0:
        return None
    vertices = []
    triangles = []
    vertex_offset = 0
    lat_segments = max(4, int(resolution))
    lon_segments = max(12, int(resolution) * 2)
    latitudes = np.linspace(-np.pi / 2.0, np.pi / 2.0, lat_segments + 1)
    longitudes = np.linspace(-np.pi, np.pi, lon_segments, endpoint=False)
    cos_v = np.cos(longitudes)
    sin_v = np.sin(longitudes)

    for quadric in params:
        sq = superquadric(
            quadric[0:2], quadric[2:5], quadric[5:8], quadric[8:11]
        )
        eps1, eps2 = sq.shape
        a, b, c = sq.scale
        interior_u = latitudes[1:-1]
        cos_u = _signed_pow(np.cos(interior_u), eps1)
        sin_u = _signed_pow(np.sin(interior_u), eps1)
        cos_v_eps = _signed_pow(cos_v, eps2)
        sin_v_eps = _signed_pow(sin_v, eps2)

        x = a * cos_u[:, None] * cos_v_eps[None, :]
        y = b * cos_u[:, None] * sin_v_eps[None, :]
        z = c * np.repeat(sin_u[:, None], lon_segments, axis=1)
        ring_vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        local_vertices = np.vstack(
            [
                np.array([[0.0, 0.0, -c]], dtype=np.float64),
                ring_vertices.astype(np.float64, copy=False),
                np.array([[0.0, 0.0, c]], dtype=np.float64),
            ]
        )

        ring_count = lat_segments - 1
        bottom_idx = vertex_offset
        top_idx = vertex_offset + local_vertices.shape[0] - 1
        first_ring_start = vertex_offset + 1

        if ring_count > 0:
            for j in range(lon_segments):
                j_next = (j + 1) % lon_segments
                triangles.append([bottom_idx, first_ring_start + j, first_ring_start + j_next])

            for ring_idx in range(ring_count - 1):
                curr_ring = first_ring_start + ring_idx * lon_segments
                next_ring = curr_ring + lon_segments
                for j in range(lon_segments):
                    j_next = (j + 1) % lon_segments
                    curr = curr_ring + j
                    curr_next = curr_ring + j_next
                    nxt = next_ring + j
                    nxt_next = next_ring + j_next
                    triangles.append([curr, curr_next, nxt_next])
                    triangles.append([curr, nxt_next, nxt])

            last_ring = first_ring_start + (ring_count - 1) * lon_segments
            for j in range(lon_segments):
                j_next = (j + 1) % lon_segments
                triangles.append([top_idx, last_ring + j_next, last_ring + j])

        world_vertices = (sq.RotM @ local_vertices.T).T + sq.translation
        vertices.append(world_vertices)
        vertex_offset += local_vertices.shape[0]
    if not vertices:
        return None
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.vstack(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh




def add_trajectory_geometry(geometries, traj_points, color):
    if not traj_points:
        return
    pts = np.asarray(traj_points)
    if len(pts) >= 2:
        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        traj = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        traj.colors = o3d.utility.Vector3dVector([color] * len(lines))
        geometries.append(traj)
    else:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.paint_uniform_color(color)
        geometries.append(pcd)


def add_object_keyframe_links(geometries, links, color, dash_length: float = 0.12, gap_length: float = 0.08):
    if not links:
        return
    points = []
    lines = []
    colors = []
    for link in links:
        if len(link) == 3:
            camera_center, object_center, link_color = link
        else:
            camera_center, object_center = link
            link_color = color
        start = np.asarray(camera_center, dtype=np.float64)
        end = np.asarray(object_center, dtype=np.float64)
        direction = end - start
        total_len = float(np.linalg.norm(direction))
        if total_len <= 1e-9:
            continue
        direction /= total_len
        step = max(dash_length + gap_length, 1e-6)
        dist = 0.0
        while dist < total_len:
            dash_start = start + direction * dist
            dash_end = start + direction * min(dist + dash_length, total_len)
            start_idx = len(points)
            points.append(dash_start)
            points.append(dash_end)
            lines.append([start_idx, start_idx + 1])
            colors.append(np.asarray(link_color, dtype=np.float64))
            dist += step
    if not lines:
        return
    link_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    link_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    geometries.append(link_set)


def make_camera_pyramid(
    transform: np.ndarray,
    size: float = 0.12,
    color=(1.0, 0.95, 0.55),
) -> o3d.geometry.LineSet:
    half_w = size * 0.55
    half_h = size * 0.4
    depth = size
    local_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [-half_w, -half_h, depth],
            [half_w, -half_h, depth],
            [half_w, half_h, depth],
            [-half_w, half_h, depth],
        ],
        dtype=np.float64,
    )
    points_h = np.hstack([local_points, np.ones((local_points.shape[0], 1), dtype=np.float64)])
    world_points = (transform @ points_h.T).T[:, :3]
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(world_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    frustum.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return frustum


def add_keyframe_pyramids(
    geometries,
    keyframe_poses,
    color,
    size: float = 0.12,
    dedup_translation_tol: float = 1e-4,
    dedup_rotation_tol: float = 1e-3,
):
    if not keyframe_poses:
        return
    unique_poses = []
    for pose in keyframe_poses:
        is_duplicate = False
        for existing in unique_poses:
            if np.linalg.norm(existing[:3, 3] - pose[:3, 3]) > dedup_translation_tol:
                continue
            delta_rot = existing[:3, :3].T @ pose[:3, :3]
            angle = np.arccos(np.clip((np.trace(delta_rot) - 1.0) * 0.5, -1.0, 1.0))
            if angle <= dedup_rotation_tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_poses.append(pose.copy())
    for pose in unique_poses:
        geometries.append(make_camera_pyramid(pose, size=size, color=color))


def add_odom_origin_marker(geometries, transform=None, size=0.8, color=(1.0, 0.0, 0.0)):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.01, size * 0.16))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    if transform is not None:
        frame.transform(transform)
        sphere.transform(transform)
    geometries.append(frame)
    geometries.append(sphere)


def load_scene(
    objects_dir: Path,
    quat_order: str,
    color,
    transform: np.ndarray | None = None,
    trajectory_transform: np.ndarray | None = None,
    normalize_trajectory_to_first: bool = False,
    enable_layer_visualization: bool = False,
    trajectory_display_offset: np.ndarray | None = None,
    mesh_offset: np.ndarray | None = None,
    sq_offset: np.ndarray | None = None,
    expanded_sq_offset: np.ndarray | None = None,
    load_sq: bool = True,
    load_expanded_sq: bool = False,
    expanded_sq_params_name: str = "obj_sq_params_expanded.npy",
    camera_traj_path: Path | None = None,
    camera_traj_pattern: str | None = None,
    pose_source: str = "sam3d",
    show_object_keyframes: bool = True,
    show_object_meshes: bool = True,
    show_object_pointclouds: bool = False,
    pointcloud_color=None,
    keyframe_pyramid_size: float = 0.05,
    show_camera_trajectory: bool = True,
    show_upper_layer_keyframe_links: bool = True,
    keyframe_link_stride: int = 1,
    upper_layer_link_object_ids=None,
    sq_resolution: int = 24,
):
    geometries = []
    traj_points = []
    keyframe_poses = []
    object_keyframe_links = []
    centers_by_id = {}
    linked_object_count = 0
    keyframe_link_stride = max(1, int(keyframe_link_stride))
    upper_layer_link_object_ids = (
        set(upper_layer_link_object_ids)
        if upper_layer_link_object_ids is not None
        else None
    )
    if not enable_layer_visualization:
        trajectory_display_offset = None
        mesh_offset = None
        sq_offset = None
        expanded_sq_offset = None

    if camera_traj_pattern is not None:
        traj = load_camera_trajectory_from_pattern(camera_traj_pattern)
        traj_base_inv = np.linalg.inv(traj[0]) if normalize_trajectory_to_first and traj.shape[0] > 0 else None
        for i in range(traj.shape[0]):
            T_WC = traj[i]
            if traj_base_inv is not None:
                T_WC = traj_base_inv @ T_WC
            if trajectory_transform is not None:
                T_WC = trajectory_transform @ T_WC
            traj_center = T_WC[:3, 3].copy()
            if trajectory_display_offset is not None:
                traj_center = traj_center + np.asarray(trajectory_display_offset, dtype=np.float64).reshape(3)
            traj_points.append(traj_center)
    elif camera_traj_path is not None:
        if camera_traj_path.exists():
            traj = load_camera_trajectory(camera_traj_path)
            traj_base_inv = np.linalg.inv(traj[0]) if normalize_trajectory_to_first and traj.shape[0] > 0 else None
            for i in range(traj.shape[0]):
                T_WC = traj[i]
                if traj_base_inv is not None:
                    T_WC = traj_base_inv @ T_WC
                if trajectory_transform is not None:
                    T_WC = trajectory_transform @ T_WC
                traj_center = T_WC[:3, 3].copy()
                if trajectory_display_offset is not None:
                    traj_center = traj_center + np.asarray(trajectory_display_offset, dtype=np.float64).reshape(3)
                traj_points.append(traj_center)
        else:
            print(f"[WARN] camera trajectory not found: {camera_traj_path}")
            camera_traj_path = None

    obj_dirs = sorted([p for p in objects_dir.iterdir() if p.is_dir() and p.name.startswith("obj_")])
    for idx, obj_dir in enumerate(obj_dirs):
        mesh_path = obj_dir / "obj_mesh.glb"
        sq_params_path = obj_dir / "obj_sq_params.npy"
        expanded_sq_params_path = obj_dir / expanded_sq_params_name
        pose_path = find_object_pose_path(obj_dir, pose_source=pose_source)
        cam_pose_path = obj_dir / "camera_pose.txt"
        if not pose_path.exists() or not cam_pose_path.exists():
            continue
        if not mesh_path.exists() and not sq_params_path.exists() and not show_object_pointclouds:
            continue

        mesh = load_geometry(mesh_path) if mesh_path.exists() else None
        pointcloud = load_object_pointcloud(obj_dir) if show_object_pointclouds else None
        params = None
        expanded_params = None
        if sq_params_path.exists():
            # Handle empty/corrupt SQ params files gracefully.
            try:
                if sq_params_path.stat().st_size > 0:
                    params = np.load(str(sq_params_path))
            except (OSError, EOFError, ValueError):
                params = None
        if load_expanded_sq and expanded_sq_params_path.exists():
            try:
                if expanded_sq_params_path.stat().st_size > 0:
                    expanded_params = np.load(str(expanded_sq_params_path))
            except (OSError, EOFError, ValueError):
                expanded_params = None
        
        sq = None
        if load_sq and params is not None:
            sq = superquadrics_to_mesh(params, resolution=sq_resolution)
        expanded_sq = None
        if load_expanded_sq and expanded_params is not None:
            expanded_sq = superquadrics_to_mesh(expanded_params, resolution=sq_resolution)
            
        if mesh is None and sq is None and expanded_sq is None and params is None and expanded_params is None and pointcloud is None:
            continue

        T_WC = load_camera_pose(cam_pose_path)
        T_CO = load_transform(pose_path)
        if pose_source == "sam3d":
            # T_WO = T_WC @ T_CO
            T_WO = compose_world_object_from_sam3d(T_WC, T_CO)
        else:
            T_WO = T_WC @ T_CO
        
        # print(f"Loaded SAM3D pose for {obj_dir.name}, T_WC:\n{T_WC}\nT_CO:\n{T_CO}\nComposed T_WO:\n{T_WO}")

        if transform is not None:
            T_WC = transform @ T_WC
            T_WO = transform @ T_WO

        if show_object_keyframes:
            T_WC_display = T_WC.copy()
            if trajectory_display_offset is not None:
                T_WC_display[:3, 3] += np.asarray(trajectory_display_offset, dtype=np.float64).reshape(3)
            keyframe_poses.append(T_WC_display)

        pointcloud_center = None
        mesh_center = None
        sq_center = None
        expanded_sq_center = None
        if pointcloud is not None:
            if transform is not None:
                pointcloud.transform(transform)
            point_color = (
                np.asarray(pointcloud_color, dtype=np.float64)
                if pointcloud_color is not None
                else np.asarray(stable_color_from_name(obj_dir.name), dtype=np.float64)
            )
            apply_color(pointcloud, point_color)
            pointcloud_center = pointcloud.get_axis_aligned_bounding_box().get_center()
            geometries.append(pointcloud)
        if mesh is not None and show_object_meshes:
            mesh.transform(T_WO)
            mesh_center = mesh.get_axis_aligned_bounding_box().get_center()
            if mesh_offset is not None:
                mesh.translate(np.asarray(mesh_offset, dtype=np.float64).reshape(3), relative=True)
                mesh_center = mesh.get_axis_aligned_bounding_box().get_center()
            geometries.append(mesh)
        if sq is not None:
            sq.transform(T_WO)
            sq_center = sq.get_axis_aligned_bounding_box().get_center()
            if sq_offset is not None:
                sq.translate(np.asarray(sq_offset, dtype=np.float64).reshape(3), relative=True)
                sq_center = sq.get_axis_aligned_bounding_box().get_center()
            apply_color(sq, color)
            geometries.append(sq)
        if expanded_sq is not None:
            expanded_sq.transform(T_WO)
            expanded_sq_center = expanded_sq.get_axis_aligned_bounding_box().get_center()
            if expanded_sq_offset is not None:
                expanded_sq.translate(np.asarray(expanded_sq_offset, dtype=np.float64).reshape(3), relative=True)
                expanded_sq_center = expanded_sq.get_axis_aligned_bounding_box().get_center()
            expanded_sq_color = np.clip(np.asarray(color, dtype=np.float64) * 0.55 + np.array([0.35, 0.3, 0.0]), 0.0, 1.0)
            apply_color(expanded_sq, expanded_sq_color)
            geometries.append(expanded_sq)

        # Track object centers for downstream overlays.
        obj_center = None
        if pointcloud_center is not None:
            obj_center = pointcloud_center
        elif mesh_center is not None:
            obj_center = mesh_center
        elif sq_center is not None:
            obj_center = sq_center
        elif expanded_sq_center is not None:
            obj_center = expanded_sq_center
        elif params is not None:
            local_center = np.mean(params[:, 8:11], axis=0)
            obj_center = (T_WO @ np.append(local_center, 1.0))[:3]
        elif expanded_params is not None:
            local_center = np.mean(expanded_params[:, 8:11], axis=0)
            obj_center = (T_WO @ np.append(local_center, 1.0))[:3]

        if obj_center is not None:
            obj_id = parse_obj_id(obj_dir, idx)
            centers_by_id[obj_id] = obj_center
            if show_object_keyframes:
                should_add_keyframe_links = (linked_object_count % keyframe_link_stride) == 0
                linked_object_count += 1
                keyframe_center = keyframe_poses[-1][:3, 3].copy()
                base_color = np.asarray(color, dtype=np.float64)
                pc_link_color = tuple(np.clip(base_color * 0.35 + np.array([0.55, 0.45, 0.15]), 0.0, 1.0).tolist())
                mesh_link_color = tuple(np.clip(base_color * 0.55 + np.array([0.15, 0.55, 0.25]), 0.0, 1.0).tolist())
                sq_link_color = tuple(np.clip(base_color * 0.35 + np.array([0.45, 0.15, 0.55]), 0.0, 1.0).tolist())
                expanded_sq_link_color = tuple(np.clip(base_color * 0.45 + np.array([0.45, 0.4, 0.05]), 0.0, 1.0).tolist())
                if enable_layer_visualization:
                    if pointcloud_center is not None:
                        object_keyframe_links.append((keyframe_center.copy(), pointcloud_center.copy(), pc_link_color))
                    should_add_upper_links = (
                        show_upper_layer_keyframe_links
                        and (
                            obj_id in upper_layer_link_object_ids
                            if upper_layer_link_object_ids is not None
                            else should_add_keyframe_links
                        )
                    )
                    if should_add_upper_links:
                        if mesh_center is not None:
                            object_keyframe_links.append((keyframe_center.copy(), mesh_center.copy(), mesh_link_color))
                        if sq_center is not None:
                            object_keyframe_links.append((keyframe_center.copy(), sq_center.copy(), sq_link_color))
                        if expanded_sq_center is not None:
                            object_keyframe_links.append((keyframe_center.copy(), expanded_sq_center.copy(), expanded_sq_link_color))
                elif should_add_keyframe_links:
                    object_keyframe_links.append((keyframe_center.copy(), obj_center.copy(), tuple(base_color.tolist())))

    if show_camera_trajectory:
        add_trajectory_geometry(geometries, traj_points, color)
    if show_object_keyframes:
        keyframe_color = np.clip(np.asarray(color, dtype=np.float64) * 0.6 + 0.4, 0.0, 1.0)
        add_keyframe_pyramids(
            geometries,
            keyframe_poses,
            color=tuple(keyframe_color.tolist()),
            size=keyframe_pyramid_size,
        )
        link_color = np.clip(np.asarray(color, dtype=np.float64) * 0.75 + 0.25, 0.0, 1.0)
        add_object_keyframe_links(
            geometries,
            object_keyframe_links,
            color=tuple(link_color.tolist()),
        )

    return geometries, centers_by_id


def main():
    parser = argparse.ArgumentParser(description="Visualize one reconstructed scene in world frame")
    parser.add_argument(
        "scene",
        nargs="?",
        default=None,
        help="Scene name under reconstruction/ (e.g., sparkal1_oneformer)",
    )
    parser.add_argument(
        "--objects-dir",
        default="output_objects_from_bag",
        help="Folder containing obj_XXX subfolders.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        default=(0.0, -5.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="XYZ display offset used to separate mesh, SQ, and expanded-SQ layers.",
    )
    parser.add_argument(
        "--quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Legacy option (unused when reading txt object poses).",
    )
    parser.add_argument(
        "--pose-source",
        choices=["sam3d", "fp"],
        default="fp",
        help="Choose object pose source. 'sam3d' uses pose_sam3d.txt with compose_world_object_from_sam3d; 'fp' uses pose_foundation.txt with T_WC @ T_CO.",
    )
    parser.add_argument(
        "--camera-trajectory",
        default=None,
        help="Optional camera trajectory file (txt/npy). For reconstructed inputs, the first pose is also used to lift the scene into world.",
    )
    parser.add_argument(
        "--camera-trajectory-pattern",
        default=None,
        help="Optional camera trajectory file pattern (e.g., dataset/7-Scenes/fire/seq-01/frame-{id}.pose.txt).",
    )
    parser.add_argument(
        "--hide-camera-trajectory",
        action="store_true",
        help="Hide the camera trajectory polyline.",
    )
    parser.add_argument(
        "--hide-odometry-origins",
        action="store_true",
        help="Hide odometry frame origin markers.",
    )
    parser.add_argument(
        "--odometry-origin-size",
        type=float,
        default=0.1,
        help="Size of odometry origin coordinate frame markers.",
    )
    parser.add_argument(
        "--hide-object-keyframes",
        action="store_true",
        help="Hide camera pyramids at keyframes that introduced objects.",
    )
    parser.add_argument(
        "--hide-upper-layer-keyframe-links",
        action="store_true",
        help="Hide dashed links from trajectory/keyframes to mesh/SQ/expanded SQ layers.",
    )
    parser.add_argument(
        "--keyframe-link-stride",
        type=int,
        default=1,
        help="Draw object keyframe links for every Nth valid object. Use 1 to draw all links.",
    )
    parser.add_argument(
        "--upper-layer-link-objects",
        nargs="+",
        default=None,
        metavar="OBJ_ID",
        help="Draw upper-layer keyframe links only for these objects, e.g. obj_032 041.",
    )
    parser.add_argument(
        "--pointcloud-only",
        action="store_true",
        help="Render only object point clouds.",
    )
    parser.add_argument(
        "--pointcloud-color",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Optional uniform RGB color for all object point clouds. If omitted, each object gets a stable distinct color.",
    )
    parser.add_argument(
        "--keyframe-pyramid-size",
        type=float,
        default=0.12,
        help="Size of camera pyramids drawn at object-introducing keyframes.",
    )
    parser.add_argument(
        "--sq-resolution",
        type=int,
        default=24,
        help="Surface resolution used for superquadric visualization.",
    )
    parser.add_argument(
        "--expanded-sq-params-name",
        default="obj_sq_params_expanded.npy",
        help="Filename used to load expanded SQ params inside each object directory.",
    )
    args = parser.parse_args()

    objects_dir = Path("reconstruction") / args.scene if args.scene is not None else Path(args.objects_dir)
    traj = Path(args.camera_trajectory) if args.camera_trajectory else find_default_camera_trajectory(objects_dir)
    traj_pattern = args.camera_trajectory_pattern
    world_from_local = None
    traj_is_explicit = (
        args.camera_trajectory is not None
        or args.camera_trajectory_pattern is not None
        or is_dataset_world_trajectory(traj)
    )
    if args.camera_trajectory or args.camera_trajectory_pattern or is_dataset_world_trajectory(traj):
        world_traj = load_camera_trajectory_from_pattern(traj_pattern) if traj_pattern else load_camera_trajectory(traj)
        if len(world_traj) > 0:
            world_from_local = np.asarray(world_traj[0], dtype=np.float64)

    load_sq = not args.pointcloud_only
    load_expanded_sq = not args.pointcloud_only
    show_object_meshes = not args.pointcloud_only
    show_object_pointclouds = True
    layer_offset = np.asarray(args.offset, dtype=np.float64).reshape(3)
    print_representation_storage(
        f"scene ({objects_dir})",
        collect_representation_storage(
            objects_dir,
            show_object_pointclouds=show_object_pointclouds,
            show_object_meshes=show_object_meshes,
            load_sq=load_sq,
            load_expanded_sq=load_expanded_sq,
            expanded_sq_params_name=args.expanded_sq_params_name,
        ),
    )
    geometries, _ = load_scene(
        objects_dir,
        quat_order=args.quat_order,
        color=[0.2, 0.6, 0.9],
        transform=world_from_local,
        trajectory_transform=world_from_local,
        normalize_trajectory_to_first=traj_is_explicit,
        enable_layer_visualization=True,
        trajectory_display_offset=0.0 * layer_offset,
        mesh_offset=-layer_offset,
        sq_offset=-2.0 * layer_offset,
        expanded_sq_offset=-3.0 * layer_offset,
        load_sq=load_sq,
        load_expanded_sq=load_expanded_sq,
        expanded_sq_params_name=args.expanded_sq_params_name,
        camera_traj_path=traj,
        camera_traj_pattern=traj_pattern,
        pose_source=args.pose_source,
        show_object_keyframes=not args.hide_object_keyframes,
        show_object_meshes=show_object_meshes,
        show_object_pointclouds=show_object_pointclouds,
        pointcloud_color=(
            np.asarray(args.pointcloud_color, dtype=np.float64)
            if args.pointcloud_color is not None
            else None
        ),
        keyframe_pyramid_size=args.keyframe_pyramid_size,
        show_camera_trajectory=not args.hide_camera_trajectory,
        show_upper_layer_keyframe_links=not args.hide_upper_layer_keyframe_links,
        keyframe_link_stride=args.keyframe_link_stride,
        upper_layer_link_object_ids=(
            parse_object_id_specs(args.upper_layer_link_objects)
            if args.upper_layer_link_objects is not None
            else None
        ),
        sq_resolution=args.sq_resolution,
    )
    if not args.hide_odometry_origins:
        add_odom_origin_marker(
            geometries,
            transform=world_from_local if world_from_local is not None else np.eye(4),
            size=args.odometry_origin_size,
            color=(0.2, 0.6, 0.9),
        )

    if not geometries:
        raise RuntimeError("No valid objects found under the provided directories.")

    o3d.visualization.draw_geometries(geometries, mesh_show_back_face=True)


if __name__ == "__main__":
    main()
