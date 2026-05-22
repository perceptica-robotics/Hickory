from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HICKORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HICKORY_ROOT.parent
for path in (PROJECT_ROOT, HICKORY_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from hickory.navigation.geometry import (
    SceneBounds,
    TriangleMeshData,
    clearance_map,
    grid_to_world,
    inflate_mask,
    load_mesh_data,
    make_height_band,
    meshdata_to_o3d,
    path_length_world,
    rasterize_mesh_height_band,
    superquadrics_to_meshdata,
    world_to_grid,
)
from hickory.navigation.planning_core import (
    PlanResult,
    RRTConfig,
    SQImplicitCollisionChecker,
    build_dense_sq_occupancy_mask,
    build_sparse_sq_display_mask,
    compute_sq_world_aabbs,
    make_path_lineset,
    make_plan_result_from_path,
    merge_linesets,
    plan_gvd,
    plan_rrt_with_collision,
    rasterize_oriented_bbox_list_height_band,
    smooth_path_chaikin_world,
    transform_sq_params_to_world,
)
try:
    from hickory.utils.frame_conventions import compose_world_object_from_sam3d
except ModuleNotFoundError:
    from frame_conventions import compose_world_object_from_sam3d

from hickory.visualization.scene import find_object_pose_path, load_camera_pose, load_camera_trajectory, load_transform


def pose_aligned_bbox_mesh_from_local_mesh(mesh_local: TriangleMeshData, object_transform: np.ndarray) -> TriangleMeshData:
    bounds = mesh_local.bounds
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    local_vertices = np.array(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ],
        dtype=np.float64,
    )
    vertices_h = np.column_stack([local_vertices, np.ones((local_vertices.shape[0],), dtype=np.float64)])
    world_vertices = (np.asarray(object_transform, dtype=np.float64) @ vertices_h.T).T[:, :3]
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return TriangleMeshData(
        vertices=world_vertices,
        faces=faces,
    )


def transform_meshdata(mesh: TriangleMeshData, transform: np.ndarray) -> TriangleMeshData:
    vertices_h = np.hstack(
        [
            np.asarray(mesh.vertices, dtype=np.float64),
            np.ones((mesh.vertices.shape[0], 1), dtype=np.float64),
        ]
    )
    transformed = (np.asarray(transform, dtype=np.float64) @ vertices_h.T).T[:, :3]
    return TriangleMeshData(vertices=transformed, faces=np.asarray(mesh.faces, dtype=np.int64).copy())


def merge_meshes(meshes: list[TriangleMeshData]) -> TriangleMeshData:
    if not meshes:
        return TriangleMeshData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
        )
    vertices = []
    faces = []
    vertex_offset = 0
    for mesh in meshes:
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            continue
        vertices.append(np.asarray(mesh.vertices, dtype=np.float64))
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + vertex_offset)
        vertex_offset += mesh.vertices.shape[0]
    if not vertices:
        return TriangleMeshData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
        )
    return TriangleMeshData(vertices=np.vstack(vertices), faces=np.vstack(faces))


def bbox_prism_from_mesh(mesh: TriangleMeshData) -> TriangleMeshData:
    bounds = mesh.bounds
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    vertices = np.array(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return TriangleMeshData(vertices=vertices, faces=faces)


def choose_scene_padding(
    mesh: TriangleMeshData,
    side_pad: float | None,
    lateral_pad: float | None,
) -> tuple[float, float]:
    bounds = mesh.bounds
    width_x = float(bounds[1, 0] - bounds[0, 0])
    width_z = float(bounds[1, 2] - bounds[0, 2])
    diag_xz = float(np.hypot(width_x, width_z))
    default_pad = max(0.6, 0.06 * diag_xz, 0.05 * max(width_x, width_z))
    out_side = float(side_pad) if side_pad is not None else default_pad
    out_lateral = float(lateral_pad) if lateral_pad is not None else default_pad
    return out_side, out_lateral


def compute_scene_bounds_from_mesh_and_trajectory(
    mesh: TriangleMeshData,
    side_pad: float,
    lateral_pad: float,
    trajectory_world: np.ndarray | None = None,
) -> SceneBounds:
    if mesh.vertices.size == 0:
        raise RuntimeError("No scene geometry loaded.")
    min_corner = np.min(mesh.vertices, axis=0)
    max_corner = np.max(mesh.vertices, axis=0)
    min_x = float(min_corner[0])
    max_x = float(max_corner[0])
    min_z = float(min_corner[2])
    max_z = float(max_corner[2])
    if trajectory_world is not None and trajectory_world.size > 0:
        min_x = min(min_x, float(np.min(trajectory_world[:, 0])))
        max_x = max(max_x, float(np.max(trajectory_world[:, 0])))
        min_z = min(min_z, float(np.min(trajectory_world[:, 2])))
        max_z = max(max_z, float(np.max(trajectory_world[:, 2])))
    return SceneBounds(
        min_x=min_x - side_pad,
        max_x=max_x + side_pad,
        min_z=min_z - lateral_pad,
        max_z=max_z + lateral_pad,
    )


def rasterize_multi_layer_mesh_occupancy(
    mesh: TriangleMeshData,
    bounds,
    resolution: int,
    robot_base_height: float,
    robot_height: float,
    num_layers: int,
) -> np.ndarray:
    num_layers = max(2, int(num_layers))
    layer_heights = np.linspace(robot_base_height, robot_base_height + robot_height, num_layers, dtype=np.float64)
    occupancy = np.zeros((resolution, resolution), dtype=bool)
    thin_band_height = max(robot_height / max(num_layers - 1, 1), 1e-3)
    for y_value in layer_heights:
        layer_band = make_height_band(float(y_value), float(thin_band_height))
        occupancy |= rasterize_mesh_height_band(mesh, bounds, resolution, layer_band)
    return occupancy


def load_scene_navigation_meshes(
    objects_dir: Path,
    camera_trajectory_path: Path | None,
    pose_source: str,
    mesh_max_triangles: int,
    sq_resolution: int,
    expanded_sq_params_name: str,
    object_limit: int | None = None,
    verbose_load: bool = False,
    load_mesh: bool = True,
    load_sq: bool = True,
    build_bbox: bool = True,
) -> tuple[TriangleMeshData, TriangleMeshData, TriangleMeshData, np.ndarray | None]:
    obj_dirs = sorted([path for path in objects_dir.iterdir() if path.is_dir() and path.name.startswith("obj_")])
    if not obj_dirs:
        raise RuntimeError(f"No obj_* folders found under {objects_dir}")
    if object_limit is not None:
        obj_dirs = obj_dirs[: max(0, int(object_limit))]

    world_from_local = None
    trajectory_world = None
    if camera_trajectory_path is not None and camera_trajectory_path.exists():
        trajectory = load_camera_trajectory(camera_trajectory_path)
        if trajectory.shape[0] > 0:
            world_from_local = np.asarray(trajectory[0], dtype=np.float64)
            trajectory_world = trajectory[:, :3, 3].astype(np.float64)

    mesh_parts: list[TriangleMeshData] = []
    sq_parts: list[TriangleMeshData] = []
    bbox_parts: list[TriangleMeshData] = []

    for obj_idx, obj_dir in enumerate(obj_dirs, start=1):
        if verbose_load:
            print(f"[load] {obj_idx}/{len(obj_dirs)} {obj_dir.name}")
        mesh_path = obj_dir / "obj_mesh.glb"
        sq_path = obj_dir / expanded_sq_params_name
        cam_pose_path = obj_dir / "camera_pose.txt"
        pose_path = find_object_pose_path(obj_dir, pose_source)
        if not cam_pose_path.exists() or not pose_path.exists():
            continue
        if not load_mesh and not load_sq:
            continue
        if load_mesh and not mesh_path.exists() and not (load_sq and sq_path.exists()):
            continue
        if load_sq and not sq_path.exists() and not (load_mesh and mesh_path.exists()):
            continue

        mesh_local = None
        if load_mesh and mesh_path.exists():
            try:
                mesh_local = load_mesh_data(mesh_path, max_triangles=mesh_max_triangles)
            except Exception:
                mesh_local = None

        T_WC = load_camera_pose(cam_pose_path)
        T_CO = load_transform(pose_path)
        if pose_source == "sam3d":
            T_WO = compose_world_object_from_sam3d(T_WC, T_CO)
        else:
            T_WO = T_WC @ T_CO
        if world_from_local is not None:
            T_WO = world_from_local @ T_WO

        if mesh_local is not None:
            mesh_world = transform_meshdata(mesh_local, T_WO)
            mesh_parts.append(mesh_world)
            if build_bbox:
                bbox_parts.append(bbox_prism_from_mesh(mesh_world))

        if load_sq and sq_path.exists():
            try:
                sq_params = np.load(sq_path)
            except Exception:
                sq_params = None
            if sq_params is not None and np.asarray(sq_params).size > 0:
                try:
                    sq_local = superquadrics_to_meshdata(np.asarray(sq_params), resolution=sq_resolution)
                    sq_world = transform_meshdata(sq_local, T_WO)
                    sq_parts.append(sq_world)
                except Exception:
                    pass

    mesh_scene = merge_meshes(mesh_parts)
    sq_scene = merge_meshes(sq_parts)
    bbox_scene = merge_meshes(bbox_parts)
    if load_mesh and mesh_scene.vertices.size == 0 and (not load_sq or sq_scene.vertices.size == 0):
        raise RuntimeError(f"No valid geometry could be loaded from {objects_dir}")
    if load_sq and sq_scene.vertices.size == 0 and (not load_mesh or mesh_scene.vertices.size == 0):
        raise RuntimeError(f"No valid geometry could be loaded from {objects_dir}")
    return mesh_scene, sq_scene, bbox_scene, trajectory_world


def load_scene_navigation_meshes_with_obb(
    objects_dir: Path,
    camera_trajectory_path: Path | None,
    pose_source: str,
    mesh_max_triangles: int,
    sq_resolution: int,
    expanded_sq_params_name: str,
    object_limit: int | None = None,
    verbose_load: bool = False,
    build_obb: bool = False,
    load_mesh: bool = True,
):
    mesh_scene, sq_scene, axis_bbox_scene, trajectory_world = load_scene_navigation_meshes(
        objects_dir=objects_dir,
        camera_trajectory_path=camera_trajectory_path,
        pose_source=pose_source,
        mesh_max_triangles=mesh_max_triangles,
        sq_resolution=sq_resolution,
        expanded_sq_params_name=expanded_sq_params_name,
        object_limit=object_limit,
        verbose_load=verbose_load,
        load_mesh=load_mesh,
        load_sq=False,
        build_bbox=load_mesh,
    )
    if not build_obb:
        return (
            mesh_scene,
            sq_scene,
            axis_bbox_scene,
            TriangleMeshData(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)),
            [],
            trajectory_world,
        )

    obj_dirs = sorted([path for path in objects_dir.iterdir() if path.is_dir() and path.name.startswith("obj_")])
    if object_limit is not None:
        obj_dirs = obj_dirs[: max(0, int(object_limit))]

    from hickory.navigation.geometry import load_mesh_data, superquadrics_to_meshdata

    world_from_local = None
    if camera_trajectory_path is not None and camera_trajectory_path.exists():
        trajectory = load_camera_trajectory(camera_trajectory_path)
        if trajectory.shape[0] > 0:
            world_from_local = np.asarray(trajectory[0], dtype=np.float64)

    obb_parts = []
    for obj_dir in obj_dirs:
        mesh_path = obj_dir / "obj_mesh.glb"
        cam_pose_path = obj_dir / "camera_pose.txt"
        pose_path = find_object_pose_path(obj_dir, pose_source)
        if not mesh_path.exists() or not cam_pose_path.exists() or not pose_path.exists():
            continue
        try:
            mesh_local = load_mesh_data(mesh_path, max_triangles=mesh_max_triangles)
        except Exception:
            continue
        T_WC = load_camera_pose(cam_pose_path)
        T_CO = load_transform(pose_path)
        if pose_source == "sam3d":
            T_WO = compose_world_object_from_sam3d(T_WC, T_CO)
        else:
            T_WO = T_WC @ T_CO
        if world_from_local is not None:
            T_WO = world_from_local @ T_WO
        obb_parts.append(pose_aligned_bbox_mesh_from_local_mesh(mesh_local, T_WO))

    obb_scene = merge_meshes(obb_parts)
    return mesh_scene, sq_scene, axis_bbox_scene, obb_scene, obb_parts, trajectory_world

def load_scene_world_sq_params(
    objects_dir: Path,
    camera_trajectory_path: Path | None,
    pose_source: str,
    expanded_sq_params_name: str,
    object_limit: int | None = None,
    verbose_load: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    obj_dirs = sorted([path for path in objects_dir.iterdir() if path.is_dir() and path.name.startswith("obj_")])
    if object_limit is not None:
        obj_dirs = obj_dirs[: max(0, int(object_limit))]

    world_from_local = None
    trajectory_world = None
    if camera_trajectory_path is not None and camera_trajectory_path.exists():
        trajectory = load_camera_trajectory(camera_trajectory_path)
        if trajectory.shape[0] > 0:
            world_from_local = np.asarray(trajectory[0], dtype=np.float64)
            trajectory_world = trajectory[:, :3, 3].astype(np.float64)

    params_world_parts = []
    for obj_idx, obj_dir in enumerate(obj_dirs, start=1):
        if verbose_load:
            print(f"[load] {obj_idx}/{len(obj_dirs)} {obj_dir.name}")
        sq_path = obj_dir / expanded_sq_params_name
        cam_pose_path = obj_dir / "camera_pose.txt"
        pose_path = find_object_pose_path(obj_dir, pose_source)
        if not sq_path.exists() or not cam_pose_path.exists() or not pose_path.exists():
            continue
        try:
            params_local = np.load(sq_path)
        except Exception:
            continue
        if np.asarray(params_local).size == 0:
            continue
        T_WC = load_camera_pose(cam_pose_path)
        T_CO = load_transform(pose_path)
        if pose_source == "sam3d":
            T_WO = compose_world_object_from_sam3d(T_WC, T_CO)
        else:
            T_WO = T_WC @ T_CO
        if world_from_local is not None:
            T_WO = world_from_local @ T_WO
        try:
            params_world_parts.append(transform_sq_params_to_world(np.asarray(params_local, dtype=np.float64), T_WO))
        except Exception:
            continue

    if not params_world_parts:
        return np.zeros((0, 11), dtype=np.float64), trajectory_world
    return np.vstack(params_world_parts), trajectory_world


def load_scene_o3d_meshes_with_original_colors(
    objects_dir: Path,
    camera_trajectory_path: Path | None,
    pose_source: str,
    object_limit: int | None,
    planner_from_scene: np.ndarray,
) -> list[o3d.geometry.TriangleMesh]:
    obj_dirs = sorted([path for path in objects_dir.iterdir() if path.is_dir() and path.name.startswith("obj_")])
    if object_limit is not None:
        obj_dirs = obj_dirs[: max(0, int(object_limit))]

    world_from_local = None
    if camera_trajectory_path is not None and camera_trajectory_path.exists():
        trajectory = load_camera_trajectory(camera_trajectory_path)
        if trajectory.shape[0] > 0:
            world_from_local = np.asarray(trajectory[0], dtype=np.float64)

    meshes = []
    for obj_dir in obj_dirs:
        mesh_path = obj_dir / "obj_mesh.glb"
        cam_pose_path = obj_dir / "camera_pose.txt"
        pose_path = find_object_pose_path(obj_dir, pose_source)
        if not mesh_path.exists() or not cam_pose_path.exists() or not pose_path.exists():
            continue

        mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
        if mesh_o3d is None or len(mesh_o3d.vertices) == 0 or len(mesh_o3d.triangles) == 0:
            continue

        T_WC = load_camera_pose(cam_pose_path)
        T_CO = load_transform(pose_path)
        if pose_source == "sam3d":
            T_WO = compose_world_object_from_sam3d(T_WC, T_CO)
        else:
            T_WO = T_WC @ T_CO
        if world_from_local is not None:
            T_WO = world_from_local @ T_WO
        if not transform_is_identity(planner_from_scene):
            T_WO = planner_from_scene @ T_WO

        mesh_o3d.transform(T_WO)
        mesh_o3d.compute_vertex_normals()
        meshes.append(mesh_o3d)
    return meshes


def make_scene_o3d_superquadrics(
    sq_params_world: np.ndarray,
    resolution: int,
    color: tuple[float, float, float],
) -> o3d.geometry.TriangleMesh | None:
    if np.asarray(sq_params_world).size == 0:
        return None
    sq_mesh = superquadrics_to_meshdata(sq_params_world, resolution=resolution)
    sq_o3d = meshdata_to_o3d(sq_mesh)
    sq_o3d.paint_uniform_color(color)
    sq_o3d.compute_vertex_normals()
    return sq_o3d


def object_xz_bounds_from_mesh(mesh: TriangleMeshData) -> SceneBounds:
    mesh_bounds = mesh.bounds
    return SceneBounds(
        min_x=float(mesh_bounds[0, 0]),
        max_x=float(mesh_bounds[1, 0]),
        min_z=float(mesh_bounds[0, 2]),
        max_z=float(mesh_bounds[1, 2]),
    )


def object_xz_bounds_from_sq_aabbs(sq_aabbs_world: np.ndarray) -> SceneBounds | None:
    sq_aabbs_world = np.asarray(sq_aabbs_world, dtype=np.float64)
    if sq_aabbs_world.size == 0:
        return None
    return SceneBounds(
        min_x=float(np.min(sq_aabbs_world[:, 0, 0])),
        max_x=float(np.max(sq_aabbs_world[:, 1, 0])),
        min_z=float(np.min(sq_aabbs_world[:, 0, 2])),
        max_z=float(np.max(sq_aabbs_world[:, 1, 2])),
    )


def choose_bounds_padding(bounds: SceneBounds, side_pad: float | None, lateral_pad: float | None) -> tuple[float, float]:
    max_dim = max(bounds.width, bounds.height)
    diag = float(np.hypot(bounds.width, bounds.height))
    default_pad = max(0.6, 0.06 * diag, 0.05 * max_dim)
    return (
        default_pad if side_pad is None else float(side_pad),
        default_pad if lateral_pad is None else float(lateral_pad),
    )


def pad_scene_bounds(
    bounds: SceneBounds,
    side_pad: float,
    lateral_pad: float,
    trajectory_world: np.ndarray | None = None,
) -> SceneBounds:
    min_x = float(bounds.min_x)
    max_x = float(bounds.max_x)
    min_z = float(bounds.min_z)
    max_z = float(bounds.max_z)
    if trajectory_world is not None and np.asarray(trajectory_world).size > 0:
        trajectory_world = np.asarray(trajectory_world, dtype=np.float64).reshape((-1, 3))
        min_x = min(min_x, float(np.min(trajectory_world[:, 0])))
        max_x = max(max_x, float(np.max(trajectory_world[:, 0])))
        min_z = min(min_z, float(np.min(trajectory_world[:, 2])))
        max_z = max(max_z, float(np.max(trajectory_world[:, 2])))
    return SceneBounds(
        min_x=min_x - float(lateral_pad),
        max_x=max_x + float(lateral_pad),
        min_z=min_z - float(side_pad),
        max_z=max_z + float(side_pad),
    )


def square_scene_bounds(bounds: SceneBounds) -> SceneBounds:
    width = float(bounds.width)
    height = float(bounds.height)
    target = max(width, height)
    if target <= 0.0 or abs(width - height) <= 1e-9:
        return bounds
    center_x = 0.5 * (bounds.min_x + bounds.max_x)
    center_z = 0.5 * (bounds.min_z + bounds.max_z)
    half = 0.5 * target
    return SceneBounds(
        min_x=center_x - half,
        max_x=center_x + half,
        min_z=center_z - half,
        max_z=center_z + half,
    )


def infer_robot_base_height(
    sq_aabbs_world: np.ndarray,
    robot_height: float,
    mesh_scene: TriangleMeshData | None = None,
) -> float:
    sq_aabbs_world = np.asarray(sq_aabbs_world, dtype=np.float64)
    if sq_aabbs_world.size > 0:
        min_y = float(np.min(sq_aabbs_world[:, 0, 1]))
        max_y = float(np.max(sq_aabbs_world[:, 1, 1]))
        height = max(float(robot_height), 1e-3)
        if max_y > min_y:
            candidates = np.linspace(min_y, max_y - height, 256, dtype=np.float64)
            counts = np.sum(
                (sq_aabbs_world[:, 0, 1][None, :] <= candidates[:, None] + height)
                & (sq_aabbs_world[:, 1, 1][None, :] >= candidates[:, None]),
                axis=1,
            )
            return float(candidates[int(np.argmax(counts))])
        return min_y
    if mesh_scene is not None and mesh_scene.vertices.size > 0:
        return float(np.percentile(np.asarray(mesh_scene.vertices, dtype=np.float64)[:, 1], 2.0))
    return 0.0


def world_xz_in_bounds(world_xz: np.ndarray, bounds: SceneBounds) -> bool:
    world_xz = np.asarray(world_xz, dtype=np.float64)
    return bool(
        bounds.min_x <= float(world_xz[0]) <= bounds.max_x
        and bounds.min_z <= float(world_xz[1]) <= bounds.max_z
    )


def transform_from_ground_plane_to_planner(ground_plane: str) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if ground_plane == "xz":
        return transform
    if ground_plane == "xy":
        # Source frame: XY ground, Z up. Planner frame: XZ ground, Y up.
        # Use Z_planner = -Y_source to preserve a right-handed rotation frame.
        transform[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        return transform
    raise ValueError(f"Unsupported ground plane: {ground_plane}")


def yaw_transform_about_planner_y(yaw_degrees: float) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_degrees))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    return transform


def transform_is_identity(transform: np.ndarray) -> bool:
    return bool(np.allclose(np.asarray(transform, dtype=np.float64), np.eye(4, dtype=np.float64)))


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.size == 0:
        return points_xyz.reshape((-1, 3))
    points_h = np.column_stack([points_xyz.reshape((-1, 3)), np.ones((points_xyz.reshape((-1, 3)).shape[0],), dtype=np.float64)])
    return (np.asarray(transform, dtype=np.float64) @ points_h.T).T[:, :3]


def occupied_mask_xz_bounds(mask: np.ndarray, planning_bounds: SceneBounds) -> SceneBounds | None:
    occupied_rc = np.argwhere(np.asarray(mask, dtype=bool))
    if occupied_rc.size == 0:
        return None
    min_row = max(float(np.min(occupied_rc[:, 0])) - 0.5, 0.0)
    max_row = min(float(np.max(occupied_rc[:, 0])) + 0.5, float(mask.shape[0] - 1))
    min_col = max(float(np.min(occupied_rc[:, 1])) - 0.5, 0.0)
    max_col = min(float(np.max(occupied_rc[:, 1])) + 0.5, float(mask.shape[1] - 1))
    corners_xz = grid_to_world(
        np.array(
            [
                [min_row, min_col],
                [max_row, max_col],
            ],
            dtype=np.float64,
        ),
        planning_bounds,
        mask.shape[0],
    )
    return SceneBounds(
        min_x=float(np.min(corners_xz[:, 0])),
        max_x=float(np.max(corners_xz[:, 0])),
        min_z=float(np.min(corners_xz[:, 1])),
        max_z=float(np.max(corners_xz[:, 1])),
    )


def outside_bounds_mask(planning_bounds: SceneBounds, navigation_bounds: SceneBounds, grid_size: int) -> np.ndarray:
    row_coords = np.arange(grid_size, dtype=np.float64)
    col_coords = np.arange(grid_size, dtype=np.float64)
    rr, cc = np.meshgrid(row_coords, col_coords, indexing="ij")
    world_xz = grid_to_world(np.column_stack([rr.reshape(-1), cc.reshape(-1)]), planning_bounds, grid_size)
    outside = (
        (world_xz[:, 0] < navigation_bounds.min_x)
        | (world_xz[:, 0] > navigation_bounds.max_x)
        | (world_xz[:, 1] < navigation_bounds.min_z)
        | (world_xz[:, 1] > navigation_bounds.max_z)
    )
    return outside.reshape(grid_size, grid_size)


def apply_planner_yaw_xz(points_xz: np.ndarray, yaw_degrees: float) -> np.ndarray:
    points_xz = np.asarray(points_xz, dtype=np.float64)
    yaw = np.deg2rad(float(yaw_degrees))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.column_stack(
        [
            c * points_xz[:, 0] + s * points_xz[:, 1],
            -s * points_xz[:, 0] + c * points_xz[:, 1],
        ]
    )


def draw_origin_marker(ax, bounds: SceneBounds, grid_size: int, planner_yaw_degrees: float):
    origin_xz = np.array([0.0, 0.0], dtype=np.float64)
    if not world_xz_in_bounds(origin_xz, bounds):
        return
    origin_rc = world_to_grid(np.asarray([origin_xz], dtype=np.float64), bounds, grid_size)[0]
    axis_len_m = 0.12 * max(bounds.width, bounds.height)
    axis_ends_xz = apply_planner_yaw_xz(
        np.asarray([[0.0, -axis_len_m], [-axis_len_m, 0.0]], dtype=np.float64),
        planner_yaw_degrees,
    )
    x_end_rc = world_to_grid(axis_ends_xz[[0]], bounds, grid_size)[0]
    y_end_rc = world_to_grid(axis_ends_xz[[1]], bounds, grid_size)[0]
    ax.scatter(
        origin_rc[1],
        origin_rc[0],
        c="#d00000",
        s=110,
        marker="+",
        linewidths=2.8,
        zorder=40,
        label="Origin",
    )
    ax.annotate(
        "",
        xy=(x_end_rc[1], x_end_rc[0]),
        xytext=(origin_rc[1], origin_rc[0]),
        arrowprops={"arrowstyle": "->", "color": "#d00000", "linewidth": 2.4},
        zorder=40,
    )
    ax.annotate(
        "",
        xy=(y_end_rc[1], y_end_rc[0]),
        xytext=(origin_rc[1], origin_rc[0]),
        arrowprops={"arrowstyle": "->", "color": "#1d4ed8", "linewidth": 2.4},
        zorder=40,
    )
    ax.text(
        origin_rc[1] + 4.0,
        origin_rc[0] - 4.0,
        "O",
        color="#d00000",
        fontsize=12,
        fontweight="bold",
        zorder=41,
    )
    ax.text(x_end_rc[1] + 3.0, x_end_rc[0], "+X", color="#d00000", fontsize=11, fontweight="bold", zorder=41)
    ax.text(y_end_rc[1] + 3.0, y_end_rc[0], "+Y", color="#1d4ed8", fontsize=11, fontweight="bold", zorder=41)


def make_robot_marker(
    center_xz: np.ndarray,
    center_y: float,
    radius: float,
    color: tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(float(radius), 0.02), resolution=16)
    marker.translate(np.array([center_xz[0], center_y, center_xz[1]], dtype=np.float64))
    marker.paint_uniform_color(color)
    marker.compute_vertex_normals()
    return marker


def update_robot_marker(marker: o3d.geometry.TriangleMesh, center_xz: np.ndarray, center_y: float):
    vertices = np.asarray(marker.vertices, dtype=np.float64)
    current_center = np.mean(vertices, axis=0)
    target_center = np.array([center_xz[0], center_y, center_xz[1]], dtype=np.float64)
    marker.translate(target_center - current_center)


def animate_robot_along_path(
    mesh_o3d: o3d.geometry.TriangleMesh,
    path_line: o3d.geometry.LineSet,
    path_world_xz: np.ndarray,
    robot_radius: float,
    robot_height: float,
    robot_base_height: float,
    method_label: str,
    animation_speed: float,
    compare_path_line: o3d.geometry.LineSet | None = None,
    destroy_window: bool = True,
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Scene RRT 3D: {method_label} | close window to return to picker",
        width=1700,
        height=980,
    )
    render = vis.get_render_option()
    render.mesh_show_back_face = True
    render.line_width = 16.0

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    robot_marker_radius = 0.1
    marker_y = robot_base_height + 0.5 * float(robot_height)
    start_robot = make_robot_marker(
        path_world_xz[0],
        marker_y,
        robot_marker_radius,
        (0.0, 0.68, 0.50),
    )
    goal_robot = make_robot_marker(
        path_world_xz[-1],
        marker_y,
        robot_marker_radius,
        (0.10, 0.25, 0.42),
    )
    moving_robot = make_robot_marker(
        path_world_xz[0],
        marker_y,
        robot_marker_radius,
        (0.18, 0.42, 0.95),
    )

    if isinstance(mesh_o3d, (list, tuple)):
        for geometry in mesh_o3d:
            vis.add_geometry(geometry)
    else:
        vis.add_geometry(mesh_o3d)
    for geometry in (path_line, axes, start_robot, goal_robot, moving_robot):
        vis.add_geometry(geometry)
    if compare_path_line is not None:
        vis.add_geometry(compare_path_line)

    view_control = vis.get_view_control()
    center_xz = np.mean(path_world_xz, axis=0)
    view_control.set_lookat(np.array([center_xz[0], robot_base_height + 0.3, center_xz[1]], dtype=np.float64))
    view_control.set_up(np.array([0.0, 0.0, -1.0], dtype=np.float64))
    view_control.set_front(np.array([0.0, 1.0, 0.0], dtype=np.float64))
    view_control.set_zoom(0.58)

    if path_world_xz.shape[0] >= 2:
        segment_vectors = np.diff(path_world_xz, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = float(cumulative[-1])
    else:
        segment_vectors = np.zeros((0, 2), dtype=np.float64)
        segment_lengths = np.zeros((0,), dtype=np.float64)
        cumulative = np.array([0.0], dtype=np.float64)
        total_length = 0.0

    def sample_position(distance_along: float) -> np.ndarray:
        if total_length <= 1e-9 or path_world_xz.shape[0] == 1:
            return path_world_xz[0]
        d = float(np.clip(distance_along, 0.0, total_length))
        seg_idx = int(np.searchsorted(cumulative, d, side="right") - 1)
        seg_idx = min(max(seg_idx, 0), len(segment_lengths) - 1)
        seg_len = float(segment_lengths[seg_idx])
        if seg_len <= 1e-9:
            return path_world_xz[seg_idx + 1]
        alpha = (d - cumulative[seg_idx]) / seg_len
        return (1.0 - alpha) * path_world_xz[seg_idx] + alpha * path_world_xz[seg_idx + 1]

    start_time = time.perf_counter()
    last_frame_time = start_time

    while vis.poll_events():
        now = time.perf_counter()
        elapsed = now - start_time
        frame_dt = now - last_frame_time
        last_frame_time = now

        if total_length <= 1e-9:
            current_xz = path_world_xz[0]
            reached_goal = True
        else:
            distance_along = min(elapsed * max(animation_speed, 1e-4), total_length)
            current_xz = sample_position(distance_along)
            reached_goal = distance_along >= total_length - 1e-9

        update_robot_marker(
            moving_robot,
            current_xz,
            marker_y,
        )
        vis.update_geometry(moving_robot)
        vis.update_renderer()
        time.sleep(max(0.0, 1.0 / 60.0 - frame_dt))

    if destroy_window:
        vis.destroy_window()


def _animate_robot_child(**kwargs):
    exit_code = 0
    try:
        animate_robot_along_path(**kwargs, destroy_window=False)
    except Exception as exc:
        exit_code = 1
        print(f"3D animation process failed: {exc}", file=sys.stderr)
    os._exit(exit_code)


def animate_robot_along_path_isolated(**kwargs):
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        animate_robot_along_path(**kwargs)
        return
    process = ctx.Process(target=_animate_robot_child, kwargs=kwargs)
    process.start()
    process.join()

class TopDownPicker:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
        self.fig.canvas.manager.set_window_title("Ground Picker")
        self._base_rgb = None
        self._grid_size = None
        self._base_title = None
        self._bounds = None
        self._planner_yaw_degrees = 0.0
        self._overlay_specs: list[dict[str, object]] = []
        self._navigation_bounds = None
        self._navigation_bounds_rc = None

    def pick_points(
        self,
        bounds,
        grid_size: int,
        navigation_bounds: SceneBounds | None,
        primary_display_mask: np.ndarray,
        compare_overlays: list[dict[str, object]] | None,
        click_is_valid_world_xz,
        method_label: str,
        obstacle_color: tuple[float, float, float],
        planner_yaw_degrees: float,
        validity_label: str | None = None,
    ) -> list[np.ndarray]:
        primary_mask = np.asarray(primary_display_mask, dtype=bool)
        overlay_specs = [
            {
                "label": method_label,
                "color": np.asarray(obstacle_color, dtype=np.float64),
                "occupied_mask": primary_mask,
            }
        ]
        for overlay in compare_overlays or []:
            overlay_specs.append(
                {
                    "label": str(overlay["label"]),
                    "color": np.asarray(overlay["color"], dtype=np.float64),
                    "occupied_mask": np.asarray(overlay["mask"], dtype=bool),
                }
            )

        base_rgb = np.ones((grid_size, grid_size, 3), dtype=np.float64)
        base_rgb[:] = np.array([0.95, 0.95, 0.95], dtype=np.float64)
        occupied_stack = np.stack([spec["occupied_mask"] for spec in overlay_specs], axis=0)
        occupied_counts = np.sum(occupied_stack, axis=0)
        for spec in overlay_specs:
            method_occ = spec["occupied_mask"]
            shared_occ = method_occ & (occupied_counts > 1)
            unique_occ = method_occ & (occupied_counts == 1)
            base_rgb[shared_occ] = 0.45 * base_rgb[shared_occ] + 0.55 * spec["color"]
            base_rgb[unique_occ] = 0.20 * base_rgb[unique_occ] + 0.80 * spec["color"]

        self._base_rgb = base_rgb
        self._grid_size = grid_size
        self._bounds = bounds
        self._planner_yaw_degrees = float(planner_yaw_degrees)
        self._navigation_bounds = navigation_bounds
        self._navigation_bounds_rc = None
        if navigation_bounds is not None:
            corners_xz = np.array(
                [
                    [navigation_bounds.min_x, navigation_bounds.min_z],
                    [navigation_bounds.max_x, navigation_bounds.max_z],
                ],
                dtype=np.float64,
            )
            self._navigation_bounds_rc = world_to_grid(corners_xz, bounds, grid_size)
        self._overlay_specs = overlay_specs
        self._base_title = (
            "Click start, Shift-click waypoints, then click goal\n"
            "Colored cells = occupied area per method | Black box = object-defined pick boundary"
        )
        self._render_picker()
        self.fig.show()
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

        for _ in range(5):
            picked = self._collect_click_points(bounds, grid_size, navigation_bounds, click_is_valid_world_xz)
            if len(picked) >= 2:
                return [np.array([point[0], 0.0, point[1]], dtype=np.float64) for point in picked]
            self._render_picker(
                "Click Start/Goal Points On Ground Map\n"
                f"Rejected selection: one or more positions are invalid under the {validity_label or method_label} collision check. Click new points."
            )
            self.fig.canvas.draw_idle()
            plt.pause(0.05)

        raise RuntimeError("Failed to collect valid ground points after 5 attempts.")

    def _collect_click_points(
        self,
        bounds,
        grid_size: int,
        navigation_bounds: SceneBounds | None,
        click_is_valid_world_xz,
    ) -> list[np.ndarray]:
        picked: list[np.ndarray] = []

        def event_to_rc(event) -> np.ndarray | None:
            if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
                return None
            return np.array([event.ydata, event.xdata], dtype=np.float64)

        def draw_point(rc: np.ndarray, color: str, label: str):
            self.ax.scatter(rc[1], rc[0], c=color, s=70, marker="o", zorder=20)
            self.ax.text(rc[1], rc[0], label, color=color, fontsize=9, fontweight="bold", zorder=22)

        def on_click(event):
            rc = event_to_rc(event)
            if rc is None:
                return
            world_xz = grid_to_world(np.asarray([rc], dtype=np.float64), bounds, grid_size)[0]
            if navigation_bounds is not None and not world_xz_in_bounds(world_xz, navigation_bounds):
                picked.clear()
                return
            if not bool(click_is_valid_world_xz(world_xz)):
                picked.clear()
                return
            is_waypoint = len(picked) > 0 and event.key == "shift"
            picked.append(world_xz)
            point_idx = len(picked) - 1
            if point_idx == 0:
                color = "#0b8f6a"
                label = "S"
            elif is_waypoint:
                color = "#7b2cbf"
                label = f"W{point_idx}"
            else:
                color = "#1d3557"
                label = "G"
            draw_point(rc, color, label)
            self.fig.canvas.draw_idle()

        cids = [self.fig.canvas.mpl_connect("button_press_event", on_click)]
        try:
            while plt.fignum_exists(self.fig.number):
                if len(picked) >= 2 and self.ax.texts and self.ax.texts[-1].get_text() == "G":
                    break
                plt.pause(0.05)
        finally:
            for cid in cids:
                self.fig.canvas.mpl_disconnect(cid)
        return picked

    def _render_picker(self, title: str | None = None):
        self.ax.clear()
        self.ax.imshow(self._base_rgb, origin="upper")
        self.ax.add_patch(
            Rectangle(
                (0.5, 0.5),
                self._grid_size - 1,
                self._grid_size - 1,
                fill=False,
                linewidth=2.5,
                edgecolor="#111111",
            )
        )
        if self._navigation_bounds_rc is not None:
            min_row = float(np.min(self._navigation_bounds_rc[:, 0]))
            max_row = float(np.max(self._navigation_bounds_rc[:, 0]))
            min_col = float(np.min(self._navigation_bounds_rc[:, 1]))
            max_col = float(np.max(self._navigation_bounds_rc[:, 1]))
            self.ax.add_patch(
                Rectangle(
                    (min_col, min_row),
                    max_col - min_col,
                    max_row - min_row,
                    fill=False,
                    linewidth=3.0,
                    linestyle="--",
                    edgecolor="#000000",
                    zorder=12,
                )
            )
        if self._bounds is not None:
            draw_origin_marker(self.ax, self._bounds, self._grid_size, self._planner_yaw_degrees)
        self.ax.set_title(title or self._base_title)
        self.ax.set_xlabel("X grid")
        self.ax.set_ylabel("Z grid")
        legend_handles = [
            Patch(
                facecolor=np.array([0.95, 0.95, 0.95], dtype=np.float64),
                edgecolor="none",
                label="Unoccupied background",
            )
        ]
        if self._navigation_bounds_rc is not None:
            legend_handles.append(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    fill=False,
                    linestyle="--",
                    linewidth=2.0,
                    edgecolor="#000000",
                    label="Object-defined pick boundary",
                )
            )
        for spec in self._overlay_specs:
            legend_handles.append(
                Patch(
                    facecolor=0.20 * np.array([0.95, 0.95, 0.95], dtype=np.float64) + 0.80 * spec["color"],
                    edgecolor="none",
                    label=f"{spec['label']} occupied area",
                )
            )
        legend = self.ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=13,
            framealpha=0.9,
            facecolor="white",
            edgecolor="#cccccc",
        )
        legend.set_zorder(10)
    def close(self):
        plt.close(self.fig)


def show_2d_plan_result(
    bounds,
    grid_size: int,
    primary_display_mask: np.ndarray,
    compare_overlays: list[dict[str, object]] | None,
    primary_path_grid: np.ndarray,
    compare_paths: list[dict[str, object]],
    method_label: str,
    obstacle_color: tuple[float, float, float],
    primary_path_color: tuple[float, float, float],
    navigation_bounds: SceneBounds | None,
    planner_yaw_degrees: float,
):
    primary_mask = np.asarray(primary_display_mask, dtype=bool)
    overlay_specs = [
        {
            "label": method_label,
            "color": np.asarray(obstacle_color, dtype=np.float64),
            "occupied_mask": primary_mask,
        }
    ]
    for overlay in compare_overlays or []:
        overlay_specs.append(
            {
                "label": str(overlay["label"]),
                "color": np.asarray(overlay["color"], dtype=np.float64),
                "occupied_mask": np.asarray(overlay["mask"], dtype=bool),
            }
        )

    base_rgb = np.ones((grid_size, grid_size, 3), dtype=np.float64)
    base_rgb[:] = np.array([0.95, 0.95, 0.95], dtype=np.float64)
    occupied_stack = np.stack([spec["occupied_mask"] for spec in overlay_specs], axis=0)
    occupied_counts = np.sum(occupied_stack, axis=0)
    for spec in overlay_specs:
        method_occ = spec["occupied_mask"]
        shared_occ = method_occ & (occupied_counts > 1)
        unique_occ = method_occ & (occupied_counts == 1)
        base_rgb[shared_occ] = 0.45 * base_rgb[shared_occ] + 0.55 * spec["color"]
        base_rgb[unique_occ] = 0.20 * base_rgb[unique_occ] + 0.80 * spec["color"]

    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    fig.canvas.manager.set_window_title("Planned Paths")
    ax.imshow(base_rgb, origin="upper")
    ax.add_patch(Rectangle((0.5, 0.5), grid_size - 1, grid_size - 1, fill=False, linewidth=2.5, edgecolor="#111111"))
    if navigation_bounds is not None:
        corners_xz = np.array(
            [[navigation_bounds.min_x, navigation_bounds.min_z], [navigation_bounds.max_x, navigation_bounds.max_z]],
            dtype=np.float64,
        )
        navigation_bounds_rc = world_to_grid(corners_xz, bounds, grid_size)
        min_row = float(np.min(navigation_bounds_rc[:, 0]))
        max_row = float(np.max(navigation_bounds_rc[:, 0]))
        min_col = float(np.min(navigation_bounds_rc[:, 1]))
        max_col = float(np.max(navigation_bounds_rc[:, 1]))
        ax.add_patch(
            Rectangle(
                (min_col, min_row),
                max_col - min_col,
                max_row - min_row,
                fill=False,
                linewidth=3.0,
                linestyle="--",
                edgecolor="#000000",
                zorder=12,
            )
        )

    draw_origin_marker(ax, bounds, grid_size, planner_yaw_degrees)
    ax.plot(
        primary_path_grid[:, 1],
        primary_path_grid[:, 0],
        color=primary_path_color,
        linewidth=3.0,
        label=f"{method_label} path",
        zorder=30,
    )
    ax.scatter(primary_path_grid[0, 1], primary_path_grid[0, 0], c="#0b8f6a", s=60, marker="o", zorder=31)
    ax.scatter(primary_path_grid[-1, 1], primary_path_grid[-1, 0], c="#1d3557", s=70, marker="x", zorder=31)
    for path_info in compare_paths:
        path_grid = np.asarray(path_info["path_grid"], dtype=np.float64)
        ax.plot(
            path_grid[:, 1],
            path_grid[:, 0],
            color=path_info["color"],
            linewidth=2.2,
            alpha=0.9,
            label=f"{path_info['label']} path",
            zorder=29,
        )

    ax.set_title("Planned Paths On Ground Map")
    ax.set_xlabel("X grid")
    ax.set_ylabel("Z grid")
    ax.legend(
        loc="upper left",
        fontsize=13,
        markerscale=1.6,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cccccc",
    )
    fig.show()
    fig.canvas.draw_idle()
    plt.pause(0.05)
    plt.show(block=True)


def method_visuals(method: str):
    if method == "mesh_obb":
        method = "oriented_bbox"
    if method == "mesh":
        color = (0.22, 0.50, 0.92)
        return "mesh", color, 0.36, color
    if method == "expanded_sq":
        color = (0.18, 0.72, 0.38)
        return "expanded_sq", color, 0.42, color
    if method == "oriented_bbox":
        color = (0.92, 0.55, 0.16)
        return "oriented_bbox", color, 0.30, color
    raise ValueError(f"Unsupported method: {method}")


def normalize_method_name(method: str | None) -> str | None:
    if method == "mesh_obb":
        return "oriented_bbox"
    return method


def format_storage_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def format_metric_value(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "nan"
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.{precision}f}"


def mean_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def undo_planner_yaw_xz(path_world_xz: np.ndarray, yaw_degrees: float) -> np.ndarray:
    path_world_xz = np.asarray(path_world_xz, dtype=np.float64)
    yaw = np.deg2rad(float(yaw_degrees))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.column_stack(
        [
            c * path_world_xz[:, 0] - s * path_world_xz[:, 1],
            s * path_world_xz[:, 0] + c * path_world_xz[:, 1],
        ]
    )


def planner_path_xz_to_output_xy(path_world_xz: np.ndarray, ground_plane: str, planner_yaw_degrees: float) -> np.ndarray:
    path_world_xz = np.asarray(path_world_xz, dtype=np.float64)
    path_world_xz = undo_planner_yaw_xz(path_world_xz, planner_yaw_degrees)
    if ground_plane == "xy":
        return np.column_stack([-path_world_xz[:, 1], -path_world_xz[:, 0]])
    if ground_plane == "xz":
        return np.column_stack([path_world_xz[:, 0], path_world_xz[:, 1]])
    raise ValueError(f"Unsupported ground plane: {ground_plane}")


def sparsify_path_by_min_step(path_xy: np.ndarray, min_step_m: float) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 2 or min_step_m <= 0.0:
        return path_xy
    kept = [path_xy[0]]
    last_kept = path_xy[0]
    for point in path_xy[1:-1]:
        if float(np.linalg.norm(point - last_kept)) >= min_step_m:
            kept.append(point)
            last_kept = point
    if float(np.linalg.norm(path_xy[-1] - last_kept)) < min_step_m and len(kept) > 1:
        kept[-1] = path_xy[-1]
    else:
        kept.append(path_xy[-1])
    return np.asarray(kept, dtype=np.float64)


def save_primary_path_csv(
    path: Path,
    path_world_xz: np.ndarray,
    ground_plane: str,
    planner_yaw_degrees: float,
    min_step_m: float,
) -> None:
    output_xy = planner_path_xz_to_output_xy(path_world_xz, ground_plane, planner_yaw_degrees)
    output_xy = sparsify_path_by_min_step(output_xy, min_step_m)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        output_xy,
        delimiter=",",
        header="x,y",
        comments="",
        fmt="%.8f",
    )


def find_default_camera_trajectory(objects_dir: Path) -> Path | None:
    candidates = [
        "camera_poses_world.txt",
        "camera_poses_world.npy",
        "camera_poses.txt",
        "camera_poses.npy",
        "camera_trajectory.txt",
        "camera_trajectory.npy",
        "trajectory.txt",
        "trajectory.npy",
        "traj.txt",
        "traj.npy",
        "poses.txt",
        "poses.npy",
    ]
    for name in candidates:
        path = objects_dir / name
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D scene planner with shared planner choices across scene representations."
    )
    parser.add_argument("--objects-dir-a", type=Path, required=True)
    parser.add_argument("--camera-trajectory", type=Path, default=None)
    parser.add_argument("--pose-source", choices=["sam3d", "fp"], default="fp")
    parser.add_argument("--ground-plane", choices=["xz", "xy"], default="xz", help="Scene ground plane before conversion to the planner frame.")
    parser.add_argument("--planner-yaw-deg", type=float, default=0.0, help="Rotate the planner/map frame about its up axis after ground-plane conversion.")
    method_choices = ["expanded_sq", "mesh", "oriented_bbox", "mesh_obb", "all"]
    parser.add_argument("--method", choices=method_choices, default="expanded_sq")
    parser.add_argument("--compare-method", choices=method_choices, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--planner", choices=["rrt", "gvd"], default="rrt")
    parser.add_argument("--expanded-sq-params-name", default="obj_sq_params_expanded.npy")
    parser.add_argument("--grid-size", type=int, default=320)
    parser.add_argument("--robot-radius", type=float, default=0.1)
    parser.add_argument("--robot-height", type=float, default=0.2)
    parser.add_argument(
        "--robot-bbox-size",
        type=float,
        nargs=3,
        default=(0.8, 0.4, 0.7),
        metavar=("X", "Y", "Z"),
        help="3D hollow robot bbox dimensions in planner-frame x/y/z. Y is vertical.",
    )
    parser.add_argument("--robot-base-height", type=float, default=0.0, help="Robot base height in planner frame. Defaults to 0.0 for xz scenes and an inferred navigable height slice for xy scenes.")
    parser.add_argument("--height-layers", type=int, default=5, help="Number of vertical samples used to build height-band occupancy maps.")
    parser.add_argument("--mesh-max-triangles", type=int, default=1200)
    parser.add_argument("--sq-resolution", type=int, default=24)
    parser.add_argument("--side-pad", type=float, default=None)
    parser.add_argument("--lateral-pad", type=float, default=None)
    parser.add_argument("--object-limit", type=int, default=None)
    parser.add_argument("--verbose-load", action="store_true")
    parser.add_argument("--rrt-step-size", type=float, default=10.0)
    parser.add_argument("--rrt-max-iters", type=int, default=7000)
    parser.add_argument("--rrt-goal-sample-rate", type=float, default=0.18)
    parser.add_argument("--rrt-connection-radius", type=float, default=18.0)
    parser.add_argument("--rrt-shortcut-iters", type=int, default=20)
    parser.add_argument("--path-smooth-iters", type=int, default=2, help="Collision-checked Chaikin smoothing iterations applied after planning.")
    parser.add_argument("--animation-speed", type=float, default=1.5, help="Robot playback speed in meters per second.")
    parser.add_argument("--show-3d", action="store_true", help="Open the 3D playback scene after showing the 2D planned path.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--benchmark-random-pairs", type=int, default=0, help="Run non-interactive benchmark with this many random start/goal pairs.")
    parser.add_argument("--benchmark-seeds", type=int, default=1, help="Planner seeds per random benchmark pair.")
    parser.add_argument("--benchmark-min-distance", type=float, default=1.0, help="Minimum straight-line distance in meters for benchmark pairs.")
    parser.add_argument("--start-xz", type=float, nargs=2, default=None, metavar=("X", "Z"))
    parser.add_argument("--goal-xz", type=float, nargs=2, default=None, metavar=("X", "Z"))
    parser.add_argument("--save-path", type=Path, default=None, help="Save the primary planned path as CSV columns x,y. For xy scenes, x is forward and y is left.")
    parser.add_argument("--save-path-min-step", type=float, default=0.5, help="Minimum adjacent waypoint spacing in meters for --save-path output.")
    args = parser.parse_args()

    if args.camera_trajectory is None:
        args.camera_trajectory = find_default_camera_trajectory(args.objects_dir_a)
        if args.verbose_load:
            if args.camera_trajectory is None:
                print(f"[load] no default camera trajectory found under {args.objects_dir_a}")
            else:
                print(f"[load] using camera trajectory: {args.camera_trajectory}")

    if (args.start_xz is None) != (args.goal_xz is None):
        raise ValueError("Provide both --start-xz and --goal-xz together.")
    robot_safety_margin = float(args.robot_radius)

    primary_planner = args.planner
    all_methods = ["expanded_sq", "mesh", "oriented_bbox"]
    requested_method = normalize_method_name(args.method)
    requested_compare_method = normalize_method_name(args.compare_method)
    if requested_method == "all":
        primary_method = "expanded_sq"
        compare_methods = [method for method in all_methods if method != primary_method]
    else:
        primary_method = requested_method
        compare_methods = []

    compare_method = requested_compare_method
    if compare_method == "all":
        compare_methods = [method for method in all_methods if method != primary_method]
    elif compare_method is not None:
        compare_methods = [compare_method]
    method_sequence = [primary_method, *compare_methods]
    needs_mesh_scene = any(method in {"mesh", "oriented_bbox"} for method in method_sequence)

    representation_timings_s: dict[str, float] = {}
    planner_from_scene = yaw_transform_about_planner_y(args.planner_yaw_deg) @ transform_from_ground_plane_to_planner(args.ground_plane)
    scene_transform_needed = not transform_is_identity(planner_from_scene)

    sq_representation_start = time.perf_counter()
    sq_params_world, trajectory_world = load_scene_world_sq_params(
        objects_dir=args.objects_dir_a,
        camera_trajectory_path=args.camera_trajectory,
        pose_source=args.pose_source,
        expanded_sq_params_name=args.expanded_sq_params_name,
        object_limit=args.object_limit,
        verbose_load=args.verbose_load and not needs_mesh_scene,
    )
    if scene_transform_needed and sq_params_world.size > 0:
        sq_params_world = transform_sq_params_to_world(sq_params_world, planner_from_scene)
    if scene_transform_needed and trajectory_world is not None:
        trajectory_world = transform_points(trajectory_world, planner_from_scene)
    sq_aabbs_world = compute_sq_world_aabbs(sq_params_world)
    representation_timings_s["expanded_sq"] = time.perf_counter() - sq_representation_start

    mesh_scene = None
    obb_parts: list[TriangleMeshData] = []
    if needs_mesh_scene:
        mesh_representation_start = time.perf_counter()
        mesh_scene, _, _, _, obb_parts, mesh_trajectory_world = load_scene_navigation_meshes_with_obb(
            objects_dir=args.objects_dir_a,
            camera_trajectory_path=args.camera_trajectory,
            pose_source=args.pose_source,
            mesh_max_triangles=args.mesh_max_triangles,
            sq_resolution=args.sq_resolution,
            expanded_sq_params_name=args.expanded_sq_params_name,
            object_limit=args.object_limit,
            verbose_load=args.verbose_load,
            build_obb=("oriented_bbox" in method_sequence),
            load_mesh=True,
        )
        if scene_transform_needed:
            mesh_scene = transform_meshdata(mesh_scene, planner_from_scene)
            obb_parts = [transform_meshdata(obb_part, planner_from_scene) for obb_part in obb_parts]
            if mesh_trajectory_world is not None:
                mesh_trajectory_world = transform_points(mesh_trajectory_world, planner_from_scene)
        if trajectory_world is None:
            trajectory_world = mesh_trajectory_world
        mesh_representation_time_s = time.perf_counter() - mesh_representation_start
        representation_timings_s["mesh"] = mesh_representation_time_s
        representation_timings_s["oriented_bbox"] = mesh_representation_time_s

    if mesh_scene is not None:
        side_pad, lateral_pad = choose_scene_padding(mesh_scene, args.side_pad, args.lateral_pad)
        bounds = compute_scene_bounds_from_mesh_and_trajectory(
            mesh=mesh_scene,
            side_pad=side_pad,
            lateral_pad=lateral_pad,
            trajectory_world=trajectory_world,
        )
    else:
        sq_bounds = object_xz_bounds_from_sq_aabbs(sq_aabbs_world)
        if sq_bounds is None:
            raise RuntimeError(f"No SQ params found under {args.objects_dir_a} using `{args.expanded_sq_params_name}`.")
        side_pad, lateral_pad = choose_bounds_padding(sq_bounds, args.side_pad, args.lateral_pad)
        bounds = pad_scene_bounds(sq_bounds, side_pad, lateral_pad, trajectory_world)

    bounds = square_scene_bounds(bounds)

    if args.robot_base_height is None:
        if args.ground_plane == "xy":
            args.robot_base_height = infer_robot_base_height(sq_aabbs_world, args.robot_height, mesh_scene)
            if args.verbose_load:
                print(f"[load] inferred robot_base_height={args.robot_base_height:.3f}")
        else:
            args.robot_base_height = 0.0

    build_timings_s: dict[str, float] = {}
    mesh_method_mask = None
    oriented_bbox_mask = None

    if mesh_scene is not None and "mesh" in method_sequence:
        mesh_build_start = time.perf_counter()
        mesh_mask_raw = rasterize_multi_layer_mesh_occupancy(
            mesh_scene,
            bounds,
            args.grid_size,
            args.robot_base_height,
            args.robot_height,
            args.height_layers,
        )
        mesh_method_mask = inflate_mask(mesh_mask_raw, robot_safety_margin, bounds)
        build_timings_s["mesh"] = time.perf_counter() - mesh_build_start

    if mesh_scene is not None and "oriented_bbox" in method_sequence:
        oriented_bbox_build_start = time.perf_counter()
        oriented_bbox_raw = np.zeros((args.grid_size, args.grid_size), dtype=bool)
        layer_heights = np.linspace(
            args.robot_base_height,
            args.robot_base_height + args.robot_height,
            max(2, int(args.height_layers)),
            dtype=np.float64,
        )
        thin_band_height = max(args.robot_height / max(int(args.height_layers) - 1, 1), 1e-3)
        for y_value in layer_heights:
            oriented_bbox_raw |= rasterize_oriented_bbox_list_height_band(
                obb_parts,
                bounds,
                args.grid_size,
                make_height_band(float(y_value), float(thin_band_height)),
            )
        oriented_bbox_mask = inflate_mask(oriented_bbox_raw, robot_safety_margin, bounds)
        build_timings_s["oriented_bbox"] = time.perf_counter() - oriented_bbox_build_start

    sq_collision_checker = None
    dense_sq_mask = None
    sq_display_mask = None
    if sq_params_world.shape[0] > 0:
        sq_build_start = time.perf_counter()
        sq_collision_checker = SQImplicitCollisionChecker(
            sq_params_world=sq_params_world,
            sq_aabbs_world=sq_aabbs_world,
            robot_base_height=args.robot_base_height,
            robot_height=args.robot_height,
            vertical_samples=args.height_layers,
        )
        if primary_method == "expanded_sq" or "expanded_sq" in compare_methods:
            dense_sq_mask_raw = build_dense_sq_occupancy_mask(bounds, args.grid_size, sq_collision_checker)
            dense_sq_mask = inflate_mask(dense_sq_mask_raw, robot_safety_margin, bounds)
        sq_display_mask_raw = build_sparse_sq_display_mask(bounds, args.grid_size, sq_collision_checker)
        sq_display_mask = inflate_mask(sq_display_mask_raw, robot_safety_margin, bounds)
        build_timings_s["expanded_sq"] = time.perf_counter() - sq_build_start

    clearance_source_mask = mesh_method_mask
    if clearance_source_mask is None:
        clearance_source_mask = dense_sq_mask if dense_sq_mask is not None else sq_display_mask
    if clearance_source_mask is None:
        clearance_source_mask = np.zeros((args.grid_size, args.grid_size), dtype=bool)
    mesh_clearance_m = clearance_map(np.asarray(clearance_source_mask, dtype=bool), bounds)

    def occupancy_collision_fn(occupancy_mask: np.ndarray):
        def _collision(points_world_xz: np.ndarray) -> np.ndarray:
            points_world_xz_arr = np.asarray(points_world_xz, dtype=np.float64)
            if points_world_xz_arr.ndim == 1:
                points_world_xz_arr = points_world_xz_arr.reshape(1, 2)
            points_rc = world_to_grid(points_world_xz_arr, bounds, args.grid_size)
            rows = np.clip(np.rint(points_rc[:, 0]).astype(np.int64), 0, args.grid_size - 1)
            cols = np.clip(np.rint(points_rc[:, 1]).astype(np.int64), 0, args.grid_size - 1)
            return np.asarray(occupancy_mask[rows, cols], dtype=bool)
        return _collision

    representation_configs = {}
    if mesh_method_mask is not None:
        representation_configs["mesh"] = {
            "display_mask": mesh_method_mask,
            "occupancy_mask": mesh_method_mask,
            "collision_fn": occupancy_collision_fn(mesh_method_mask),
            "sample_points_rc": np.argwhere(~mesh_method_mask).astype(np.float64),
            "storage_bytes": int(mesh_scene.vertices.nbytes + mesh_scene.faces.nbytes),
            "representation_build_time_s": float(representation_timings_s["mesh"]),
            "occupancy_map_build_time_s": float(build_timings_s["mesh"]),
        }
    if oriented_bbox_mask is not None:
        representation_configs["oriented_bbox"] = {
            "display_mask": oriented_bbox_mask,
            "occupancy_mask": oriented_bbox_mask,
            "collision_fn": occupancy_collision_fn(oriented_bbox_mask),
            "sample_points_rc": np.argwhere(~oriented_bbox_mask).astype(np.float64),
            "storage_bytes": int(len(obb_parts) * 15 * 8),
            "representation_build_time_s": float(representation_timings_s["oriented_bbox"]),
            "occupancy_map_build_time_s": float(build_timings_s["oriented_bbox"]),
        }
    if sq_collision_checker is not None:
        representation_configs["expanded_sq"] = {
            "display_mask": dense_sq_mask if dense_sq_mask is not None else sq_display_mask,
            "occupancy_mask": dense_sq_mask,
            "collision_fn": (
                occupancy_collision_fn(dense_sq_mask)
                if dense_sq_mask is not None
                else (lambda points_world_xz: sq_collision_checker.collides(points_world_xz))
            ),
            "sample_points_rc": None if sq_display_mask is None else np.argwhere(~sq_display_mask).astype(np.float64),
            "storage_bytes": int(sq_params_world.nbytes),
            "representation_build_time_s": float(representation_timings_s["expanded_sq"]),
            "occupancy_map_build_time_s": float(build_timings_s["expanded_sq"]),
        }

    if primary_method not in representation_configs:
        raise RuntimeError(f"No geometry available for method `{primary_method}`.")
    for method in compare_methods:
        if method not in representation_configs:
            raise RuntimeError(f"No geometry available for compare method `{method}`.")

    boundary_source_mask = oriented_bbox_mask
    if boundary_source_mask is None:
        boundary_source_mask = dense_sq_mask if dense_sq_mask is not None else sq_display_mask
    navigation_bounds = occupied_mask_xz_bounds(boundary_source_mask, bounds) if boundary_source_mask is not None else None
    if navigation_bounds is None:
        navigation_bounds = object_xz_bounds_from_mesh(mesh_scene) if mesh_scene is not None else object_xz_bounds_from_sq_aabbs(sq_aabbs_world)
    if navigation_bounds is None:
        raise RuntimeError("Could not infer a navigation boundary from the selected representation.")
    boundary_mask = outside_bounds_mask(bounds, navigation_bounds, args.grid_size)
    inside_boundary_mask = ~boundary_mask

    def free_space_ratio_inside_boundary(mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool)
        inside_count = int(np.count_nonzero(inside_boundary_mask))
        if inside_count == 0:
            return float("nan")
        free_inside = np.count_nonzero((~mask) & inside_boundary_mask)
        return float(free_inside / inside_count)

    for rep_cfg in representation_configs.values():
        rep_cfg["display_mask"] = np.asarray(rep_cfg["display_mask"], dtype=bool) | boundary_mask
        if rep_cfg["occupancy_mask"] is not None:
            rep_cfg["occupancy_mask"] = np.asarray(rep_cfg["occupancy_mask"], dtype=bool) | boundary_mask
            rep_cfg["collision_fn"] = occupancy_collision_fn(rep_cfg["occupancy_mask"])
        else:
            base_collision_fn = rep_cfg["collision_fn"]

            def bounded_collision_fn(points_world_xz, base_collision_fn=base_collision_fn):
                points_world_xz_arr = np.asarray(points_world_xz, dtype=np.float64)
                if points_world_xz_arr.ndim == 1:
                    points_world_xz_arr = points_world_xz_arr.reshape(1, 2)
                outside = np.array(
                    [not world_xz_in_bounds(point, navigation_bounds) for point in points_world_xz_arr],
                    dtype=bool,
                )
                return outside | np.asarray(base_collision_fn(points_world_xz_arr), dtype=bool)

            rep_cfg["collision_fn"] = bounded_collision_fn
        rep_cfg["sample_points_rc"] = np.argwhere(~np.asarray(rep_cfg["display_mask"], dtype=bool)).astype(np.float64)

    method_label, obstacle_color, _, path_color = method_visuals(primary_method)
    mesh_o3d = None
    if args.show_3d:
        # Keep the 3D execution view simple and consistent: always show the
        # expanded SQ object model when available, independent of planner method.
        mesh_o3d = make_scene_o3d_superquadrics(
            sq_params_world=sq_params_world,
            resolution=args.sq_resolution,
            color=method_visuals("expanded_sq")[1],
        )
        if mesh_o3d is None and mesh_scene is not None:
            mesh_o3d = load_scene_o3d_meshes_with_original_colors(
                objects_dir=args.objects_dir_a,
                camera_trajectory_path=args.camera_trajectory,
                pose_source=args.pose_source,
                object_limit=args.object_limit,
                planner_from_scene=planner_from_scene,
            )
            if not mesh_o3d:
                mesh_o3d = meshdata_to_o3d(mesh_scene)
                mesh_o3d.compute_vertex_normals()
    compare_infos = []
    for method in compare_methods:
        compare_label, compare_obstacle_color, _, compare_path_color = method_visuals(method)
        compare_infos.append(
            {
                "method": method,
                "label": compare_label,
                "obstacle_color": compare_obstacle_color,
                "path_color": compare_path_color,
            }
        )
    plan_config = RRTConfig(
        step_size_cells=args.rrt_step_size,
        max_iterations=args.rrt_max_iters,
        goal_sample_rate=args.rrt_goal_sample_rate,
        connection_radius_cells=args.rrt_connection_radius,
        shortcut_iterations=args.rrt_shortcut_iters,
        random_seed=args.seed,
    )

    def plan_for_query(
        rep_method: str,
        planner_name: str,
        query_start_rc: np.ndarray,
        query_goal_rc: np.ndarray,
        rng_seed: int,
    ):
        rep_cfg = representation_configs[rep_method]
        if planner_name == "rrt":
            result = plan_rrt_with_collision(
                start_rc=query_start_rc,
                goal_rc=query_goal_rc,
                bounds=bounds,
                grid_size=args.grid_size,
                mesh_clearance_m=mesh_clearance_m,
                config=plan_config,
                rng=np.random.default_rng(rng_seed),
                collision_fn=rep_cfg["collision_fn"],
                sample_points_rc=rep_cfg["sample_points_rc"],
            )
        else:
            occupancy_mask = rep_cfg["occupancy_mask"]
            if occupancy_mask is None:
                raise RuntimeError(f"Planner `gvd` requires a dense occupancy map for method `{rep_method}`.")
            result = plan_gvd(
                occupancy_mask=occupancy_mask,
                start_rc=query_start_rc,
                goal_rc=query_goal_rc,
                bounds=bounds,
                grid_size=args.grid_size,
                mesh_clearance_m=mesh_clearance_m,
            )
        if result.path_found and result.path_grid is not None and args.path_smooth_iters > 0:
            smoothed_grid = smooth_path_chaikin_world(
                np.asarray(result.path_grid, dtype=np.float64),
                args.path_smooth_iters,
                bounds,
                grid_size=args.grid_size,
                collision_fn=rep_cfg["collision_fn"],
            )
            result = make_plan_result_from_path(
                result.method,
                smoothed_grid,
                bounds,
                args.grid_size,
                mesh_clearance_m,
            )
        occupancy_mask = rep_cfg["occupancy_mask"]
        free_space_ratio = free_space_ratio_inside_boundary(
            occupancy_mask if occupancy_mask is not None else rep_cfg["display_mask"]
        )
        return {
            "result": result,
            "total_build_time_s": float(rep_cfg["representation_build_time_s"] + rep_cfg["occupancy_map_build_time_s"]),
            "free_space_ratio": free_space_ratio,
            "storage_bytes": int(rep_cfg["storage_bytes"]),
            "success_rate": 1.0 if result.path_found else 0.0,
        }

    if args.benchmark_random_pairs > 0:
        benchmark_seeds = max(1, int(args.benchmark_seeds))
        common_free_mask = np.ones((args.grid_size, args.grid_size), dtype=bool)
        for method in method_sequence:
            rep_cfg = representation_configs[method]
            mask = rep_cfg["occupancy_mask"] if rep_cfg["occupancy_mask"] is not None else rep_cfg["display_mask"]
            common_free_mask &= ~np.asarray(mask, dtype=bool)
        free_rc = np.argwhere(common_free_mask).astype(np.float64)
        if free_rc.shape[0] < 2:
            raise RuntimeError("Benchmark could not find at least two common-free cells.")

        rng = np.random.default_rng(args.seed)
        query_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        max_attempts = max(1000, int(args.benchmark_random_pairs) * 200)
        attempts = 0
        while len(query_pairs) < int(args.benchmark_random_pairs) and attempts < max_attempts:
            attempts += 1
            start_rc = free_rc[int(rng.integers(0, free_rc.shape[0]))]
            goal_rc = free_rc[int(rng.integers(0, free_rc.shape[0]))]
            if np.allclose(start_rc, goal_rc):
                continue
            start_xz, goal_xz = grid_to_world(np.vstack([start_rc, goal_rc]), bounds, args.grid_size)
            if float(np.linalg.norm(goal_xz - start_xz)) < float(args.benchmark_min_distance):
                continue
            query_pairs.append((start_rc, goal_rc))
        if len(query_pairs) < int(args.benchmark_random_pairs):
            raise RuntimeError(
                f"Only sampled {len(query_pairs)} benchmark pairs after {attempts} attempts. "
                "Lower --benchmark-random-pairs or --benchmark-min-distance."
            )

        summaries = {
            method: {
                "efficiency_ratio": [],
                "success_rate": [],
                "total_build_time_s": [],
                "map_storage": [],
                "free_space_ratio": [],
            }
            for method in method_sequence
        }
        for pair_idx, (start_rc, goal_rc) in enumerate(query_pairs):
            for seed_idx in range(benchmark_seeds):
                for method_idx, method in enumerate(method_sequence):
                    plan_entry = plan_for_query(
                        method,
                        args.planner,
                        start_rc,
                        goal_rc,
                        args.seed + pair_idx * 10000 + seed_idx * 100 + method_idx,
                    )
                    result = plan_entry["result"]
                    summaries[method]["success_rate"].append(plan_entry["success_rate"])
                    summaries[method]["total_build_time_s"].append(plan_entry["total_build_time_s"])
                    summaries[method]["map_storage"].append(float(plan_entry["storage_bytes"]))
                    summaries[method]["free_space_ratio"].append(plan_entry["free_space_ratio"])
                    if result.path_found and result.efficiency_ratio is not None:
                        summaries[method]["efficiency_ratio"].append(float(result.efficiency_ratio))

        print("------------------------------------------------------------------------------")
        print(
            f"Benchmark: pairs={len(query_pairs)}, seeds_per_pair={benchmark_seeds}, "
            f"planner={args.planner}, common_free_sampling=True"
        )
        for method in method_sequence:
            summary = summaries[method]
            print("------------------------------------------------------------------------------")
            print(
                f"{method}+{args.planner}: \n"
                f"mean_efficiency_ratio={format_metric_value(mean_or_nan(summary['efficiency_ratio']))}, \n"
                f"success_rate={format_metric_value(mean_or_nan(summary['success_rate']))}, \n"
                f"mean_total_build_time_s={format_metric_value(mean_or_nan(summary['total_build_time_s']))}, \n"
                f"map_storage={format_storage_bytes(int(round(mean_or_nan(summary['map_storage']))))}, \n"
                f"free_space_ratio={format_metric_value(mean_or_nan(summary['free_space_ratio']))}"
            )
        return

    fixed_query = args.start_xz is not None
    iteration = 0
    while True:
        if fixed_query:
            start_world = np.array([args.start_xz[0], args.robot_base_height, args.start_xz[1]], dtype=np.float64)
            goal_world = np.array([args.goal_xz[0], args.robot_base_height, args.goal_xz[1]], dtype=np.float64)
            waypoint_worlds = [start_world, goal_world]
        else:
            print("Opening top-down ground picker with the confined planning boundary and pickable floor area.")
            primary_collision_fn = representation_configs[primary_method]["collision_fn"]
            def picker_primary_valid(world_xz: np.ndarray) -> bool:
                return (
                    world_xz_in_bounds(world_xz, navigation_bounds)
                    and not bool(primary_collision_fn(np.asarray([world_xz], dtype=np.float64))[0])
                )
            compare_overlays = [
                {
                    "mask": representation_configs[info["method"]]["display_mask"],
                    "label": info["label"],
                    "color": info["obstacle_color"],
                }
                for info in compare_infos
            ]
            picker = TopDownPicker()
            try:
                waypoint_worlds = picker.pick_points(
                    bounds=bounds,
                    grid_size=args.grid_size,
                    navigation_bounds=navigation_bounds,
                    primary_display_mask=representation_configs[primary_method]["display_mask"],
                    compare_overlays=compare_overlays,
                    click_is_valid_world_xz=picker_primary_valid,
                    method_label=method_label,
                    obstacle_color=obstacle_color,
                    planner_yaw_degrees=args.planner_yaw_deg,
                    validity_label=primary_method,
                )
            finally:
                picker.close()
                plt.pause(0.05)
            for waypoint_world in waypoint_worlds:
                waypoint_world[1] = args.robot_base_height
            start_world = waypoint_worlds[0]
            goal_world = waypoint_worlds[-1]

        waypoint_xz = np.asarray([[point[0], point[2]] for point in waypoint_worlds], dtype=np.float64)
        waypoint_rcs = world_to_grid(waypoint_xz, bounds, args.grid_size)
        primary_collision_fn = representation_configs[primary_method]["collision_fn"]
        for waypoint_idx, waypoint in enumerate(waypoint_xz):
            label = "start" if waypoint_idx == 0 else ("goal" if waypoint_idx == len(waypoint_xz) - 1 else f"waypoint {waypoint_idx}")
            if not world_xz_in_bounds(waypoint, navigation_bounds):
                raise RuntimeError(f"Selected {label} lies outside the object-defined navigation boundary.")
            if bool(primary_collision_fn(np.asarray([waypoint], dtype=np.float64))[0]):
                raise RuntimeError(f"Selected {label} collides with the `{primary_method}` occupancy for the robot safety radius.")

        def plan_for(rep_method: str, planner_name: str, seed_offset: int) -> PlanResult:
            rep_cfg = representation_configs[rep_method]

            def make_plan_entry(result: PlanResult):
                occupancy_mask = rep_cfg["occupancy_mask"]
                if occupancy_mask is None:
                    free_space_ratio = free_space_ratio_inside_boundary(rep_cfg["display_mask"])
                else:
                    free_space_ratio = free_space_ratio_inside_boundary(occupancy_mask)
                return {
                    "result": result,
                    "representation_build_time_s": float(rep_cfg["representation_build_time_s"]),
                    "occupancy_map_build_time_s": float(rep_cfg["occupancy_map_build_time_s"]),
                    "total_build_time_s": float(
                        rep_cfg["representation_build_time_s"] + rep_cfg["occupancy_map_build_time_s"]
                    ),
                    "free_space_ratio": free_space_ratio,
                    "storage_bytes": int(rep_cfg["storage_bytes"]),
                    "success_rate": 1.0 if result.path_found else 0.0,
                }

            segment_paths = []
            for segment_idx, (segment_start_rc, segment_goal_rc) in enumerate(zip(waypoint_rcs[:-1], waypoint_rcs[1:])):
                if planner_name == "rrt":
                    segment_result = plan_rrt_with_collision(
                        start_rc=segment_start_rc,
                        goal_rc=segment_goal_rc,
                        bounds=bounds,
                        grid_size=args.grid_size,
                        mesh_clearance_m=mesh_clearance_m,
                        config=plan_config,
                        rng=np.random.default_rng(args.seed + seed_offset + iteration + 10000 * segment_idx),
                        collision_fn=rep_cfg["collision_fn"],
                        sample_points_rc=rep_cfg["sample_points_rc"],
                    )
                else:
                    occupancy_mask = rep_cfg["occupancy_mask"]
                    if occupancy_mask is None:
                        raise RuntimeError(f"Planner `gvd` requires a dense occupancy map for method `{rep_method}`.")
                    segment_result = plan_gvd(
                        occupancy_mask=occupancy_mask,
                        start_rc=segment_start_rc,
                        goal_rc=segment_goal_rc,
                        bounds=bounds,
                        grid_size=args.grid_size,
                        mesh_clearance_m=mesh_clearance_m,
                    )
                if not segment_result.path_found or segment_result.path_grid is None:
                    return make_plan_entry(segment_result)
                segment_path = np.asarray(segment_result.path_grid, dtype=np.float64)
                if segment_idx > 0:
                    segment_path = segment_path[1:]
                segment_paths.append(segment_path)
            path_grid = np.vstack(segment_paths)
            result = make_plan_result_from_path(planner_name, path_grid, bounds, args.grid_size, mesh_clearance_m)
            if result.path_found and result.path_grid is not None and args.path_smooth_iters > 0:
                smoothed_grid = smooth_path_chaikin_world(
                    np.asarray(result.path_grid, dtype=np.float64),
                    args.path_smooth_iters,
                    bounds,
                    grid_size=args.grid_size,
                    collision_fn=rep_cfg["collision_fn"],
                )
                result = make_plan_result_from_path(
                    result.method,
                    smoothed_grid,
                    bounds,
                    args.grid_size,
                    mesh_clearance_m,
                )
            return make_plan_entry(result)

        primary_plan = plan_for(primary_method, primary_planner, 0)
        result: PlanResult = primary_plan["result"]
        if not result.path_found:
            raise RuntimeError(f"Planning failed for method `{primary_method}` with planner `{primary_planner}`: {result.reason}")

        compare_results = []
        for compare_idx, method in enumerate(compare_methods):
            compare_planner = args.planner
            compare_results.append(
                {
                    "method": method,
                    "planner": compare_planner,
                    **plan_for(method, compare_planner, 1000 + 100 * compare_idx),
                }
            )
        print("------------------------------------------------------------------------------")
        print(
            f"{method_label}+{primary_planner}: \n"
            f"representation_build_time_s={primary_plan['representation_build_time_s']:.3f}, \n"
            f"occupancy_2d_build_time_s={primary_plan['occupancy_map_build_time_s']:.3f}, \n"
            f"total_build_time_s={primary_plan['total_build_time_s']:.3f}, \n"
            f"map_storage={format_storage_bytes(primary_plan['storage_bytes'])}, \n"
            f"free_space_ratio={primary_plan['free_space_ratio']:.3f}, \n"
            f"path_length_m={format_metric_value(result.path_length_m if result.path_found else None)}, \n"
            f"success_rate={primary_plan['success_rate']:.3f}"
        )
        for compare_entry in compare_results:
            compare_result = compare_entry["result"]
            print("------------------------------------------------------------------------------")
            print(
                f"{compare_entry['method']}+{compare_entry['planner']}: \n"
                f"representation_build_time_s={compare_entry['representation_build_time_s']:.3f}, \n"
                f"occupancy_2d_build_time_s={compare_entry['occupancy_map_build_time_s']:.3f}, \n"
                f"total_build_time_s={compare_entry['total_build_time_s']:.3f}, \n"
                f"map_storage={format_storage_bytes(compare_entry['storage_bytes'])}, \n"
                f"free_space_ratio={compare_entry['free_space_ratio']:.3f}, \n"
                f"path_length_m={format_metric_value(compare_result.path_length_m if compare_result.path_found else None)}, \n"
                f"success_rate={compare_entry['success_rate']:.3f}"
                + ("" if compare_result.path_found else f", reason={compare_result.reason}")
            )

        path_world_xz = np.asarray(result.path_world, dtype=np.float64)
        if args.save_path is not None:
            save_primary_path_csv(
                args.save_path,
                path_world_xz,
                args.ground_plane,
                args.planner_yaw_deg,
                args.save_path_min_step,
            )
            print(f"Saved primary path to {args.save_path}")
        result_compare_paths = []
        for compare_entry, compare_info in zip(compare_results, compare_infos):
            compare_result = compare_entry["result"]
            if compare_result.path_found and compare_result.path_grid is not None:
                result_compare_paths.append(
                    {
                        "path_grid": compare_result.path_grid,
                        "label": compare_info["label"],
                        "color": compare_info["path_color"],
                    }
                )
        compare_overlays = [
            {
                "mask": representation_configs[info["method"]]["display_mask"],
                "label": info["label"],
                "color": info["obstacle_color"],
            }
            for info in compare_infos
        ]
        show_2d_plan_result(
            bounds=bounds,
            grid_size=args.grid_size,
            primary_display_mask=representation_configs[primary_method]["display_mask"],
            compare_overlays=compare_overlays,
            primary_path_grid=np.asarray(result.path_grid, dtype=np.float64),
            compare_paths=result_compare_paths,
            method_label=method_label,
            obstacle_color=obstacle_color,
            primary_path_color=path_color,
            navigation_bounds=navigation_bounds,
            planner_yaw_degrees=args.planner_yaw_deg,
        )
        if not args.show_3d:
            if fixed_query:
                break
            time.sleep(0.15)
            iteration += 1
            continue

        compare_path_lines = []
        path_center_y = args.robot_base_height + 0.5 * args.robot_height
        for compare_entry, compare_info in zip(compare_results, compare_infos):
            compare_result = compare_entry["result"]
            if not compare_result.path_found:
                continue
            compare_path_world_xz = np.asarray(compare_result.path_world, dtype=np.float64)
            compare_path_lines.append(
                make_path_lineset(compare_path_world_xz, path_center_y + 0.01, compare_info["path_color"])
            )
        compare_path_line = merge_linesets(compare_path_lines)
        path_line = make_path_lineset(path_world_xz, args.robot_base_height + 0.5 * args.robot_height, path_color)
        print("------------------------------------------------------------------------------")
        print("Opening 3D scene for this plan.")
        animate_robot_along_path_isolated(
            mesh_o3d=mesh_o3d,
            path_line=path_line,
            path_world_xz=path_world_xz,
            robot_radius=args.robot_radius,
            robot_height=args.robot_height,
            robot_base_height=args.robot_base_height,
            method_label=f"{args.objects_dir_a.name} [{method_label}]",
            animation_speed=args.animation_speed,
            compare_path_line=compare_path_line,
        )
        if fixed_query:
            break
        time.sleep(0.15)
        iteration += 1


if __name__ == "__main__":
    main()
