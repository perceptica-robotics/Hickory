from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
import open3d as o3d
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R

from hickory.navigation.geometry import TriangleMeshData, grid_to_world, world_to_grid


SQ_EPS = 1e-9


@dataclass(frozen=True)
class RRTConfig:
    step_size_cells: float
    max_iterations: int
    goal_sample_rate: float
    connection_radius_cells: float
    shortcut_iterations: int
    random_seed: int


@dataclass
class PlanResult:
    method: str
    path_found: bool
    reason: str | None = None
    path_grid: np.ndarray | None = None
    path_world: np.ndarray | None = None
    path_yaw: np.ndarray | None = None
    path_world_pose: np.ndarray | None = None
    path_length_m: float | None = None
    straight_line_m: float | None = None
    efficiency_ratio: float | None = None
    min_clearance_to_mesh_m: float | None = None
    mean_clearance_to_mesh_m: float | None = None


def transform_sq_params_to_world(params_local: np.ndarray, object_transform: np.ndarray) -> np.ndarray:
    params_local = np.asarray(params_local, dtype=np.float64)
    if params_local.ndim == 1:
        params_local = params_local.reshape(1, -1)
    if params_local.ndim != 2 or params_local.shape[1] < 11:
        raise ValueError(f"Expected SQ params with shape (N, 11+), got {params_local.shape}")

    object_rot = np.asarray(object_transform[:3, :3], dtype=np.float64)
    object_trans = np.asarray(object_transform[:3, 3], dtype=np.float64)
    params_world = params_local.copy()
    for i, params in enumerate(params_local):
        local_rot = R.from_euler("ZYX", params[5:8]).as_matrix()
        local_trans = np.asarray(params[8:11], dtype=np.float64)
        world_rot = object_rot @ local_rot
        world_trans = object_rot @ local_trans + object_trans
        params_world[i, 5:8] = R.from_matrix(world_rot).as_euler("ZYX")
        params_world[i, 8:11] = world_trans
    return params_world


def world_to_local_sq(points_world: np.ndarray, params: np.ndarray) -> np.ndarray:
    rot = R.from_euler("ZYX", params[5:8]).as_matrix()
    trans = np.asarray(params[8:11], dtype=np.float64)
    return (rot.T @ (points_world - trans).T).T


def sq_iof_local(local_points: np.ndarray, params: np.ndarray) -> np.ndarray:
    e1 = max(float(params[0]), SQ_EPS)
    e2 = max(float(params[1]), SQ_EPS)
    ax = max(float(params[2]), SQ_EPS)
    ay = max(float(params[3]), SQ_EPS)
    az = max(float(params[4]), SQ_EPS)

    x = np.abs(local_points[:, 0]) / ax
    y = np.abs(local_points[:, 1]) / ay
    z = np.abs(local_points[:, 2]) / az
    with np.errstate(over="ignore", invalid="ignore"):
        xy_term = (x ** (2.0 / e2) + y ** (2.0 / e2)) ** (e2 / e1)
        z_term = z ** (2.0 / e1)
        values = xy_term + z_term
    return np.nan_to_num(values, nan=np.inf, posinf=np.inf)


def compute_sq_world_aabbs(params_world: np.ndarray) -> np.ndarray:
    params_world = np.asarray(params_world, dtype=np.float64)
    if params_world.ndim == 1:
        params_world = params_world.reshape(1, -1)
    if params_world.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float64)
    bounds = np.empty((params_world.shape[0], 2, 3), dtype=np.float64)
    for i, params in enumerate(params_world):
        rot = R.from_euler("ZYX", params[5:8]).as_matrix()
        half_extents = np.abs(rot) @ np.asarray(params[2:5], dtype=np.float64)
        center = np.asarray(params[8:11], dtype=np.float64)
        bounds[i, 0] = center - half_extents
        bounds[i, 1] = center + half_extents
    return bounds


class SQImplicitCollisionChecker:
    def __init__(
        self,
        sq_params_world: np.ndarray,
        sq_aabbs_world: np.ndarray,
        robot_base_height: float,
        robot_height: float,
        vertical_samples: int = 5,
    ):
        self.sq_params_world = np.asarray(sq_params_world, dtype=np.float64)
        self.sq_aabbs_world = np.asarray(sq_aabbs_world, dtype=np.float64)
        self.vertical_offsets = np.linspace(
            float(robot_base_height),
            float(robot_base_height + robot_height),
            max(2, int(vertical_samples)),
            dtype=np.float64,
        )

    def collides(self, points_world_xz: np.ndarray) -> np.ndarray:
        points_world_xz = np.asarray(points_world_xz, dtype=np.float64)
        if points_world_xz.ndim == 1:
            points_world_xz = points_world_xz.reshape(1, 2)
        if points_world_xz.size == 0 or self.sq_params_world.size == 0:
            return np.zeros((points_world_xz.shape[0],), dtype=bool)

        num_points = points_world_xz.shape[0]
        num_heights = self.vertical_offsets.shape[0]
        points_world = np.column_stack(
            [
                np.repeat(points_world_xz[:, 0], num_heights),
                np.tile(self.vertical_offsets, num_points),
                np.repeat(points_world_xz[:, 1], num_heights),
            ]
        )
        parent_indices = np.repeat(np.arange(num_points, dtype=np.int64), num_heights)
        occupied = np.zeros((num_points,), dtype=bool)

        inside_aabb = np.all(
            (self.sq_aabbs_world[None, :, 0, :] <= points_world[:, None, :])
            & (points_world[:, None, :] <= self.sq_aabbs_world[None, :, 1, :]),
            axis=2,
        )
        active_sq_indices = np.flatnonzero(np.any(inside_aabb, axis=0))
        for sq_idx in active_sq_indices:
            sample_indices = np.flatnonzero(inside_aabb[:, sq_idx])
            if sample_indices.size == 0:
                continue
            params = self.sq_params_world[sq_idx]
            local_points = world_to_local_sq(points_world[sample_indices], params)
            collided_samples = sq_iof_local(local_points, params) <= 1.0
            if np.any(collided_samples):
                occupied[parent_indices[sample_indices[collided_samples]]] = True
        return occupied


def build_sparse_sq_display_mask(bounds, grid_size: int, collision_checker, coarse_resolution: int = 56) -> np.ndarray:
    coarse_resolution = int(np.clip(coarse_resolution, 8, grid_size))
    row_coords = np.linspace(0.0, float(grid_size - 1), coarse_resolution, dtype=np.float64)
    col_coords = np.linspace(0.0, float(grid_size - 1), coarse_resolution, dtype=np.float64)
    rr, cc = np.meshgrid(row_coords, col_coords, indexing="ij")
    coarse_rc = np.column_stack([rr.reshape(-1), cc.reshape(-1)])
    coarse_world_xz = grid_to_world(coarse_rc, bounds, grid_size)
    coarse_mask = collision_checker.collides(coarse_world_xz).reshape(coarse_resolution, coarse_resolution)

    full_rows = np.clip(np.rint(np.linspace(0, coarse_resolution - 1, grid_size)).astype(np.int64), 0, coarse_resolution - 1)
    full_cols = np.clip(np.rint(np.linspace(0, coarse_resolution - 1, grid_size)).astype(np.int64), 0, coarse_resolution - 1)
    return coarse_mask[np.ix_(full_rows, full_cols)]


def build_dense_sq_occupancy_mask(bounds, grid_size: int, collision_checker) -> np.ndarray:
    occupancy = np.zeros((grid_size, grid_size), dtype=bool)
    if collision_checker.sq_params_world.size == 0:
        return occupancy

    for sq_idx, params in enumerate(collision_checker.sq_params_world):
        sq_bounds = collision_checker.sq_aabbs_world[sq_idx]
        active_heights = collision_checker.vertical_offsets[
            (collision_checker.vertical_offsets >= sq_bounds[0, 1])
            & (collision_checker.vertical_offsets <= sq_bounds[1, 1])
        ]
        if active_heights.size == 0:
            continue

        corners_xz = np.array(
            [
                [sq_bounds[0, 0], sq_bounds[0, 2]],
                [sq_bounds[1, 0], sq_bounds[1, 2]],
            ],
            dtype=np.float64,
        )
        corners_rc = world_to_grid(corners_xz, bounds, grid_size)
        min_row = max(0, int(np.floor(np.min(corners_rc[:, 0]))))
        max_row = min(grid_size - 1, int(np.ceil(np.max(corners_rc[:, 0]))))
        min_col = max(0, int(np.floor(np.min(corners_rc[:, 1]))))
        max_col = min(grid_size - 1, int(np.ceil(np.max(corners_rc[:, 1]))))
        if min_row > max_row or min_col > max_col:
            continue

        rr, cc = np.meshgrid(
            np.arange(min_row, max_row + 1, dtype=np.float64),
            np.arange(min_col, max_col + 1, dtype=np.float64),
            indexing="ij",
        )
        query_rc = np.column_stack([rr.reshape(-1), cc.reshape(-1)])
        query_xz = grid_to_world(query_rc, bounds, grid_size)
        num_cells = query_xz.shape[0]
        points_world = np.column_stack(
            [
                np.repeat(query_xz[:, 0], active_heights.size),
                np.tile(active_heights, num_cells),
                np.repeat(query_xz[:, 1], active_heights.size),
            ]
        )
        local_points = world_to_local_sq(points_world, params)
        occupied = np.any(sq_iof_local(local_points, params).reshape(num_cells, active_heights.size) <= 1.0, axis=1)
        occupancy[min_row:max_row + 1, min_col:max_col + 1] |= occupied.reshape(rr.shape)
    return occupancy


def rasterize_oriented_bbox_height_band(mesh: TriangleMeshData, bounds, resolution: int, band) -> np.ndarray:
    occupancy = np.zeros((resolution, resolution), dtype=bool)
    if mesh.vertices.size == 0:
        return occupancy
    mesh_bounds = mesh.bounds
    if not bool(mesh_bounds[0, 1] <= band.y_max and mesh_bounds[1, 1] >= band.y_min):
        return occupancy

    projected_xz = np.asarray(mesh.vertices[:, [0, 2]], dtype=np.float64)
    projected_xz = np.unique(np.round(projected_xz, decimals=8), axis=0)
    if projected_xz.shape[0] < 3:
        return occupancy

    centroid = np.mean(projected_xz, axis=0)
    angles = np.arctan2(projected_xz[:, 1] - centroid[1], projected_xz[:, 0] - centroid[0])
    polygon_xz = projected_xz[np.argsort(angles)]
    polygon_rc = world_to_grid(polygon_xz, bounds, resolution)

    min_row = max(0, int(np.floor(np.min(polygon_rc[:, 0]))))
    max_row = min(resolution - 1, int(np.ceil(np.max(polygon_rc[:, 0]))))
    min_col = max(0, int(np.floor(np.min(polygon_rc[:, 1]))))
    max_col = min(resolution - 1, int(np.ceil(np.max(polygon_rc[:, 1]))))
    if min_row > max_row or min_col > max_col:
        return occupancy

    rr, cc = np.meshgrid(
        np.arange(min_row, max_row + 1, dtype=np.float64),
        np.arange(min_col, max_col + 1, dtype=np.float64),
        indexing="ij",
    )
    query_rc = np.column_stack([rr.reshape(-1), cc.reshape(-1)])
    polygon_path = MplPath(polygon_rc, closed=True)
    inside = polygon_path.contains_points(query_rc, radius=0.5)
    occupancy[min_row:max_row + 1, min_col:max_col + 1] = inside.reshape(rr.shape)
    return occupancy


def rasterize_oriented_bbox_list_height_band(meshes: list[TriangleMeshData], bounds, resolution: int, band) -> np.ndarray:
    occupancy = np.zeros((resolution, resolution), dtype=bool)
    for mesh in meshes:
        occupancy |= rasterize_oriented_bbox_height_band(mesh, bounds, resolution, band)
    return occupancy


def make_path_lineset(path_world_xz: np.ndarray, y_value: float, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    points = np.column_stack(
        [
            path_world_xz[:, 0],
            np.full((path_world_xz.shape[0],), float(y_value), dtype=np.float64),
            path_world_xz[:, 1],
        ]
    )
    lines = np.column_stack(
        [
            np.arange(0, path_world_xz.shape[0] - 1, dtype=np.int32),
            np.arange(1, path_world_xz.shape[0], dtype=np.int32),
        ]
    )
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64).reshape(1, 3), (lines.shape[0], 1))
    )
    return line_set


def merge_linesets(linesets: list[o3d.geometry.LineSet]) -> o3d.geometry.LineSet | None:
    valid_linesets = [line for line in linesets if line is not None]
    if not valid_linesets:
        return None
    out = o3d.geometry.LineSet()
    points_parts = []
    lines_parts = []
    colors_parts = []
    point_offset = 0
    for line in valid_linesets:
        points = np.asarray(line.points, dtype=np.float64)
        lines = np.asarray(line.lines, dtype=np.int32)
        colors = np.asarray(line.colors, dtype=np.float64)
        if points.size == 0 or lines.size == 0:
            continue
        points_parts.append(points)
        lines_parts.append(lines + point_offset)
        colors_parts.append(colors)
        point_offset += points.shape[0]
    if not points_parts:
        return None
    out.points = o3d.utility.Vector3dVector(np.vstack(points_parts))
    out.lines = o3d.utility.Vector2iVector(np.vstack(lines_parts))
    out.colors = o3d.utility.Vector3dVector(np.vstack(colors_parts))
    return out


def points_are_free_world(points_rc: np.ndarray, bounds, grid_size: int, collision_fn, collision_cache: dict[tuple[int, int], bool] | None = None) -> np.ndarray:
    points_rc = np.asarray(points_rc, dtype=np.float64)
    rounded_rc = np.clip(np.rint(points_rc).astype(np.int64), 0, grid_size - 1)
    free_mask = np.zeros((rounded_rc.shape[0],), dtype=bool)
    uncached_indices = []
    uncached_keys = []
    for idx, (row, col) in enumerate(rounded_rc):
        key = (int(row), int(col))
        if collision_cache is not None and key in collision_cache:
            free_mask[idx] = not collision_cache[key]
        else:
            uncached_indices.append(idx)
            uncached_keys.append(key)
    if uncached_indices:
        uncached_points_world_xz = grid_to_world(points_rc[np.asarray(uncached_indices, dtype=np.int64)], bounds, grid_size)
        collided = np.asarray(collision_fn(uncached_points_world_xz), dtype=bool)
        for local_idx, collided_value in enumerate(collided):
            global_idx = uncached_indices[local_idx]
            free_mask[global_idx] = not collided_value
            if collision_cache is not None:
                collision_cache[uncached_keys[local_idx]] = bool(collided_value)
    return free_mask


def is_free_world(point_rc: np.ndarray, bounds, grid_size: int, collision_fn, collision_cache: dict[tuple[int, int], bool] | None = None) -> bool:
    return bool(points_are_free_world(np.asarray([point_rc], dtype=np.float64), bounds, grid_size, collision_fn, collision_cache)[0])


def segment_is_free_world(start_rc: np.ndarray, end_rc: np.ndarray, bounds, grid_size: int, collision_fn, collision_cache: dict[tuple[int, int], bool] | None = None) -> bool:
    delta = np.asarray(end_rc, dtype=np.float64) - np.asarray(start_rc, dtype=np.float64)
    samples = max(2, int(np.ceil(np.linalg.norm(delta) * 2.5)))
    t_values = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    points_rc = (1.0 - t_values[:, None]) * np.asarray(start_rc, dtype=np.float64) + t_values[:, None] * np.asarray(end_rc, dtype=np.float64)
    return bool(np.all(points_are_free_world(points_rc, bounds, grid_size, collision_fn, collision_cache)))


def shortcut_path_world(path_rc: np.ndarray, rng: np.random.Generator, iterations: int, bounds, grid_size: int, collision_fn, collision_cache: dict[tuple[int, int], bool] | None = None) -> np.ndarray:
    if path_rc.shape[0] < 3 or iterations <= 0:
        return path_rc
    path = [point.copy() for point in path_rc]
    for _ in range(iterations):
        if len(path) < 3:
            break
        i, j = sorted(rng.integers(0, len(path), size=2).tolist())
        if j - i < 2:
            continue
        if segment_is_free_world(path[i], path[j], bounds, grid_size, collision_fn, collision_cache):
            path = path[: i + 1] + path[j:]
    return np.asarray(path, dtype=np.float64)


def path_is_free_world(path_rc: np.ndarray, bounds, grid_size: int, collision_fn, collision_cache: dict[tuple[int, int], bool] | None = None) -> bool:
    path_rc = np.asarray(path_rc, dtype=np.float64)
    if path_rc.shape[0] == 0:
        return False
    if not bool(np.all(points_are_free_world(path_rc, bounds, grid_size, collision_fn, collision_cache))):
        return False
    for idx in range(path_rc.shape[0] - 1):
        if not segment_is_free_world(path_rc[idx], path_rc[idx + 1], bounds, grid_size, collision_fn, collision_cache):
            return False
    return True


def smooth_path_chaikin_world(
    path_rc: np.ndarray,
    iterations: int,
    bounds,
    grid_size: int,
    collision_fn,
    collision_cache: dict[tuple[int, int], bool] | None = None,
) -> np.ndarray:
    path = np.asarray(path_rc, dtype=np.float64)
    if path.shape[0] < 3 or iterations <= 0:
        return path
    for _ in range(int(iterations)):
        smoothed = [path[0]]
        for idx in range(path.shape[0] - 1):
            p0 = path[idx]
            p1 = path[idx + 1]
            smoothed.append(0.75 * p0 + 0.25 * p1)
            smoothed.append(0.25 * p0 + 0.75 * p1)
        smoothed.append(path[-1])
        candidate = np.asarray(smoothed, dtype=np.float64)
        if path_is_free_world(candidate, bounds, grid_size, collision_fn, collision_cache):
            path = candidate
        else:
            break
    return path


def make_plan_result_from_path(method: str, path_grid: np.ndarray, bounds, grid_size: int, mesh_clearance_m: np.ndarray) -> PlanResult:
    path_grid = np.asarray(path_grid, dtype=np.float64)
    path_world = grid_to_world(path_grid, bounds, grid_size)
    length_m = float(np.sum(np.linalg.norm(np.diff(path_world, axis=0), axis=1))) if path_world.shape[0] >= 2 else 0.0
    straight_line_m = float(np.linalg.norm(path_world[-1] - path_world[0])) if path_world.shape[0] >= 2 else 0.0
    path_indices = np.clip(np.rint(path_grid).astype(np.int64), 0, grid_size - 1)
    clearances = mesh_clearance_m[path_indices[:, 0], path_indices[:, 1]]
    return PlanResult(
        method=method,
        path_found=True,
        path_grid=path_grid,
        path_world=path_world,
        path_length_m=length_m,
        straight_line_m=straight_line_m,
        efficiency_ratio=(length_m / straight_line_m) if straight_line_m > 0.0 else 1.0,
        min_clearance_to_mesh_m=float(np.min(clearances)),
        mean_clearance_to_mesh_m=float(np.mean(clearances)),
    )


def grid_neighbors_8(node_rc: tuple[int, int], shape: tuple[int, int]):
    row, col = node_rc
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row == 0 and d_col == 0:
                continue
            next_row = row + d_row
            next_col = col + d_col
            if 0 <= next_row < shape[0] and 0 <= next_col < shape[1]:
                yield next_row, next_col, float(np.hypot(d_row, d_col))


def astar_grid_path(allowed_mask: np.ndarray, start_rc: np.ndarray, goal_rc: np.ndarray, cost_scale: np.ndarray | None = None) -> np.ndarray | None:
    allowed_mask = np.asarray(allowed_mask, dtype=bool)
    start = tuple(np.clip(np.rint(start_rc).astype(np.int64), 0, np.array(allowed_mask.shape) - 1).tolist())
    goal = tuple(np.clip(np.rint(goal_rc).astype(np.int64), 0, np.array(allowed_mask.shape) - 1).tolist())
    if not allowed_mask[start] or not allowed_mask[goal]:
        return None

    frontier = [(float(np.hypot(goal[0] - start[0], goal[1] - start[1])), 0.0, start)]
    came_from = {start: None}
    g_score = {start: 0.0}

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == goal:
            break
        if current_cost > g_score[current] + 1e-9:
            continue
        for next_row, next_col, step_distance in grid_neighbors_8(current, allowed_mask.shape):
            if not allowed_mask[next_row, next_col]:
                continue
            scale = 1.0 if cost_scale is None else float(cost_scale[next_row, next_col])
            trial_cost = current_cost + step_distance * scale
            next_node = (next_row, next_col)
            if trial_cost + 1e-9 < g_score.get(next_node, np.inf):
                g_score[next_node] = trial_cost
                came_from[next_node] = current
                heuristic = float(np.hypot(goal[0] - next_row, goal[1] - next_col))
                heapq.heappush(frontier, (trial_cost + heuristic, trial_cost, next_node))

    if goal not in came_from:
        return None
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return np.asarray(path, dtype=np.float64)


def dijkstra_to_mask(free_mask: np.ndarray, start_rc: np.ndarray, target_mask: np.ndarray) -> np.ndarray | None:
    free_mask = np.asarray(free_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool) & free_mask
    start = tuple(np.clip(np.rint(start_rc).astype(np.int64), 0, np.array(free_mask.shape) - 1).tolist())
    if not free_mask[start]:
        return None
    if target_mask[start]:
        return np.asarray([start], dtype=np.float64)

    frontier = [(0.0, start)]
    came_from = {start: None}
    dist = {start: 0.0}

    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current_cost > dist[current] + 1e-9:
            continue
        if target_mask[current]:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return np.asarray(path, dtype=np.float64)
        for next_row, next_col, step_distance in grid_neighbors_8(current, free_mask.shape):
            if not free_mask[next_row, next_col]:
                continue
            next_node = (next_row, next_col)
            trial_cost = current_cost + step_distance
            if trial_cost + 1e-9 < dist.get(next_node, np.inf):
                dist[next_node] = trial_cost
                came_from[next_node] = current
                heapq.heappush(frontier, (trial_cost, next_node))
    return None


def compute_gvd_mask(occupancy_mask: np.ndarray) -> np.ndarray:
    occupancy_mask = np.asarray(occupancy_mask, dtype=bool)
    free_mask = ~occupancy_mask
    if not np.any(free_mask):
        return np.zeros_like(occupancy_mask, dtype=bool)
    _, nearest_indices = distance_transform_edt(free_mask, return_indices=True)
    nearest_labels = nearest_indices[0] * occupancy_mask.shape[1] + nearest_indices[1]
    gvd_mask = np.zeros_like(occupancy_mask, dtype=bool)
    for d_row, d_col in ((1, 0), (0, 1), (1, 1), (1, -1)):
        src_row_start = max(0, -d_row)
        src_row_end = occupancy_mask.shape[0] - max(0, d_row)
        src_col_start = max(0, -d_col)
        src_col_end = occupancy_mask.shape[1] - max(0, d_col)
        dst_row_start = src_row_start + d_row
        dst_row_end = src_row_end + d_row
        dst_col_start = src_col_start + d_col
        dst_col_end = src_col_end + d_col
        src_free = free_mask[src_row_start:src_row_end, src_col_start:src_col_end]
        dst_free = free_mask[dst_row_start:dst_row_end, dst_col_start:dst_col_end]
        differing = src_free & dst_free & (
            nearest_labels[src_row_start:src_row_end, src_col_start:src_col_end]
            != nearest_labels[dst_row_start:dst_row_end, dst_col_start:dst_col_end]
        )
        gvd_mask[src_row_start:src_row_end, src_col_start:src_col_end] |= differing
        gvd_mask[dst_row_start:dst_row_end, dst_col_start:dst_col_end] |= differing
    return gvd_mask & free_mask


def plan_gvd(occupancy_mask: np.ndarray, start_rc: np.ndarray, goal_rc: np.ndarray, bounds, grid_size: int, mesh_clearance_m: np.ndarray) -> PlanResult:
    occupancy_mask = np.asarray(occupancy_mask, dtype=bool)
    free_mask = ~occupancy_mask
    start_idx = tuple(np.clip(np.rint(start_rc).astype(np.int64), 0, grid_size - 1).tolist())
    goal_idx = tuple(np.clip(np.rint(goal_rc).astype(np.int64), 0, grid_size - 1).tolist())
    if not free_mask[start_idx]:
        return PlanResult(method="gvd", path_found=False, reason="start lies inside obstacle")
    if not free_mask[goal_idx]:
        return PlanResult(method="gvd", path_found=False, reason="goal lies inside obstacle")

    gvd_mask = compute_gvd_mask(occupancy_mask)
    if not np.any(gvd_mask):
        return PlanResult(method="gvd", path_found=False, reason="GVD skeleton is empty")

    start_connector = dijkstra_to_mask(free_mask, start_rc, gvd_mask)
    goal_connector = dijkstra_to_mask(free_mask, goal_rc, gvd_mask)
    if start_connector is None or goal_connector is None:
        return PlanResult(method="gvd", path_found=False, reason="could not connect start/goal to GVD")

    skeleton_start = start_connector[-1]
    skeleton_goal = goal_connector[-1]
    clearance_cost = 1.0 / np.maximum(mesh_clearance_m, 1e-3)
    skeleton_path = astar_grid_path(gvd_mask, skeleton_start, skeleton_goal, cost_scale=clearance_cost)
    if skeleton_path is None:
        return PlanResult(method="gvd", path_found=False, reason="no path on GVD skeleton")

    merged_path = [start_connector]
    merged_path.append(skeleton_path[1:] if start_connector.shape[0] > 0 and skeleton_path.shape[0] > 0 and np.allclose(start_connector[-1], skeleton_path[0]) else skeleton_path)
    goal_connector_rev = goal_connector[::-1]
    merged_path.append(goal_connector_rev[1:] if skeleton_path.shape[0] > 0 and goal_connector_rev.shape[0] > 0 and np.allclose(skeleton_path[-1], goal_connector_rev[0]) else goal_connector_rev)

    path_grid = np.vstack([segment for segment in merged_path if segment.size > 0])
    dedup = [path_grid[0]]
    for point in path_grid[1:]:
        if not np.allclose(point, dedup[-1]):
            dedup.append(point)
    return make_plan_result_from_path("gvd", np.asarray(dedup, dtype=np.float64), bounds, grid_size, mesh_clearance_m)


def plan_rrt_with_collision(
    start_rc: np.ndarray,
    goal_rc: np.ndarray,
    bounds,
    grid_size: int,
    mesh_clearance_m: np.ndarray,
    config: RRTConfig,
    rng: np.random.Generator,
    collision_fn,
    sample_points_rc: np.ndarray | None = None,
) -> PlanResult:
    collision_cache: dict[tuple[int, int], bool] = {}
    if not is_free_world(start_rc, bounds, grid_size, collision_fn, collision_cache):
        return PlanResult(method="rrt", path_found=False, reason="start lies inside obstacle")
    if not is_free_world(goal_rc, bounds, grid_size, collision_fn, collision_cache):
        return PlanResult(method="rrt", path_found=False, reason="goal lies inside obstacle")

    nodes = [np.asarray(start_rc, dtype=np.float64)]
    parents = [-1]
    goal_index = None

    if sample_points_rc is not None:
        sample_points_rc = np.asarray(sample_points_rc, dtype=np.float64)
        if sample_points_rc.ndim != 2 or sample_points_rc.shape[1] != 2 or sample_points_rc.shape[0] == 0:
            sample_points_rc = None

    for _ in range(config.max_iterations):
        if rng.random() < config.goal_sample_rate:
            sample = np.asarray(goal_rc, dtype=np.float64)
        elif sample_points_rc is not None:
            sample = sample_points_rc[int(rng.integers(0, sample_points_rc.shape[0]))].astype(np.float64)
        else:
            sample = rng.uniform(0.0, float(grid_size - 1), size=2).astype(np.float64)
        nearest_idx = int(np.argmin([np.linalg.norm(node - sample) for node in nodes]))
        nearest = nodes[nearest_idx]
        direction = sample - nearest
        norm = float(np.linalg.norm(direction))
        candidate = sample if norm <= config.step_size_cells else nearest + direction * (config.step_size_cells / max(norm, 1e-9))

        if not is_free_world(candidate, bounds, grid_size, collision_fn, collision_cache):
            continue
        if not segment_is_free_world(nearest, candidate, bounds, grid_size, collision_fn, collision_cache):
            continue

        nodes.append(candidate)
        parents.append(nearest_idx)
        new_idx = len(nodes) - 1

        goal_distance = float(np.linalg.norm(candidate - goal_rc))
        if goal_distance <= config.step_size_cells and segment_is_free_world(candidate, goal_rc, bounds, grid_size, collision_fn, collision_cache):
            nodes.append(np.asarray(goal_rc, dtype=np.float64))
            parents.append(new_idx)
            goal_index = len(nodes) - 1
            break

    if goal_index is None:
        closest_to_goal = int(np.argmin([np.linalg.norm(node - goal_rc) for node in nodes]))
        if segment_is_free_world(nodes[closest_to_goal], goal_rc, bounds, grid_size, collision_fn, collision_cache):
            nodes.append(np.asarray(goal_rc, dtype=np.float64))
            parents.append(closest_to_goal)
            goal_index = len(nodes) - 1
        else:
            return PlanResult(method="rrt", path_found=False, reason="RRT did not connect to goal")

    path_nodes = []
    index = goal_index
    while index >= 0:
        path_nodes.append(nodes[index])
        index = parents[index]
    path_nodes.reverse()
    path_grid = shortcut_path_world(
        np.asarray(path_nodes, dtype=np.float64),
        rng,
        config.shortcut_iterations,
        bounds,
        grid_size,
        collision_fn,
        collision_cache,
    )
    return make_plan_result_from_path("rrt", path_grid, bounds, grid_size, mesh_clearance_m)
