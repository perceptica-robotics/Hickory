from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt
from scipy.spatial.transform import Rotation as R


EPS = 1e-9


@dataclass(frozen=True)
class SceneBounds:
    min_x: float
    max_x: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_z - self.min_z


@dataclass(frozen=True)
class TriangleMeshData:
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def bounds(self) -> np.ndarray:
        min_corner = np.min(self.vertices, axis=0)
        max_corner = np.max(self.vertices, axis=0)
        return np.stack([min_corner, max_corner], axis=0)

    @property
    def extents(self) -> np.ndarray:
        bounds = self.bounds
        return bounds[1] - bounds[0]


@dataclass(frozen=True)
class HeightBand:
    y_min: float
    y_max: float


def meshdata_to_o3d(mesh: TriangleMeshData) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    out.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    out.compute_vertex_normals()
    return out


def load_mesh_data(mesh_path, max_triangles: int | None = None) -> TriangleMeshData:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Failed to load mesh from {mesh_path}")
    if max_triangles is not None and max_triangles > 0 and len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(int(max_triangles))
    return TriangleMeshData(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.triangles, dtype=np.int64),
    )


def _signed_pow(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(values) * (np.abs(values) ** exponent)


def superquadrics_to_meshdata(params: np.ndarray, resolution: int = 40) -> TriangleMeshData:
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] < 11:
        raise ValueError(f"Expected (N, 11+) SQ params, got {params.shape}")

    vertices = []
    faces = []
    vertex_offset = 0
    lat_segments = max(4, int(resolution))
    lon_segments = max(12, int(resolution) * 2)
    latitudes = np.linspace(-np.pi / 2.0, np.pi / 2.0, lat_segments + 1)
    longitudes = np.linspace(-np.pi, np.pi, lon_segments, endpoint=False)
    cos_v = np.cos(longitudes)
    sin_v = np.sin(longitudes)

    for quadric in params:
        eps1 = max(float(quadric[0]), EPS)
        eps2 = max(float(quadric[1]), EPS)
        ax, ay, az = np.maximum(quadric[2:5].astype(np.float64), EPS)
        rot = R.from_euler("ZYX", quadric[5:8]).as_matrix()
        trans = quadric[8:11].astype(np.float64)

        interior_u = latitudes[1:-1]
        cos_u = _signed_pow(np.cos(interior_u), eps1)
        sin_u = _signed_pow(np.sin(interior_u), eps1)
        cos_v_eps = _signed_pow(cos_v, eps2)
        sin_v_eps = _signed_pow(sin_v, eps2)

        x = ax * cos_u[:, None] * cos_v_eps[None, :]
        y = ay * cos_u[:, None] * sin_v_eps[None, :]
        z = az * np.repeat(sin_u[:, None], lon_segments, axis=1)
        ring_vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        local_vertices = np.vstack(
            [
                np.array([[0.0, 0.0, -az]], dtype=np.float64),
                ring_vertices,
                np.array([[0.0, 0.0, az]], dtype=np.float64),
            ]
        )
        world_vertices = (rot @ local_vertices.T).T + trans
        vertices.append(world_vertices)

        ring_count = lat_segments - 1
        bottom_idx = vertex_offset
        top_idx = vertex_offset + local_vertices.shape[0] - 1
        first_ring_start = vertex_offset + 1

        for lon_idx in range(lon_segments):
            next_lon = (lon_idx + 1) % lon_segments
            faces.append([bottom_idx, first_ring_start + next_lon, first_ring_start + lon_idx])

        for ring_idx in range(max(0, ring_count - 1)):
            ring_start = first_ring_start + ring_idx * lon_segments
            next_ring_start = ring_start + lon_segments
            for lon_idx in range(lon_segments):
                next_lon = (lon_idx + 1) % lon_segments
                a = ring_start + lon_idx
                b = ring_start + next_lon
                c = next_ring_start + lon_idx
                d = next_ring_start + next_lon
                faces.append([a, b, c])
                faces.append([b, d, c])

        if ring_count > 0:
            last_ring_start = first_ring_start + (ring_count - 1) * lon_segments
            for lon_idx in range(lon_segments):
                next_lon = (lon_idx + 1) % lon_segments
                faces.append([top_idx, last_ring_start + lon_idx, last_ring_start + next_lon])

        vertex_offset += local_vertices.shape[0]

    return TriangleMeshData(vertices=np.vstack(vertices), faces=np.asarray(faces, dtype=np.int64))


def mesh_bbox_prism(mesh: TriangleMeshData) -> TriangleMeshData:
    bounds = mesh.bounds
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    return bbox_prism_from_bounds(min_x, min_y, min_z, max_x, max_y, max_z)


def bbox_prism_from_bounds(
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float,
) -> TriangleMeshData:
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
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return TriangleMeshData(vertices=vertices, faces=faces)


def expanded_bbox_prism(mesh: TriangleMeshData, margin_xz: float, margin_y: float = 0.0) -> TriangleMeshData:
    bounds = mesh.bounds
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    margin_xz = max(0.0, float(margin_xz))
    margin_y = max(0.0, float(margin_y))
    return bbox_prism_from_bounds(
        min_x=min_x - margin_xz,
        min_y=min_y - margin_y,
        min_z=min_z - margin_xz,
        max_x=max_x + margin_xz,
        max_y=max_y + margin_y,
        max_z=max_z + margin_xz,
    )


def compute_scene_bounds(mesh: TriangleMeshData, side_pad: float, lateral_pad: float) -> SceneBounds:
    bounds = mesh.bounds
    min_corner = bounds[0]
    max_corner = bounds[1]
    return SceneBounds(
        min_x=float(min_corner[0] - side_pad),
        max_x=float(max_corner[0] + side_pad),
        min_z=float(min_corner[2] - lateral_pad),
        max_z=float(max_corner[2] + lateral_pad),
    )


def world_to_grid(points_xz: np.ndarray, bounds: SceneBounds, resolution: int) -> np.ndarray:
    points_xz = np.asarray(points_xz, dtype=np.float64)
    scale_x = (resolution - 1) / max(bounds.width, EPS)
    scale_z = (resolution - 1) / max(bounds.height, EPS)
    col = (points_xz[:, 0] - bounds.min_x) * scale_x
    row = (points_xz[:, 1] - bounds.min_z) * scale_z
    return np.stack([row, col], axis=1)


def grid_to_world(indices_rc: np.ndarray, bounds: SceneBounds, resolution: int) -> np.ndarray:
    indices_rc = np.asarray(indices_rc, dtype=np.float64)
    x = bounds.min_x + indices_rc[:, 1] * bounds.width / max(resolution - 1, 1)
    z = bounds.min_z + indices_rc[:, 0] * bounds.height / max(resolution - 1, 1)
    return np.stack([x, z], axis=1)


def make_height_band(base_height: float, robot_height: float) -> HeightBand:
    if robot_height <= 0.0:
        raise ValueError("robot_height must be > 0.")
    return HeightBand(y_min=float(base_height), y_max=float(base_height + robot_height))


def mesh_intersects_height_band(mesh: TriangleMeshData, band: HeightBand) -> bool:
    bounds = mesh.bounds
    return bool(bounds[0, 1] <= band.y_max and bounds[1, 1] >= band.y_min)


def face_band_overlap_mask(mesh: TriangleMeshData, band: HeightBand) -> np.ndarray:
    triangles_y = mesh.vertices[mesh.faces][:, :, 1]
    tri_y_min = np.min(triangles_y, axis=1)
    tri_y_max = np.max(triangles_y, axis=1)
    return (tri_y_min <= band.y_max) & (tri_y_max >= band.y_min)


def rasterize_mesh_height_band(
    mesh: TriangleMeshData,
    bounds: SceneBounds,
    resolution: int,
    band: HeightBand,
) -> np.ndarray:
    occupancy = np.zeros((resolution, resolution), dtype=bool)
    band_faces = face_band_overlap_mask(mesh, band)
    if not np.any(band_faces):
        return occupancy
    band_face_indices = mesh.faces[band_faces]
    used_vertex_ids = np.unique(band_face_indices.reshape(-1))
    vertices_xz = mesh.vertices[used_vertex_ids][:, [0, 2]]
    centroids_xz = np.mean(mesh.vertices[band_face_indices][:, :, [0, 2]], axis=1)
    samples_xz = np.vstack([vertices_xz, centroids_xz])
    samples_rc = np.rint(world_to_grid(samples_xz, bounds, resolution)).astype(np.int64)
    valid = (
        (samples_rc[:, 0] >= 0)
        & (samples_rc[:, 0] < resolution)
        & (samples_rc[:, 1] >= 0)
        & (samples_rc[:, 1] < resolution)
    )
    occupancy[samples_rc[valid, 0], samples_rc[valid, 1]] = True
    occupancy = binary_dilation(occupancy, iterations=1)
    occupancy = binary_fill_holes(occupancy)
    return occupancy


def rasterize_bounds_height_band(
    mesh: TriangleMeshData,
    bounds: SceneBounds,
    resolution: int,
    band: HeightBand,
) -> np.ndarray:
    occupancy = np.zeros((resolution, resolution), dtype=bool)
    if not mesh_intersects_height_band(mesh, band):
        return occupancy
    mesh_bounds = mesh.bounds
    corners_xz = np.array(
        [
            [mesh_bounds[0, 0], mesh_bounds[0, 2]],
            [mesh_bounds[1, 0], mesh_bounds[1, 2]],
        ],
        dtype=np.float64,
    )
    corners_rc = world_to_grid(corners_xz, bounds, resolution)
    min_row = max(0, int(np.floor(np.min(corners_rc[:, 0]))))
    max_row = min(resolution - 1, int(np.ceil(np.max(corners_rc[:, 0]))))
    min_col = max(0, int(np.floor(np.min(corners_rc[:, 1]))))
    max_col = min(resolution - 1, int(np.ceil(np.max(corners_rc[:, 1]))))
    occupancy[min_row:max_row + 1, min_col:max_col + 1] = True
    return occupancy


def inflate_mask(mask: np.ndarray, inflation_m: float, bounds: SceneBounds) -> np.ndarray:
    if inflation_m <= 0.0:
        return mask.copy()
    meters_per_cell_x = bounds.width / max(mask.shape[1] - 1, 1)
    meters_per_cell_z = bounds.height / max(mask.shape[0] - 1, 1)
    radius_cells = int(np.ceil(inflation_m / max(min(meters_per_cell_x, meters_per_cell_z), EPS)))
    if radius_cells <= 0:
        return mask.copy()
    yy, xx = np.mgrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    kernel = (xx * xx + yy * yy) <= radius_cells * radius_cells
    return binary_dilation(mask, structure=kernel)


def find_edge_start_goal(occupancy: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    mid_row = occupancy.shape[0] // 2
    candidate_rows = sorted(range(occupancy.shape[0]), key=lambda r: abs(r - mid_row))

    def _pick(col: int) -> tuple[int, int]:
        for row in candidate_rows:
            if not occupancy[row, col]:
                return (row, col)
        raise RuntimeError("Could not find a free start/goal cell on the scene edge.")

    return _pick(1), _pick(occupancy.shape[1] - 2)


def astar_grid(occupancy: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    if occupancy[start] or occupancy[goal]:
        return None

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        dr = a[0] - b[0]
        dc = a[1] - b[1]
        return float(np.hypot(dr, dc))

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    ]
    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    visited = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dr, dc, step_cost in neighbors:
            nr = current[0] + dr
            nc = current[1] + dc
            if nr < 0 or nr >= occupancy.shape[0] or nc < 0 or nc >= occupancy.shape[1]:
                continue
            if occupancy[nr, nc]:
                continue
            nxt = (nr, nc)
            tentative = current_g + step_cost
            if tentative >= g_score.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            g_score[nxt] = tentative
            heapq.heappush(open_heap, (tentative + heuristic(nxt, goal), tentative, nxt))
    return None


def path_length_world(path_world: np.ndarray) -> float:
    if path_world.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path_world, axis=0), axis=1)))


def clearance_map(mask: np.ndarray, bounds: SceneBounds) -> np.ndarray:
    meters_per_cell_x = bounds.width / max(mask.shape[1] - 1, 1)
    meters_per_cell_z = bounds.height / max(mask.shape[0] - 1, 1)
    return distance_transform_edt(~mask) * min(meters_per_cell_x, meters_per_cell_z)
