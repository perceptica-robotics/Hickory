import argparse
from pathlib import Path

import numpy as np
from scipy.special import beta
from scipy.spatial.transform import Rotation as R
import pyvista as pv

from mps.superquadrics import superquadric
from mps.utils import superquadric_to_mesh


EPS = 1e-9


def read_mesh(mesh_path: Path) -> pv.PolyData:
    mesh = pv.read(str(mesh_path))
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    mesh = mesh.extract_surface().triangulate()
    if mesh.n_points == 0:
        raise ValueError(f"Failed to load mesh from {mesh_path}")
    return mesh

def evaluate_all_sq_radial_distances(points_world: np.ndarray, params_array: np.ndarray) -> np.ndarray:
    """
    Signed Radial Distance: ||x||_2 * (1 - f^(-e1/2)(x))
    Inside the SQ => f(x) < 1 => distance < 0
    Outside the SQ => f(x) > 1 => distance > 0
    """
    distances = np.empty((points_world.shape[0], params_array.shape[0]), dtype=np.float64)
    
    for i, params in enumerate(params_array):
        local_points = world_to_local(points_world, params)
        f_values = sq_iof_local(local_points, params)
        
        e1 = max(float(params[0]), EPS)
        norms = np.linalg.norm(local_points, axis=1)
        
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            f_term = f_values ** (-e1 / 2.0)
        
            dist = norms * (1.0 - f_term)
            
        dist[norms == 0] = -np.inf
        dist = np.nan_to_num(dist, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        
        distances[:, i] = dist
        
    return distances


def load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    mesh = read_mesh(mesh_path)
    vertices = np.asarray(mesh.points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Invalid mesh vertices in {mesh_path}")
    return vertices


def compute_sq_centroid_from_params(params_array: np.ndarray) -> np.ndarray:
    params_array = np.asarray(params_array, dtype=np.float64)
    if params_array.ndim == 1:
        params_array = params_array.reshape(1, -1)
    if params_array.ndim != 2 or params_array.shape[1] < 11:
        raise ValueError(f"Expected SQ params with shape (N, 11), got {params_array.shape}")

    e1 = np.maximum(params_array[:, 0], EPS)
    e2 = np.maximum(params_array[:, 1], EPS)
    ax = np.maximum(params_array[:, 2], EPS)
    ay = np.maximum(params_array[:, 3], EPS)
    az = np.maximum(params_array[:, 4], EPS)
    translations = params_array[:, 8:11]

    term_scale = 2.0 * ax * ay * az
    term_shape = e1 * e2
    term_beta = beta(e1 / 2.0 + 1.0, e1) * beta(e2 / 2.0, e2 / 2.0)
    volumes = np.nan_to_num(term_scale * term_shape * term_beta)
    total_volume = float(np.sum(volumes))
    if total_volume <= EPS:
        return np.mean(translations, axis=0)

    weighted_positions = volumes[:, np.newaxis] * translations
    return np.sum(weighted_positions, axis=0) / total_volume


def load_mesh_vertices_with_margin(
    mesh_path: Path,
    metric_margin: float,
    centroid_world: np.ndarray | None = None,
) -> np.ndarray:
    mesh = read_mesh(mesh_path)
    vertices = np.asarray(mesh.points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Invalid mesh vertices in {mesh_path}")

    if metric_margin <= 0.0:
        return vertices

    if "Normals" not in mesh.point_data:
        mesh = mesh.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,
            inplace=False,
        )
    normals = np.array(mesh.point_data["Normals"], dtype=np.float64, copy=True)
    if normals.shape != vertices.shape:
        raise ValueError(f"Invalid mesh vertex normals in {mesh_path}")

    # Offset the target surface itself instead of approximating the margin
    # from SQ-centric radial distances.
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_normals = normal_norms[:, 0] > EPS
    if not np.all(valid_normals):
        if centroid_world is None:
            centroid_world = vertices.mean(axis=0)
        centroid_world = np.asarray(centroid_world, dtype=np.float64).reshape(1, 3)
        fallback = vertices - centroid_world
        fallback_norms = np.linalg.norm(fallback, axis=1, keepdims=True)
        fallback_valid = fallback_norms[:, 0] > EPS
        normals[valid_normals] /= normal_norms[valid_normals]
        if np.any(fallback_valid & ~valid_normals):
            normals[fallback_valid & ~valid_normals] = (
                fallback[fallback_valid & ~valid_normals] / fallback_norms[fallback_valid & ~valid_normals]
            )
        if np.any(~valid_normals & ~fallback_valid):
            normals[~valid_normals & ~fallback_valid] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normals /= normal_norms

    return vertices + metric_margin * normals


def world_to_local(points_world: np.ndarray, params: np.ndarray) -> np.ndarray:
    rot = R.from_euler("ZYX", params[5:8]).as_matrix()
    trans = params[8:11]
    return (rot.T @ (points_world - trans).T).T


def sq_iof_local(local_points: np.ndarray, params: np.ndarray) -> np.ndarray:
    e1 = max(float(params[0]), EPS)
    e2 = max(float(params[1]), EPS)
    ax = max(float(params[2]), EPS)
    ay = max(float(params[3]), EPS)
    az = max(float(params[4]), EPS)

    x = np.abs(local_points[:, 0]) / ax
    y = np.abs(local_points[:, 1]) / ay
    z = np.abs(local_points[:, 2]) / az

    with np.errstate(over="ignore", invalid="ignore"):
        xy_term = (x ** (2.0 / e2) + y ** (2.0 / e2)) ** (e2 / e1)
        z_term = z ** (2.0 / e1)
        values = xy_term + z_term
    return np.nan_to_num(values, nan=np.inf, posinf=np.inf)


def sq_axis_scale_from_fmax(fmax: float, e1: float) -> float:
    if fmax <= 1.0:
        return 1.0
    return float(fmax) ** (max(float(e1), EPS) / 2.0)


def evaluate_all_sq_iof(points_world: np.ndarray, params_array: np.ndarray) -> np.ndarray:
    values = np.empty((points_world.shape[0], params_array.shape[0]), dtype=np.float64)
    for i, params in enumerate(params_array):
        local_points = world_to_local(points_world, params)
        values[:, i] = sq_iof_local(local_points, params)
    return values


def expand_sq_params_for_mesh(
    points_world: np.ndarray,
    params_array: np.ndarray,
    metric_margin: float = 0.0,
    f_quantile: float = 1.0,
    max_gamma: float | None = None,
    min_assigned_verts: int = 1,
) -> tuple[np.ndarray, dict]:
    params_array = np.asarray(params_array, dtype=np.float64)
    if params_array.ndim == 1:
        params_array = params_array.reshape(1, -1)
    if params_array.ndim != 2 or params_array.shape[1] < 11:
        raise ValueError(f"Expected SQ params with shape (N, 11), got {params_array.shape}")

    original_iof = evaluate_all_sq_iof(points_world, params_array)
    original_union_max = float(np.min(original_iof, axis=1).max())
    radial_distances = evaluate_all_sq_radial_distances(points_world, params_array)
    assignments = np.argmin(radial_distances, axis=1)

    expanded = params_array.copy()
    per_sq_stats = []
    for i in range(params_array.shape[0]):
        mask = assignments == i
        assigned_count = int(mask.sum())
        if assigned_count < min_assigned_verts:
            per_sq_stats.append(
                {
                    "index": i,
                    "assigned_vertices": assigned_count,
                    "fmax_before": 0.0,
                    "f_target": 0.0,
                    "gamma": 1.0,
                    "skipped": True,
                }
            )
            continue

        local_points = world_to_local(points_world[mask], params_array[i])
        iof_values = sq_iof_local(local_points, params_array[i])
        fmax_before = float(iof_values.max())
        f_target = float(np.quantile(iof_values, f_quantile))
        gamma_from_f = sq_axis_scale_from_fmax(f_target, params_array[i, 0])

        gamma = max(1.0, gamma_from_f)
        if max_gamma is not None:
            gamma = min(gamma, max_gamma)
        expanded[i, 2:5] *= gamma

        per_sq_stats.append(
            {
                "index": i,
                "assigned_vertices": assigned_count,
                "fmax_before": fmax_before,
                "f_target": f_target,
                "metric_margin": metric_margin,
                "gamma_from_f": gamma_from_f,
                "gamma": gamma,
                "skipped": False,
            }
        )

    expanded_iof = evaluate_all_sq_iof(points_world, expanded)
    expanded_union_max = float(np.min(expanded_iof, axis=1).max())

    stats = {
        "num_sq": int(params_array.shape[0]),
        "num_vertices": int(points_world.shape[0]),
        "original_union_max": original_union_max,
        "expanded_union_max": expanded_union_max,
        "per_sq": per_sq_stats,
    }
    return expanded, stats


def iter_object_dirs(root: Path, params_name: str, mesh_name: str):
    params_path = root / params_name
    mesh_path = root / mesh_name
    if params_path.exists() and mesh_path.exists():
        yield root, params_path, mesh_path
        return

    for params_path in sorted(root.rglob(params_name)):
        obj_dir = params_path.parent
        mesh_path = obj_dir / mesh_name
        if mesh_path.exists():
            yield obj_dir, params_path, mesh_path


def output_path_for(params_path: Path, output_name: str | None, in_place: bool) -> Path:
    if in_place:
        return params_path
    if output_name:
        return params_path.with_name(output_name)
    return params_path.with_name("obj_sq_params_expanded.npy")


def screenshot_path_for(screenshot_dir: Path, root: Path, obj_dir: Path) -> Path:
    try:
        relative_obj_dir = obj_dir.relative_to(root)
    except ValueError:
        relative_obj_dir = Path(obj_dir.name)
    if str(relative_obj_dir) in {"", "."}:
        return screenshot_dir / "sq_expansion_check.png"
    return screenshot_dir / relative_obj_dir / "sq_expansion_check.png"


def build_sq_meshes(params_array: np.ndarray, resolution: int):
    meshes = []
    for params in np.asarray(params_array):
        sq = superquadric(params[0:2], params[2:5], params[5:8], params[8:11])
        mesh = superquadric_to_mesh(sq, resolution=resolution)
        meshes.append(pv.wrap(mesh).extract_surface().triangulate())
    return meshes


def visualize_comparison(
    mesh_path: Path,
    original_params: np.ndarray,
    expanded_params: np.ndarray,
    resolution: int,
    screenshot_path: Path | None,
    max_mesh_triangles: int | None,
):
    mesh = read_mesh(mesh_path)
    if max_mesh_triangles is not None and max_mesh_triangles > 0:
        triangle_count = int(mesh.n_cells)
        if triangle_count > max_mesh_triangles:
            reduction = 1.0 - (float(max_mesh_triangles) / float(triangle_count))
            mesh = mesh.decimate(max(0.0, min(reduction, 0.99)))

    mesh_poly = mesh
    mesh_kwargs = {"opacity": 1.0}
    color_array = None
    for key in ("RGBA", "COLOR_0"):
        if key in mesh_poly.point_data:
            color_array = np.asarray(mesh_poly.point_data[key])
            break
    if color_array is not None and color_array.ndim == 2 and color_array.shape[0] == mesh_poly.n_points:
        if color_array.shape[1] == 3:
            if np.issubdtype(color_array.dtype, np.floating):
                rgb = np.clip(np.round(color_array * 255.0), 0.0, 255.0).astype(np.uint8)
            else:
                rgb = np.clip(color_array, 0, 255).astype(np.uint8)
            alpha = np.full((mesh_poly.n_points, 1), 255, dtype=np.uint8)
            color_array = np.hstack([rgb, alpha])
        elif color_array.shape[1] == 4 and np.issubdtype(color_array.dtype, np.floating):
            color_array = np.clip(np.round(color_array * 255.0), 0.0, 255.0).astype(np.uint8)
        mesh_poly.point_data["rgba"] = color_array
        mesh_kwargs["scalars"] = "rgba"
        mesh_kwargs["rgba"] = True
    else:
        mesh_kwargs["color"] = "lightgray"

    plotter = pv.Plotter(shape=(1, 2), off_screen=screenshot_path is not None)
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_text("Original SQ", font_size=12, color="black")

    for i, sq_mesh in enumerate(build_sq_meshes(original_params, resolution)):
        plotter.add_mesh(
            sq_mesh,
            color="#4f83ff",
            opacity=0.28,
            show_edges=True,
            edge_color="#1d4ed8",
            smooth_shading=True,
            label="Original SQs" if i == 0 else None,
        )
    plotter.add_legend()
    plotter.camera_position = "iso"

    plotter.subplot(0, 1)
    plotter.add_text("Expanded SQ", font_size=12, color="black")
    plotter.add_mesh(mesh_poly.copy(), label="Original mesh", **mesh_kwargs)

    for i, sq_mesh in enumerate(build_sq_meshes(expanded_params, resolution)):
        plotter.add_mesh(
            sq_mesh,
            color="#ff9f1c",
            opacity=0.28,
            show_edges=True,
            edge_color="#c2410c",
            smooth_shading=True,
            label="Expanded SQs" if i == 0 else None,
        )

    plotter.add_legend()
    plotter.camera_position = "iso"
    plotter.link_views()

    if screenshot_path is not None:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot_path))
        print(f"    screenshot: {screenshot_path}")
    else:
        plotter.show()
    plotter.close()


def main():
    parser = argparse.ArgumentParser(
        description="Expand saved superquadric axes to better contain each reconstructed object mesh."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("reconstruction"),
        help="Scene/root directory, or a single object directory containing the params and mesh files.",
    )
    parser.add_argument("--params-name", default="obj_sq_params.npy")
    parser.add_argument("--mesh-name", default="obj_mesh.glb")
    parser.add_argument(
        "--output-name",
        default="obj_sq_params_expanded.npy",
        help="Ignored when --in-place is set.",
    )
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute objects even when the expanded params output already exists.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the mesh with original SQs and inflated SQs overlaid.",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=None,
        help="If set with --visualize, save screenshots here instead of opening an interactive window.",
    )
    parser.add_argument("--viz-resolution", type=int, default=50)
    parser.add_argument(
        "--viz-max-mesh-triangles",
        type=int,
        default=200000,
        help="Cap visualization mesh density by decimating large meshes before rendering. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--assignment-mode",
        choices=["closest", "all"],
        default="closest",
        help="Use 'closest' for multi-SQ objects so each vertex expands only its best-fitting primitive.",
    )
    parser.add_argument(
        "--f-quantile",
        type=float,
        default=1.0,
        help="Per-SQ IOF quantile used as the expansion target. 1.0 means strict max; lower is more robust.",
    )
    parser.add_argument(
        "--metric-margin",
        type=float,
        default=0.0,
        help="Additive outward clearance in world units (for example meters) applied beyond the mesh surface.",
    )
    parser.add_argument(
        "--max-gamma",
        type=float,
        default=1.5,
        help="Optional upper bound on per-SQ gamma to prevent blow-up.",
    )
    parser.add_argument(
        "--min-assigned-verts",
        type=int,
        default=1,
        help="Skip expansion for SQs with fewer assigned vertices than this threshold.",
    )
    args = parser.parse_args()

    if not (0.0 < args.f_quantile <= 1.0):
        raise ValueError("--f-quantile must be in (0, 1].")
    if args.metric_margin < 0.0:
        raise ValueError("--metric-margin must be >= 0.0")
    if args.max_gamma is not None and args.max_gamma < 1.0:
        raise ValueError("--max-gamma must be >= 1.0 when set.")
    if args.min_assigned_verts < 1:
        raise ValueError("--min-assigned-verts must be >= 1.")

    processed = 0
    skipped_existing = 0
    for obj_dir, params_path, mesh_path in iter_object_dirs(args.root, args.params_name, args.mesh_name):
        if args.limit is not None and processed >= args.limit:
            break

        destination = output_path_for(params_path, args.output_name, args.in_place)
        if (
            not args.in_place
            and not args.overwrite_existing
            and destination.exists()
            and destination != params_path
        ):
            skipped_existing += 1
            print(f"skip existing expanded SQ: {obj_dir} -> {destination}")
            continue

        params_array = np.load(params_path)
        sq_centroid = compute_sq_centroid_from_params(params_array)
        vertices = load_mesh_vertices_with_margin(
            mesh_path,
            args.metric_margin,
            centroid_world=sq_centroid,
        )
        expanded, stats = expand_sq_params_for_mesh(
            vertices,
            params_array,
            metric_margin=args.metric_margin,
            f_quantile=args.f_quantile,
            max_gamma=args.max_gamma,
            min_assigned_verts=args.min_assigned_verts,
        )

        if not args.dry_run:
            np.save(destination, expanded)

        processed += 1
        gamma_values = [item["gamma"] for item in stats["per_sq"]]
        gamma_max = max(gamma_values) if gamma_values else 1.0
        skipped_count = sum(1 for item in stats["per_sq"] if item["skipped"])
        print(
            f"[{processed}] {obj_dir} | SQs={stats['num_sq']} | verts={stats['num_vertices']} "
            f"| union max: {stats['original_union_max']:.6f} -> {stats['expanded_union_max']:.6f} "
            f"| largest per-SQ gamma={gamma_max:.6f} | q={args.f_quantile:.3f} "
            f"| metric_margin={args.metric_margin:.3f} | skipped={skipped_count}"
        )
        if args.dry_run:
            print("    dry-run: no file written")
        else:
            print(f"    wrote: {destination}")

        if args.visualize:
            screenshot_path = None
            if args.screenshot_dir is not None:
                screenshot_path = screenshot_path_for(args.screenshot_dir, args.root, obj_dir)
            visualize_comparison(
                mesh_path=mesh_path,
                original_params=params_array,
                expanded_params=expanded,
                resolution=max(16, int(args.viz_resolution)),
                screenshot_path=screenshot_path,
                max_mesh_triangles=args.viz_max_mesh_triangles,
            )

    if processed == 0 and skipped_existing == 0:
        print(f"No matching objects found under {args.root}")
    elif skipped_existing > 0:
        print(f"Skipped {skipped_existing} object(s) with existing expanded SQ params.")


if __name__ == "__main__":
    main()
