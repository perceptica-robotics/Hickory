"""FoundationPose adapter for Hickory object-map directories."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from hickory.utils.third_party import FOUNDATIONPOSE_ROOT


def _add_foundationpose_path() -> None:
    path = str(FOUNDATIONPOSE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_intrinsics(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Camera intrinsics not found: {path}")
    values = np.load(path) if path.suffix.lower() == ".npy" else np.loadtxt(path)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 16:
        values = values.reshape(4, 4)[:3, :3]
    elif values.size == 9:
        values = values.reshape(3, 3)
    else:
        raise ValueError(f"Expected a 3x3 or 4x4 intrinsics matrix in {path}, got {values.shape}")
    if not np.isfinite(values).all() or values[0, 0] <= 0 or values[1, 1] <= 0:
        raise ValueError(f"Invalid camera intrinsics in {path}")
    return values


def _simplify_mesh(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Return a colored pose-estimation mesh without modifying the saved GLB."""
    if len(mesh.faces) <= max_faces:
        return mesh

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            f"Mesh has {len(mesh.faces)} faces, but Open3D is unavailable for simplification"
        ) from exc

    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None and hasattr(mesh.visual, "to_color"):
        colors = mesh.visual.to_color().vertex_colors
    colors = np.asarray(colors) if colors is not None else np.empty((0, 4))
    if colors.shape[0] != len(mesh.vertices):
        colors = np.full((len(mesh.vertices), 4), 128, dtype=np.uint8)

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.asarray(colors[:, :3], dtype=np.float64) / 255.0
    )
    simplified = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(max_faces)
    )
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_unreferenced_vertices()

    vertices = np.asarray(simplified.vertices)
    faces = np.asarray(simplified.triangles)
    vertex_colors = np.asarray(simplified.vertex_colors)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("Open3D produced an empty simplified mesh")
    if vertex_colors.shape != (len(vertices), 3):
        vertex_colors = np.full((len(vertices), 3), 0.5, dtype=np.float64)

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=np.clip(vertex_colors * 255.0, 0, 255).astype(np.uint8),
        process=False,
    )


def _load_mesh(path: Path, max_faces: int) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(f"Object mesh not found: {path}")
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected one triangle mesh in {path}, got {type(mesh).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Object mesh is empty: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Object mesh contains non-finite vertices: {path}")
    original_faces = len(mesh.faces)
    mesh = _simplify_mesh(mesh, max_faces)
    if len(mesh.faces) < original_faces:
        print(
            f"[FoundationPose] Simplified {path.name}: "
            f"{original_faces:,} -> {len(mesh.faces):,} faces",
            flush=True,
        )
    return mesh


def _load_observation(
    obj_dir: Path,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth_scale <= 0:
        raise ValueError(f"depth_scale must be positive, got {depth_scale}")

    rgb_path = obj_dir / "view.png"
    depth_path = obj_dir / "depth.png"
    mask_path = obj_dir / "mask.png"
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    if mask_raw is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")

    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[..., -1]
    if depth_raw.shape != rgb_bgr.shape[:2] or mask_raw.shape != rgb_bgr.shape[:2]:
        raise ValueError(
            "RGB, depth, and mask sizes must match: "
            f"rgb={rgb_bgr.shape[:2]}, depth={depth_raw.shape}, mask={mask_raw.shape}"
        )

    rgb = np.ascontiguousarray(rgb_bgr[..., ::-1])
    depth = np.asarray(depth_raw, dtype=np.float32) / float(depth_scale)
    depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
    mask = np.asarray(mask_raw > 0, dtype=bool)
    valid = mask & (depth >= 0.001)
    if int(valid.sum()) < 4:
        raise ValueError(
            f"Too few valid masked depth pixels in {obj_dir}: {int(valid.sum())}"
        )
    return rgb, depth, mask


def _select_object_dirs(map_dir: Path, obj: str) -> list[Path]:
    selectors = [part.strip() for part in obj.split(",") if part.strip()]
    if not selectors:
        raise ValueError("Object selector cannot be empty")
    if selectors == ["all"]:
        object_dirs = sorted(
            path for path in map_dir.iterdir()
            if path.is_dir() and path.name.startswith("obj_")
        )
    else:
        if "all" in selectors:
            raise ValueError("'all' cannot be combined with individual object selectors")
        object_dirs = []
        for selector in selectors:
            name = f"obj_{int(selector):03d}" if selector.isdigit() else selector
            if not name.startswith("obj_") or Path(name).name != name:
                raise ValueError(f"Invalid object selector: {selector!r}")
            object_dirs.append(map_dir / name)
    if not object_dirs:
        raise FileNotFoundError(f"No matching object directories under {map_dir}")
    for path in object_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Object directory not found: {path}")
    return object_dirs


def _validate_weight_layout() -> None:
    weights_root = FOUNDATIONPOSE_ROOT / "weights"
    required = ("2023-10-28-18-33-37", "2024-01-11-20-02-45")
    missing = [
        run_name
        for run_name in required
        if not (weights_root / run_name / "config.yml").is_file()
        or not (weights_root / run_name / "model_best.pth").is_file()
    ]
    if missing:
        expected = "\n  - ".join(str(weights_root / name) for name in missing)
        raise FileNotFoundError(
            "FoundationPose weights are missing or have an extra archive directory level. "
            f"Expected:\n  - {expected}"
        )


def _validate_pose(pose: np.ndarray, obj_dir: Path) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError(f"FoundationPose returned a non-finite pose for {obj_dir.name}")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"FoundationPose returned an invalid homogeneous pose for {obj_dir.name}")
    det = float(np.linalg.det(pose[:3, :3]))
    if not np.isclose(det, 1.0, atol=1e-2):
        raise ValueError(f"FoundationPose returned an invalid rotation for {obj_dir.name}: det={det}")
    return pose


def run_map(
    map_dir: Path,
    *,
    obj: str = "all",
    depth_scale: float = 1000.0,
    est_refine_iter: int = 5,
    max_mesh_faces: int = 100_000,
    save_detailed_results: bool = False,
) -> list[Path]:
    map_dir = Path(map_dir).expanduser().resolve()
    if not map_dir.is_dir():
        raise FileNotFoundError(f"Object map directory not found: {map_dir}")
    if est_refine_iter < 1:
        raise ValueError("est_refine_iter must be at least 1")
    if max_mesh_faces < 1_000:
        raise ValueError("max_mesh_faces must be at least 1000")

    camera_k = _load_intrinsics(map_dir / "camera_K.txt")
    object_dirs = _select_object_dirs(map_dir, obj)
    _validate_weight_layout()

    _add_foundationpose_path()
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor
    from Utils import dr, set_logging_format, set_seed

    set_logging_format(logging.INFO)
    set_seed(0)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    saved_paths: list[Path] = []
    for index, obj_dir in enumerate(object_dirs, start=1):
        print(f"[FoundationPose] {index}/{len(object_dirs)} {obj_dir.name}", flush=True)
        mesh = _load_mesh(obj_dir / "obj_mesh.glb", max_mesh_faces)
        rgb, depth, mask = _load_observation(obj_dir, depth_scale)

        temp_context = None
        if save_detailed_results:
            debug_dir = obj_dir / "foundationpose"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_level = 2
        else:
            temp_context = tempfile.TemporaryDirectory(prefix=f"hickory-fp-{obj_dir.name}-")
            debug_dir = Path(temp_context.name)
            debug_level = 0

        estimator = None
        try:
            estimator = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=str(debug_dir),
                debug=debug_level,
                glctx=glctx,
            )
            pose = estimator.register(
                K=camera_k,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=est_refine_iter,
            )
            pose = _validate_pose(pose, obj_dir)
            pose_path = obj_dir / "pose_foundation.txt"
            np.savetxt(pose_path, pose, fmt="%.10f")
            saved_paths.append(pose_path)

            if save_detailed_results:
                posed_mesh = mesh.copy()
                posed_mesh.apply_transform(pose)
                posed_mesh.export(debug_dir / "model_in_camera.ply")
        finally:
            if estimator is not None and estimator.mesh_path:
                try:
                    Path(estimator.mesh_path).unlink(missing_ok=True)
                except OSError:
                    pass
            estimator = None
            if temp_context is not None:
                temp_context.cleanup()
            gc.collect()
            torch.cuda.empty_cache()

    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine Hickory object poses with FoundationPose."
    )
    parser.add_argument("--map_dir", type=Path, required=True)
    parser.add_argument(
        "--obj",
        default="all",
        help="Object folder, numeric ID, comma-separated objects, or 'all'.",
    )
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--max_mesh_faces", type=int, default=100_000)
    parser.add_argument("--save_detailed_results", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    saved = run_map(
        args.map_dir,
        obj=args.obj,
        depth_scale=args.depth_scale,
        est_refine_iter=args.est_refine_iter,
        max_mesh_faces=args.max_mesh_faces,
        save_detailed_results=args.save_detailed_results,
    )
    for path in saved:
        print(f"Saved FoundationPose pose: {path}")


if __name__ == "__main__":
    main()
