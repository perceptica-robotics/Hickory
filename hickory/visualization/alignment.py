import argparse
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

from hickory.visualization.scene import (
    add_odom_origin_marker,
    load_camera_trajectory,
    load_scene,
    load_transform,
)


def resolve_scene_dir(root: Path, scene: str | None, explicit_dir: str | None) -> Path:
    if explicit_dir is not None:
        return Path(explicit_dir)
    if scene is None:
        raise ValueError("Provide a scene name or an explicit objects directory.")
    return root / scene


def resolve_alignment_file(
    root: Path,
    scene_a: str,
    scene_b: str,
    shape_mode: str,
    explicit_path: str | None,
    suffix: str,
) -> Path | None:
    if explicit_path is not None:
        return Path(explicit_path)

    scene_a_dir = root / scene_a
    scene_b_dir = root / scene_b
    shared_parts = []
    for part_a, part_b in zip(Path(scene_a).parts, Path(scene_b).parts):
        if part_a != part_b:
            break
        shared_parts.append(part_a)
    shared_dir = root / Path(*shared_parts) if shared_parts else root
    label_a_parts = Path(scene_a).parts[len(shared_parts):] or (scene_a_dir.name,)
    label_b_parts = Path(scene_b).parts[len(shared_parts):] or (scene_b_dir.name,)
    shared_stem = f"{'_'.join(label_a_parts)}_{'_'.join(label_b_parts)}"

    candidates = [
        shared_dir / f"{shared_stem}_{shape_mode}{suffix}",
        shared_dir / f"{shared_stem}{suffix}",
        root / f"{scene_a}_{scene_b}_{shape_mode}{suffix}",
        root / f"{scene_a}_{scene_b}{suffix}",
        Path("reconstruction") / f"{scene_a}_{scene_b}_{shape_mode}{suffix}",
        Path("reconstruction") / f"{scene_a}_{scene_b}{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_alignment_camera_trajectory(objects_dir: Path, source: str) -> Path | None:
    if source == "world":
        candidates = [
            "camera_poses_world.txt",
            "camera_poses_world.npy",
        ]
    elif source == "odom":
        candidates = [
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
    else:
        raise ValueError(f"Unsupported trajectory source: {source}")

    fallback_candidates = [
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
        "camera_poses_world.txt",
        "camera_poses_world.npy",
    ]
    for name in candidates + [name for name in fallback_candidates if name not in candidates]:
        path = objects_dir / name
        if path.exists():
            return path
    return None


def load_first_pose(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    traj = load_camera_trajectory(path)
    if traj.shape[0] == 0:
        return None
    return np.asarray(traj[0], dtype=np.float64)


def compute_world_from_odom(objects_dir: Path) -> np.ndarray:
    odom_pose = load_first_pose(objects_dir / "camera_poses.txt")
    world_pose = load_first_pose(objects_dir / "camera_poses_world.txt")
    if odom_pose is None or world_pose is None:
        return np.eye(4)
    return world_pose @ np.linalg.inv(odom_pose)


def with_display_offset(transform: np.ndarray, offset: np.ndarray) -> np.ndarray:
    out = np.asarray(transform, dtype=np.float64).copy()
    if np.any(offset):
        out[:3, 3] += offset
    return out


def make_translation(offset: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = offset
    return transform


def add_association_lines(geometries, associations_path: Path, centers_a, centers_b):
    pairs = np.load(str(associations_path))
    assoc_points = []
    assoc_lines = []
    for id_a, id_b in pairs:
        id_a = int(id_a)
        id_b = int(id_b)
        if id_a not in centers_a or id_b not in centers_b:
            continue
        start_idx = len(assoc_points)
        assoc_points.append(centers_a[id_a])
        assoc_points.append(centers_b[id_b])
        assoc_lines.append([start_idx, start_idx + 1])

    if not assoc_lines:
        print(f"Loaded 0 drawable associations from {associations_path}")
        return

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(assoc_points, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(assoc_lines, dtype=np.int32)),
    )
    line_set.paint_uniform_color([0.2, 1.0, 0.1])
    geometries.append(line_set)
    print(f"Loaded {len(assoc_lines)} drawable associations from {associations_path}")


def normalize_pose_source(pose_source: str) -> str:
    if pose_source in {"foundationpose", "foundation", "fp"}:
        return "fp"
    if pose_source == "sam3d":
        return "sam3d"
    raise ValueError(f"Unsupported pose source: {pose_source}")


def main():
    parser = argparse.ArgumentParser(description="Visualize a two-scene CLIPPER alignment.")
    parser.add_argument("--root", type=Path, default=Path("reconstruction/REPLICA"))
    parser.add_argument("--scene-a", required=True, help="Scene A folder under --root.")
    parser.add_argument("--scene-b", required=True, help="Scene B folder under --root.")
    parser.add_argument("--objects-dir-a", default=None, help="Direct path for scene A objects.")
    parser.add_argument("--objects-dir-b", default=None, help="Direct path for scene B objects.")
    parser.add_argument(
        "--transform-b-to-a",
        default=None,
        help="4x4 transform that maps scene B into scene A. Defaults to the CLIPPER output under --root.",
    )
    parser.add_argument(
        "--associations",
        default=None,
        help="Optional [id_a, id_b] association .npy. Defaults to the CLIPPER associations under --root.",
    )
    parser.add_argument(
        "--shape-mode",
        choices=["sq", "coarse", "none"],
        default="sq",
        help="Transform filename suffix used when auto-resolving CLIPPER outputs.",
    )
    parser.add_argument(
        "--pose-source",
        choices=["foundationpose", "foundation", "fp", "sam3d"],
        default="foundationpose",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        default=(0.0, 5.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Optional display offset applied to scene B after alignment.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Show scene B directly over scene A by forcing the display offset to zero.",
    )
    parser.add_argument(
        "--camera-trajectory-a",
        default=None,
        help="Optional trajectory path for scene A.",
    )
    parser.add_argument(
        "--camera-trajectory-b",
        default=None,
        help="Optional trajectory path for scene B.",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=["world", "odom"],
        default="world",
        help="Default trajectory source. 'world' uses camera_poses_world.txt and renders objects in world frame.",
    )
    parser.add_argument("--hide-associations", action="store_true")
    parser.add_argument("--hide-camera-trajectory", action="store_true")
    parser.add_argument("--hide-object-keyframes", action="store_true")
    parser.add_argument("--hide-odometry-origins", action="store_true")
    parser.add_argument("--odometry-origin-size", type=float, default=0.1)
    parser.add_argument("--keyframe-pyramid-size", type=float, default=0.12)
    parser.add_argument("--sq-resolution", type=int, default=24)
    args = parser.parse_args()

    objects_dir_a = resolve_scene_dir(args.root, args.scene_a, args.objects_dir_a)
    objects_dir_b = resolve_scene_dir(args.root, args.scene_b, args.objects_dir_b)
    transform_path = resolve_alignment_file(
        args.root,
        args.scene_a,
        args.scene_b,
        args.shape_mode,
        args.transform_b_to_a,
        ".txt",
    )
    if transform_path is None:
        raise FileNotFoundError(
            "Could not find alignment transform. Run clipper_solve.py first or pass --transform-b-to-a."
        )

    associations_path = None
    if not args.hide_associations:
        associations_path = resolve_alignment_file(
            args.root,
            args.scene_a,
            args.scene_b,
            args.shape_mode,
            args.associations,
            "_associations.npy",
        )

    pose_source = normalize_pose_source(args.pose_source)
    traj_a = (
        Path(args.camera_trajectory_a)
        if args.camera_trajectory_a
        else find_alignment_camera_trajectory(objects_dir_a, args.trajectory_source)
    )
    traj_b = (
        Path(args.camera_trajectory_b)
        if args.camera_trajectory_b
        else find_alignment_camera_trajectory(objects_dir_b, args.trajectory_source)
    )
    estimated_b_to_a = load_transform(transform_path)
    offset = np.zeros(3, dtype=np.float64) if args.overlay else np.asarray(args.offset, dtype=np.float64).reshape(3)

    if args.trajectory_source == "world":
        world_from_a_odom = compute_world_from_odom(objects_dir_a)
        transform_a = world_from_a_odom
        transform_b = with_display_offset(world_from_a_odom @ estimated_b_to_a, offset)
        trajectory_transform_a = None
        trajectory_transform_b = make_translation(offset) if np.any(offset) else None
    else:
        transform_a = None
        transform_b = with_display_offset(estimated_b_to_a, offset)
        trajectory_transform_a = None
        trajectory_transform_b = transform_b

    print(f"Scene A: {objects_dir_a}")
    print(f"Scene B: {objects_dir_b}")
    print(f"Transform B->A: {transform_path}")
    print(f"Trajectory source: {args.trajectory_source}")
    print(f"Trajectory A: {traj_a}")
    print(f"Trajectory B: {traj_b}")
    if associations_path is not None:
        print(f"Associations: {associations_path}")

    geometries, centers_a = load_scene(
        objects_dir_a,
        quat_order="xyzw",
        color=[0.2, 0.6, 0.9],
        transform=transform_a,
        trajectory_transform=trajectory_transform_a,
        normalize_trajectory_to_first=False,
        enable_layer_visualization=False,
        load_sq=False,
        load_expanded_sq=False,
        camera_traj_path=traj_a,
        pose_source=pose_source,
        show_object_keyframes=not args.hide_object_keyframes,
        keyframe_pyramid_size=args.keyframe_pyramid_size,
        show_camera_trajectory=not args.hide_camera_trajectory,
        sq_resolution=args.sq_resolution,
    )

    geoms_b, centers_b = load_scene(
        objects_dir_b,
        quat_order="xyzw",
        color=[0.95, 0.55, 0.2],
        transform=transform_b,
        trajectory_transform=trajectory_transform_b,
        normalize_trajectory_to_first=False,
        enable_layer_visualization=False,
        load_sq=False,
        load_expanded_sq=False,
        camera_traj_path=traj_b,
        pose_source=pose_source,
        show_object_keyframes=not args.hide_object_keyframes,
        keyframe_pyramid_size=args.keyframe_pyramid_size,
        show_camera_trajectory=not args.hide_camera_trajectory,
        sq_resolution=args.sq_resolution,
    )
    geometries += geoms_b

    if not args.hide_odometry_origins:
        add_odom_origin_marker(
            geometries,
            transform=transform_a if transform_a is not None else np.eye(4),
            size=args.odometry_origin_size,
            color=(0.2, 0.6, 0.9),
        )
        add_odom_origin_marker(
            geometries,
            transform=transform_b,
            size=args.odometry_origin_size,
            color=(0.95, 0.55, 0.2),
        )

    if associations_path is not None:
        add_association_lines(geometries, associations_path, centers_a, centers_b)

    if not geometries:
        raise RuntimeError("No valid visualization geometry was loaded.")

    o3d.visualization.draw_geometries(geometries, mesh_show_back_face=True)


if __name__ == "__main__":
    main()
