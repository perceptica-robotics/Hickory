import argparse
from pathlib import Path
import sys
import time

import numpy as np
import trimesh

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _path in (_REPO_ROOT, _THIS_DIR, _REPO_ROOT / "hickory" / "utils"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

try:
    from hickory.align import evaluate_clipper_transform as ect
except ModuleNotFoundError:
    import evaluate_clipper_transform as ect

try:
    from hickory.utils.frame_conventions import compose_world_object_from_sam3d
except ModuleNotFoundError:
    from frame_conventions import compose_world_object_from_sam3d

from roman.align.sq_registration import SQRegistration, SQObject
try:
    from hickory.align.calculate_centroid import compute_sq_centroid as _compute_sq_centroid
except Exception:
    try:
        from calculate_centroid import compute_sq_centroid as _compute_sq_centroid
    except Exception:
        _compute_sq_centroid = None

DEFAULT_SCENES = {
    "A": "Kimera-Multi/demo_sparkal1_test",
    # "A": "ReplicaCAD/apt0/1/demo",
    "B": "Kimera-Multi/demo_sparkal2_test",
    # "B": "ReplicaCAD/apt0/2/demo",
}


def build_output_stem(root_dir: Path, scene_a_dir: Path, scene_b_dir: Path):
    """Return a stable output directory and filename stem for a scene pair."""
    out_dir = root_dir
    try:
        rel_a = scene_a_dir.resolve().relative_to(root_dir.resolve())
        rel_b = scene_b_dir.resolve().relative_to(root_dir.resolve())
        shared_parts = []
        for part_a, part_b in zip(rel_a.parts, rel_b.parts):
            if part_a != part_b:
                break
            shared_parts.append(part_a)

        if shared_parts:
            out_dir = root_dir / Path(*shared_parts)

        label_a_parts = rel_a.parts[len(shared_parts):] or (scene_a_dir.name,)
        label_b_parts = rel_b.parts[len(shared_parts):] or (scene_b_dir.name,)
        stem = f"{'_'.join(label_a_parts)}_{'_'.join(label_b_parts)}"
        return out_dir, stem
    except Exception:
        pass
    stem = f"{scene_a_dir.name}_{scene_b_dir.name}"
    return out_dir, stem

def load_camera_pose(path: Path) -> np.ndarray:
    mat = np.loadtxt(str(path))
    if mat.size == 12:
        mat = mat.reshape(3, 4)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
    if mat.shape != (4, 4):
        raise ValueError(f"Unexpected camera pose shape in {path}: {mat.shape}")
    return mat

def load_feature(feature_path: Path):
    raw_feat = np.load(feature_path, allow_pickle=True)
    if isinstance(raw_feat, np.ndarray) and raw_feat.ndim == 0:
        raw_feat = raw_feat.item()
    if isinstance(raw_feat, dict):
        return raw_feat.get("mean", raw_feat)
    return raw_feat

def parse_obj_id(obj_dir: Path, fallback: int):
    suffix = obj_dir.name.split("_")[-1]
    return int(suffix) if suffix.isdigit() else fallback

def compute_centroid(sq_path: Path, sq_data: np.ndarray):
    if _compute_sq_centroid is not None:
        centroid = _compute_sq_centroid(str(sq_path))
        if centroid is not None:
            return centroid
    if sq_data.ndim == 2 and sq_data.shape[1] >= 11:
        return np.mean(sq_data[:, 8:11], axis=0)
    return np.zeros(3)


def compute_coarse_shape_attrs(mesh_path: Path):
    """Compute coarse shape attributes [f1, f2, f3, f4].

    Uses an axis-aligned bounding-box volume and caps the number of vertices
    used for covariance to keep large meshes from exhausting CPU or memory.
    """
    if not mesh_path.exists():
        return None
    try:
        mesh = trimesh.load(str(mesh_path), force="mesh")
        pts = np.asarray(mesh.vertices)
        if pts.shape[0] < 6:
            return None

        if pts.shape[0] > 50000:
            sample_idx = np.linspace(0, pts.shape[0] - 1, 50000, dtype=int)
            pts = pts[sample_idx]

        try:
            bounds = np.asarray(mesh.bounds, dtype=float)
            if bounds.shape != (2, 3):
                return None
            extents = np.maximum(bounds[1] - bounds[0], 0.0)
            volume = float(np.prod(extents))
        except Exception:
            volume = 0.0

        cov = np.cov(pts.T)
        if cov.shape != (3, 3):
            return None
        eigvals = np.asarray(np.linalg.eigvalsh(cov), dtype=float)[::-1]
        s = float(np.sum(eigvals))
        if s <= 1e-12:
            return None
        e = eigvals / s
        e0 = max(float(e[0]), 1e-12)
        linearity = (float(e[0]) - float(e[1])) / e0
        planarity = (float(e[1]) - float(e[2])) / e0
        scattering = float(e[2]) / e0
        return np.array([volume, linearity, planarity, scattering], dtype=float)
    except Exception:
        return None

def load_scene_objects(scene_dir: Path, pose_source: str, shape_mode: str = "coarse"):
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    objects = []
    obj_dirs = sorted(p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("obj_"))
    for i, obj_dir in enumerate(obj_dirs):
        sq_path = obj_dir / "obj_sq_params.npy"
        ft_path = obj_dir / "dino_feature.npy"
        mesh_path = obj_dir / "obj_mesh.glb"
        if pose_source == "sam3d":
            pose_path = obj_dir / "pose_sam3d.txt"
        elif pose_source == "foundationpose":
            pose_path = obj_dir / "pose_foundation.txt"
        else:
            raise ValueError(f"Unsupported pose_source: {pose_source}")
        cam_pose_path = obj_dir / "camera_pose.txt"
        if not sq_path.exists() or not ft_path.exists():
            continue

        sq_data = np.load(sq_path, allow_pickle=True)
        centroid = np.asarray(compute_centroid(sq_path, sq_data), dtype=float)
        if pose_path.exists() and cam_pose_path.exists():
            T_WC = load_camera_pose(cam_pose_path)
            T_CO = load_camera_pose(pose_path)
            if pose_source == "sam3d":
                T_WO = compose_world_object_from_sam3d(T_WC, T_CO)
            else:
                T_WO = T_WC @ T_CO
            centroid_h = np.concatenate([centroid[:3], [1.0]])
            centroid = (T_WO @ centroid_h)[:3]

        coarse_shape_attrs = None
        if str(shape_mode).lower() == "coarse":
            coarse_shape_attrs = compute_coarse_shape_attrs(mesh_path)

        obj_kwargs = {
            "centroid": centroid,
            "sq_params": sq_data,
            "semantic_feature": load_feature(ft_path),
            "id": parse_obj_id(obj_dir, i),
        }
        try:
            obj = SQObject(**obj_kwargs, coarse_shape_attrs=coarse_shape_attrs)
        except TypeError as e:
            if "coarse_shape_attrs" not in str(e):
                raise
            obj = SQObject(**obj_kwargs)
            if coarse_shape_attrs is not None:
                obj.coarse_shape_attrs = coarse_shape_attrs
        objects.append(obj)
    return objects


def add_evaluation_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only solve the transform and skip the GT evaluation phase.",
    )
    parser.add_argument(
        "--apt-name",
        type=str,
        default=None,
        help="ReplicaCAD apartment name (e.g. apt0). Use this when the saved transform path does not encode the apartment.",
    )
    parser.add_argument(
        "--gt-seq-a-poses",
        type=Path,
        default=None,
        help="GT world-frame camera_poses.txt or Kimera GT config for seq A. If omitted, infer from the saved transform path.",
    )
    parser.add_argument(
        "--gt-seq-b-poses",
        type=Path,
        default=None,
        help="GT world-frame camera_poses.txt or Kimera GT config for seq B. If omitted, infer from the saved transform path.",
    )
    parser.add_argument(
        "--recon-seq-a-poses",
        type=Path,
        default=None,
        help="Reconstruction odom-frame camera_poses.txt for seq A. If omitted, infer from the saved transform path.",
    )
    parser.add_argument(
        "--recon-seq-b-poses",
        type=Path,
        default=None,
        help="Reconstruction odom-frame camera_poses.txt for seq B. If omitted, infer from the saved transform path.",
    )
    parser.add_argument(
        "--transform-direction",
        choices=["b_to_a", "a_to_b"],
        default="b_to_a",
        help="Direction of the solved transform. clipper_solve.py returns b_to_a.",
    )
    parser.add_argument(
        "--save-alignments-csv",
        type=Path,
        default=None,
        help="Optional CSV path to save relative-transform error and alignment drift diagnostics.",
    )


def run_evaluation(args, est_transform_path: Path, est_transform: np.ndarray):
    dataset_family = ect.infer_dataset_family(est_transform_path)

    eval_args = argparse.Namespace(
        apt_name=args.apt_name,
        gt_seq_a_poses=args.gt_seq_a_poses,
        gt_seq_b_poses=args.gt_seq_b_poses,
        recon_seq_a_poses=args.recon_seq_a_poses,
        recon_seq_b_poses=args.recon_seq_b_poses,
        est_transform=est_transform_path,
        demo_align_pkl=None,
        demo_submap_pair=None,
        transform_direction=args.transform_direction,
        save_alignments_csv=args.save_alignments_csv,
    )

    if dataset_family == "kimera" and not all(
        path is not None
        for path in (
            eval_args.gt_seq_a_poses,
            eval_args.gt_seq_b_poses,
            eval_args.recon_seq_a_poses,
            eval_args.recon_seq_b_poses,
        )
    ):
        gt_seq_a_poses, gt_seq_b_poses, recon_seq_a_poses, recon_seq_b_poses = ect.resolve_kimera_pose_sources(eval_args)
    else:
        gt_seq_a_poses, gt_seq_b_poses, recon_seq_a_poses, recon_seq_b_poses = ect.resolve_pose_paths(eval_args)

    kimera_gt_formats = {".csv", ".yaml", ".yml"}
    use_kimera_pose_data = dataset_family == "kimera" and (
        gt_seq_a_poses.suffix in kimera_gt_formats or gt_seq_b_poses.suffix in kimera_gt_formats
    )

    if use_kimera_pose_data:
        gt_a = ect.load_kimera_gt_pose_data(gt_seq_a_poses)
        gt_b = ect.load_kimera_gt_pose_data(gt_seq_b_poses)
        recon_a_records = ect.load_camera_pose_records(recon_seq_a_poses)
        recon_b_records = ect.load_camera_pose_records(recon_seq_b_poses)

        rows_a, T_wo_a_all = ect.estimate_world_from_odom_pose_data(gt_a, recon_a_records)
        rows_b, T_wo_b_all = ect.estimate_world_from_odom_pose_data(gt_b, recon_b_records)
    else:
        gt_a = ect.load_camera_pose_trajectory(gt_seq_a_poses)
        gt_b = ect.load_camera_pose_trajectory(gt_seq_b_poses)
        recon_a = ect.load_camera_pose_trajectory(recon_seq_a_poses)
        recon_b = ect.load_camera_pose_trajectory(recon_seq_b_poses)

        rows_a, T_wo_a_all = ect.estimate_world_from_odom(gt_a, recon_a)
        rows_b, T_wo_b_all = ect.estimate_world_from_odom(gt_b, recon_b)

    ref_frame_a = rows_a[0][0]
    ref_frame_b = rows_b[0][0]
    T_wo_a_ref = ect.aggregate_reference_transform(T_wo_a_all)
    T_wo_b_ref = ect.aggregate_reference_transform(T_wo_b_all)

    a_trans_drift, a_rot_drift = ect.transform_stability(T_wo_a_ref, T_wo_a_all)
    b_trans_drift, b_rot_drift = ect.transform_stability(T_wo_b_ref, T_wo_b_all)

    T_rel_gt = ect.relative_transform_from_world_alignments(
        T_wo_a=T_wo_a_ref,
        T_wo_b=T_wo_b_ref,
        transform_direction=args.transform_direction,
    )
    rel_t_err, rel_r_err = ect.pose_error(est_transform, T_rel_gt)
    gt_transform_path = est_transform_path.with_name(f"{est_transform_path.stem}_gt{est_transform_path.suffix}")
    np.savetxt(gt_transform_path, T_rel_gt, fmt="%.8f")

    print("\nStep 3: Evaluating Registration...")
    print("Clipper Relative-Transform Evaluation")
    print(f"GT seq A poses:    {gt_seq_a_poses}")
    print(f"GT seq B poses:    {gt_seq_b_poses}")
    print(f"Recon seq A poses: {recon_seq_a_poses}")
    print(f"Recon seq B poses: {recon_seq_b_poses}")
    print(f"Estimated transform: {est_transform_path}")
    print(f"Saved GT transform: {gt_transform_path}")
    print(f"Input transform direction: {args.transform_direction}")

    print("\nSeq A world<-odom alignment")
    print(f"  Common frames: {len(rows_a)}")
    print(f"  Aggregate reference built from {len(rows_a)} frames; first common frame was {ref_frame_a}")
    print(f"  Drift translation mean/max (m): {np.mean(a_trans_drift):.6f} / {np.max(a_trans_drift):.6f}")
    print(f"  Drift rotation    mean/max (deg): {np.mean(a_rot_drift):.6f} / {np.max(a_rot_drift):.6f}")

    print("\nSeq B world<-odom alignment")
    print(f"  Common frames: {len(rows_b)}")
    print(f"  Aggregate reference built from {len(rows_b)} frames; first common frame was {ref_frame_b}")
    print(f"  Drift translation mean/max (m): {np.mean(b_trans_drift):.6f} / {np.max(b_trans_drift):.6f}")
    print(f"  Drift rotation    mean/max (deg): {np.mean(b_rot_drift):.6f} / {np.max(b_rot_drift):.6f}")

    print("\nRelative Transform Error")
    print("  Estimated transform produced by clipper_solve.py")
    print(f"  Ground-truth direction: {args.transform_direction}")
    print(f"  Translation error (m): {rel_t_err:.6f}")
    print(f"  Rotation error (deg): {rel_r_err:.6f}")

    if args.save_alignments_csv is not None:
        args.save_alignments_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_alignments_csv.open("w", newline="") as f:
            writer = ect.csv.DictWriter(
                f,
                fieldnames=[
                    "sequence",
                    "frame_idx",
                    "relative_translation_error_m",
                    "relative_rotation_error_deg",
                    "alignment_translation_drift_m",
                    "alignment_rotation_drift_deg",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sequence": "relative_transform",
                    "frame_idx": "",
                    "relative_translation_error_m": rel_t_err,
                    "relative_rotation_error_deg": rel_r_err,
                    "alignment_translation_drift_m": "",
                    "alignment_rotation_drift_deg": "",
                }
            )
            for (frame_idx, _), t_drift, r_drift in zip(rows_a, a_trans_drift, a_rot_drift):
                writer.writerow(
                    {
                        "sequence": "A_alignment",
                        "frame_idx": frame_idx,
                        "relative_translation_error_m": "",
                        "relative_rotation_error_deg": "",
                        "alignment_translation_drift_m": t_drift,
                        "alignment_rotation_drift_deg": r_drift,
                    }
                )
            for (frame_idx, _), t_drift, r_drift in zip(rows_b, b_trans_drift, b_rot_drift):
                writer.writerow(
                    {
                        "sequence": "B_alignment",
                        "frame_idx": frame_idx,
                        "relative_translation_error_m": "",
                        "relative_rotation_error_deg": "",
                        "alignment_translation_drift_m": t_drift,
                        "alignment_rotation_drift_deg": r_drift,
                    }
                )
        print(f"\nSaved relative-transform error and alignment drift CSV to: {args.save_alignments_csv}")


def main():
    parser = argparse.ArgumentParser(description="Register two reconstructed scenes with SQ features.")
    parser.add_argument("--root", default="reconstruction", help="Root directory with scene folders")
    parser.add_argument("--scene-a", default=DEFAULT_SCENES["A"], help="Scene A folder name")
    parser.add_argument("--scene-b", default=DEFAULT_SCENES["B"], help="Scene B folder name")
    parser.add_argument("--map-a", default=None, help="Optional direct path to map A folder (overrides --root/--scene-a)")
    parser.add_argument("--map-b", default=None, help="Optional direct path to map B folder (overrides --root/--scene-b)")
    parser.add_argument(
        "--pose-source",
        choices=["sam3d", "foundationpose"],
        default="foundationpose",
        help="Choose object pose source. 'sam3d' uses pose_sam3d.txt with compose_world_object_from_sam3d; 'foundationpose' uses pose_foundation.txt with T_WC @ T_CO.",
    )
    parser.add_argument(
        "--shape-mode",
        choices=["sq", "coarse", "none"],
        default="sq",
        help="Shape ablation mode: 'sq' exact SQ shape, 'coarse' [volume+PCA(linearity/planarity/scattering)], 'none' no shape term (semantic only).",
    )
    parser.add_argument(
        "--geom-sigma",
        type=float,
        default=0.05,
        help="Geometric pairwise distance sigma used by CLIPPER.",
    )
    parser.add_argument(
        "--shape-sigma",
        type=float,
        default=2.0,
        help="SQ exponent similarity sigma. Only used when --shape-mode=sq.",
    )
    add_evaluation_arguments(parser)
    args = parser.parse_args()

    root_dir = Path(args.root)
    scene_a_dir = Path(args.map_a) if args.map_a is not None else root_dir / args.scene_a
    scene_b_dir = Path(args.map_b) if args.map_b is not None else root_dir / args.scene_b
    output_dir, output_stem = build_output_stem(root_dir, scene_a_dir, scene_b_dir)

    time_start = time.time()

    print("Step 1: Loading Data...")
    mapA = load_scene_objects(scene_a_dir, pose_source=args.pose_source, shape_mode=args.shape_mode)
    mapB = load_scene_objects(scene_b_dir, pose_source=args.pose_source, shape_mode=args.shape_mode)
    if not mapA or not mapB:
        raise RuntimeError("No valid objects found in one or both scenes.")
    print(f"Loaded: Map A ({len(mapA)} objs), Map B ({len(mapB)} objs)")

    print("Step 2: Running Registration...")
    try:
        registrar = SQRegistration(
            geom_sigma=args.geom_sigma,
            shape_sigma=args.shape_sigma,
            shape_mode=args.shape_mode,
        )
    except TypeError as e:
        if "shape_mode" not in str(e):
            raise
        registrar = SQRegistration(
            geom_sigma=args.geom_sigma,
            shape_sigma=args.shape_sigma,
        )
        if args.shape_mode != "sq":
            print(
                "Warning: installed SQRegistration does not support --shape-mode; "
                "using its built-in SQ registration behavior."
            )
    try:
        try:
            T, corrs = registrar.register_sq_maps(mapA, mapB, return_corrs=True)
        except TypeError as e:
            if "return_corrs" not in str(e):
                raise
            T = registrar.register_sq_maps(mapA, mapB)
            corrs = None
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{output_stem}_{args.shape_mode}.txt"
        np.savetxt(out_path, T, fmt="%.8f")

        if corrs is not None:
            assoc_pairs = []
            for u, v in corrs:
                assoc_pairs.append([mapA[u].id, mapB[v].id])
            assoc_path = output_dir / f"{output_stem}_{args.shape_mode}_associations.npy"
            np.save(assoc_path, np.asarray(assoc_pairs, dtype=int))
            print(f"Saved {len(assoc_pairs)} associations to {assoc_path}")
        else:
            print("Associations were printed by SQRegistration but not returned by this installed roman version.")

        print("\n" + "="*40)
        print("SUCCESS: Calculated Transformation Matrix")
        print("="*40)
        print(T)
        print("="*40)
        
        translation_dist = np.linalg.norm(T[:3, 3])
        print(f"Translation magnitude: {translation_dist:.4f} meters")
        rot_trace = np.trace(T[:3, :3])
        rotation_angle_rad = np.arccos(np.clip((rot_trace - 1.0) / 2.0, -1.0, 1.0))
        rotation_angle_deg = np.degrees(rotation_angle_rad)
        print(f"Rotation magnitude from identity: {rotation_angle_deg:.4f} degrees")
        time_end = time.time()
        print(f"Registration runtime: {time_end - time_start:.2f} seconds")
        if args.skip_eval:
            print("Evaluation skipped (--skip-eval).")
        else:
            run_evaluation(args, out_path, T)
    except Exception as e:
        print(f"\nFAILED: {e}")

if __name__ == "__main__":
    main()
