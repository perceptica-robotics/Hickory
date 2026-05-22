import argparse
from pathlib import Path
import time
import gc
import contextlib
import io

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("BLIS_NUM_THREADS", "8")

import shutil
import subprocess
import sys

from hickory.utils.third_party import FOUNDATIONPOSE_ROOT, add_third_party_paths

add_third_party_paths()


def ensure_native_runtime():
    """Re-exec once with Conda libstdc++ ahead of system libs for compiled extensions like mesh2sdf."""
    if os.environ.get("OBJECT_SLAM_NATIVE_BOOTSTRAP") == "1":
        return

    env = os.environ.copy()
    lib_candidates = []

    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        lib_candidates.append(Path(conda_prefix) / "lib")

    conda_exe = env.get("CONDA_EXE")
    if conda_exe:
        lib_candidates.append(Path(conda_exe).resolve().parent.parent / "lib")

    lib_dirs = []
    for lib_dir in lib_candidates:
        lib_dir_str = str(lib_dir)
        if lib_dir.is_dir() and lib_dir_str not in lib_dirs:
            lib_dirs.append(lib_dir_str)

    existing_ld_library_path = env.get("LD_LIBRARY_PATH", "")
    if existing_ld_library_path:
        for path in existing_ld_library_path.split(":"):
            if path and path not in lib_dirs:
                lib_dirs.append(path)

    if not lib_dirs:
        return

    new_ld_library_path = ":".join(lib_dirs)
    libstdcxx_path = None
    for lib_dir in lib_dirs:
        candidate = Path(lib_dir) / "libstdc++.so.6"
        if candidate.is_file():
            libstdcxx_path = str(candidate)
            break

    if (
        env.get("LD_LIBRARY_PATH") == new_ld_library_path
        and env.get("LD_PRELOAD") == libstdcxx_path
    ):
        return

    env["OBJECT_SLAM_NATIVE_BOOTSTRAP"] = "1"
    env["LD_LIBRARY_PATH"] = new_ld_library_path
    if libstdcxx_path is not None:
        env["LD_PRELOAD"] = libstdcxx_path
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def iter_object_dirs(output_dir: Path):
    return sorted([p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("obj_")])


def is_valid_triangle_mesh(mesh_path: Path) -> bool:
    if not mesh_path.exists() or mesh_path.stat().st_size == 0:
        return False
    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        return len(mesh.vertices) > 0 and len(mesh.triangles) > 0
    except Exception:
        return False


def cleanup_selection_output_cache_dirs(output_dir: Path):
    for name in ("rgb_imgs", "depth_imgs", "_cache_masks", "_cache_features"):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def resolve_param_file(param_dir: Path | None, filename: str) -> Path | None:
    if param_dir is None:
        return None
    path = param_dir / filename
    return path if path.exists() else None


def detect_scene_format(scene_dir: Path) -> str:
    """Detect supported extracted dataset layout for --scene."""
    if (
        (any(scene_dir.glob("*_rgb.jpg")) or any(scene_dir.glob("*_rgb.png")))
        and any(scene_dir.glob("*_depth.png"))
        and any(p for p in scene_dir.glob("*.json") if p.stem.isdigit())
    ):
        return "hope"
    if (
        (scene_dir / "results").exists()
        and (scene_dir / "traj.txt").exists()
    ) or (
        (scene_dir / "rgb").exists()
        and (scene_dir / "depth").exists()
        and (scene_dir / "camera_poses.txt").exists()
    ):
        return "replica"
    return "unknown"


def resolve_scene_path(scene_path: Path) -> Path:
    """Resolve common shorthand forms for dataset sequence paths."""
    if scene_path.exists():
        return scene_path

    if scene_path.name.isdigit():
        parent = scene_path.parent
        candidate = parent / f"{parent.name}_{scene_path.name}"
        if candidate.exists():
            return candidate

    return scene_path


def is_hope_sequence_dir(scene_dir: Path) -> bool:
    return (
        scene_dir.is_dir()
        and (any(scene_dir.glob("*_rgb.jpg")) or any(scene_dir.glob("*_rgb.png")))
        and any(scene_dir.glob("*_depth.png"))
        and any(p for p in scene_dir.glob("*.json") if p.stem.isdigit())
    )


def is_replica_sequence_dir(scene_dir: Path) -> bool:
    return scene_dir.is_dir() and detect_scene_format(scene_dir) == "replica"


def list_hope_sequence_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.is_dir():
        return []
    seq_dirs = []
    for p in sorted(root_dir.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.isdigit():
            continue
        if is_hope_sequence_dir(p):
            seq_dirs.append(p)
    return seq_dirs


def list_replica_sequence_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.is_dir():
        return []
    seq_dirs = []
    for p in sorted(root_dir.iterdir()):
        if not is_replica_sequence_dir(p):
            continue
        seq_dirs.append(p)
    return seq_dirs


def run_sam3d_on_objects(
    config_path: Path | None,
    output_dir: Path,
    run: str | None,
    tag: str,
    compile_model: bool,
    limit: int | None,
    intrinsics_path: Path | None = None,
    depth_scale: float = 1000.0,
):
    def cleanup_partial_outputs(obj_dir: Path):
        gc.collect()
        release_vram()

        partial_mesh = obj_dir / "obj_mesh.glb"
        if partial_mesh.exists():
            try:
                partial_mesh.unlink()
            except Exception:
                pass

    obj_dirs = iter_object_dirs(output_dir)
    if limit is not None:
        obj_dirs = obj_dirs[:limit]

    max_attempts = 5
    runner = None

    def build_runner():
        from hickory.reconstruction.sam3d_demo import Sam3DRunner
        return Sam3DRunner(
            config_path,
            run=run,
            tag=tag,
            compile_model=compile_model,
            intrinsics_path=intrinsics_path,
            depth_scale=depth_scale,
            verbose=False,
        )

    for obj_dir in obj_dirs:
        mesh_path = obj_dir / "obj_mesh.glb"
        if mesh_path.exists():
            if is_valid_triangle_mesh(mesh_path):
                print(f"[SKIP] {obj_dir.name}: {mesh_path.name} already exists.")
                continue
            print(f"[WARN] {obj_dir.name}: removing invalid {mesh_path.name}; rerunning SAM3D.")
            cleanup_partial_outputs(obj_dir)

        for attempt in range(1, max_attempts + 1):
            try:
                if runner is None:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        runner = build_runner()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    summary = runner.run_on_object(obj_dir)
                if summary is not None:
                    print(f"Translation (x, y, z): {summary['translation']}")
                    print(f"Rotation (Matrix/Quaternion): {summary['rotation']}")
                    print(f"Scale: {summary['scale']}")
                    print(f"Pose saved as: {summary['pose_path']}")
                    if summary.get("mesh_path") is not None:
                        print(f"Object mesh saved as {summary['mesh_path']}")
                break
            except Exception as exc:
                cleanup_partial_outputs(obj_dir)
                runner = None
                print(
                    f"[WARN] SAM3D failed on {obj_dir.name}: {exc} "
                    f"(attempt {attempt}/{max_attempts})."
                )
                if attempt == max_attempts:
                    print(f"[ERROR] Giving up on {obj_dir.name} after {max_attempts} attempts.")
                    break
                time.sleep(2.0)


def run_foundationpose_refinement(
    output_dir: Path,
    limit: int | None = None,
    depth_scale: float | None = None,
    save_detailed_results: bool = False,
):
    output_dir = output_dir.resolve()
    obj_dirs = iter_object_dirs(output_dir)
    if limit is not None:
        obj_dirs = obj_dirs[:limit]

    if len(obj_dirs) == 0:
        print(f"[WARN] No object folders found in {output_dir}; skipping FoundationPose refinement.")
        return

    camera_k_path = output_dir / "camera_K.txt"
    if not camera_k_path.exists():
        print(f"[WARN] Missing {camera_k_path}; skipping FoundationPose refinement.")
        return

    run_demo_path = FOUNDATIONPOSE_ROOT / "run_demo.py"
    if not run_demo_path.exists():
        print(f"[WARN] Missing {run_demo_path}; skipping FoundationPose refinement.")
        return

    if depth_scale is None:
        depth_scale = 1000.0

    pending_obj_dirs = []
    for obj_dir in obj_dirs:
        pose_path = obj_dir / "pose_foundation.txt"
        if pose_path.exists():
            print(f"[SKIP] {obj_dir.name}: {pose_path.name} already exists.")
            continue
        mesh_path = obj_dir / "obj_mesh.glb"
        if not is_valid_triangle_mesh(mesh_path):
            print(f"[WARN] {obj_dir.name}: missing or invalid obj_mesh.glb; skipping FoundationPose refinement.")
            continue
        pending_obj_dirs.append(obj_dir)

    if len(pending_obj_dirs) == 0:
        print(f"[INFO] All objects already have pose_foundation.txt in {output_dir}; skipping FoundationPose refinement.")
        return

    if limit is None and len(pending_obj_dirs) == len(obj_dirs):
        cmd = [
            sys.executable,
            str(run_demo_path),
            "--map_dir",
            str(output_dir),
            "--obj",
            "all",
            "--depth_scale",
            str(depth_scale),
        ]
        if save_detailed_results:
            cmd.append("--save_detailed_results")
        subprocess.run(cmd, check=True, cwd=str(run_demo_path.parent))
        return

    for obj_dir in pending_obj_dirs:
        cmd = [
            sys.executable,
            str(run_demo_path),
            "--map_dir",
            str(output_dir),
            "--obj",
            obj_dir.name,
            "--depth_scale",
            str(depth_scale),
        ]
        if save_detailed_results:
            cmd.append("--save_detailed_results")
        subprocess.run(cmd, check=True, cwd=str(run_demo_path.parent))

def run_sq_fitting(
    mesh_path: str,
    sq_params_path: str,
    grid_resolution: int = 64,
    level: float = 2.0,
    sq_mesh_resolution: int = 50,
) -> bool:
    """Run SQ fitting via lazy import after native runtime bootstrap in main()."""
    try:
        from hickory.sq.fitting import fit_sq_params_for_mesh

        fit_sq_params_for_mesh(
            mesh_path=mesh_path,
            sq_params_path=sq_params_path,
            grid_resolution=int(grid_resolution),
            level=float(level),
            sq_mesh_resolution=int(sq_mesh_resolution),
        )
        return True
    except Exception as e:
        print(f"SQ fitting failed for {mesh_path}: {e}")
        return False


def run_sq_expansion(
    output_dir: Path,
    limit: int | None = None,
    overwrite_existing: bool = False,
    f_quantile: float = 1.0,
    metric_margin: float = 0.0,
    max_gamma: float | None = 1.5,
    min_assigned_verts: int = 1,
) -> None:
    """Expand fitted SQ axes so obj_sq_params_expanded.npy is available after fitting."""
    from hickory.sq.expansion import (
        compute_sq_centroid_from_params,
        expand_sq_params_for_mesh,
        iter_object_dirs as iter_expansion_object_dirs,
        load_mesh_vertices_with_margin,
        output_path_for,
    )
    import numpy as np

    if not (0.0 < f_quantile <= 1.0):
        raise ValueError("--expansion-f-quantile must be in (0, 1].")
    if metric_margin < 0.0:
        raise ValueError("--expansion-metric-margin must be >= 0.0")
    if max_gamma is not None and max_gamma < 1.0:
        raise ValueError("--expansion-max-gamma must be >= 1.0 when set.")
    if min_assigned_verts < 1:
        raise ValueError("--expansion-min-assigned-verts must be >= 1.")

    processed = 0
    skipped_existing = 0
    for obj_dir, params_path, mesh_path in iter_expansion_object_dirs(
        output_dir,
        params_name="obj_sq_params.npy",
        mesh_name="obj_mesh.glb",
    ):
        if limit is not None and processed >= limit:
            break

        destination = output_path_for(
            params_path,
            output_name="obj_sq_params_expanded.npy",
            in_place=False,
        )
        if destination.exists() and not overwrite_existing:
            skipped_existing += 1
            print(f"Skipping {obj_dir.name}, expanded SQ params already exist.")
            continue

        params_array = np.load(params_path)
        sq_centroid = compute_sq_centroid_from_params(params_array)
        vertices = load_mesh_vertices_with_margin(
            mesh_path,
            metric_margin,
            centroid_world=sq_centroid,
        )
        expanded, stats = expand_sq_params_for_mesh(
            vertices,
            params_array,
            metric_margin=metric_margin,
            f_quantile=f_quantile,
            max_gamma=max_gamma,
            min_assigned_verts=min_assigned_verts,
        )
        np.save(destination, expanded)

        processed += 1
        gamma_values = [item["gamma"] for item in stats["per_sq"]]
        gamma_max = max(gamma_values) if gamma_values else 1.0
        skipped_count = sum(1 for item in stats["per_sq"] if item["skipped"])
        print(
            f"Expanded SQ for {obj_dir.name}: SQs={stats['num_sq']} "
            f"| verts={stats['num_vertices']} "
            f"| union max {stats['original_union_max']:.6f}->{stats['expanded_union_max']:.6f} "
            f"| largest gamma={gamma_max:.6f} | skipped={skipped_count}"
        )

    if processed == 0 and skipped_existing == 0:
        print(f"[WARN] No fitted SQ params and meshes found under {output_dir}; skipping SQ expansion.")
    elif skipped_existing > 0:
        print(f"Skipped {skipped_existing} object(s) with existing expanded SQ params.")


def release_vram():
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def main():
    view_selection_modes = ["best", "bbox", "bbox2d", "mask", "first"]
    parser = argparse.ArgumentParser(
        description="Run scene extraction, SAM3D reconstruction, pose refinement, and SQ fitting."
    )
    parser.add_argument(
        "run_or_config",
        nargs="?",
        default=None,
        help="Run name (e.g., map1) or YAML config path",
    )
    parser.add_argument(
        "--config",
        default="config_ros1.yaml",
        help="Default YAML config path when using a run name",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Path to scene folder or dataset root (Replica: dataset/Replica/office0_1 or dataset/Replica, HOPE: dataset/HOPE/0002).",
    )
    parser.add_argument(
        "--param",
        default=None,
        help=(
            "Dataset parameter directory. If fastsam.yaml and mapper.yaml are present, "
            "mapper extraction is used so those files are applied; without run_or_config, "
            "data.yaml is used as the config."
        ),
    )
    parser.add_argument(
        "--intrinsics",
        default=None,
        help="Path to intrinsics txt/npy for SAM3D (default: auto-detect camera_K.txt).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame rate used to assign timestamps from frame indices.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Optional depth scale override for extraction (e.g., 1000.0).",
    )
    parser.add_argument(
        "--use-mapper-extraction",
        "--use-ours-demo",
        dest="use_mapper_extraction",
        action="store_true",
        help="Use unified mapper extraction. Compatibility alias: --use-ours-demo. Also enabled automatically when --param has fastsam.yaml and mapper.yaml.",
    )
    parser.add_argument(
        "--seg-model",
        choices=["fastsam", "oneformer"],
        default="fastsam",
        help="Segmentation backend for extraction.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for extracted objects and reconstruction artifacts.",
    )
    parser.add_argument(
        "--view-selection",
        choices=view_selection_modes,
        default="best",
        help="Representative-view strategy used during extraction.",
    )
    parser.add_argument(
        "--all-view-selections",
        action="store_true",
        help="Run extraction/reconstruction once per view-selection mode, each in its own subfolder.",
    )
    parser.add_argument(
        "--output-group-by-suffix",
        action="store_true",
        help="If run is like '<sequence>_<group>', save to reconstruction/<group>/<sequence> (default: enabled).",
    )
    parser.add_argument(
        "--output-group-sep",
        default="_",
        help="Separator used by --output-group-by-suffix (default: '_').",
    )
    parser.add_argument("--run", default=None, help="Optional run name in the YAML config")
    parser.add_argument("--tag", default="hf", help="Checkpoint tag (e.g., hf)")
    parser.add_argument("--compile", action="store_true", help="Enable model compilation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of objects to reconstruct",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction and only run reconstruction",
    )
    parser.add_argument(
        "--skip-reconst",
        action="store_true",
        help="Skip extraction and SAM3D; only run SQ fitting",
    )
    parser.add_argument(
        "--skip-pose",
        action="store_true",
        help="Skip FoundationPose pose refinement after mesh scale refinement",
    )
    parser.add_argument(
        "--save-foundationpose-detailed-results",
        action="store_true",
        help="Persist FoundationPose debug artifacts under each obj_XXX/foundationpose folder.",
    )
    parser.add_argument(
        "--skip-fitting",
        action="store_true",
        help="Skip SQ fitting after reconstruction",
    )
    parser.add_argument(
        "--skip-expansion",
        action="store_true",
        help="Skip SQ expansion after fitting",
    )
    parser.add_argument(
        "--force-expansion",
        action="store_true",
        help="Recompute obj_sq_params_expanded.npy even when it already exists.",
    )
    parser.add_argument(
        "--expansion-f-quantile",
        type=float,
        default=1.0,
        help="Per-SQ IOF quantile used as the expansion target. 1.0 means strict max.",
    )
    parser.add_argument(
        "--expansion-metric-margin",
        type=float,
        default=0.0,
        help="Additive outward clearance in world units applied beyond the mesh surface.",
    )
    parser.add_argument(
        "--expansion-max-gamma",
        type=float,
        default=1.5,
        help="Upper bound on per-SQ expansion scale. Use <=0 to disable the cap.",
    )
    parser.add_argument(
        "--expansion-min-assigned-verts",
        type=int,
        default=1,
        help="Skip expansion for SQs with fewer assigned mesh vertices than this threshold.",
    )
    args = parser.parse_args()

    expansion_max_gamma = args.expansion_max_gamma
    if expansion_max_gamma is not None and expansion_max_gamma <= 0.0:
        expansion_max_gamma = None

    param_dir = Path(args.param) if args.param is not None else None
    if param_dir is not None and not param_dir.is_dir():
        raise FileNotFoundError(f"--param must be a directory: {param_dir}")
    param_data_config = resolve_param_file(param_dir, "data.yaml")
    fastsam_params_path = resolve_param_file(param_dir, "fastsam.yaml")
    mapper_params_path = resolve_param_file(param_dir, "mapper.yaml")
    has_mapper_param_files = fastsam_params_path is not None and mapper_params_path is not None
    use_mapper_extraction = args.use_mapper_extraction or has_mapper_param_files
    if param_dir is not None and use_mapper_extraction:
        missing_param_files = [
            name
            for name, path in (
                ("fastsam.yaml", fastsam_params_path),
                ("mapper.yaml", mapper_params_path),
            )
            if path is None
        ]
        if missing_param_files:
            missing = ", ".join(missing_param_files)
            raise FileNotFoundError(f"--param {param_dir} is missing required mapper extraction files: {missing}")

    effective_fps = args.fps
    if param_dir is not None and has_mapper_param_files and not args.use_mapper_extraction:
        print("[INFO] --param uses mapper extraction so fastsam.yaml/mapper.yaml can be applied.")

    run_name = args.run
    replica_scene = resolve_scene_path(Path(args.scene)) if args.scene else None
    candidate_path = Path(args.run_or_config) if args.run_or_config is not None else None
    config_path = None

    if replica_scene is None:
        if args.run_or_config is None and not args.skip_extract and param_data_config is None:
            raise ValueError("run_or_config is required unless --scene is provided.")
        if args.run_or_config is not None:
            if candidate_path.is_file() and candidate_path.suffix in {".yaml", ".yml"}:
                config_path = candidate_path
            else:
                run_name = args.run_or_config if run_name is None else run_name
                config_path = Path(args.config)
        elif param_data_config is not None:
            config_path = param_data_config

    hope_seq_dirs: list[Path] = []
    replica_seq_dirs: list[Path] = []
    if replica_scene is not None:
        hope_seq_dirs = list_hope_sequence_dirs(replica_scene)
        if len(hope_seq_dirs) == 0:
            replica_seq_dirs = list_replica_sequence_dirs(replica_scene)

    if args.output_dir is not None:
        base_output_dir = Path(args.output_dir)
    elif replica_scene is not None:
        scene_format_hint = detect_scene_format(replica_scene)
        if scene_format_hint == "hope":
            base_output_dir = Path("reconstruction") / replica_scene.parent.name / replica_scene.name
        else:
            base_output_dir = Path("reconstruction") / replica_scene.name
    elif args.output_group_by_suffix and run_name is not None and args.output_group_sep in run_name:
        seq_name, group_name = run_name.rsplit(args.output_group_sep, 1)
        base_output_dir = Path("reconstruction") / group_name / seq_name
    elif run_name is not None:
        base_output_dir = Path("reconstruction") / run_name
    else:
        base_output_dir = Path("reconstruction")

    scene_entries: list[tuple[Path | None, Path]] = []
    if replica_scene is not None:
        if len(hope_seq_dirs) > 0:
            for seq_dir in hope_seq_dirs:
                scene_entries.append((seq_dir, Path(seq_dir.name)))
            print(f"Detected HOPE dataset root with {len(scene_entries)} sequences: {replica_scene}")
        elif len(replica_seq_dirs) > 0:
            for seq_dir in replica_seq_dirs:
                scene_entries.append((seq_dir, Path(seq_dir.name)))
            print(f"Detected Replica dataset root with {len(scene_entries)} sequences: {replica_scene}")
        else:
            scene_entries.append((replica_scene, Path()))
    else:
        scene_entries.append((None, Path()))

    selected_view_modes = view_selection_modes if args.all_view_selections else [args.view_selection]

    shared_extract_prepared = False
    scene_jobs: list[tuple[str, Path | None, Path]] = []
    if args.all_view_selections and not args.skip_extract:
        from hickory.mapping.extraction_utils import (
            load_hope_dataset,
            load_replica_dataset,
            materialize_representative_views_from_cache,
            run_extract_from_data,
            run_extract_rosbag,
        )
        from hickory.mapping.scene_extractor import run_scene_extraction, run_scene_extraction_from_data
        from roman.params.data_params import DataParams

        cache_root = base_output_dir / "_extract_cache"
        for scene_idx, (scene_path, scene_suffix) in enumerate(scene_entries, start=1):
            cache_output_dir = cache_root / scene_suffix if str(scene_suffix) else cache_root
            scene_format = detect_scene_format(scene_path) if scene_path is not None else None

            if scene_path is not None:
                print(f"\n[{scene_idx}/{len(scene_entries)}] Preparing shared extraction cache for scene: {scene_path}")
            else:
                print(f"\n[{scene_idx}/{len(scene_entries)}] Preparing shared extraction cache for rosbag/config input")

            if cache_output_dir.exists() and any(cache_output_dir.iterdir()):
                print(f"[SKIP] Existing shared extraction cache is non-empty: {cache_output_dir}")
            else:
                if scene_path is not None:
                    if use_mapper_extraction:
                        if scene_format in {"replica", "hope"}:
                            run_scene_extraction(
                                scene_dir=scene_path,
                                output_dir=cache_output_dir,
                                fps=effective_fps,
                                view_selection=args.view_selection,
                                fastsam_params_path=fastsam_params_path or "params/demo/fastsam.yaml",
                                mapper_params_path=mapper_params_path or "params/demo/mapper.yaml",
                                seg_model=args.seg_model,
                                shared_cache_layout=True,
                                depth_scale_override=args.depth_scale,
                            )
                            dt = None
                            depth_scale = None
                        else:
                            raise FileNotFoundError(
                                f"Could not infer scene format from --scene path: {scene_path}. "
                                "If you meant a ReplicaCAD shorthand like scene0/1, use the extracted "
                                "sequence directory or rely on auto-resolution to scene0/scene0_1. "
                                "Expected Replica layout (results+traj.txt or rgb+depth+camera_poses.txt) "
                                "or HOPE layout (*_rgb.jpg|png, *_depth.png, and frame *.json files)."
                            )
                    elif scene_format == "replica":
                        img_data, depth_data, pose_data, dt, depth_scale = load_replica_dataset(
                            scene_dir=scene_path,
                            fps=args.fps,
                        )
                    elif scene_format == "hope":
                        img_data, depth_data, pose_data, dt, depth_scale = load_hope_dataset(
                            scene_dir=scene_path,
                            fps=args.fps,
                        )
                    else:
                        raise FileNotFoundError(
                            f"Could not infer scene format from --scene path: {scene_path}. "
                            "If you meant a ReplicaCAD shorthand like scene0/1, use the extracted "
                            "sequence directory or rely on auto-resolution to scene0/scene0_1. "
                            "Expected Replica layout (results+traj.txt or rgb+depth+camera_poses.txt) "
                            "or HOPE layout (*_rgb.jpg|png, *_depth.png, and frame *.json files)."
                        )
                    if not use_mapper_extraction:
                        run_extract_from_data(
                            img_data=img_data,
                            depth_data=depth_data,
                            pose_data=pose_data,
                            output_dir=cache_output_dir,
                            view_selection=args.view_selection,
                            seg_model=args.seg_model,
                            dt=dt,
                            depth_scale_override=args.depth_scale if args.depth_scale is not None else depth_scale,
                            shared_cache_layout=True,
                        )
                else:
                    if config_path is None:
                        raise ValueError("Could not resolve rosbag config path for extraction.")
                    if use_mapper_extraction:
                        params = DataParams.from_yaml(config_path, run=run_name)
                        run_scene_extraction_from_data(
                            img_data=params.load_img_data(),
                            depth_data=params.load_depth_data(),
                            pose_data=params.load_pose_data(),
                            output_dir=cache_output_dir,
                            dataset_name=f"{config_path}:{run_name}" if run_name is not None else str(config_path),
                            dt=params.dt,
                            view_selection=args.view_selection,
                            seg_model=args.seg_model,
                            fastsam_params_path=fastsam_params_path or "params/demo/fastsam.yaml",
                            mapper_params_path=mapper_params_path or "params/demo/mapper.yaml",
                            depth_scale_override=args.depth_scale,
                            shared_cache_layout=True,
                        )
                    else:
                        run_extract_rosbag(
                            config_path,
                            run=run_name,
                            output_dir=cache_output_dir,
                            view_selection=args.view_selection,
                            seg_model=args.seg_model,
                            depth_scale_override=args.depth_scale,
                            shared_cache_layout=True,
                        )

            for view_selection in selected_view_modes:
                mode_output_root = base_output_dir / view_selection
                output_dir = mode_output_root / scene_suffix if str(scene_suffix) else mode_output_root
                if output_dir.exists() and any(output_dir.iterdir()):
                    print(f"[SKIP] Existing output directory is non-empty: {output_dir}")
                    continue
                shutil.copytree(cache_output_dir, output_dir, dirs_exist_ok=True)
                materialize_representative_views_from_cache(output_dir, view_selection)
                cleanup_selection_output_cache_dirs(output_dir)
                scene_jobs.append((view_selection, scene_path, output_dir))

        shared_extract_prepared = True
    else:
        for view_selection in selected_view_modes:
            mode_output_root = base_output_dir / view_selection if args.all_view_selections else base_output_dir
            for scene_path, scene_suffix in scene_entries:
                output_dir = mode_output_root / scene_suffix if str(scene_suffix) else mode_output_root
                scene_jobs.append((view_selection, scene_path, output_dir))
    
    # record the running time
    t0 = time.time()

    for scene_idx, (view_selection, scene_path, output_dir) in enumerate(scene_jobs, start=1):
        mode_msg = f" [view_selection={view_selection}]"
        if scene_path is not None:
            print(f"\n[{scene_idx}/{len(scene_jobs)}] Processing scene: {scene_path}{mode_msg}")
        else:
            print(f"\n[{scene_idx}/{len(scene_jobs)}] Processing rosbag/config input{mode_msg}")

        if not args.skip_extract and not shared_extract_prepared and output_dir.exists() and any(output_dir.iterdir()):
            print(f"[SKIP] Existing output directory is non-empty: {output_dir}")
            continue

        scene_format = detect_scene_format(scene_path) if scene_path is not None else None

        if not args.skip_extract and not shared_extract_prepared:
            if scene_path is not None:
                from hickory.mapping.scene_extractor import run_scene_extraction
                from hickory.mapping.extraction_utils import load_replica_dataset, load_hope_dataset, run_extract_from_data
                if use_mapper_extraction:
                    if scene_format in {"replica", "hope"}:
                        run_scene_extraction(
                            scene_dir=scene_path,
                            output_dir=output_dir,
                            fps=effective_fps,
                            view_selection=view_selection,
                            fastsam_params_path=fastsam_params_path or "params/demo/fastsam.yaml",
                            mapper_params_path=mapper_params_path or "params/demo/mapper.yaml",
                            seg_model=args.seg_model,
                            depth_scale_override=args.depth_scale,
                        )
                    else:
                        raise FileNotFoundError(
                            f"Could not infer scene format from --scene path: {scene_path}. "
                            "If you meant a ReplicaCAD shorthand like scene0/1, use the extracted "
                            "sequence directory or rely on auto-resolution to scene0/scene0_1. "
                            "Expected Replica layout (results+traj.txt or rgb+depth+camera_poses.txt) "
                            "or HOPE layout (*_rgb.jpg|png, *_depth.png, and frame *.json files)."
                        )
                elif scene_format == "hope":
                    img_data, depth_data, pose_data, dt, depth_scale = load_hope_dataset(
                        scene_dir=scene_path,
                        fps=args.fps,
                    )
                elif scene_format == "replica":
                    img_data, depth_data, pose_data, dt, depth_scale = load_replica_dataset(
                        scene_dir=scene_path,
                        fps=args.fps,
                    )
                else:
                    raise FileNotFoundError(
                        f"Could not infer scene format from --scene path: {scene_path}. "
                        "If you meant a ReplicaCAD shorthand like scene0/1, use the extracted "
                        "sequence directory or rely on auto-resolution to scene0/scene0_1. "
                        "Expected Replica layout (results+traj.txt or rgb+depth+camera_poses.txt) "
                        "or HOPE layout (*_rgb.jpg|png, *_depth.png, and frame *.json files)."
                    )
                if not use_mapper_extraction:
                    run_extract_from_data(
                        img_data=img_data,
                        depth_data=depth_data,
                        pose_data=pose_data,
                        output_dir=output_dir,
                        view_selection=view_selection,
                        seg_model=args.seg_model,
                        dt=dt,
                        depth_scale_override=args.depth_scale if args.depth_scale is not None else depth_scale,
                    )
            else:
                if config_path is None:
                    raise ValueError("Could not resolve rosbag config path for extraction.")
                if use_mapper_extraction:
                    from hickory.mapping.scene_extractor import run_scene_extraction_from_data
                    from roman.params.data_params import DataParams
                    params = DataParams.from_yaml(config_path, run=run_name)
                    run_scene_extraction_from_data(
                        img_data=params.load_img_data(),
                        depth_data=params.load_depth_data(),
                        pose_data=params.load_pose_data(),
                        output_dir=output_dir,
                        dataset_name=f"{config_path}:{run_name}" if run_name is not None else str(config_path),
                        dt=params.dt,
                        view_selection=view_selection,
                        seg_model=args.seg_model,
                        fastsam_params_path=fastsam_params_path or "params/demo/fastsam.yaml",
                        mapper_params_path=mapper_params_path or "params/demo/mapper.yaml",
                        depth_scale_override=args.depth_scale,
                    )
                else:
                    from hickory.mapping.extraction_utils import run_extract_rosbag
                    run_extract_rosbag(
                        config_path,
                        run=run_name,
                        output_dir=output_dir,
                        view_selection=view_selection,
                        seg_model=args.seg_model,
                        depth_scale_override=args.depth_scale,
                    )

        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        if not args.skip_reconst:
            depth_scale = 1000.0
            depth_scale_txt = output_dir / "depth_scale.txt"
            if depth_scale_txt.exists():
                try:
                    depth_scale = float(depth_scale_txt.read_text().strip())
                except Exception:
                    pass
            elif scene_path is not None and scene_format == "replica":
                cam_params_candidates = [
                    scene_path / "cam_params.json",
                    scene_path.parent / "cam_params.json",
                ]
                for cp in cam_params_candidates:
                    if cp.exists():
                        try:
                            import json
                            cam_json = json.loads(cp.read_text())
                            cam = cam_json.get("camera", cam_json)
                            if "scale" in cam:
                                depth_scale = float(cam["scale"])
                        except Exception:
                            pass
                        break

            intrinsics_path = None
            if args.intrinsics is not None:
                intrinsics_path = Path(args.intrinsics)
            elif scene_path is not None:
                default_replica_intrinsics = scene_path.parent / "camera_K.txt"
                if default_replica_intrinsics.exists():
                    intrinsics_path = default_replica_intrinsics
            if intrinsics_path is None:
                output_k = output_dir / "camera_K.txt"
                if output_k.exists():
                    intrinsics_path = output_k

            run_sam3d_on_objects(
                config_path=config_path,
                output_dir=output_dir,
                run=run_name,
                tag=args.tag,
                compile_model=args.compile,
                limit=args.limit,
                intrinsics_path=intrinsics_path,
                depth_scale=depth_scale,
            )
            release_vram()

        if not args.skip_pose:
            refine_depth_scale = args.depth_scale
            if refine_depth_scale is None:
                depth_scale_txt = output_dir / "depth_scale.txt"
                if depth_scale_txt.exists():
                    try:
                        refine_depth_scale = float(depth_scale_txt.read_text().strip())
                    except Exception:
                        refine_depth_scale = None
            run_foundationpose_refinement(
                output_dir=output_dir,
                limit=args.limit,
                depth_scale=refine_depth_scale,
                save_detailed_results=args.save_foundationpose_detailed_results,
            )
            release_vram()

        if not args.skip_fitting:
            obj_dirs = iter_object_dirs(output_dir)
            for obj_dir in obj_dirs:
                mesh_path = obj_dir / "obj_mesh.glb"
                sq_params_path = obj_dir / "obj_sq_params.npy"

                if not mesh_path.exists():
                    print(f"[WARN] Missing scaled mesh for {obj_dir.name}; skipping SQ fitting.")
                    continue
                if sq_params_path.exists():
                    print(f"Skipping {obj_dir.name}, SQ params already exist.")
                    continue

                print(f"Fitting SQ for {obj_dir.name}...")
                run_sq_fitting(
                    str(mesh_path),
                    str(sq_params_path)
                )
                # Force garbage collection after each heavy object fitting to prevent RAM bloat
                gc.collect()

        if not args.skip_expansion:
            run_sq_expansion(
                output_dir=output_dir,
                limit=args.limit,
                overwrite_existing=args.force_expansion,
                f_quantile=args.expansion_f_quantile,
                metric_margin=args.expansion_metric_margin,
                max_gamma=expansion_max_gamma,
                min_assigned_verts=args.expansion_min_assigned_verts,
            )

    print(f"Total running time: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    ensure_native_runtime()
    main()
