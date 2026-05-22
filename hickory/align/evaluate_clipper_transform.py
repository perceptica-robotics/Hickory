import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import yaml

try:
    from roman.params.data_params import DataParams
    from roman.align.results import submaps_from_align_results
except ModuleNotFoundError:
    DataParams = None
    submaps_from_align_results = None


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
    return mat.astype(np.float64)


def load_demo_align_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Demo align result not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def select_demo_submap_pair(results, pair: tuple[int, int] | None):
    if pair is not None:
        return tuple(pair)

    assoc = np.asarray(results.clipper_num_associations, dtype=np.float64)
    if assoc.size == 0 or np.all(np.isnan(assoc)):
        raise ValueError("No valid association counts found in demo align results.")
    return np.unravel_index(np.nanargmax(assoc), assoc.shape)


def compute_demo_world_from_odom_b(results, T_wo_a: np.ndarray, pair: tuple[int, int] | None = None):
    if submaps_from_align_results is None:
        raise ModuleNotFoundError(
            "roman.align.results.submaps_from_align_results is unavailable. "
            "Run this script from the project environment."
        )

    idx_a, idx_b = select_demo_submap_pair(results, pair)
    assoc_count = float(results.clipper_num_associations[idx_a, idx_b])
    if assoc_count <= 0:
        raise ValueError(
            f"Selected demo submap pair {(idx_a, idx_b)} has no associations."
        )

    submaps = submaps_from_align_results(results)
    sm_a = submaps[0][idx_a]
    sm_b = submaps[1][idx_b]
    T_oa_ca = np.asarray(sm_a.pose_gravity_aligned, dtype=np.float64)
    T_ob_cb = np.asarray(sm_b.pose_gravity_aligned, dtype=np.float64)
    T_ca_cb = np.asarray(results.T_ij_hat_mat[idx_a, idx_b], dtype=np.float64)
    T_wo_b = T_wo_a @ T_oa_ca @ T_ca_cb @ np.linalg.inv(T_ob_cb)
    return T_wo_b, (idx_a, idx_b), assoc_count


def iter_valid_demo_submap_pairs(results):
    assoc = np.asarray(results.clipper_num_associations, dtype=np.float64)
    T_hat = np.asarray(results.T_ij_hat_mat, dtype=np.float64)
    valid_pairs = []
    for idx_a in range(assoc.shape[0]):
        for idx_b in range(assoc.shape[1]):
            assoc_count = float(assoc[idx_a, idx_b])
            if not np.isfinite(assoc_count) or assoc_count <= 0:
                continue
            if not np.all(np.isfinite(T_hat[idx_a, idx_b])):
                continue
            valid_pairs.append(((idx_a, idx_b), assoc_count))
    return valid_pairs


def load_camera_pose_trajectory(camera_poses_path: Path) -> dict[int, np.ndarray]:
    if not camera_poses_path.exists():
        raise FileNotFoundError(f"Camera pose file not found: {camera_poses_path}")

    records = {}
    for line in camera_poses_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 16:
            continue
        try:
            frame_idx = int(parts[0])
            T_wc = np.asarray([float(x) for x in parts[-16:]], dtype=np.float64).reshape(4, 4)
        except ValueError:
            continue
        records[frame_idx] = T_wc

    if not records:
        raise ValueError(f"No valid 4x4 camera poses found in {camera_poses_path}")
    return records


def load_camera_pose_records(camera_poses_path: Path) -> list[tuple[int, float, np.ndarray]]:
    if not camera_poses_path.exists():
        raise FileNotFoundError(f"Camera pose file not found: {camera_poses_path}")

    records = []
    for line in camera_poses_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 16:
            continue
        try:
            frame_idx = int(parts[0])
            timestamp = float(parts[1])
            T_wc = np.asarray([float(x) for x in parts[-16:]], dtype=np.float64).reshape(4, 4)
        except ValueError:
            continue
        records.append((frame_idx, timestamp, T_wc))

    if not records:
        raise ValueError(f"No valid 4x4 camera poses found in {camera_poses_path}")
    return records


def quaternion_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quaternion_slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


class TimestampedPoseData:
    def __init__(self, times: np.ndarray, translations: np.ndarray, quaternions_xyzw: np.ndarray, time_tol: float):
        self.times = times
        self.translations = translations
        self.quaternions_xyzw = quaternions_xyzw
        self.time_tol = time_tol

    def pose(self, timestamp: float) -> np.ndarray:
        idx = int(np.searchsorted(self.times, timestamp))
        if len(self.times) == 0:
            raise ValueError("No timestamps available.")

        # For in-range queries, interpolate between adjacent GT rows instead of snapping to
        # the nearest sample. Snapping with the Kimera 10 s time tolerance causes visible
        # pose drift even when reconstruction poses were generated from the same GT source.
        if idx < len(self.times) and np.isclose(self.times[idx], timestamp):
            q = self.quaternions_xyzw[idx]
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = quaternion_to_matrix_xyzw(q)
            T[:3, 3] = self.translations[idx]
            return T

        if idx > 0 and np.isclose(self.times[idx - 1], timestamp):
            q = self.quaternions_xyzw[idx - 1]
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = quaternion_to_matrix_xyzw(q)
            T[:3, 3] = self.translations[idx - 1]
            return T

        if idx <= 0:
            if abs(timestamp - self.times[0]) <= self.time_tol:
                q = self.quaternions_xyzw[0]
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = quaternion_to_matrix_xyzw(q)
                T[:3, 3] = self.translations[0]
                return T
            raise ValueError(f"Timestamp {timestamp} outside interpolation range.")

        if idx >= len(self.times):
            if abs(timestamp - self.times[-1]) <= self.time_tol:
                q = self.quaternions_xyzw[-1]
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = quaternion_to_matrix_xyzw(q)
                T[:3, 3] = self.translations[-1]
                return T
            raise ValueError(f"Timestamp {timestamp} outside interpolation range.")

        t0 = self.times[idx - 1]
        t1 = self.times[idx]
        if (timestamp < self.times[0] - self.time_tol) or (timestamp > self.times[-1] + self.time_tol):
            raise ValueError(f"Timestamp {timestamp} outside tolerated interpolation range.")

        alpha = 0.0 if t1 == t0 else (timestamp - t0) / (t1 - t0)
        trans = (1.0 - alpha) * self.translations[idx - 1] + alpha * self.translations[idx]
        quat = quaternion_slerp_xyzw(self.quaternions_xyzw[idx - 1], self.quaternions_xyzw[idx], alpha)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = quaternion_to_matrix_xyzw(quat)
        T[:3, 3] = trans
        return T


def load_kimera_gt_pose_data(config_path: Path) -> TimestampedPoseData:
    if config_path.suffix in {".yaml", ".yml"} and DataParams is not None:
        return DataParams.from_yaml(str(config_path)).load_pose_data()

    if config_path.suffix == ".csv":
        gt_path = config_path
        time_scale = 1e-9
        time_tol = 10.0
    else:
        config = yaml.safe_load(config_path.read_text())
        pose_cfg = config["pose_data"]
        gt_path = Path(pose_cfg["path"])
        time_scale = float(pose_cfg["csv_options"].get("timescale", 1.0))
        time_tol = float(pose_cfg.get("time_tol", 10.0))

    raw = np.loadtxt(str(gt_path), delimiter=",", comments="#")
    times = raw[:, 0].astype(np.float64) * time_scale
    translations = raw[:, 1:4].astype(np.float64)
    quaternions_xyzw = raw[:, [5, 6, 7, 4]].astype(np.float64)

    # extract_sam_bag uses pose_data.pose(t) from DataParams.load_pose_data(), and the
    # GT comparison elsewhere in the repo uses inv(T_camera_flu). For these Kimera CSVs,
    # that means converting the GT FLU/base pose into the camera frame with inv(T_FLURDF).
    T_camera_flu = np.eye(4, dtype=np.float64)
    T_camera_flu[:3, :3] = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    T_flu_camera = np.linalg.inv(T_camera_flu)

    camera_translations = []
    camera_quats = []
    for trans, quat_xyzw in zip(translations, quaternions_xyzw):
        T_w_flu = np.eye(4, dtype=np.float64)
        T_w_flu[:3, :3] = quaternion_to_matrix_xyzw(quat_xyzw)
        T_w_flu[:3, 3] = trans
        T_w_camera = T_w_flu @ T_flu_camera
        camera_translations.append(T_w_camera[:3, 3])

        R = T_w_camera[:3, :3]
        trace = np.trace(R)
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
        camera_quats.append(np.asarray([qx, qy, qz, qw], dtype=np.float64))

    return TimestampedPoseData(
        times=times,
        translations=np.asarray(camera_translations, dtype=np.float64),
        quaternions_xyzw=np.asarray(camera_quats, dtype=np.float64),
        time_tol=time_tol,
    )


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    T_err = np.linalg.inv(T_est) @ T_gt
    t_err = float(np.linalg.norm(T_err[:3, 3]))
    cos_theta = (np.trace(T_err[:3, :3]) - 1.0) / 2.0
    r_err = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    return t_err, r_err


def infer_scene_pair(est_transform: Path) -> tuple[str, str, str]:
    path_str = est_transform.as_posix()
    apt_match = re.search(r"/(apt\d+)/", path_str)
    apt_name = apt_match.group(1) if apt_match is not None else None

    stem = est_transform.stem
    suffix_match = re.search(r"_(sq|coarse|none)$", stem)
    if suffix_match is None:
        raise ValueError(
            f"Could not infer pose suffix from {est_transform.name}. "
            "Expected names like 1_best_2.txt."
        )
    pair_stem = stem[:suffix_match.start()]
    scene_dirs = sorted(
        p.name
        for p in est_transform.parent.iterdir()
        if p.is_dir() and (p / "camera_poses.txt").exists()
    )
    candidate_pairs = []
    for scene_name in scene_dirs:
        prefix = f"{scene_name}_"
        if not pair_stem.startswith(prefix):
            continue
        other_scene = pair_stem[len(prefix) :]
        if other_scene in scene_dirs:
            candidate_pairs.append((scene_name, other_scene))
    if len(candidate_pairs) == 1:
        scene_a, scene_b = candidate_pairs[0]
    else:
        split_idx = pair_stem.rfind("_")
        if split_idx < 0:
            raise ValueError(
                f"Could not infer scene pair from {est_transform.name}. "
                "Expected names like 1_best_2.txt."
            )
        scene_a = pair_stem[:split_idx]
        scene_b = pair_stem[split_idx + 1:]
    return apt_name, scene_a, scene_b


def infer_dataset_family(est_transform: Path) -> str:
    parts = est_transform.parts
    if "reconstruction" in parts and "ReplicaCAD" in parts:
        return "replicacad"
    if "reconstruction" in parts and "REPLICA" in parts:
        return "replica"
    if "reconstruction" in parts and any(part.startswith("Kimera") for part in parts):
        return "kimera"
    return "unknown"


def parse_replica_scene_name(scene_name: str) -> tuple[str, str]:
    match = re.match(r"apt_?(\d+)_([0-9]+)(?:_|$)", scene_name)
    if match is None:
        raise ValueError(
            f"Could not infer REPLICA apartment/sequence from scene '{scene_name}'. "
            "Expected a name like apt_0_1_test."
        )
    return f"apt{int(match.group(1))}", str(int(match.group(2)))


def resolve_replica_recon_scene(scene_name: str, recon_root: Path) -> tuple[Path, str, str]:
    recon_path = recon_root / scene_name / "camera_poses.txt"
    if not recon_path.exists():
        raise FileNotFoundError(f"Could not find reconstruction poses for scene '{scene_name}' under {recon_root}.")
    apt_name, seq_name = parse_replica_scene_name(scene_name)
    return recon_path, apt_name, seq_name


def resolve_recon_scene(apt_name: str, scene_name: str) -> tuple[Path, str]:
    recon_root = Path("reconstruction/ReplicaCAD") / apt_name

    direct_path = recon_root / scene_name / "camera_poses.txt"
    if direct_path.exists():
        return direct_path, scene_name

    scene_parts = tuple(part for part in scene_name.split("_") if part)
    if len(scene_parts) > 1:
        nested_from_name = recon_root.joinpath(*scene_parts) / "camera_poses.txt"
        if nested_from_name.exists():
            return nested_from_name, scene_parts[0]

    nested_matches = sorted(recon_root.glob(f"*/{scene_name}/camera_poses.txt"))
    if len(nested_matches) == 1:
        recon_path = nested_matches[0]
        return recon_path, recon_path.parent.parent.name
    if len(nested_matches) > 1:
        raise ValueError(
            f"Ambiguous reconstruction scene '{scene_name}' under {recon_root}: "
            f"{[str(p.parent) for p in nested_matches]}"
        )

    raise FileNotFoundError(
        f"Could not find reconstruction poses for scene '{scene_name}' under {recon_root}."
    )


def infer_apartment_from_recon(scene_a: str, scene_b: str) -> str:
    recon_root = Path("reconstruction/ReplicaCAD")
    candidates = []
    for apt_dir in sorted(p for p in recon_root.glob("apt*") if p.is_dir()):
        try:
            resolve_recon_scene(apt_dir.name, scene_a)
            resolve_recon_scene(apt_dir.name, scene_b)
        except (FileNotFoundError, ValueError):
            continue
        candidates.append(apt_dir.name)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"Could not infer apartment for scenes '{scene_a}' and '{scene_b}' from {recon_root}. "
            "Pass --apt-name or explicit --gt-seq-*-poses and --recon-seq-*-poses."
        )
    raise ValueError(
        f"Ambiguous apartment for scenes '{scene_a}' and '{scene_b}': {candidates}. "
        "Pass --apt-name or explicit --gt-seq-*-poses and --recon-seq-*-poses."
    )


def resolve_kimera_scene(scene_name: str, recon_root: Path) -> Path:
    recon_path = recon_root / scene_name / "camera_poses.txt"
    if not recon_path.exists():
        raise FileNotFoundError(f"Could not find reconstruction poses for scene '{scene_name}' under {recon_root}.")
    return recon_path


def infer_kimera_run_name(scene_name: str) -> str:
    candidates = [scene_name]
    for suffix in ("_test", "_train"):
        if scene_name.endswith(suffix):
            candidates.append(scene_name[: -len(suffix)])

    expanded = []
    seen = set()
    for run_name in candidates:
        variants = [run_name]
        stripped = run_name.rstrip("_")
        if stripped != run_name:
            variants.append(stripped)
        else:
            variants.append(f"{run_name}_")
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                expanded.append(variant)

    for run_name in expanded:
        if (Path("ros_config") / f"{run_name}.yaml").exists():
            return run_name
    raise FileNotFoundError(
        f"Could not infer a ros_config run for scene '{scene_name}'. "
        "Pass explicit GT and reconstruction pose paths."
    )


def infer_kimera_gt_config(recon_records_a, recon_records_b, run_name_a: str, run_name_b: str) -> tuple[Path, Path]:
    config_a_candidates = sorted(Path("ros_config").glob(f"{run_name_a}*.yaml"))
    config_b_candidates = sorted(Path("ros_config").glob(f"{run_name_b}*.yaml"))
    if not config_a_candidates or not config_b_candidates:
        raise FileNotFoundError(
            f"Missing ros_config YAML for '{run_name_a}' or '{run_name_b}'. "
            "Pass explicit GT and reconstruction pose paths."
        )

    t_min_a = min(ts for _, ts, _ in recon_records_a)
    t_max_a = max(ts for _, ts, _ in recon_records_a)
    t_min_b = min(ts for _, ts, _ in recon_records_b)
    t_max_b = max(ts for _, ts, _ in recon_records_b)

    matches = []
    for config_a in config_a_candidates:
        pose_a = load_kimera_gt_pose_data(config_a)
        if pose_a.times[0] > t_max_a or pose_a.times[-1] < t_min_a:
            continue
        for config_b in config_b_candidates:
            pose_b = load_kimera_gt_pose_data(config_b)
            if pose_b.times[0] > t_max_b or pose_b.times[-1] < t_min_b:
                continue
            matches.append((config_a, config_b))

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Could not infer matching Kimera GT configs for '{run_name_a}' and '{run_name_b}' "
            f"from ros_config using reconstruction timestamp ranges."
        )
    raise ValueError(
        f"Ambiguous Kimera GT configs for '{run_name_a}' and '{run_name_b}': "
        f"{[(str(a), str(b)) for a, b in matches]}"
    )


def resolve_pose_paths(args):
    if all(
        path is not None
        for path in (
            args.gt_seq_a_poses,
            args.gt_seq_b_poses,
            args.recon_seq_a_poses,
            args.recon_seq_b_poses,
        )
    ):
        return (
            args.gt_seq_a_poses,
            args.gt_seq_b_poses,
            args.recon_seq_a_poses,
            args.recon_seq_b_poses,
        )

    apt_name, scene_a, scene_b = infer_scene_pair(args.est_transform)
    dataset_family = infer_dataset_family(args.est_transform)

    if dataset_family == "replica":
        recon_root = args.est_transform.parent
        if args.recon_seq_a_poses is None:
            recon_seq_a, apt_a, seq_a = resolve_replica_recon_scene(scene_a, recon_root)
        else:
            recon_seq_a = args.recon_seq_a_poses
            apt_a, seq_a = parse_replica_scene_name(recon_seq_a.parent.name)

        if args.recon_seq_b_poses is None:
            recon_seq_b, apt_b, seq_b = resolve_replica_recon_scene(scene_b, recon_root)
        else:
            recon_seq_b = args.recon_seq_b_poses
            apt_b, seq_b = parse_replica_scene_name(recon_seq_b.parent.name)

        if apt_a != apt_b:
            raise ValueError(f"REPLICA scenes must be from the same apartment, got {apt_a} and {apt_b}.")

        gt_seq_a = args.gt_seq_a_poses or Path("dataset/REPLICA") / apt_a / seq_a / "camera_poses.txt"
        gt_seq_b = args.gt_seq_b_poses or Path("dataset/REPLICA") / apt_b / seq_b / "camera_poses.txt"
        return gt_seq_a, gt_seq_b, recon_seq_a, recon_seq_b

    if args.apt_name is not None:
        apt_name = args.apt_name
    if apt_name is None:
        apt_name = infer_apartment_from_recon(scene_a, scene_b)

    if args.recon_seq_a_poses is None:
        recon_seq_a, seq_a = resolve_recon_scene(apt_name, scene_a)
    else:
        recon_seq_a = args.recon_seq_a_poses
        seq_a = recon_seq_a.parent.name if recon_seq_a.parent.name.isdigit() else recon_seq_a.parent.parent.name

    if args.recon_seq_b_poses is None:
        recon_seq_b, seq_b = resolve_recon_scene(apt_name, scene_b)
    else:
        recon_seq_b = args.recon_seq_b_poses
        seq_b = recon_seq_b.parent.name if recon_seq_b.parent.name.isdigit() else recon_seq_b.parent.parent.name

    gt_seq_a = args.gt_seq_a_poses or Path("dataset/ReplicaCAD") / apt_name / seq_a / "camera_poses.txt"
    gt_seq_b = args.gt_seq_b_poses or Path("dataset/ReplicaCAD") / apt_name / seq_b / "camera_poses.txt"
    return gt_seq_a, gt_seq_b, recon_seq_a, recon_seq_b


def resolve_kimera_pose_sources(args):
    _, scene_a, scene_b = infer_scene_pair(args.est_transform)
    recon_root = args.est_transform.parent

    recon_seq_a = args.recon_seq_a_poses or resolve_kimera_scene(scene_a, recon_root)
    recon_seq_b = args.recon_seq_b_poses or resolve_kimera_scene(scene_b, recon_root)
    recon_records_a = load_camera_pose_records(recon_seq_a)
    recon_records_b = load_camera_pose_records(recon_seq_b)

    gt_seq_a = args.gt_seq_a_poses
    gt_seq_b = args.gt_seq_b_poses
    if gt_seq_a is None or gt_seq_b is None:
        run_name_a = infer_kimera_run_name(scene_a)
        run_name_b = infer_kimera_run_name(scene_b)
        config_a, config_b = infer_kimera_gt_config(recon_records_a, recon_records_b, run_name_a, run_name_b)
        gt_seq_a = gt_seq_a or config_a
        gt_seq_b = gt_seq_b or config_b

    return gt_seq_a, gt_seq_b, recon_seq_a, recon_seq_b


def estimate_world_from_odom(gt_records: dict[int, np.ndarray], recon_records: dict[int, np.ndarray]):
    common_frames = sorted(set(gt_records) & set(recon_records))
    if not common_frames:
        raise ValueError("No common frame indices between GT and reconstruction trajectories.")

    rows = []
    transforms = []
    for frame_idx in common_frames:
        T_wc_gt = gt_records[frame_idx]
        T_oc_recon = recon_records[frame_idx]
        T_wo = T_wc_gt @ np.linalg.inv(T_oc_recon)
        rows.append((frame_idx, T_wo))
        transforms.append(T_wo)
    return rows, transforms


def estimate_world_from_odom_pose_data(gt_pose_data, recon_records: list[tuple[int, float, np.ndarray]]):
    rows = []
    transforms = []
    for frame_idx, timestamp, T_oc_recon in recon_records:
        try:
            T_wc_gt = np.asarray(gt_pose_data.pose(timestamp), dtype=np.float64)
        except Exception:
            continue
        T_wo = T_wc_gt @ np.linalg.inv(T_oc_recon)
        rows.append((frame_idx, T_wo))
        transforms.append(T_wo)

    if not rows:
        raise ValueError("No overlapping timestamps between GT pose data and reconstruction trajectories.")
    return rows, transforms


def transform_stability(reference: np.ndarray, transforms: list[np.ndarray]):
    trans_deltas = []
    rot_deltas = []
    for T in transforms:
        t_err, r_err = pose_error(reference, T)
        trans_deltas.append(t_err)
        rot_deltas.append(r_err)
    return np.asarray(trans_deltas, dtype=np.float64), np.asarray(rot_deltas, dtype=np.float64)


def aggregate_reference_transform(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("No transforms available to aggregate.")

    translations = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
    rotations = np.asarray([T[:3, :3] for T in transforms], dtype=np.float64)

    T_ref = np.eye(4, dtype=np.float64)
    T_ref[:3, 3] = np.median(translations, axis=0)

    rot_mean = np.mean(rotations, axis=0)
    U, _, Vt = np.linalg.svd(rot_mean)
    R_ref = U @ Vt
    if np.linalg.det(R_ref) < 0.0:
        U[:, -1] *= -1.0
        R_ref = U @ Vt
    T_ref[:3, :3] = R_ref
    return T_ref


def relative_transform_from_world_alignments(
    T_wo_a: np.ndarray,
    T_wo_b: np.ndarray,
    transform_direction: str,
) -> np.ndarray:
    if transform_direction == "b_to_a":
        return np.linalg.inv(T_wo_a) @ T_wo_b
    return np.linalg.inv(T_wo_b) @ T_wo_a


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a CLIPPER seq-to-seq transform against the ground-truth relative "
            "transform inferred from GT world poses and reconstruction odometry poses."
        )
    )
    parser.add_argument(
        "--apt-name",
        type=str,
        default=None,
        help="ReplicaCAD apartment name (e.g. apt0). Use this when --est-transform path does not encode the apartment.",
    )
    parser.add_argument(
        "--gt-seq-a-poses",
        type=Path,
        default=None,
        help="GT world-frame camera_poses.txt for seq A. If omitted, infer from --est-transform.",
    )
    parser.add_argument(
        "--gt-seq-b-poses",
        type=Path,
        default=None,
        help="GT world-frame camera_poses.txt for seq B. If omitted, infer from --est-transform.",
    )
    parser.add_argument(
        "--recon-seq-a-poses",
        type=Path,
        default=None,
        help="Reconstruction odom-frame camera_poses.txt for seq A. If omitted, infer from --est-transform.",
    )
    parser.add_argument(
        "--recon-seq-b-poses",
        type=Path,
        default=None,
        help="Reconstruction odom-frame camera_poses.txt for seq B. If omitted, infer from --est-transform.",
    )
    parser.add_argument(
        "--est-transform",
        type=Path,
        default=None,
        help="Estimated 4x4 transform from clipper_solve.py.",
    )
    parser.add_argument(
        "--demo-align-pkl",
        type=Path,
        default=None,
        help=(
            "Optional demo align.pkl from demo.py. If provided, the script derives "
            "world<-odom_B from the selected submap pair instead of requiring --est-transform."
        ),
    )
    parser.add_argument(
        "--demo-submap-pair",
        type=int,
        nargs=2,
        default=None,
        metavar=("I", "J"),
        help=(
            "Submap pair to use from --demo-align-pkl. Defaults to the pair with the "
            "maximum number of associations."
        ),
    )
    parser.add_argument(
        "--demo-all-submap-pairs",
        action="store_true",
        help=(
            "When used with --demo-align-pkl, evaluate every valid submap pair with "
            "nonzero associations and report aggregate error statistics."
        ),
    )
    parser.add_argument(
        "--transform-direction",
        choices=["b_to_a", "a_to_b"],
        default="b_to_a",
        help="Direction of --est-transform. clipper_solve.py returns b_to_a.",
    )
    parser.add_argument(
        "--save-alignments-csv",
        type=Path,
        default=None,
        help="Optional CSV path to save relative-transform error and alignment drift diagnostics.",
    )
    args = parser.parse_args()

    if args.est_transform is None and args.demo_align_pkl is None:
        parser.error("Provide either --est-transform or --demo-align-pkl.")
    if args.est_transform is not None and args.demo_align_pkl is not None:
        parser.error("Use either --est-transform or --demo-align-pkl, not both.")

    dataset_family = infer_dataset_family(args.est_transform) if args.est_transform is not None else "unknown"
    if args.demo_align_pkl is not None:
        if not all(
            path is not None
            for path in (
                args.gt_seq_a_poses,
                args.gt_seq_b_poses,
                args.recon_seq_a_poses,
                args.recon_seq_b_poses,
            )
        ):
            raise ValueError(
                "--demo-align-pkl requires explicit --gt-seq-a-poses, --gt-seq-b-poses, "
                "--recon-seq-a-poses, and --recon-seq-b-poses."
            )
        gt_seq_a_poses, gt_seq_b_poses = args.gt_seq_a_poses, args.gt_seq_b_poses
        recon_seq_a_poses, recon_seq_b_poses = args.recon_seq_a_poses, args.recon_seq_b_poses
        kimera_gt_formats = {".csv", ".yaml", ".yml"}
        if gt_seq_a_poses.suffix in kimera_gt_formats or gt_seq_b_poses.suffix in kimera_gt_formats:
            dataset_family = "kimera"
    elif dataset_family == "kimera" and not all(
        path is not None
        for path in (
            args.gt_seq_a_poses,
            args.gt_seq_b_poses,
            args.recon_seq_a_poses,
            args.recon_seq_b_poses,
        )
    ):
        gt_seq_a_poses, gt_seq_b_poses, recon_seq_a_poses, recon_seq_b_poses = resolve_kimera_pose_sources(args)
    else:
        gt_seq_a_poses, gt_seq_b_poses, recon_seq_a_poses, recon_seq_b_poses = resolve_pose_paths(args)

    kimera_gt_formats = {".csv", ".yaml", ".yml"}
    use_kimera_pose_data = dataset_family == "kimera" and (
        gt_seq_a_poses.suffix in kimera_gt_formats or gt_seq_b_poses.suffix in kimera_gt_formats
    )

    if use_kimera_pose_data and not args.demo_align_pkl:
        gt_a = load_kimera_gt_pose_data(gt_seq_a_poses)
        gt_b = load_kimera_gt_pose_data(gt_seq_b_poses)
        recon_a_records = load_camera_pose_records(recon_seq_a_poses)
        recon_b_records = load_camera_pose_records(recon_seq_b_poses)

        rows_a, T_wo_a_all = estimate_world_from_odom_pose_data(gt_a, recon_a_records)
        rows_b, T_wo_b_all = estimate_world_from_odom_pose_data(gt_b, recon_b_records)
    elif use_kimera_pose_data and args.demo_align_pkl:
        gt_a = load_kimera_gt_pose_data(gt_seq_a_poses)
        gt_b = load_kimera_gt_pose_data(gt_seq_b_poses)
        recon_a_records = load_camera_pose_records(recon_seq_a_poses)
        recon_b_records = load_camera_pose_records(recon_seq_b_poses)

        rows_a, T_wo_a_all = estimate_world_from_odom_pose_data(gt_a, recon_a_records)
        rows_b, T_wo_b_all = estimate_world_from_odom_pose_data(gt_b, recon_b_records)
    else:
        gt_a = load_camera_pose_trajectory(gt_seq_a_poses)
        gt_b = load_camera_pose_trajectory(gt_seq_b_poses)
        recon_a = load_camera_pose_trajectory(recon_seq_a_poses)
        recon_b = load_camera_pose_trajectory(recon_seq_b_poses)

        rows_a, T_wo_a_all = estimate_world_from_odom(gt_a, recon_a)
        rows_b, T_wo_b_all = estimate_world_from_odom(gt_b, recon_b)

    ref_frame_a = rows_a[0][0]
    ref_frame_b = rows_b[0][0]
    T_wo_a_ref = aggregate_reference_transform(T_wo_a_all)
    T_wo_b_ref = aggregate_reference_transform(T_wo_b_all)

    a_trans_drift, a_rot_drift = transform_stability(T_wo_a_ref, T_wo_a_all)
    b_trans_drift, b_rot_drift = transform_stability(T_wo_b_ref, T_wo_b_all)

    selected_demo_pair = None
    selected_demo_assoc = None
    demo_pair_metrics = []
    T_rel_gt = relative_transform_from_world_alignments(
        T_wo_a=T_wo_a_ref,
        T_wo_b=T_wo_b_ref,
        transform_direction=args.transform_direction,
    )
    if args.demo_align_pkl is not None:
        results = load_demo_align_results(args.demo_align_pkl)
        if args.demo_all_submap_pairs:
            for pair, assoc_count in iter_valid_demo_submap_pairs(results):
                T_wo_b_pred, _, _ = compute_demo_world_from_odom_b(
                    results=results,
                    T_wo_a=T_wo_a_ref,
                    pair=pair,
                )
                T_rel_est_pair = relative_transform_from_world_alignments(
                    T_wo_a=T_wo_a_ref,
                    T_wo_b=T_wo_b_pred,
                    transform_direction=args.transform_direction,
                )
                rel_t_err_pair, rel_r_err_pair = pose_error(T_rel_est_pair, T_rel_gt)
                demo_pair_metrics.append(
                    {
                        "pair_i": int(pair[0]),
                        "pair_j": int(pair[1]),
                        "associations": float(assoc_count),
                        "relative_translation_error_m": float(rel_t_err_pair),
                        "relative_rotation_error_deg": float(rel_r_err_pair),
                    }
                )
            if not demo_pair_metrics:
                raise ValueError("No valid demo submap pairs with nonzero associations were found.")

            best_pair = min(
                demo_pair_metrics,
                key=lambda row: (row["relative_translation_error_m"], row["relative_rotation_error_deg"]),
            )
            selected_demo_pair = (best_pair["pair_i"], best_pair["pair_j"])
            selected_demo_assoc = best_pair["associations"]
            T_wo_b_pred, _, _ = compute_demo_world_from_odom_b(
                results=results,
                T_wo_a=T_wo_a_ref,
                pair=selected_demo_pair,
            )
            T_rel_est = relative_transform_from_world_alignments(
                T_wo_a=T_wo_a_ref,
                T_wo_b=T_wo_b_pred,
                transform_direction=args.transform_direction,
            )
        else:
            T_wo_b_pred, selected_demo_pair, selected_demo_assoc = compute_demo_world_from_odom_b(
                results=results,
                T_wo_a=T_wo_a_ref,
                pair=tuple(args.demo_submap_pair) if args.demo_submap_pair is not None else None,
            )
            T_rel_est = relative_transform_from_world_alignments(
                T_wo_a=T_wo_a_ref,
                T_wo_b=T_wo_b_pred,
                transform_direction=args.transform_direction,
            )
    else:
        T_rel_est = load_transform(args.est_transform)

    rel_t_err, rel_r_err = pose_error(T_rel_est, T_rel_gt)
    if demo_pair_metrics:
        rel_t_err = float(np.mean([row["relative_translation_error_m"] for row in demo_pair_metrics]))
        rel_r_err = float(np.mean([row["relative_rotation_error_deg"] for row in demo_pair_metrics]))

    print("Clipper Relative-Transform Evaluation")
    print(f"GT seq A poses:    {gt_seq_a_poses}")
    print(f"GT seq B poses:    {gt_seq_b_poses}")
    print(f"Recon seq A poses: {recon_seq_a_poses}")
    print(f"Recon seq B poses: {recon_seq_b_poses}")
    if args.demo_align_pkl is not None:
        print(f"Demo align pickle: {args.demo_align_pkl}")
        if args.demo_all_submap_pairs:
            print(f"Evaluated submap pairs: {len(demo_pair_metrics)}")
            print(f"Best submap pair: {selected_demo_pair}")
            print(f"Best pair associations: {selected_demo_assoc:.0f}")
        else:
            print(f"Selected submap pair: {selected_demo_pair}")
            print(f"Selected pair associations: {selected_demo_assoc:.0f}")
    else:
        print(f"Estimated transform: {args.est_transform}")
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
    if args.demo_align_pkl is not None:
        if args.demo_all_submap_pairs:
            pair_t_err = np.asarray([row["relative_translation_error_m"] for row in demo_pair_metrics], dtype=np.float64)
            pair_r_err = np.asarray([row["relative_rotation_error_deg"] for row in demo_pair_metrics], dtype=np.float64)
            print(
                f"  Evaluated all valid demo submap pairs; {len(demo_pair_metrics)} pairs "
                f"with nonzero associations"
            )
            print(
                f"  Mean translation / rotation error: "
                f"{np.mean(pair_t_err):.6f} m / {np.mean(pair_r_err):.6f} deg"
            )
            print(
                f"  Median translation / rotation error: "
                f"{np.median(pair_t_err):.6f} m / {np.median(pair_r_err):.6f} deg"
            )
            print(
                f"  Best pair {selected_demo_pair} error: "
                f"{np.min(pair_t_err):.6f} m / {pair_r_err[np.argmin(pair_t_err)]:.6f} deg"
            )
        else:
            print(
                f"  Estimated transform derived from demo submap pair {selected_demo_pair} "
                f"and aggregated seq A alignment over {len(rows_a)} frames"
            )
    else:
        print("  Estimated transform loaded from --est-transform")
    print(f"  Ground-truth direction: {args.transform_direction}")
    print(f"  Translation error (m): {rel_t_err:.6f}")
    print(f"  Rotation error (deg): {rel_r_err:.6f}")

    if args.save_alignments_csv is not None:
        args.save_alignments_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_alignments_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sequence",
                    "frame_idx",
                    "pair_i",
                    "pair_j",
                    "associations",
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
                    "pair_i": "",
                    "pair_j": "",
                    "associations": "",
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
                        "pair_i": "",
                        "pair_j": "",
                        "associations": "",
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
                        "pair_i": "",
                        "pair_j": "",
                        "associations": "",
                        "relative_translation_error_m": "",
                        "relative_rotation_error_deg": "",
                        "alignment_translation_drift_m": t_drift,
                        "alignment_rotation_drift_deg": r_drift,
                    }
                )
            for row in demo_pair_metrics:
                writer.writerow(
                    {
                        "sequence": "demo_pair",
                        "frame_idx": "",
                        "pair_i": row["pair_i"],
                        "pair_j": row["pair_j"],
                        "associations": row["associations"],
                        "relative_translation_error_m": row["relative_translation_error_m"],
                        "relative_rotation_error_deg": row["relative_rotation_error_deg"],
                        "alignment_translation_drift_m": "",
                        "alignment_rotation_drift_deg": "",
                    }
                )
        print(f"\nSaved relative-transform error and alignment drift CSV to: {args.save_alignments_csv}")


if __name__ == "__main__":
    main()
