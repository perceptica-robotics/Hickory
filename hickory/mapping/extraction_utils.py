import os
import sys
import warnings
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse
import json
import logging
import re
import torch
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

def configure_quiet_logs():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        from loguru import logger as loguru_logger
        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level="ERROR")
        loguru_logger.disable("sam3d_objects")
    except Exception:
        pass

configure_quiet_logs()


def iter_object_dirs(output_dir: Path):
    return sorted([p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("obj_")])

# ================= Data loading imports =================
from roman.params.data_params import DataParams
# Assumes these classes are provided by robotdatapy and installed in the env.
# from robotdatapy.data import ImgData, PoseData

# ================= FastSAM-related imports =================
from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.map.oneformer_wrapper import OneFormerWrapper

# ================= Global config =================
DEFAULT_OUTPUT_DIR = "reconstruction"

# Keep imgsz aligned to Ultralytics stride to avoid repetitive resize warnings.
# IMGSZ = (736, 1280) # (480, 640) for HOPE dataset & Kimera-Multi and (736, 1280) for ReplicaCAD dataset
IMGSZ = (480, 640) # (480, 640) for HOPE dataset & Kimera-Multi and (736, 1280) for ReplicaCAD dataset

# Depth params
MAX_DEPTH = 1.0 # 1.0 for HOPE, 5.0 for ReplicaCAD, 7.5 for Kimera dataset
MERGE_VOXEL_SIZE = 0.01 # # 0.01 for HOPE dataset and 0.05 for ReplicaCAD dataset
MIN_PC_POINTS = 50 # 50 for Kimera dataset and 100 for ReplicaCAD and 75 for Hope dataset

# FastSAM config
FASTSAM_WEIGHTS = "weights/FastSAM-x.pt"
FASTSAM_CONF_THRESH = 0.9 # 0.95 for HOPE dataset and 0.4 for ReplicaCAD dataset and 0.5 for Kimera dataset 

MIN_VIEWS = 2 # 2 for HOPE & Kimera-Multi and 6 for ReplicaCAD dataset

DEVICE = 'cuda'
# OneFormer config
ONEFORMER_MODEL = "shi-labs/oneformer_ade20k_dinat_large"
EXCLUDE_CLASSES = [
    "person",
    "light",
    "ceiling",
    "door",
    "sky",
    "glass",
    "building",
    "fence",
    "grass",
    "wall",
    "window",
    "earth",
    "ground",
    "floor",
    "stairs",
    "step",
    "stair",
    "stairway",
    "staircase",
    "road",""
    "route",
    "sidewalk", 
    "pavement",
    "bannister",
    "curtain",
    "column"
]
ONEFORMER_POST_THRESHOLD = 0.5
ONEFORMER_POST_MASK_THRESHOLD = 0.5
ONEFORMER_POST_OVERLAP_MASK_AREA_THRESHOLD = 0.3
ONEFORMER_FILTER_MIN_OVERLAP = 0.3

# Registration/matching params
MATCH_IOU_THRESHOLD = 0.1 # 0.05 for HOPE & ReplicaCAD dataset and 0.1 for Kimera dataset
FINAL_DEDUP_VOXEL_IOU_THRESHOLD = 0.05 # 0.1 for ReplicaCAD and 0.05 for Kimera & HOPEdataset
VIEW_SELECTION = "best"  # "best", "bbox", "bbox2d", "mask", or "first"
SEG_MODEL = "fastsam"  # "fastsam" or "oneformer"

MAX_FRAMES_PER_SECOND = 6 # 6 for Kimera-Multi

HOPE_POSE_MODE = "raw_inv"

def compute_aabb(points):
    if points is None or len(points) == 0:
        return None
    pts = np.asarray(points)
    return pts.min(axis=0), pts.max(axis=0)

def expand_aabb(aabb, margin):
    if aabb is None:
        return None
    min_a, max_a = aabb
    m = float(margin)
    return (min_a - m, max_a + m)

def aabb_iou(aabb_a, aabb_b):
    min_a, max_a = aabb_a
    min_b, max_b = aabb_b
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]
    vol_a = np.prod(max_a - min_a)
    vol_b = np.prod(max_b - min_b)
    union = vol_a + vol_b - inter_vol
    if union <= 0:
        return 0.0
    return inter_vol / union

def mask_bbox_area(mask: np.ndarray) -> int:
    if mask is None:
        return 0
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.ndim != 2 or not np.any(mask_bool):
        return 0
    ys, xs = np.nonzero(mask_bool)
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    return h * w

def voxel_downsample(points, voxel_size):
    if points is None or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(points)
    vox = np.floor(pts / voxel_size).astype(np.int64)
    _, idx = np.unique(vox, axis=0, return_index=True)
    return pts[idx]


def _prepare_image_for_imwrite(img: np.ndarray, is_depth: bool = False):
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.size == 0:
        return None

    if arr.ndim == 3:
        c = arr.shape[2]
        if c == 1:
            arr = arr[:, :, 0]
        elif c in (3, 4):
            pass
        elif c > 4:
            arr = arr[:, :, :4]
        else:
            arr = arr[:, :, 0]

    if is_depth:
        if arr.dtype == np.uint16:
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(arr, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def infer_saved_depth_scale(depth_sample: np.ndarray, configured_scale: float) -> float:
    arr = np.asarray(depth_sample)
    if np.issubdtype(arr.dtype, np.floating):
        return 1000.0
    return float(configured_scale)


def prepare_depth_for_png(depth_img: np.ndarray, configured_scale: float) -> np.ndarray | None:
    arr = np.asarray(depth_img)
    if arr.size == 0:
        return None
    if arr.dtype == np.uint16:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # FoundationPose consumes PNG depth with a divisor; persist float-meter
        # depth as uint16 millimeters so the saved scene is self-consistent.
        arr = np.rint(np.clip(arr, 0.0, np.iinfo(np.uint16).max / 1000.0) * 1000.0)
        return arr.astype(np.uint16)
    return _prepare_image_for_imwrite(arr, is_depth=True)


class SimpleCameraParams:
    def __init__(self, width, height, fx, fy, cx, cy, frame_id="camera"):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.frame_id = frame_id


def save_cam_params_json(output_dir, cam_params: SimpleCameraParams, depth_scale=None):
    output_dir = Path(output_dir)
    cam_json = {
        "camera": {
            "w": int(cam_params.width),
            "h": int(cam_params.height),
            "fx": float(cam_params.fx),
            "fy": float(cam_params.fy),
            "cx": float(cam_params.cx),
            "cy": float(cam_params.cy),
            "pose_frame": str(cam_params.frame_id),
        }
    }
    if depth_scale is not None:
        cam_json["camera"]["scale"] = float(depth_scale)
    cam_params_path = output_dir / "cam_params.json"
    cam_params_path.write_text(json.dumps(cam_json, indent=2) + "\n", encoding="utf-8")
    return cam_params_path


def camera_params_to_K(cam_params: SimpleCameraParams) -> np.ndarray:
    return np.array(
        [
            [cam_params.fx, 0.0, cam_params.cx],
            [0.0, cam_params.fy, cam_params.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class SimpleFrameData:
    def __init__(self, times, paths, loader, camera_params=None):
        self.times = np.asarray(times, dtype=float)
        self._paths = list(paths)
        self._loader = loader
        self.camera_params = camera_params

    def img(self, t):
        idx = int(np.argmin(np.abs(self.times - t)))
        return self._loader(self._paths[idx])

    def to_mp4(self, output_path, fps=30):
        write_frame_paths_to_mp4(self._paths, output_path, fps=fps, loader=self._loader)


def _prepare_frame_for_video(frame, size=None):
    if frame is None:
        return None

    frame = np.asarray(frame)
    if frame.size == 0:
        return None

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        return None

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            max_val = float(np.nanmax(frame)) if np.isfinite(frame).any() else 0.0
            if max_val <= 1.0:
                frame = frame * 255.0
        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if size is not None and (frame.shape[1], frame.shape[0]) != size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(frame)


def write_frame_paths_to_mp4(frame_paths, output_path, fps=30, loader=None):
    frame_paths = list(frame_paths)
    if len(frame_paths) == 0:
        print(f"[WARN] No RGB frames available; skipping overview video: {output_path}")
        return False

    if loader is None:
        loader = lambda p: cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

    first_frame = None
    for path in frame_paths:
        first_frame = _prepare_frame_for_video(loader(path))
        if first_frame is not None:
            break

    if first_frame is None:
        print(f"[WARN] Could not load any valid RGB frames; skipping overview video: {output_path}")
        return False

    h, w = first_frame.shape[:2]
    size = (w, h)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(output_path), fc, float(fps), size)
    if not video.isOpened():
        print(f"[WARN] OpenCV could not open VideoWriter; skipping overview video: {output_path}")
        return False

    written = 0
    for path in frame_paths:
        frame = _prepare_frame_for_video(loader(path), size=size)
        if frame is None:
            continue
        video.write(frame)
        written += 1
    video.release()

    if written == 0:
        print(f"[WARN] No valid frames were written to overview video: {output_path}")
        return False

    return True


class SimplePoseData:
    def __init__(self, times, poses):
        self.times = np.asarray(times, dtype=float)
        self._poses = list(poses)

    def pose(self, t):
        idx = int(np.argmin(np.abs(self.times - t)))
        return self._poses[idx]


def _label_aliases(label_name: str) -> set[str]:
    label_name = label_name.strip().lower()
    aliases = {label_name} if label_name else set()
    aliases.update(part.strip() for part in label_name.split(",") if part.strip())
    return aliases


def should_exclude_class(label_name: str, exclude_classes) -> bool:
    label_aliases = _label_aliases(label_name)
    for excluded in exclude_classes:
        if excluded.strip().lower() in label_aliases:
            return True
    return False


class OneFormerClassMaskFilter:
    def __init__(
        self,
        model_name=ONEFORMER_MODEL,
        device=DEVICE,
        filtered_classes=None,
        post_threshold=ONEFORMER_POST_THRESHOLD,
        post_mask_threshold=ONEFORMER_POST_MASK_THRESHOLD,
        post_overlap_mask_area_threshold=ONEFORMER_POST_OVERLAP_MASK_AREA_THRESHOLD
    ):
        self.device = device
        self.filtered_classes = set(c.strip().lower() for c in (filtered_classes or []))
        self.post_threshold = post_threshold
        self.post_mask_threshold = post_mask_threshold
        self.post_overlap_mask_area_threshold = post_overlap_mask_area_threshold

        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def run(self, img_bgr: np.ndarray, return_all_masks: bool = False):
        if len(img_bgr.shape) == 2:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=img_rgb, task_inputs=["panoptic"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        post = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[img_rgb.shape[:2]],
            threshold=self.post_threshold,
            mask_threshold=self.post_mask_threshold,
            overlap_mask_area_threshold=self.post_overlap_mask_area_threshold
        )[0]

        seg_map = post["segmentation"]
        segments_info = post["segments_info"]
        id2label = getattr(self.model.config, "id2label", {})

        if isinstance(seg_map, torch.Tensor):
            seg_map = seg_map.cpu().numpy()

        filtered_masks = []
        all_masks = []
        for info in segments_info:
            seg_id = info.get("id")
            if seg_id is None:
                continue
            mask = (seg_map == seg_id).astype(np.uint8)
            all_masks.append(mask)
            label_id = info.get("label_id")
            label_name = id2label.get(label_id, "") if label_id is not None else ""
            if not should_exclude_class(label_name, self.filtered_classes):
                continue
            filtered_masks.append(mask)

        if return_all_masks:
            return filtered_masks, all_masks
        return filtered_masks


def should_reject_mask_by_oneformer(
    candidate_mask: np.ndarray,
    filtered_masks,
    overlap_threshold: float = ONEFORMER_FILTER_MIN_OVERLAP,
) -> bool:
    if candidate_mask is None or len(filtered_masks) == 0:
        return False

    cand = candidate_mask.astype(bool)
    cand_area = int(cand.sum())
    if cand_area == 0:
        return False

    for fm in filtered_masks:
        filt = fm.astype(bool)
        filt_area = int(filt.sum())
        if filt_area == 0:
            continue

        inter = int(np.logical_and(cand, filt).sum())
        if inter == 0:
            continue

        # "contained": candidate substantially covered by filtered-class mask
        contain_ratio = inter / float(cand_area)
        if contain_ratio >= overlap_threshold:
            return True

        # "intersect": substantial overlap by IoU
        union = int(np.logical_or(cand, filt).sum())
        if union > 0 and (inter / float(union)) >= overlap_threshold:
            return True

    return False


def load_depth_png(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    return depth


def load_color_png(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Color image not found: {path}")
    return img


def _extract_frame_index(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else -1


def _parse_replica_traj(traj_path: Path):
    lines = [ln.strip() for ln in traj_path.read_text().splitlines() if ln.strip()]
    if len(lines) == 0:
        raise ValueError(f"Empty trajectory file: {traj_path}")

    poses = []
    # Replica commonly stores one 4x4 matrix (16 floats) per line.
    if all(len(ln.split()) == 16 for ln in lines):
        for ln in lines:
            vals = np.array([float(x) for x in ln.split()], dtype=np.float64)
            poses.append(vals.reshape(4, 4))
        return poses

    # Fallback for variants using 4 lines per matrix.
    vals = []
    for ln in lines:
        vals.extend(float(x) for x in ln.split())
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size % 16 != 0:
        raise ValueError(f"Could not parse 4x4 poses from {traj_path}")
    mats = vals.reshape(-1, 16)
    for row in mats:
        poses.append(row.reshape(4, 4))
    return poses


def _parse_extracted_camera_poses(poses_path: Path):
    records = []
    for ln in poses_path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 22:
            continue
        try:
            frame_idx = int(parts[0])
            rgb_time = float(parts[1])
            rgb_name = parts[2]
            depth_name = parts[4]
            mat_vals = [float(x) for x in parts[-16:]]
            T_wc = np.asarray(mat_vals, dtype=np.float64).reshape(4, 4)
            records.append((frame_idx, rgb_time, rgb_name, depth_name, T_wc))
        except Exception:
            continue
    records.sort(key=lambda x: x[0])
    return records


def load_replica_dataset(scene_dir, fps=30.0):
    scene_dir = Path(scene_dir)
    if not scene_dir.exists():
        raise FileNotFoundError(f"Replica scene folder not found: {scene_dir}")

    cam_params_candidates = [
        scene_dir / "cam_params.json",
        scene_dir.parent / "cam_params.json",
    ]
    cam_params_path = next((p for p in cam_params_candidates if p.exists()), None)
    if cam_params_path is None:
        raise FileNotFoundError(
            f"Missing cam_params.json. Tried: {cam_params_candidates}"
        )

    cam_json = json.loads(cam_params_path.read_text())
    cam = cam_json.get("camera", cam_json)
    cam_params = SimpleCameraParams(
        width=int(cam["w"]),
        height=int(cam["h"]),
        fx=float(cam["fx"]),
        fy=float(cam["fy"]),
        cx=float(cam["cx"]),
        cy=float(cam["cy"]),
    )

    results_dir = scene_dir / "results"
    traj_path = scene_dir / "traj.txt"
    extracted_rgb_dir = scene_dir / "rgb"
    extracted_depth_dir = scene_dir / "depth"
    extracted_pose_path = scene_dir / "camera_poses.txt"

    if results_dir.exists() and traj_path.exists():
        rgb_paths = sorted(results_dir.glob("frame*.jpg"), key=_extract_frame_index)
        depth_paths = sorted(results_dir.glob("depth*.png"), key=_extract_frame_index)
        if len(rgb_paths) == 0 or len(depth_paths) == 0:
            raise FileNotFoundError(f"No Replica frame/depth files found in {results_dir}")

        rgb_by_idx = {_extract_frame_index(p): p for p in rgb_paths if _extract_frame_index(p) >= 0}
        depth_by_idx = {_extract_frame_index(p): p for p in depth_paths if _extract_frame_index(p) >= 0}
        common_idx = sorted(set(rgb_by_idx.keys()) & set(depth_by_idx.keys()))
        if len(common_idx) == 0:
            raise ValueError("No aligned RGB/depth frame indices found in Replica results folder")

        poses = _parse_replica_traj(traj_path)
        max_pose_idx = len(poses) - 1
        common_idx = [i for i in common_idx if i <= max_pose_idx]
        if len(common_idx) == 0:
            raise ValueError("No frame indices overlap between results frames and traj poses")

        rgb_paths = [rgb_by_idx[i] for i in common_idx]
        depth_paths = [depth_by_idx[i] for i in common_idx]
        pose_list = [poses[i] for i in common_idx]
        times = np.asarray(common_idx, dtype=np.float64) / float(fps)
    elif extracted_rgb_dir.exists() and extracted_depth_dir.exists() and extracted_pose_path.exists():
        records = _parse_extracted_camera_poses(extracted_pose_path)
        if len(records) == 0:
            raise ValueError(f"No valid pose rows found in {extracted_pose_path}")

        rgb_paths = []
        depth_paths = []
        pose_list = []
        times = []
        for frame_idx, rgb_time, rgb_name, depth_name, T_wc in records:
            rgb_path = extracted_rgb_dir / rgb_name
            depth_path = extracted_depth_dir / depth_name
            if not rgb_path.exists() or not depth_path.exists():
                continue
            rgb_paths.append(rgb_path)
            depth_paths.append(depth_path)
            pose_list.append(T_wc)
            times.append(float(rgb_time))

        if len(times) == 0:
            raise ValueError(
                f"No rows from {extracted_pose_path} had matching rgb/depth files in "
                f"{extracted_rgb_dir} and {extracted_depth_dir}"
            )
        times = np.asarray(times, dtype=np.float64)
    else:
        raise FileNotFoundError(
            "Scene format not recognized. Expected either "
            f"Replica-style ({results_dir}, {traj_path}) or extracted-style "
            f"({extracted_rgb_dir}, {extracted_depth_dir}, {extracted_pose_path})."
        )

    img_data = SimpleFrameData(
        times=times,
        paths=rgb_paths,
        loader=load_color_png,
        camera_params=cam_params,
    )
    depth_data = SimpleFrameData(
        times=times,
        paths=depth_paths,
        loader=load_depth_png,
        camera_params=cam_params,
    )
    pose_data = SimplePoseData(
        times=times,
        poses=pose_list,
    )
    if len(times) > 1:
        dt = float(np.median(np.diff(times)))
    else:
        dt = (1.0 / float(fps)) if fps > 0 else None
    depth_scale = cam.get("scale", None)
    return img_data, depth_data, pose_data, dt, depth_scale


def _extract_hope_index(path: Path, suffix: str):
    m = re.match(rf"^(\d+)_{re.escape(suffix)}$", path.stem)
    return int(m.group(1)) if m else -1


def _extract_hope_json_index(path: Path):
    m = re.match(r"^(\d+)$", path.stem)
    return int(m.group(1)) if m else -1


def _load_hope_frame_pose_and_intrinsics(json_path: Path):
    data = json.loads(json_path.read_text())
    cam = data.get("camera", {})

    K = np.asarray(cam.get("intrinsics", []), dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics in {json_path}, got {K.shape}")

    T_wc = np.asarray(cam.get("extrinsics", []), dtype=np.float64)
    if T_wc.shape != (4, 4):
        raise ValueError(f"Expected 4x4 extrinsics in {json_path}, got {T_wc.shape}")
    if HOPE_POSE_MODE == "raw":
        pass
    elif HOPE_POSE_MODE == "raw_inv":
        T_wc = np.linalg.inv(T_wc)
    else:
        raise ValueError(
            f"Invalid HOPE_POSE_MODE={HOPE_POSE_MODE!r}. Use one of: raw, raw_inv."
        )

    width = int(cam.get("width", 0))
    height = int(cam.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Missing valid width/height in {json_path}")

    return T_wc, K, width, height


def load_hope_dataset(scene_dir, fps=30.0):
    scene_dir = Path(scene_dir)
    if not scene_dir.exists():
        raise FileNotFoundError(f"HOPE scene folder not found: {scene_dir}")

    rgb_paths_all = sorted(
        list(scene_dir.glob("*_rgb.jpg")) + list(scene_dir.glob("*_rgb.png")),
        key=lambda p: _extract_hope_index(p, "rgb"),
    )
    depth_paths_all = sorted(
        scene_dir.glob("*_depth.png"),
        key=lambda p: _extract_hope_index(p, "depth"),
    )
    json_paths_all = sorted(
        scene_dir.glob("*.json"),
        key=_extract_hope_json_index,
    )
    if len(rgb_paths_all) == 0 or len(depth_paths_all) == 0 or len(json_paths_all) == 0:
        raise FileNotFoundError(
            f"HOPE sequence is missing required files in {scene_dir}. "
            "Expected *_rgb.jpg|png, *_depth.png, and frame *.json files."
        )

    rgb_by_idx = {
        _extract_hope_index(p, "rgb"): p
        for p in rgb_paths_all
        if _extract_hope_index(p, "rgb") >= 0
    }
    depth_by_idx = {
        _extract_hope_index(p, "depth"): p
        for p in depth_paths_all
        if _extract_hope_index(p, "depth") >= 0
    }
    json_by_idx = {
        _extract_hope_json_index(p): p
        for p in json_paths_all
        if _extract_hope_json_index(p) >= 0
    }

    common_idx = sorted(set(rgb_by_idx.keys()) & set(depth_by_idx.keys()) & set(json_by_idx.keys()))
    if len(common_idx) == 0:
        raise ValueError(f"No aligned RGB/depth/json frame indices found in {scene_dir}")

    T0, K0, width0, height0 = _load_hope_frame_pose_and_intrinsics(json_by_idx[common_idx[0]])
    cam_params = SimpleCameraParams(
        width=width0,
        height=height0,
        fx=float(K0[0, 0]),
        fy=float(K0[1, 1]),
        cx=float(K0[0, 2]),
        cy=float(K0[1, 2]),
    )

    rgb_paths = []
    depth_paths = []
    pose_list = []
    times = []
    for idx in common_idx:
        rgb_paths.append(rgb_by_idx[idx])
        depth_paths.append(depth_by_idx[idx])
        T_wc, _, _, _ = _load_hope_frame_pose_and_intrinsics(json_by_idx[idx])
        pose_list.append(T_wc)
        times.append(float(idx) / float(fps))

    times = np.asarray(times, dtype=np.float64)
    img_data = SimpleFrameData(
        times=times,
        paths=rgb_paths,
        loader=load_color_png,
        camera_params=cam_params,
    )
    depth_data = SimpleFrameData(
        times=times,
        paths=depth_paths,
        loader=load_depth_png,
        camera_params=cam_params,
    )
    pose_data = SimplePoseData(
        times=times,
        poses=pose_list,
    )
    if len(times) > 1:
        dt = float(np.median(np.diff(times)))
    else:
        dt = (1.0 / float(fps)) if fps > 0 else None
    depth_scale = 1000.0
    return img_data, depth_data, pose_data, dt, depth_scale


VIEW_SELECTION_MODES = ("best", "bbox", "bbox2d", "mask", "first")


def select_view_record(history, view_selection):
    if view_selection not in VIEW_SELECTION_MODES:
        raise ValueError(
            "view_selection must be one of: " + ", ".join(VIEW_SELECTION_MODES)
        )
    if not history:
        return None

    if view_selection == "best":
        return max(history, key=lambda x: x.get('score', float("-inf")))

    if view_selection == "bbox":
        return max(
            history,
            key=lambda x: (
                x.get('volume', float("-inf")),
                x.get('score', float("-inf")),
            ),
        )

    if view_selection == "bbox2d":
        return max(
            history,
            key=lambda x: (
                x.get('mask_bbox_area', float("-inf")),
                x.get('mask_area', float("-inf")),
                x.get('score', float("-inf")),
            ),
        )

    if view_selection == "mask":
        return max(
            history,
            key=lambda x: (
                x.get('mask_area', float("-inf")),
                x.get('score', float("-inf")),
            ),
        )

    if view_selection == "first":
        def first_key(x):
            fid = x.get('frame_id')
            return (
                fid if fid is not None else float("inf"),
                str(x.get('frame_name', '')),
            )

        return min(history, key=first_key)

    raise ValueError(f"Unsupported view_selection: {view_selection}")


def _frame_stem(frame_name):
    if frame_name is None:
        return None
    return Path(str(frame_name)).stem


def _remove_file_if_exists(path: Path):
    if path.exists() and path.is_file():
        path.unlink()


def _copy_file_or_clear(src: Path | None, dst: Path):
    if src is None or not src.exists():
        _remove_file_if_exists(dst)
        return
    shutil.copyfile(src, dst)


def _find_rgb_frame_path(output_root: Path, frame_name: str | None) -> Path | None:
    frame_stem = _frame_stem(frame_name)
    if frame_stem is None:
        return None
    for rgb_dir in (output_root / "rgb", output_root / "rgb_imgs"):
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = rgb_dir / f"{frame_stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _find_depth_frame_path(output_root: Path, frame_name: str | None) -> Path | None:
    frame_stem = _frame_stem(frame_name)
    if frame_stem is None:
        return None
    for depth_dir in (output_root / "depth", output_root / "depth_imgs"):
        candidate = depth_dir / f"{frame_stem}.png"
        if candidate.exists():
            return candidate
    return None


def _parse_score_records(all_scores_path: Path):
    records = []
    if not all_scores_path.exists():
        return records

    for raw_line in all_scores_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('saved_id') or line.startswith('original_id'):
            continue
        if line.startswith('frame_name'):
            continue

        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 6:
            frame_name, frame_id_text, score_text, mask_ratio_text, bbox2d_ratio_text, bbox_vol_text = parts[:6]
        elif len(parts) >= 5:
            frame_name, score_text, mask_ratio_text, bbox2d_ratio_text, bbox_vol_text = parts[:5]
            frame_id_text = ''
        else:
            continue

        frame_id = None
        if frame_id_text:
            try:
                frame_id = int(frame_id_text)
            except ValueError:
                frame_id = None

        try:
            score = float(score_text)
        except ValueError:
            score = float('-inf')
        try:
            mask_ratio = float(mask_ratio_text)
        except ValueError:
            mask_ratio = float('-inf')
        try:
            bbox2d_ratio = float(bbox2d_ratio_text)
        except ValueError:
            bbox2d_ratio = float('-inf')
        try:
            bbox_volume = float(bbox_vol_text)
        except ValueError:
            bbox_volume = float('-inf')

        records.append({
            'frame_name': frame_name,
            'frame_id': frame_id,
            'score': score,
            'mask_area': mask_ratio,
            'mask_bbox_area': bbox2d_ratio,
            'volume': bbox_volume,
        })

    return records


def _load_camera_pose_index(poses_path: Path):
    pose_map = {}
    if not poses_path.exists():
        return pose_map

    with open(poses_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 19:
                continue
            frame_name = parts[2]
            matrix_vals = parts[3:19]
            try:
                matrix = np.asarray([float(v) for v in matrix_vals], dtype=np.float64).reshape(4, 4)
            except ValueError:
                continue
            pose_map[frame_name] = matrix
    return pose_map


def _shared_object_cache_dir(output_root: Path, cache_name: str, obj_dir: Path) -> Path:
    return output_root / cache_name / obj_dir.name


def _materialize_object_selected_outputs(
    output_root: Path,
    obj_dir: Path,
    selected_frame_name: str | None,
    selected_pose=None,
    selected_feature=None,
):
    frame_stem = _frame_stem(selected_frame_name)
    rgb_src = _find_rgb_frame_path(output_root, selected_frame_name)
    depth_src = _find_depth_frame_path(output_root, selected_frame_name)

    shared_masks_dir = _shared_object_cache_dir(output_root, '_cache_masks', obj_dir)
    shared_features_dir = _shared_object_cache_dir(output_root, '_cache_features', obj_dir)

    mask_src = (shared_masks_dir / f'{frame_stem}.png') if frame_stem is not None else None
    if mask_src is not None and not mask_src.exists():
        mask_src = (obj_dir / 'all_masks' / f'{frame_stem}.png') if frame_stem is not None else None
    if mask_src is not None and not mask_src.exists():
        mask_src = None

    feature_src = (shared_features_dir / f'{frame_stem}.npy') if frame_stem is not None else None
    if feature_src is not None and not feature_src.exists():
        feature_src = (obj_dir / 'all_features' / f'{frame_stem}.npy') if frame_stem is not None else None
    if feature_src is not None and not feature_src.exists():
        feature_src = None

    _copy_file_or_clear(mask_src, obj_dir / 'mask.png')
    _copy_file_or_clear(rgb_src, obj_dir / 'view.png')
    _copy_file_or_clear(depth_src, obj_dir / 'depth.png')

    feature_dst = obj_dir / 'dino_feature.npy'
    if feature_src is not None:
        shutil.copyfile(feature_src, feature_dst)
    elif selected_feature is not None:
        np.save(feature_dst, np.asarray(selected_feature))
    else:
        _remove_file_if_exists(feature_dst)

    pose_dst = obj_dir / 'camera_pose.txt'
    if selected_pose is not None:
        np.savetxt(str(pose_dst), np.asarray(selected_pose), fmt='%.8f')
    else:
        _remove_file_if_exists(pose_dst)


def materialize_representative_views_from_cache(output_dir, view_selection):
    output_root = Path(output_dir)
    pose_map = _load_camera_pose_index(output_root / 'camera_poses.txt')

    for obj_dir in iter_object_dirs(output_root):
        records = _parse_score_records(obj_dir / 'all_scores.txt')
        selected_record = select_view_record(records, view_selection)
        if selected_record is None:
            continue
        selected_frame_name = selected_record.get('frame_name')
        selected_pose = pose_map.get(selected_frame_name)
        _materialize_object_selected_outputs(
            output_root=output_root,
            obj_dir=obj_dir,
            selected_frame_name=selected_frame_name,
            selected_pose=selected_pose,
            selected_feature=None,
        )


class SceneLibrary:
    def __init__(
        self,
        iou_threshold=0.5,
        view_selection=VIEW_SELECTION,
        allow_tblr_edges=None,
    ):
        self.objects = []
        self.iou_threshold = iou_threshold
        self.obj_counter = 0
        self.allow_tblr_edges = (
            [True, True, True, True]
            if allow_tblr_edges is None
            else list(allow_tblr_edges)
        )
        if view_selection not in VIEW_SELECTION_MODES:
            raise ValueError(
                "view_selection must be one of: " + ", ".join(VIEW_SELECTION_MODES)
            )
        self.view_selection = view_selection

    def process_observation(self, obs, rgb_img, depth_img, frame_name, frame_id, forced_match_idx=None):

        # 1. Read score and features
        if isinstance(obs.semantic_descriptor, dict):
            score = obs.semantic_descriptor.get('score', 0.0)
            feature_vector = obs.semantic_descriptor.get('mean', None)
        else:
            score = 0.0
            feature_vector = None

        centroid = np.mean(obs.transformed_points, axis=0)
        current_aabb = compute_aabb(obs.transformed_points)
        if current_aabb is None:
            return None
        cur_min, cur_max = current_aabb
        current_volume = np.prod(cur_max - cur_min)

        # 2. Build full-size RGBA for history saving and best-view candidates
        current_full_rgba = None
        current_mask_area = 0
        current_mask_area_ratio = 0.0
        current_mask_bbox_area = 0
        current_mask_bbox_area_ratio = 0.0
        if obs.mask is not None:
            mask_uint8 = (obs.mask * 255).astype(np.uint8)
            current_mask_area = int(np.count_nonzero(obs.mask))
            current_mask_area_ratio = (
                float(current_mask_area) / float(obs.mask.size) if obs.mask.size > 0 else 0.0
            )
            current_mask_bbox_area = int(mask_bbox_area(obs.mask))
            current_mask_bbox_area_ratio = (
                float(current_mask_bbox_area) / float(obs.mask.size) if obs.mask.size > 0 else 0.0
            )
            current_full_rgba = np.dstack((rgb_img, mask_uint8))

        current_depth = depth_img.copy() if depth_img is not None else None

        # 3. Object matching logic (expanded AABB IoU only)
        match_idx = -1
        best_iou = 0.0
        best_aabb_idx = -1
        aabb_margin = 0.1
        current_aabb_expanded = expand_aabb(current_aabb, aabb_margin)
        if forced_match_idx is None:
            for idx, obj in enumerate(self.objects):
                if obj.get('aabb') is None:
                    continue
                iou = aabb_iou(expand_aabb(obj['aabb'], aabb_margin), current_aabb_expanded)
                if iou > best_iou:
                    best_iou = iou
                    best_aabb_idx = idx

        if forced_match_idx is not None and 0 <= forced_match_idx < len(self.objects):
            match_idx = forced_match_idx
        else:
            if best_iou >= self.iou_threshold:
                match_idx = best_aabb_idx

        if match_idx is not None and match_idx >= 0:
            existing_obj = self.objects[match_idx]

            found_in_history = False
            for record in existing_obj['view_history']:
                if record['frame_id'] == frame_id:
                    found_in_history = True
                    if (current_mask_area, score) > (
                        record.get('mask_area', -1),
                        record.get('score', float('-inf')),
                    ):
                        record['score'] = score
                        record['volume'] = current_volume
                        record['mask_area'] = current_mask_area
                        record['mask_area_ratio'] = current_mask_area_ratio
                        record['mask_bbox_area'] = current_mask_bbox_area
                        record['mask_bbox_area_ratio'] = current_mask_bbox_area_ratio
                        record['feature'] = feature_vector
                        record['rgba'] = current_full_rgba
                        record['depth'] = current_depth
                        record['pose'] = np.asarray(obs.pose).copy()
                        record['points_world'] = voxel_downsample(
                            obs.transformed_points, MERGE_VOXEL_SIZE
                        )
                    break

            if not found_in_history:
                existing_obj['view_history'].append({
                    'frame_id': frame_id,
                    'frame_name': frame_name,
                    'score': score,
                    'volume': current_volume,
                    'mask_area': current_mask_area,
                    'mask_area_ratio': current_mask_area_ratio,
                    'mask_bbox_area': current_mask_bbox_area,
                    'mask_bbox_area_ratio': current_mask_bbox_area_ratio,
                    'feature': feature_vector,
                    'rgba': current_full_rgba,
                    'depth': current_depth,
                    'pose': np.asarray(obs.pose).copy(),
                    'points_world': voxel_downsample(obs.transformed_points, MERGE_VOXEL_SIZE),
                })

            obj_min, obj_max = existing_obj['aabb']
            new_aabb = (np.minimum(obj_min, cur_min), np.maximum(obj_max, cur_max))

            if score >= existing_obj.get('best_score', float('-inf')):
                existing_obj['best_score'] = score
                existing_obj['centroid'] = centroid
                existing_obj['obs'] = obs
                existing_obj['aabb'] = new_aabb
                existing_obj['frame_name'] = frame_name
                existing_obj['best_feature'] = feature_vector
                existing_obj['best_rgba_full'] = current_full_rgba
                existing_obj['best_rgb_original'] = rgb_img
                existing_obj['best_depth'] = current_depth
                existing_obj['best_points_world'] = voxel_downsample(
                    obs.transformed_points, MERGE_VOXEL_SIZE
                )
            else:
                existing_obj['aabb'] = new_aabb
            return match_idx

        new_obj = {
            'id': self.obj_counter,
            'centroid': centroid,
            'aabb': current_aabb,
            'best_score': score,
            'obs': obs,
            'frame_name': frame_name,
            'best_feature': feature_vector,
            'best_rgba_full': current_full_rgba,
            'best_rgb_original': rgb_img,
            'best_depth': current_depth,
            'best_points_world': voxel_downsample(obs.transformed_points, MERGE_VOXEL_SIZE),
            'view_history': [{
                'frame_id': frame_id,
                'frame_name': frame_name,
                'score': score,
                'volume': current_volume,
                'mask_area': current_mask_area,
                'mask_area_ratio': current_mask_area_ratio,
                'mask_bbox_area': current_mask_bbox_area,
                'mask_bbox_area_ratio': current_mask_bbox_area_ratio,
                'feature': feature_vector,
                'rgba': current_full_rgba,
                'depth': current_depth,
                'pose': np.asarray(obs.pose).copy(),
                'points_world': voxel_downsample(obs.transformed_points, MERGE_VOXEL_SIZE),
            }],
        }
        self.objects.append(new_obj)
        self.obj_counter += 1
        return self.obj_counter - 1

    def _select_view_record(self, obj):
        history = obj.get('view_history', [])
        return select_view_record(history, self.view_selection)

    def save_library(self, output_dir, shared_cache_layout=False):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        eligible_objects = [
            obj for obj in self.objects
            if MIN_VIEWS <= 1
            or len(
                {
                    record.get('frame_id')
                    for record in obj.get('view_history', [])
                    if record.get('frame_id') is not None
                }
            )
            >= MIN_VIEWS
        ]
        eligible_objects = self._deduplicate_objects_by_voxel_iou(
            eligible_objects,
            voxel_size=MERGE_VOXEL_SIZE,
            iou_threshold=FINAL_DEDUP_VOXEL_IOU_THRESHOLD,
        )
        print(
            f"\nSaving {len(eligible_objects)} objects "
            f"(view_selection={self.view_selection} + metadata)..."
        )

        eligible_objects = sorted(eligible_objects, key=lambda obj: obj.get('id', 0))

        for saved_id, obj in enumerate(eligible_objects):
            obj_dir = output_path / f"obj_{saved_id:03d}"
            obj_dir.mkdir(exist_ok=True)
            selected_record = self._select_view_record(obj)

            with open(str(obj_dir / 'all_scores.txt'), 'w', encoding='utf-8') as f:
                f.write(f"saved_id, {saved_id}\n")
                f.write(f"original_id, {obj.get('id', saved_id)}\n")
                f.write(
                    'frame_name, frame_id, score, mask_area_ratio, '
                    'bbox2d_area_ratio, bbox_volume_cm3\n'
                )
                sorted_history = sorted(
                    obj['view_history'],
                    key=lambda x: (
                        x.get('score', float('-inf')),
                        x.get('mask_area', float('-inf')),
                    ),
                    reverse=True,
                )
                for record in sorted_history:
                    bbox_volume_cm3 = float(record.get('volume', 0.0)) * 1e6
                    frame_id = record.get('frame_id')
                    frame_id_str = '' if frame_id is None else str(int(frame_id))
                    f.write(
                        f"{record['frame_name']}, {frame_id_str}, {record['score']:.6f}, "
                        f"{record.get('mask_area_ratio', 0.0):.6f}, "
                        f"{record.get('mask_bbox_area_ratio', 0.0):.6f}, "
                        f"{bbox_volume_cm3:.6f}\n"
                    )

            if shared_cache_layout:
                all_masks_dir = _shared_object_cache_dir(output_path, '_cache_masks', obj_dir)
                all_features_dir = _shared_object_cache_dir(output_path, '_cache_features', obj_dir)
                all_masks_dir.mkdir(parents=True, exist_ok=True)
                all_features_dir.mkdir(parents=True, exist_ok=True)
            else:
                all_masks_dir = obj_dir / 'all_masks'
                all_masks_dir.mkdir(exist_ok=True)
                all_features_dir = obj_dir / 'all_features'
                all_features_dir.mkdir(exist_ok=True)
            history_by_time = sorted(
                obj.get('view_history', []),
                key=lambda r: (
                    r.get('frame_id') if r.get('frame_id') is not None else 10**9,
                    r.get('frame_name', ''),
                ),
            )
            for idx, record in enumerate(history_by_time):
                frame_name = str(record.get('frame_name', f'view_{idx:05d}'))
                frame_stem = Path(frame_name).stem

                rgba = record.get('rgba')
                if rgba is not None:
                    rgba_arr = np.asarray(rgba)
                    if rgba_arr.ndim == 3 and rgba_arr.shape[2] >= 4:
                        alpha = rgba_arr[:, :, 3]
                        mask_img = _prepare_image_for_imwrite(alpha, is_depth=False)
                        if mask_img is not None:
                            cv2.imwrite(str(all_masks_dir / f'{frame_stem}.png'), mask_img)

                feature = record.get('feature')
                if feature is not None:
                    np.save(all_features_dir / f'{frame_stem}.npy', np.asarray(feature))

            selected_points_world = (
                selected_record.get('points_world')
                if selected_record is not None
                else None
            )
            if not shared_cache_layout:
                _materialize_object_selected_outputs(
                    output_root=output_path,
                    obj_dir=obj_dir,
                    selected_frame_name=selected_record.get('frame_name') if selected_record is not None else None,
                    selected_pose=selected_record.get('pose') if selected_record is not None else None,
                    selected_feature=selected_record.get('feature') if selected_record is not None else None,
                )

            points_world = selected_points_world
            if (not shared_cache_layout) and points_world is not None and len(points_world) > 0:
                pts = np.asarray(points_world, dtype=np.float64)
                pcd_world = o3d.geometry.PointCloud()
                pcd_world.points = o3d.utility.Vector3dVector(pts)
                o3d.io.write_point_cloud(str(obj_dir / 'point_cloud_world.ply'), pcd_world)
                np.save(obj_dir / 'point_cloud_world.npy', pts.astype(np.float32))

    def _deduplicate_objects_by_voxel_iou(self, objects, voxel_size, iou_threshold):
        if objects is None or len(objects) <= 1:
            return objects

        def get_points(obj):
            pts = obj.get('best_points_world')
            if pts is None or len(pts) == 0:
                obs = obj.get('obs', None)
                pts = getattr(obs, 'transformed_points', None) if obs is not None else None
            return None if pts is None else np.asarray(pts)

        def size_in_voxels(pts):
            if pts is None or len(pts) == 0:
                return 0
            vox = np.floor(pts / float(voxel_size)).astype(np.int64)
            return int(np.unique(vox, axis=0).shape[0])

        def voxel_set(pts):
            if pts is None or len(pts) == 0:
                return set()
            vox = np.floor(pts / float(voxel_size)).astype(np.int64)
            return set(map(tuple, np.unique(vox, axis=0)))

        sizes = [size_in_voxels(get_points(obj)) for obj in objects]
        voxel_sets = [voxel_set(get_points(obj)) for obj in objects]
        order = sorted(range(len(objects)), key=lambda i: sizes[i], reverse=True)
        keep = np.ones(len(objects), dtype=bool)

        for i_pos, i in enumerate(order):
            if not keep[i]:
                continue
            vox_i = voxel_sets[i]
            if not vox_i:
                continue
            for j in order[i_pos + 1:]:
                if not keep[j]:
                    continue
                vox_j = voxel_sets[j]
                if not vox_j:
                    continue

                inter = len(vox_i & vox_j)
                if inter == 0:
                    continue

                containment_j = float(inter) / float(len(vox_j))
                if containment_j >= float(iou_threshold):
                    keep[j] = False

        deduped = [obj for idx, obj in enumerate(objects) if keep[idx]]
        removed = len(objects) - len(deduped)
        print(
            f"Deduplicated by voxel containment >= {iou_threshold:.2f}: "
            f"kept {len(deduped)} / {len(objects)} (removed {removed})"
        )
        return deduped

def get_sync_index(target_t, timestamp_array, tol=0.05):
    """
    Find the index closest to a target timestamp.
    """
    idx = np.searchsorted(timestamp_array, target_t)
    if idx > 0 and (idx == len(timestamp_array) or abs(target_t - timestamp_array[idx-1]) < abs(target_t - timestamp_array[idx])):
        idx = idx - 1
    if idx < len(timestamp_array):
        if abs(timestamp_array[idx] - target_t) < tol:
            return idx
    return None


def run_extract_from_data(
    img_data,
    depth_data,
    pose_data,
    output_dir=DEFAULT_OUTPUT_DIR,
    seg_model=SEG_MODEL,
    view_selection=VIEW_SELECTION,
    view_selection_seed=0,
    dt=None,
    max_frames_override=None,
    depth_scale_override=None,
    shared_cache_layout=False,
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract camera intrinsics
    # ImgData provides intrinsics via camera_params (K/width/height)
    try:
        if hasattr(img_data, 'camera_params') and img_data.camera_params is not None:
            cp = img_data.camera_params
            cam_params = SimpleCameraParams(
                width=cp.width, height=cp.height, 
                fx=cp.fx, fy=cp.fy, cx=cp.cx, cy=cp.cy
            )
        elif hasattr(img_data, 'K') and img_data.K is not None:
            K = np.asarray(img_data.K, dtype=np.float32)
            if K.shape != (3, 3):
                raise ValueError(f"Expected img_data.K shape (3, 3), got {K.shape}")
            width = getattr(img_data, "width", None)
            height = getattr(img_data, "height", None)
            if width is None or height is None:
                first_img = img_data.img(img_data.times[0])
                height, width = first_img.shape[:2]
            cam_params = SimpleCameraParams(
                width=width,
                height=height,
                fx=K[0, 0],
                fy=K[1, 1],
                cx=K[0, 2],
                cy=K[1, 2],
            )
        elif len(img_data.times) > 0:
            # Fallback: derive resolution from the first frame
            K = np.eye(3)
            first_img = img_data.img(img_data.times[0])
            height, width = first_img.shape[:2]
             
            cam_params = SimpleCameraParams(
                width=width, height=height, 
                fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2]
            )
        else:
            print("No image data found.")
            return
    except Exception as e:
        print(f"Failed to extract camera intrinsics: {e}")
        return

    K = camera_params_to_K(cam_params)
    camera_k_path = Path(output_dir) / "camera_K.txt"
    np.savetxt(str(camera_k_path), K, fmt="%.8f")
    print(f"Saved camera intrinsics: {camera_k_path}")

    oneformer_class_filter = None

    # 3. Initialize segmentation model
    if seg_model == "fastsam":
        print("Step 2: Initializing FastSAM...")
        wrapper = FastSAMWrapper(
            weights=FASTSAM_WEIGHTS,
            conf=FASTSAM_CONF_THRESH,
            imgsz=IMGSZ,
            device=DEVICE,
            mask_downsample_factor=8,
            use_pointcloud=False,
            verbose=False,
        )
    elif seg_model == "oneformer":
        print("Step 2: Initializing OneFormer...")
        wrapper = OneFormerWrapper(
            model_name=ONEFORMER_MODEL,
            device=DEVICE,
            mask_downsample_factor=8,
            use_pointcloud=False,
            exclude_classes=EXCLUDE_CLASSES,
            post_threshold=ONEFORMER_POST_THRESHOLD,
            post_mask_threshold=ONEFORMER_POST_MASK_THRESHOLD,
            post_overlap_mask_area_threshold=ONEFORMER_POST_OVERLAP_MASK_AREA_THRESHOLD
        )
    else:
        raise ValueError("seg_model must be 'fastsam', or 'oneformer'.")

    if seg_model == "fastsam":
        print("Step 2.1: Initializing OneFormer class filter for FastSAM masks...")
        oneformer_class_filter = OneFormerClassMaskFilter(
            model_name=ONEFORMER_MODEL,
            device=DEVICE,
            filtered_classes=EXCLUDE_CLASSES,
            post_threshold=ONEFORMER_POST_THRESHOLD,
            post_mask_threshold=ONEFORMER_POST_MASK_THRESHOLD,
            post_overlap_mask_area_threshold=ONEFORMER_POST_OVERLAP_MASK_AREA_THRESHOLD
        )
    
    # Infer depth units from the data type (uint16 is usually millimeters)
    first_depth_frame = depth_data.img(depth_data.times[0])
    if depth_scale_override is not None:
        depth_scale = float(depth_scale_override)
    else:
        depth_scale = 1000.0 if first_depth_frame.dtype == np.uint16 else 1.0
    saved_depth_scale = infer_saved_depth_scale(first_depth_frame, depth_scale)
    print(f"Using depth scale: {depth_scale}")
    print(f"Saving depth PNGs with scale: {saved_depth_scale}")
    (Path(output_dir) / "depth_scale.txt").write_text(f"{saved_depth_scale:.10f}\n", encoding="utf-8")
    save_cam_params_json(output_dir, cam_params, depth_scale=saved_depth_scale)

    wrapper.setup_rgbd_params(
        depth_cam_params=cam_params,
        max_depth=MAX_DEPTH,
        depth_scale=depth_scale,
        voxel_size=MERGE_VOXEL_SIZE,
    )

    if seg_model == "fastsam":
        wrapper.setup_filtering(
            # area_bounds=[img_area / (30**2), img_area / (30**2)],
            allow_tblr_edges=[True, True, False, False], # for Kimera-Multi dataset
            # allow_tblr_edges=[False, False, False, False],
            semantics='dino',
        )
    elif seg_model == "oneformer":
        wrapper.setup_filtering(
            allow_tblr_edges=[True, True, False, False], # for Kimera-Multi dataset
            # allow_tblr_edges=[False, False, False, False],
            semantics='dino',
        )

    # 4. Initialize scene library
    scene_lib = SceneLibrary(
        iou_threshold=MATCH_IOU_THRESHOLD,
        view_selection=view_selection,
        allow_tblr_edges=wrapper.allow_tblr_edges,
    )

    # 5. Main loop: iterate RGB frames
    # Uniformly sample to a max rate to limit workload
    all_times = np.asarray(img_data.times)
    if len(all_times) > 1:
        duration_s = float(all_times[-1] - all_times[0])
    else:
        duration_s = 0.0
    max_frames = max(1, int(np.ceil(duration_s * MAX_FRAMES_PER_SECOND))) if duration_s > 0 else len(all_times)
    if max_frames_override is not None:
        max_frames = min(max_frames, max(1, int(max_frames_override)))
    if len(all_times) > max_frames:
        sample_idx = np.linspace(0, len(all_times) - 1, max_frames, dtype=int)
    else:
        sample_idx = np.arange(len(all_times))
    sample_times = all_times[sample_idx]

    print(f"Step 3: Processing {len(sample_times)} frames from bag...")
    rgb_dir = Path(output_dir) / "rgb"
    depth_dir = Path(output_dir) / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_records = []
    T_odom_from_world = None
    
    for i, t in enumerate(tqdm(sample_times)):
        frame_name = f"frame_{i:05d}.png"
        
        # A. Get current RGB frame
        img_bgr = img_data.img(t)
        if img_bgr is None: continue
        
        # B. Find synced depth frame
        sync_tol = dt if dt is not None else 0.05
        depth_idx = get_sync_index(t, depth_data.times, tol=sync_tol)
        if depth_idx is None: continue
        depth_img = depth_data.img(depth_data.times[depth_idx])
        if depth_img is None or np.asarray(depth_img).size == 0:
            print("Detected empty depth_img and skipped this frame...")
            continue
        
        # Force resolution alignment
        h_rgb, w_rgb = img_bgr.shape[:2]
        h_d, w_d = depth_img.shape[:2]
        if (h_d != h_rgb) or (w_d != w_rgb):
            depth_img = cv2.resize(depth_img, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
            
        # C. Get interpolated 4x4 pose (T_WC)
        try:
            # pose_data.pose(t) interpolates based on timestamps in the TF tree
            T_WC_raw = np.asarray(pose_data.pose(t), dtype=np.float64)
        except Exception:
            continue
        if T_WC_raw.shape != (4, 4):
            continue
        if T_odom_from_world is None:
            # Rebase the trajectory so the first valid camera pose becomes identity.
            T_odom_from_world = np.linalg.inv(T_WC_raw)
        T_WC = T_odom_from_world @ T_WC_raw

        pose_records.append((i, t, frame_name, T_WC))
            
        # Save sampled RGB/depth frames so representative views can be rematerialized later.
        cv2.imwrite(str(rgb_dir / f"frame_{i:05d}.png"), img_bgr)
        depth_img_to_save = prepare_depth_for_png(depth_img, saved_depth_scale)
        if depth_img_to_save is not None:
            cv2.imwrite(str(depth_dir / f"frame_{i:05d}.png"), depth_img_to_save)

        # 6. Run FastSAM
        observations, _ = wrapper.run(t=t, pose=T_WC, img=img_bgr, depth_data=depth_img)
        filtered_class_masks = []
        oneformer_all_masks = []
        if oneformer_class_filter is not None:
            filtered_class_masks, oneformer_all_masks = oneformer_class_filter.run(
                img_bgr, return_all_masks=True
            )
        
        frame_records = []
        for obs in observations:
            if should_reject_mask_by_oneformer(obs.mask, filtered_class_masks):
                continue
            if obs.point_cloud is None or len(obs.point_cloud) < MIN_PC_POINTS:
                continue
            obj_idx = scene_lib.process_observation(
                obs, img_bgr, depth_img, frame_name, i)
            if obj_idx is not None and obs.mask is not None:
                frame_records.append({"mask": obs.mask, "obj_idx": obj_idx})

    # 7. Save results
    print(f"Step 4: Saving representative-view results (mode={view_selection})...")
    scene_lib.save_library(output_dir, shared_cache_layout=shared_cache_layout)
    print("Done!")

    # Save all sampled camera poses
    poses_path = Path(output_dir) / "camera_poses.txt"
    with open(poses_path, "w") as f:
        f.write("# frame_index timestamp frame_name T_WC_first_view_relative (row-major 4x4)\n")
        for frame_idx, ts, fname, T_WC in pose_records:
            flat = " ".join([f"{v:.8f}" for v in np.asarray(T_WC).reshape(-1)])
            f.write(f"{frame_idx} {ts:.6f} {fname} {flat}\n")

    # Save an RGB overview video after saving results (avoid being deleted by save_library)
    output_video = Path(output_dir) / f"rgb_views_{Path(output_dir).name}.mp4"
    output_video.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving RGB overview video: {output_video}")
    rgb_frame_paths = sorted((Path(output_dir) / "rgb_imgs").glob("frame_*.png"))
    if len(rgb_frame_paths) == 0:
        rgb_frame_paths = sorted((Path(output_dir) / "rgb").glob("frame_*.png"))
    if not write_frame_paths_to_mp4(rgb_frame_paths, str(output_video), fps=30):
        img_data.to_mp4(str(output_video), fps=30)


def run_extract_rosbag(
    config_path,
    run=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    seg_model=SEG_MODEL,
    view_selection=VIEW_SELECTION,
    view_selection_seed=0,
    depth_scale_override=None,
    shared_cache_layout=False,
):
    # 1. Load config via DataParams
    print(f"Loading data from config: {config_path}")
    params = DataParams.from_yaml(config_path, run=run)
    img_data = params.load_img_data()
    depth_data = params.load_depth_data()
    pose_data = params.load_pose_data()
    run_extract_from_data(
        img_data=img_data,
        depth_data=depth_data,
        pose_data=pose_data,
        output_dir=output_dir,
        seg_model=seg_model,
        view_selection=view_selection,
        view_selection_seed=view_selection_seed,
        dt=params.dt,
        depth_scale_override=depth_scale_override,
        shared_cache_layout=shared_cache_layout,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract objects from bag data")
    parser.add_argument("config", nargs="?", default=None, help="YAML config path")
    parser.add_argument("--run", default=None, help="Optional run name in the YAML config")
    parser.add_argument(
        "--replica-scene",
        default=None,
        help="Path to Replica scene folder (e.g., dataset/Replica/office0).",
    )
    parser.add_argument(
        "--replica-fps",
        type=float,
        default=15.0,
        help="Replica frame rate used to assign timestamps from frame indices.",
    )
    parser.add_argument(
        "--hope-scene",
        default=None,
        help="Path to HOPE scene folder (e.g., dataset/HOPE/0002).",
    )
    parser.add_argument(
        "--hope-fps",
        type=float,
        default=15.0,
        help="HOPE frame rate used to assign timestamps from frame indices.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output folder for results")
    parser.add_argument(
        "--seg-model",
        choices=["fastsam", "oneformer"],
        default=SEG_MODEL,
        help="Segmentation backend to use",
    )
    parser.add_argument(
        "--view-selection",
        choices=["best", "bbox", "bbox2d", "mask", "first"],
        default=VIEW_SELECTION,
        help="How to choose the representative per-object view for saved outputs",
    )
    parser.add_argument(
        "--view-selection-seed",
        type=int,
        default=0,
        help="Unused legacy argument kept for CLI compatibility.",
    )
    args = parser.parse_args()

    if args.replica_scene is not None:
        img_data, depth_data, pose_data, dt, depth_scale = load_replica_dataset(
            scene_dir=args.replica_scene,
            fps=args.replica_fps,
        )
        print(f"Loaded Replica dataset from: {args.replica_scene}")
        print(f"Frames: rgb={len(img_data.times)}, depth={len(depth_data.times)}, poses={len(pose_data.times)}")
        run_extract_from_data(
            img_data=img_data,
            depth_data=depth_data,
            pose_data=pose_data,
            output_dir=args.output_dir,
            seg_model=args.seg_model,
            view_selection=args.view_selection,
            view_selection_seed=args.view_selection_seed,
            dt=dt,
            depth_scale_override=depth_scale,
        )
    elif args.hope_scene is not None:
        img_data, depth_data, pose_data, dt, depth_scale = load_hope_dataset(
            scene_dir=args.hope_scene,
            fps=args.hope_fps,
        )
        print(f"Loaded HOPE dataset from: {args.hope_scene}")
        print(f"Frames: rgb={len(img_data.times)}, depth={len(depth_data.times)}, poses={len(pose_data.times)}")
        run_extract_from_data(
            img_data=img_data,
            depth_data=depth_data,
            pose_data=pose_data,
            output_dir=args.output_dir,
            seg_model=args.seg_model,
            view_selection=args.view_selection,
            view_selection_seed=args.view_selection_seed,
            dt=dt,
            depth_scale_override=depth_scale,
        )
    else:
        if args.config is None:
            parser.error(
                "Provide one input source: positional 'config', --replica-scene, or "
                "--hope-scene."
            )
        run_extract_rosbag(
            args.config,
            run=args.run,
            output_dir=args.output_dir,
            seg_model=args.seg_model,
            view_selection=args.view_selection,
            view_selection_seed=args.view_selection_seed,
        )
