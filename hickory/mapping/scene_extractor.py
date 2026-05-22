import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from robotdatapy.exceptions import NoDataNearTimeException
from tqdm import tqdm

from hickory.mapping.extraction_utils import (
    DEFAULT_OUTPUT_DIR,
    EXCLUDE_CLASSES,
    ONEFORMER_MODEL,
    ONEFORMER_POST_MASK_THRESHOLD,
    ONEFORMER_POST_OVERLAP_MASK_AREA_THRESHOLD,
    ONEFORMER_POST_THRESHOLD,
    OneFormerClassMaskFilter,
    SimpleCameraParams,
    VIEW_SELECTION_MODES,
    _materialize_object_selected_outputs,
    _prepare_image_for_imwrite,
    camera_params_to_K,
    get_sync_index,
    infer_saved_depth_scale,
    load_hope_dataset,
    load_replica_dataset,
    mask_bbox_area,
    prepare_depth_for_png,
    save_cam_params_json,
    select_view_record,
    write_frame_paths_to_mp4,
)
from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.map.mapper import Mapper
from roman.map.observation import Observation
from roman.map.oneformer_wrapper import OneFormerWrapper
from roman.params.fastsam_params import FastSAMParams
from roman.params.mapper_params import MapperParams
from hickory.mapping.extraction_common import CameraPoseRecorder


DEFAULT_FASTSAM_PARAMS = "params/demo/fastsam.yaml"
DEFAULT_MAPPER_PARAMS = "params/demo/mapper.yaml"
DEFAULT_SEG_MODEL = "fastsam"
ONEFORMER_FILTER_MIN_OVERLAP = 0.3
FINAL_DEDUP_VOXEL_OVERLAP_RATIO = 0.025
FRAME_MASK_DEDUP_OVERLAP_THRESHOLD = 0.5


def detect_scene_format(scene_dir) -> str:
    scene_dir = Path(scene_dir)
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


def extract_camera_params(img_data) -> SimpleCameraParams:
    if hasattr(img_data, "camera_params") and img_data.camera_params is not None:
        cp = img_data.camera_params
        return SimpleCameraParams(
            width=cp.width,
            height=cp.height,
            fx=cp.fx,
            fy=cp.fy,
            cx=cp.cx,
            cy=cp.cy,
        )

    if hasattr(img_data, "K") and img_data.K is not None:
        k = np.asarray(img_data.K, dtype=np.float32)
        if k.shape != (3, 3):
            raise ValueError(f"Expected img_data.K shape (3, 3), got {k.shape}")
        width = getattr(img_data, "width", None)
        height = getattr(img_data, "height", None)
        if width is None or height is None:
            first_img = img_data.img(img_data.times[0])
            height, width = first_img.shape[:2]
        return SimpleCameraParams(
            width=width,
            height=height,
            fx=k[0, 0],
            fy=k[1, 1],
            cx=k[0, 2],
            cy=k[1, 2],
        )

    if len(img_data.times) == 0:
        raise ValueError("No image data found.")

    first_img = img_data.img(img_data.times[0])
    height, width = first_img.shape[:2]
    return SimpleCameraParams(
        width=width,
        height=height,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
    )


def sample_times_by_dt(times, dt: float | None):
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0 or dt is None or dt <= 0:
        return times

    sampled = [times[0]]
    last_t = float(times[0])
    min_gap = float(dt) - 1e-9
    for t in times[1:]:
        t = float(t)
        if t - last_t >= min_gap:
            sampled.append(t)
            last_t = t
    return np.asarray(sampled, dtype=np.float64)


def infer_depth_scale_from_data(depth_data, configured_scale: float) -> float:
    times = np.asarray(getattr(depth_data, "times", []), dtype=np.float64)
    if times.size == 0:
        return configured_scale

    probe_count = min(3, int(times.size))
    probe_indices = np.linspace(0, times.size - 1, probe_count, dtype=int)
    for idx in probe_indices:
        depth_img = depth_data.img(float(times[idx]))
        if depth_img is None:
            continue
        depth_arr = np.asarray(depth_img)
        if depth_arr.dtype.kind != "f":
            continue

        valid_depth = depth_arr[np.isfinite(depth_arr) & (depth_arr > 0)]
        if valid_depth.size == 0:
            continue

        if not np.isclose(float(configured_scale), 1.0):
            median_depth = float(np.median(valid_depth))
            print(
                "[INFO] Detected floating-point depth frames; "
                f"overriding depth scale {configured_scale} -> 1.0 "
                f"(median valid depth {median_depth:.3f} m)."
            )
        return 1.0

    return configured_scale


def create_demo_fastsam_wrapper(
    cam_params: SimpleCameraParams,
    fastsam_params: FastSAMParams,
) -> FastSAMWrapper:
    wrapper = FastSAMWrapper(
        weights=os.path.expandvars(fastsam_params.weights_path),
        conf=fastsam_params.conf,
        iou=fastsam_params.iou,
        imgsz=tuple(fastsam_params.imgsz),
        device=fastsam_params.device,
        mask_downsample_factor=fastsam_params.mask_downsample_factor,
        rotate_img=fastsam_params.rotate_img,
        use_pointcloud=False,
        verbose=False,
    )

    img_h = int(cam_params.height)
    img_w = int(cam_params.width)
    min_mask_len = max(1, min(img_h, img_w) // int(fastsam_params.min_mask_len_div))
    max_mask_len = max(img_h, img_w) / max(float(fastsam_params.max_mask_len_div), 1.0)
    area_bounds = np.array([float(min_mask_len ** 2), float(max_mask_len ** 2)], dtype=np.float64)

    wrapper.setup_filtering(
        ignore_labels=[],
        use_keep_labels=fastsam_params.use_keep_labels,
        keep_labels=list(fastsam_params.keep_labels),
        keep_labels_option=(
            "intersect"
            if fastsam_params.keep_labels_option is None
            else fastsam_params.keep_labels_option
        ),
        yolo_weights=os.path.expandvars(fastsam_params.yolo_weights_path),
        yolo_det_img_size=tuple(fastsam_params.yolo_imgsz),
        area_bounds=area_bounds,
        # allow_tblr_edges=[True, True, True, True],
        allow_tblr_edges=[False, False, False, False],
        semantics=fastsam_params.semantics,
        frame_descriptor=fastsam_params.frame_descriptor,
        triangle_ignore_masks=fastsam_params.triangle_ignore_masks,
    )
    wrapper.setup_rgbd_params(
        depth_cam_params=cam_params,
        max_depth=fastsam_params.max_depth,
        depth_scale=fastsam_params.depth_scale,
        voxel_size=fastsam_params.voxel_size,
        erosion_size=fastsam_params.erosion_size,
        plane_filter_ratio=fastsam_params.plane_filter_params,
    )
    return wrapper


def create_demo_oneformer_wrapper(
    cam_params: SimpleCameraParams,
    fastsam_params: FastSAMParams,
) -> OneFormerWrapper:
    wrapper = OneFormerWrapper(
        model_name=ONEFORMER_MODEL,
        device=fastsam_params.device,
        mask_downsample_factor=fastsam_params.mask_downsample_factor,
        rotate_img=fastsam_params.rotate_img,
        use_pointcloud=False,
        exclude_classes=EXCLUDE_CLASSES,
        post_threshold=ONEFORMER_POST_THRESHOLD,
        post_mask_threshold=ONEFORMER_POST_MASK_THRESHOLD,
        post_overlap_mask_area_threshold=ONEFORMER_POST_OVERLAP_MASK_AREA_THRESHOLD,
    )

    img_h = int(cam_params.height)
    img_w = int(cam_params.width)
    min_mask_len = max(1, min(img_h, img_w) // int(fastsam_params.min_mask_len_div))
    max_mask_len = max(img_h, img_w) / max(float(fastsam_params.max_mask_len_div), 1.0)
    area_bounds = np.array([float(min_mask_len ** 2), float(max_mask_len ** 2)], dtype=np.float64)

    wrapper.setup_filtering(
        allow_tblr_edges=[False, False, False, False],
        area_bounds=area_bounds,
        semantics=fastsam_params.semantics,
        frame_descriptor=fastsam_params.frame_descriptor,
    )
    wrapper.setup_rgbd_params(
        depth_cam_params=cam_params,
        max_depth=fastsam_params.max_depth,
        depth_scale=fastsam_params.depth_scale,
        voxel_size=fastsam_params.voxel_size,
        erosion_size=fastsam_params.erosion_size,
    )
    return wrapper


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

        contain_ratio = inter / float(cand_area)
        if contain_ratio >= overlap_threshold:
            return True

        union = int(np.logical_or(cand, filt).sum())
        if union > 0 and (inter / float(union)) >= overlap_threshold:
            return True

    return False


def deduplicate_observations_by_mask_overlap(
    observations,
    overlap_threshold: float = FRAME_MASK_DEDUP_OVERLAP_THRESHOLD,
):
    if observations is None or len(observations) <= 1:
        return observations, 0

    ranked = []
    for idx, obs in enumerate(observations):
        mask = getattr(obs, "mask", None)
        if mask is None:
            continue

        mask_bool = np.asarray(mask).astype(bool)
        mask_area = int(mask_bool.sum())
        if mask_area == 0:
            continue

        score = float(getattr(obs, "score", 0.0) or 0.0)
        ranked.append((idx, obs, mask_bool, mask_area, score))

    if len(ranked) <= 1:
        return observations, 0

    ranked.sort(key=lambda item: (item[4], item[3]), reverse=True)
    keep = np.ones(len(ranked), dtype=bool)

    for i, (_, _, mask_i, area_i, _) in enumerate(ranked):
        if not keep[i]:
            continue

        for j in range(i + 1, len(ranked)):
            if not keep[j]:
                continue

            _, _, mask_j, area_j, _ = ranked[j]
            inter = int(np.logical_and(mask_i, mask_j).sum())
            if inter == 0:
                continue

            contain_ratio = inter / float(area_j)
            if contain_ratio >= overlap_threshold:
                keep[j] = False
                continue

            union = area_i + area_j - inter
            if union > 0 and (inter / float(union)) >= overlap_threshold:
                keep[j] = False

    kept_obs = [item[1] for item, is_kept in zip(ranked, keep) if is_kept]
    removed = len(ranked) - len(kept_obs)
    return kept_obs, removed


def normalize_observations_for_mapper(observations):
    normalized = []
    for obs in observations:
        semantic_descriptor = obs.semantic_descriptor
        score = getattr(obs, "score", None)
        if isinstance(semantic_descriptor, dict):
            score = float(semantic_descriptor.get("score", 0.0))
            semantic_descriptor = semantic_descriptor.get("mean")
        normalized.append(
            Observation(
                time=obs.time,
                pose=obs.pose,
                mask=obs.mask,
                mask_downsampled=obs.mask_downsampled,
                point_cloud=obs.point_cloud,
                semantic_descriptor=semantic_descriptor,
                score=score,
            )
        )
    return normalized


def mask_area(mask) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(mask))


def segment_volume_cm3(segment) -> float:
    try:
        return float(segment.volume) * 1e6
    except Exception:
        return 0.0


def find_frame_record(frame_records, obs_time, atol=1e-6):
    if not frame_records:
        return None
    for record in frame_records:
        if abs(float(record["time"]) - float(obs_time)) <= atol:
            return record
    return min(frame_records, key=lambda rec: abs(float(rec["time"]) - float(obs_time)))


def observation_world_points(obs):
    if obs is None or getattr(obs, "point_cloud", None) is None:
        return None
    pts = np.asarray(obs.transformed_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None
    return pts.copy()


def deduplicate_segments_by_voxel_overlap(segments, voxel_size, overlap_ratio):
    if segments is None or len(segments) <= 1:
        return segments

    def segment_points(seg):
        pts = getattr(seg, "points", None)
        if pts is None or len(pts) == 0:
            return None
        return np.asarray(pts)

    def voxel_set(pts):
        if pts is None or len(pts) == 0:
            return set()
        vox = np.floor(pts / float(voxel_size)).astype(np.int64)
        return set(map(tuple, np.unique(vox, axis=0)))

    voxel_sets = [voxel_set(segment_points(seg)) for seg in segments]
    sizes = [len(vox) for vox in voxel_sets]
    order = sorted(range(len(segments)), key=lambda i: sizes[i], reverse=True)
    keep = np.ones(len(segments), dtype=bool)

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
            if containment_j >= float(overlap_ratio):
                keep[j] = False

    deduped = [seg for idx, seg in enumerate(segments) if keep[idx]]
    removed = len(segments) - len(deduped)
    print(
        f"Deduplicated mapper segments by voxel containment >= {overlap_ratio:.2f}: "
        f"kept {len(deduped)} / {len(segments)} (removed {removed})"
    )
    return deduped


def export_mapper_segments(
    output_dir,
    mapper,
    frame_records,
    segment_view_records,
    merge_voxel_size,
    view_selection="best",
    shared_cache_layout=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = sorted(mapper.get_segment_map(), key=lambda seg: int(seg.id))
    segments = deduplicate_segments_by_voxel_overlap(
        segments,
        voxel_size=merge_voxel_size,
        overlap_ratio=FINAL_DEDUP_VOXEL_OVERLAP_RATIO,
    )
    print(f"\nSaving {len(segments)} mapper segments (view_selection={view_selection})...")

    for saved_id, seg in enumerate(segments):
        obj_dir = output_dir / f"obj_{saved_id:03d}"
        obj_dir.mkdir(exist_ok=True)

        history = segment_view_records.get(int(seg.id), [])
        selected_record = select_view_record(history, view_selection) if history else None
        if selected_record is not None:
            frame_name = selected_record["frame_name"]
            frame_id = selected_record["frame_id"]
            selected_pose = np.asarray(selected_record["pose"]).copy()
            selected_mask = selected_record["mask"]
            selected_feature = (
                np.asarray(selected_record["feature"]).copy()
                if selected_record.get("feature") is not None
                else None
            )
            selected_points_world = (
                np.asarray(selected_record["points_world"], dtype=np.float64).copy()
                if selected_record.get("points_world") is not None
                else None
            )
        else:
            obs = seg.last_observation
            frame_record = find_frame_record(frame_records, seg.last_seen)
            frame_name = frame_record["frame_name"] if frame_record is not None else f"frame_{saved_id:05d}.jpg"
            frame_id = frame_record["frame_id"] if frame_record is not None else saved_id
            selected_pose = np.asarray(obs.pose).copy()
            selected_mask = obs.mask
            selected_feature = (
                np.asarray(obs.semantic_descriptor).copy()
                if obs.semantic_descriptor is not None
                else None
            )
            selected_points_world = observation_world_points(obs)

        with open(obj_dir / "all_scores.txt", "w", encoding="utf-8") as f:
            f.write(f"saved_id, {saved_id}\n")
            f.write(f"original_id, {int(seg.id)}\n")
            f.write("frame_name, frame_id, score, mask_area_ratio, bbox2d_area_ratio, bbox_volume_cm3\n")
            sorted_history = sorted(
                history,
                key=lambda x: (
                    x.get("score", float("-inf")),
                    x.get("mask_area", float("-inf")),
                ),
                reverse=True,
            )
            if sorted_history:
                for record in sorted_history:
                    f.write(
                        f"{record['frame_name']}, {record['frame_id']}, {record['score']:.6f}, "
                        f"{record.get('mask_area_ratio', 0.0):.6f}, "
                        f"{record.get('mask_bbox_area_ratio', 0.0):.6f}, "
                        f"{record.get('volume_cm3', 0.0):.6f}\n"
                    )
            else:
                f.write(
                    f"{frame_name}, {frame_id}, {float(seg.num_sightings):.6f}, "
                    f"0.000000, 0.000000, {segment_volume_cm3(seg):.6f}\n"
                )

        if shared_cache_layout:
            all_masks_dir = output_dir / "_cache_masks" / obj_dir.name
            all_features_dir = output_dir / "_cache_features" / obj_dir.name
            all_masks_dir.mkdir(parents=True, exist_ok=True)
            all_features_dir.mkdir(parents=True, exist_ok=True)
        else:
            all_masks_dir = obj_dir / "all_masks"
            all_features_dir = obj_dir / "all_features"
            all_masks_dir.mkdir(exist_ok=True)
            all_features_dir.mkdir(exist_ok=True)

        history_by_time = sorted(
            history,
            key=lambda r: (
                r.get("frame_id") if r.get("frame_id") is not None else 10**9,
                r.get("frame_name", ""),
            ),
        )
        for idx, record in enumerate(history_by_time):
            frame_name_i = str(record.get("frame_name", f"view_{idx:05d}"))
            frame_stem_i = Path(frame_name_i).stem

            record_mask = record.get("mask")
            if record_mask is not None:
                mask_img = _prepare_image_for_imwrite(
                    (np.asarray(record_mask) > 0).astype(np.uint8) * 255,
                    is_depth=False,
                )
                if mask_img is not None:
                    cv2.imwrite(str(all_masks_dir / f"{frame_stem_i}.png"), mask_img)

            record_feature = record.get("feature")
            if record_feature is not None:
                np.save(all_features_dir / f"{frame_stem_i}.npy", np.asarray(record_feature))

        _materialize_object_selected_outputs(
            output_root=output_dir,
            obj_dir=obj_dir,
            selected_frame_name=frame_name,
            selected_pose=selected_pose,
            selected_feature=selected_feature,
        )

        if selected_points_world is not None and selected_points_world.shape[0] > 0:
            pcd_world = o3d.geometry.PointCloud()
            pcd_world.points = o3d.utility.Vector3dVector(selected_points_world)
            o3d.io.write_point_cloud(str(obj_dir / "point_cloud_world.ply"), pcd_world)
            np.save(obj_dir / "point_cloud_world.npy", selected_points_world.astype(np.float32))

        if seg.num_points > 0:
            pts = np.asarray(seg.points, dtype=np.float64)
            pcd_world = o3d.geometry.PointCloud()
            pcd_world.points = o3d.utility.Vector3dVector(pts)
            o3d.io.write_point_cloud(str(obj_dir / "point_cloud_accumulated_world.ply"), pcd_world)
            np.save(obj_dir / "point_cloud_accumulated_world.npy", pts.astype(np.float32))


def run_scene_extraction_from_data(
    img_data,
    depth_data,
    pose_data,
    output_dir=DEFAULT_OUTPUT_DIR,
    dataset_name="dataset",
    dt=None,
    depth_scale=None,
    fps=15.0,
    view_selection="best",
    fastsam_params_path=DEFAULT_FASTSAM_PARAMS,
    mapper_params_path=DEFAULT_MAPPER_PARAMS,
    seg_model=DEFAULT_SEG_MODEL,
    max_frames=None,
    shared_cache_layout=False,
    depth_scale_override=None,
):
    fastsam_params = FastSAMParams.from_yaml(fastsam_params_path)
    mapper_params = MapperParams.from_yaml(mapper_params_path)
    if depth_scale_override is not None:
        fastsam_params.depth_scale = float(depth_scale_override)
    elif depth_scale is not None:
        fastsam_params.depth_scale = depth_scale
    else:
        fastsam_params.depth_scale = infer_depth_scale_from_data(
            depth_data,
            float(fastsam_params.depth_scale),
        )

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_params = extract_camera_params(img_data)
    cam_params.K = camera_params_to_K(cam_params)
    first_depth_sample = None
    depth_times = np.asarray(getattr(depth_data, "times", []), dtype=np.float64)
    for t in depth_times[:3]:
        sample = depth_data.img(float(t))
        if sample is not None and np.asarray(sample).size > 0:
            first_depth_sample = np.asarray(sample)
            break
    saved_depth_scale = (
        infer_saved_depth_scale(first_depth_sample, float(fastsam_params.depth_scale))
        if first_depth_sample is not None
        else float(fastsam_params.depth_scale)
    )
    camera_k_path = output_dir / "camera_K.txt"
    np.savetxt(str(camera_k_path), cam_params.K, fmt="%.8f")
    (output_dir / "depth_scale.txt").write_text(
        f"{saved_depth_scale:.10f}\n",
        encoding="utf-8",
    )
    save_cam_params_json(output_dir, cam_params, depth_scale=saved_depth_scale)

    print(f"Loaded dataset from: {dataset_name}")
    print(f"Using segmentation model: {seg_model}")
    print(f"Using FastSAM params: {fastsam_params_path}")
    print(f"Using mapper params: {mapper_params_path}")
    print(f"Saved camera intrinsics: {camera_k_path}")
    print(f"Using depth scale: {fastsam_params.depth_scale}")
    print(f"Using merge voxel size: {fastsam_params.voxel_size}")

    if seg_model == "fastsam":
        wrapper = create_demo_fastsam_wrapper(cam_params, fastsam_params)
        oneformer_class_filter = OneFormerClassMaskFilter(
            model_name=ONEFORMER_MODEL,
            device=fastsam_params.device,
            filtered_classes=EXCLUDE_CLASSES,
            post_threshold=ONEFORMER_POST_THRESHOLD,
            post_mask_threshold=ONEFORMER_POST_MASK_THRESHOLD,
            post_overlap_mask_area_threshold=ONEFORMER_POST_OVERLAP_MASK_AREA_THRESHOLD,
        )
    elif seg_model == "oneformer":
        wrapper = create_demo_oneformer_wrapper(cam_params, fastsam_params)
        oneformer_class_filter = None
    else:
        raise ValueError("seg_model must be 'fastsam' or 'oneformer'.")
    mapper = Mapper(mapper_params, cam_params)

    all_times = np.asarray(img_data.times, dtype=np.float64)
    should_sample_by_dt = not hasattr(img_data, "_paths")
    if should_sample_by_dt and dt is not None and dt > 0.0 and all_times.size > 0:
        # Bag-backed streams can contain every recorded frame; respect the configured
        # cadence while tolerating small timestamp jitter. File-backed extracted
        # datasets such as ReplicaCAD already provide the intended frame samples.
        sampled = [float(all_times[0])]
        next_time = float(all_times[0]) + float(dt)
        for t in all_times[1:]:
            t_val = float(t)
            if t_val + 1e-9 < next_time:
                continue
            sampled.append(t_val)
            next_time = t_val + float(dt)
        sample_times = np.asarray(sampled, dtype=np.float64)
    else:
        sample_times = all_times

    if max_frames is not None and len(sample_times) > max_frames:
        sample_idx = np.linspace(0, len(sample_times) - 1, int(max_frames), dtype=int)
        sample_times = sample_times[sample_idx]

    rgb_dir = output_dir / "rgb_imgs"
    depth_dir = output_dir / "depth_imgs"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    pose_recorder = CameraPoseRecorder()
    frame_records = []
    segment_view_records = {}
    total_raw_observations = 0
    total_oneformer_rejected = 0
    total_empty_rejected = 0
    total_mask_dedup_rejected = 0
    total_passed_to_mapper = 0
    skipped_missing_pose = 0

    print(f"Processing {len(sample_times)} frames...")
    for i, t in enumerate(tqdm(sample_times)):
        frame_name = f"frame_{i:05d}.png"

        img_bgr = img_data.img(t)
        if img_bgr is None:
            continue

        sync_tol = dt if dt is not None else 0.05
        depth_idx = get_sync_index(t, depth_data.times, tol=sync_tol)
        if depth_idx is None:
            continue
        depth_img = depth_data.img(depth_data.times[depth_idx])
        if depth_img is None or depth_img.size == 0:
            print("Detected empty depth_img and skipped this frame...")
            continue

        h_rgb, w_rgb = img_bgr.shape[:2]
        h_d, w_d = depth_img.shape[:2]
        if (h_d != h_rgb) or (w_d != w_rgb):
            depth_img = cv2.resize(depth_img, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)

        try:
            T_WC_raw = np.asarray(pose_data.pose(t), dtype=np.float64)
        except NoDataNearTimeException:
            skipped_missing_pose += 1
            continue
        except Exception:
            skipped_missing_pose += 1
            continue
        if T_WC_raw.shape != (4, 4):
            continue
        T_WC = pose_recorder.add(i, t, frame_name, T_WC_raw)
        frame_records.append(
            {
                "time": float(t),
                "frame_id": i,
                "frame_name": frame_name,
                "pose": T_WC,
            }
        )

        cv2.imwrite(str(rgb_dir / f"frame_{i:05d}.png"), img_bgr)
        depth_img_to_save = prepare_depth_for_png(depth_img, saved_depth_scale)
        if depth_img_to_save is not None:
            cv2.imwrite(str(depth_dir / f"frame_{i:05d}.png"), depth_img_to_save)

        observations, frame_descriptor = wrapper.run(t=t, pose=T_WC, img=img_bgr, depth_data=depth_img)
        total_raw_observations += len(observations)
        filtered_class_masks = (
            oneformer_class_filter.run(img_bgr) if oneformer_class_filter is not None else []
        )

        filtered_observations = []
        for obs in observations:
            if should_reject_mask_by_oneformer(obs.mask, filtered_class_masks):
                total_oneformer_rejected += 1
                continue
            if getattr(obs, "transformed_points", None) is None or len(obs.transformed_points) == 0:
                total_empty_rejected += 1
                continue
            filtered_observations.append(obs)

        filtered_observations, dedup_removed = deduplicate_observations_by_mask_overlap(
            filtered_observations
        )
        total_mask_dedup_rejected += dedup_removed
        total_passed_to_mapper += len(filtered_observations)

        mapper.update(
            float(t),
            T_WC,
            normalize_observations_for_mapper(filtered_observations),
            frame_descriptor,
        )

        for seg in mapper.segments + mapper.segment_nursery:
            if not np.isclose(float(seg.last_seen), float(t)):
                continue
            if seg.last_observation.mask is None:
                continue
            obs = seg.last_observation
            current_mask = np.asarray(obs.mask)
            current_area = mask_area(current_mask)
            current_bbox_area = int(mask_bbox_area(current_mask))
            current_mask_area_ratio = (
                float(current_area) / float(current_mask.size) if current_mask.size > 0 else 0.0
            )
            current_mask_bbox_area_ratio = (
                float(current_bbox_area) / float(current_mask.size) if current_mask.size > 0 else 0.0
            )

            feature = obs.semantic_descriptor
            score = float(getattr(obs, "score", 0.0) or 0.0)
            if isinstance(feature, dict):
                score = float(feature.get("score", score))
                feature = feature.get("mean")

            record = {
                "frame_name": frame_name,
                "frame_id": i,
                "score": score,
                "volume_cm3": segment_volume_cm3(seg),
                "mask_area": current_area,
                "mask_area_ratio": current_mask_area_ratio,
                "mask_bbox_area": current_bbox_area,
                "mask_bbox_area_ratio": current_mask_bbox_area_ratio,
                "pose": np.asarray(obs.pose).copy(),
                "mask": current_mask.copy(),
                "feature": np.asarray(feature).copy() if feature is not None else None,
                "points_world": observation_world_points(obs),
            }

            history = segment_view_records.setdefault(int(seg.id), [])
            updated = False
            for existing in history:
                if existing.get("frame_id") != i:
                    continue
                if (current_area, score) > (
                    existing.get("mask_area", -1),
                    existing.get("score", float("-inf")),
                ):
                    existing.update(record)
                updated = True
                break
            if not updated:
                history.append(record)

    if skipped_missing_pose > 0:
        print(
            f"[WARN] Skipped {skipped_missing_pose} frame(s) with no pose sample near the RGB timestamp."
        )
    if len(frame_records) == 0:
        pose_times = getattr(pose_data, "times", None)
        pose_range = "unknown"
        if pose_times is not None and len(pose_times) > 0:
            pose_times = np.asarray(pose_times, dtype=np.float64)
            pose_range = f"[{pose_times[0]:.6f}, {pose_times[-1]:.6f}]"
        rgb_range = "unknown"
        if all_times.size > 0:
            rgb_range = f"[{all_times[0]:.6f}, {all_times[-1]:.6f}]"
        raise RuntimeError(
            "No frames had synchronized RGB, depth, and pose data for scene_extractor. "
            f"RGB range: {rgb_range}, pose range: {pose_range}. "
            "Check the ros config topics and timestamp alignment."
        )

    print(f"Saving representative-view results (mode={view_selection})...")
    print(
        "Observation stats: "
        f"raw={total_raw_observations}, "
        f"oneformer_rejected={total_oneformer_rejected}, "
        f"empty_3d_rejected={total_empty_rejected}, "
        f"mask_dedup_rejected={total_mask_dedup_rejected}, "
        f"passed_to_mapper={total_passed_to_mapper}"
    )
    print(
        "Mapper stats before export: "
        f"nursery={len(mapper.segment_nursery)}, "
        f"active={len(mapper.segments)}, "
        f"inactive={len(mapper.inactive_segments)}, "
        f"graveyard={len(mapper.segment_graveyard)}"
    )
    export_mapper_segments(
        output_dir=output_dir,
        mapper=mapper,
        frame_records=frame_records,
        segment_view_records=segment_view_records,
        view_selection=view_selection,
        shared_cache_layout=shared_cache_layout,
        merge_voxel_size=float(fastsam_params.voxel_size),
    )

    pose_recorder.write(output_dir)

    output_video = output_dir / f"rgb_views_{output_dir.name}.mp4"
    print(f"Saving RGB overview video: {output_video}")
    rgb_frame_paths = sorted(rgb_dir.glob("frame_*.png"))
    if len(rgb_frame_paths) == 0:
        rgb_frame_paths = sorted((output_dir / "rgb").glob("frame_*.png"))
    if not write_frame_paths_to_mp4(rgb_frame_paths, str(output_video), fps=30):
        img_data.to_mp4(str(output_video), fps=30)
    print("Done!")


def run_scene_extraction(
    scene_dir,
    output_dir=DEFAULT_OUTPUT_DIR,
    fps=15.0,
    view_selection="best",
    fastsam_params_path=DEFAULT_FASTSAM_PARAMS,
    mapper_params_path=DEFAULT_MAPPER_PARAMS,
    seg_model=DEFAULT_SEG_MODEL,
    max_frames=None,
    shared_cache_layout=False,
    depth_scale_override=None,
):
    scene_format = detect_scene_format(scene_dir)
    if scene_format == "replica":
        img_data, depth_data, pose_data, dt, depth_scale = load_replica_dataset(
            scene_dir=scene_dir,
            fps=fps,
        )
    elif scene_format == "hope":
        img_data, depth_data, pose_data, dt, depth_scale = load_hope_dataset(
            scene_dir=scene_dir,
            fps=fps,
        )
    else:
        raise FileNotFoundError(
            f"Could not infer scene format from: {scene_dir}. "
            "Expected Replica or HOPE extracted layouts."
        )

    run_scene_extraction_from_data(
        img_data=img_data,
        depth_data=depth_data,
        pose_data=pose_data,
        output_dir=output_dir,
        dataset_name=str(scene_dir),
        dt=dt,
        depth_scale=depth_scale,
        fps=fps,
        view_selection=view_selection,
        fastsam_params_path=fastsam_params_path,
        mapper_params_path=mapper_params_path,
        seg_model=seg_model,
        max_frames=max_frames,
        shared_cache_layout=shared_cache_layout,
        depth_scale_override=depth_scale_override,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Dataset extraction using selectable FastSAM or OneFormer segmentation "
            "and ROMAN mapper tracking, with legacy extraction-style outputs."
        )
    )
    parser.add_argument(
        "--scene",
        "--replica-scene",
        dest="scene",
        required=True,
        help="Path to a ReplicaCAD or HOPE scene folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for extracted objects.",
    )
    parser.add_argument(
        "--replica-fps",
        type=float,
        default=15.0,
        help="Frame rate used to assign timestamps when needed.",
    )
    parser.add_argument(
        "--view-selection",
        default="best",
        choices=VIEW_SELECTION_MODES,
        help="Representative view selection mode.",
    )
    parser.add_argument(
        "--fastsam-params",
        default=DEFAULT_FASTSAM_PARAMS,
        help="FastSAM params YAML. Defaults to params/demo/fastsam.yaml.",
    )
    parser.add_argument(
        "--mapper-params",
        default=DEFAULT_MAPPER_PARAMS,
        help="Mapper params YAML. Defaults to params/demo/mapper.yaml.",
    )
    parser.add_argument(
        "--seg-model",
        default=DEFAULT_SEG_MODEL,
        choices=["fastsam", "oneformer"],
        help="Segmentation backend. FastSAM keeps the OneFormer class-filter pass.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on sampled frames.",
    )
    parser.add_argument(
        "--shared-cache-layout",
        action="store_true",
        help="Write object cache assets into shared cache folders.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Optional depth scale override.",
    )
    args = parser.parse_args()

    run_scene_extraction(
        scene_dir=args.scene,
        output_dir=args.output_dir,
        fps=args.replica_fps,
        view_selection=args.view_selection,
        fastsam_params_path=args.fastsam_params,
        mapper_params_path=args.mapper_params,
        seg_model=args.seg_model,
        max_frames=args.max_frames,
        shared_cache_layout=args.shared_cache_layout,
        depth_scale_override=args.depth_scale,
    )
