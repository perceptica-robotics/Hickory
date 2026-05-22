#########################################
# 
# oneformer_wrapper.py
#
# A Python wrapper for sending RGBD images to OneFormer and using segmentation 
# masks to create object observations.
# 
# Authors: Jouko Kinnari, Mason Peterson, Lucas Jia, Annika Thomas, Qingyuan Li
# 
# Feb. 2, 2026
#
#########################################

import cv2 as cv
import numpy as np
import open3d as o3d
import copy
import math
import torch
import logging
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import AutoImageProcessor, AutoModel

from roman.map.observation import Observation

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


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


class OneFormerWrapper():
    def __init__(
        self,
        model_name="shi-labs/oneformer_ade20k_swin_large",
        device="cuda",
        mask_downsample_factor=1,
        rotate_img=None,
        use_pointcloud=False,
        area_bounds=np.array([0, np.inf]),
        allow_tblr_edges=None,
        exclude_classes=None,
        post_threshold=0.5,
        post_mask_threshold=0.5,
        post_overlap_mask_area_threshold=0.3,
        post_label_ids_to_fuse=None,
    ):
        self.device = device
        self.mask_downsample_factor = mask_downsample_factor
        self.rotate_img = rotate_img
        self.use_pointcloud = use_pointcloud
        self.area_bounds = area_bounds
        self.allow_tblr_edges = allow_tblr_edges
        self.exclude_classes = set(c.strip().lower() for c in exclude_classes) if exclude_classes else set()
        self.post_threshold = post_threshold
        self.post_mask_threshold = post_mask_threshold
        self.post_overlap_mask_area_threshold = post_overlap_mask_area_threshold
        self.post_label_ids_to_fuse = post_label_ids_to_fuse
        self.semantics = None
        self.semantics_model = None
        self.semantics_preprocess = None
        self.semantic_patches_shape = None
        self.frame_descriptor_type = None

        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.observations = []

        assert self.device == "cuda" or self.device == "cpu", "Device should be 'cuda' or 'cpu'."
        assert self.rotate_img is None or self.rotate_img == "CW" or self.rotate_img == "CCW" \
            or self.rotate_img == "180", "Invalid rotate_img option."

    def setup_filtering(
        self,
        allow_tblr_edges=None,
        area_bounds=None,
        semantics: str = None,
        frame_descriptor: str = None,
    ):
        if allow_tblr_edges is not None:
            self.allow_tblr_edges = allow_tblr_edges
        if area_bounds is not None:
            self.area_bounds = area_bounds

        self.semantics = semantics
        if semantics is None or semantics.lower() == "none":
            self.semantics_model = None
            self.semantics_preprocess = None
        elif semantics.lower() == "dino":
            self.semantics_preprocess = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-base", do_center_crop=False
            )
            self.semantics_model = AutoModel.from_pretrained("facebook/dinov2-base")
            self.semantics_model.eval()
            self.semantics_model.to(self.device)
        else:
            raise ValueError(f"Invalid semantics option: {semantics}. Choose from 'dino' or 'none'.")

        self.semantic_patches_shape = None
        self.frame_descriptor_type = frame_descriptor
        if frame_descriptor is not None:
            assert self.semantics is not None and self.semantics.lower() == "dino", \
                "Frame descriptor only supported with DINO semantics."

    def setup_rgbd_params(
        self,
        depth_cam_params,
        max_depth,
        depth_scale=1e3,
        voxel_size=0.05,
        within_depth_frac=0.5,
        pcd_stride=4,
        erosion_size=0,
    ):
        self.depth_cam_params = depth_cam_params
        self.max_depth = max_depth
        self.within_depth_frac = within_depth_frac
        self.depth_scale = depth_scale
        if not self.use_pointcloud:
            self.depth_cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=int(depth_cam_params.width),
                height=int(depth_cam_params.height),
                fx=depth_cam_params.fx,
                fy=depth_cam_params.fy,
                cx=depth_cam_params.cx,
                cy=depth_cam_params.cy,
            )
        self.voxel_size = voxel_size
        self.pcd_stride = pcd_stride
        if erosion_size > 0:
            erosion_shape = cv.MORPH_ELLIPSE
            self.erosion_element = cv.getStructuringElement(
                erosion_shape,
                (2 * erosion_size + 1, 2 * erosion_size + 1),
                (erosion_size, erosion_size),
            )
        else:
            self.erosion_element = None

    def run(self, t, pose, img, depth_data=None):
        """
        Takes an image and returns OneFormer masks as Observations.

        Returns:
            self.observations (list): list of Observations
            frame_descriptor (np.ndarray): always None for OneFormer
        """
        self.observations = []

        img_orig = img
        img = self.apply_rotation(img)

        if self.use_pointcloud and depth_data is not None:
            pcl, pcl_proj = depth_data

        if len(img.shape) == 2:
            img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        inputs = self.processor(images=img_rgb, task_inputs=["panoptic"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_size = [img_rgb.shape[:2]]
        post = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=target_size,
            threshold=self.post_threshold,
            mask_threshold=self.post_mask_threshold,
            overlap_mask_area_threshold=self.post_overlap_mask_area_threshold,
            label_ids_to_fuse=self.post_label_ids_to_fuse,
        )[0]
        seg_map = post["segmentation"]
        segments_info = post["segments_info"]
        id2label = getattr(self.model.config, "id2label", {})

        if isinstance(seg_map, torch.Tensor):
            seg_map = seg_map.cpu().numpy()

        masks = []
        for info in segments_info:
            seg_id = info.get("id")
            if seg_id is None:
                continue
            label_id = info.get("label_id")
            label_name = id2label.get(label_id, "") if label_id is not None else ""
            if should_exclude_class(label_name, self.exclude_classes):
                continue
            mask = (seg_map == seg_id).astype(np.uint8)
            if self.area_bounds is not None:
                area = np.sum(mask)
                if area < self.area_bounds[0] or area > self.area_bounds[1]:
                    continue
            masks.append(mask)

        if masks and not np.all(self.allow_tblr_edges):
            masks = self._delete_edge_masks(np.stack(masks, axis=0))
            masks = list(masks)

        dino_features = None
        dino_output_patches = None
        if self.semantics is not None and self.semantics.lower() == "dino":
            dino_shape = 768
            preprocessed = self.semantics_preprocess(images=img_rgb, return_tensors="pt").to(self.device)
            dino_output = self.semantics_model(**preprocessed)
            dino_output_patches = self.get_output_patches(
                model_output=dino_output.last_hidden_state,
                img_shape=img.shape,
                feature_dim=dino_shape,
            )
            dino_features = self.get_per_pixel_features(
                model_output_patches=dino_output_patches,
                img_shape=img.shape,
            )
            dino_features = self.unapply_rotation(dino_features)

        frame_descriptor = None
        if self.frame_descriptor_type is not None and dino_output_patches is not None:
            frame_descriptor = self.get_frame_descriptor(dino_output_patches)

        for mask in masks:
            mask = self.unapply_rotation(mask)

            ptcld = None
            if depth_data is not None:
                if self.use_pointcloud:
                    inside_mask = mask[pcl_proj[:, 1], pcl_proj[:, 0]] == 1
                    inside_mask_points = pcl[inside_mask]
                    pre_truncate_len = len(inside_mask_points)
                    ptcld_test = inside_mask_points[inside_mask_points[:, 2] < self.max_depth]

                    if len(ptcld_test) < self.within_depth_frac * pre_truncate_len:
                        continue

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(inside_mask_points)
                else:
                    depth_obj = copy.deepcopy(depth_data)
                    if self.erosion_element is not None:
                        eroded_mask = cv.erode(mask, self.erosion_element)
                        depth_obj[eroded_mask == 0] = 0
                    else:
                        depth_obj[mask == 0] = 0

                    pcd_test = o3d.geometry.PointCloud.create_from_depth_image(
                        o3d.geometry.Image(np.ascontiguousarray(depth_obj).astype(np.dtype(depth_obj.dtype).type)),
                        self.depth_cam_intrinsics,
                        depth_scale=self.depth_scale,
                        stride=self.pcd_stride,
                        project_valid_depth_only=True,
                    )
                    ptcld_test = np.asarray(pcd_test.points)
                    pre_truncate_len = len(ptcld_test)
                    ptcld_test = ptcld_test[ptcld_test[:, 2] < self.max_depth]
                    if len(ptcld_test) < self.within_depth_frac * pre_truncate_len:
                        continue

                    pcd = o3d.geometry.PointCloud.create_from_depth_image(
                        o3d.geometry.Image(np.ascontiguousarray(depth_obj).astype(np.dtype(depth_obj.dtype).type)),
                        self.depth_cam_intrinsics,
                        depth_scale=self.depth_scale,
                        depth_trunc=self.max_depth,
                        stride=self.pcd_stride,
                        project_valid_depth_only=True,
                    )

                pcd.remove_non_finite_points()
                pcd_sampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
                if not pcd_sampled.is_empty():
                    ptcld = np.asarray(pcd_sampled.points)
                if ptcld is None:
                    continue

            mask_downsampled = np.array(
                cv.resize(
                    mask,
                    (mask.shape[1] // self.mask_downsample_factor, mask.shape[0] // self.mask_downsample_factor),
                    interpolation=cv.INTER_NEAREST,
                )
            ).astype("uint8")

            if self.semantics is not None and self.semantics.lower() == "dino":
                mask_bool = torch.from_numpy(mask.astype(bool)).to(dino_features.device)
                dino_mask_tensor = dino_features[mask_bool]
                variances = torch.var(dino_mask_tensor, dim=0, unbiased=True)
                total_variance = torch.sum(variances).item()
                area_mask = float(mask_bool.sum().item())
                area_img = float(mask_bool.numel())
                area_ratio = area_mask / area_img if area_img > 0 else 0.0
                score = total_variance * area_ratio
                mean_feat = torch.mean(dino_mask_tensor, dim=0).detach().cpu().numpy()
                semantic_data = {
                    "mean": mean_feat.astype(np.float32),
                    "score": float(score),
                    # "variance": float(total_variance),
                    # "area_ratio": float(area_ratio),
                }
                self.observations.append(Observation(
                    t, pose, mask, mask_downsampled, ptcld, semantic_descriptor=semantic_data
                ))
            else:
                self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld))

        return self.observations, frame_descriptor

    def apply_rotation(self, img, unrotate=False):
        if self.rotate_img is None:
            return img
        elif self.rotate_img == "CW":
            k = 3 if not unrotate else 1
        elif self.rotate_img == "CCW":
            k = 1 if not unrotate else 3
        elif self.rotate_img == "180":
            k = 2
        else:
            raise Exception("Invalid rotate_img option.")
        return np.rot90(img, k)

    def unapply_rotation(self, img):
        return self.apply_rotation(img, unrotate=True)

    def _delete_edge_masks(self, segmask):
        [numMasks, h, w] = segmask.shape
        edge_width = min(5, h, w)
        if edge_width <= 0:
            return segmask
        contains_edge = np.zeros(numMasks, dtype=bool)
        for i in range(numMasks):
            mask = segmask[i, :, :].astype(bool)
            touches_left = np.any(mask[:, :edge_width])
            touches_right = np.any(mask[:, -edge_width:])
            touches_top = np.any(mask[:edge_width, :])
            touches_bottom = np.any(mask[-edge_width:, :])
            contains_edge[i] = (touches_left and not self.allow_tblr_edges[2]) or \
                               (touches_right and not self.allow_tblr_edges[3]) or \
                               (touches_top and not self.allow_tblr_edges[0]) or \
                               (touches_bottom and not self.allow_tblr_edges[1])
        return np.delete(segmask, contains_edge, axis=0)

    def get_output_patches(self, model_output, img_shape, feature_dim):
        model_output_flat_patches = model_output[:, 1:, :]
        if self.semantic_patches_shape is None:
            ratio = img_shape[1] / img_shape[0]
            num_patches = model_output_flat_patches.shape[1]
            h = np.round(np.sqrt(num_patches / ratio)).astype(int)
            w = np.round(np.sqrt(num_patches * ratio)).astype(int)
            self.semantic_patches_shape = (1, h, w, feature_dim)
        model_output_patches = model_output_flat_patches.reshape(self.semantic_patches_shape)
        return model_output_patches

    def get_per_pixel_features(self, model_output_patches, img_shape):
        per_pixel_features = torch.nn.functional.interpolate(
            model_output_patches.permute(0, 3, 1, 2),
            size=(img_shape[0], img_shape[1]),
            mode="bilinear",
        )
        per_pixel_features = per_pixel_features[0].permute(1, 2, 0)
        return per_pixel_features

    def get_frame_descriptor(self, dino_features):
        with torch.no_grad():
            dino_features_flat = dino_features.view(-1, dino_features.shape[-1])
            if self.frame_descriptor_type == "dino-gap":
                frame_descriptor = torch.sum(dino_features_flat, dim=0)
            elif self.frame_descriptor_type == "dino-gmp":
                frame_descriptor = torch.max(dino_features_flat, dim=0).values
            elif self.frame_descriptor_type == "dino-gem":
                cubed_descriptor = torch.mean(dino_features_flat ** 3, dim=0)
                frame_descriptor = torch.sign(cubed_descriptor) * \
                                   (torch.abs(cubed_descriptor).clamp(min=1e-12) ** (1.0 / 3))
            else:
                raise ValueError("frame descriptor must be one of 'dino-gap', 'dino-gmp', or 'dino-gem'.")
            frame_descriptor /= torch.norm(frame_descriptor)
        return frame_descriptor.cpu().detach().numpy()
