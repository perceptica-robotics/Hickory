#########################################
# 
# SAM2_wrapper.py
#
# A Python wrapper for sending RGBD images to SAM2 and using segmentation 
# masks to create object observations.
# 
# Authors: Jouko Kinnari, Mason Peterson, Lucas Jia, Annika Thomas, Qingyuan Li
# 
# Dec. 21, 2024
#
#########################################


import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike
import open3d as o3d
import torch
from yolov7_package import Yolov7Detector
import math
import time
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import clip
import logging
from transformers import AutoImageProcessor, AutoModel

from roman.map.observation import Observation
# from roman.utils import expandvars_recursive
# from roman.viz import viz_pointcloud_on_img

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

class SAM2Wrapper():

    def __init__(self, 
        model_cfg,  # SAM2 需要配置文件 (.yaml)
        weights,    # SAM2 权重 (.pt)
        conf, 
        imgsz=(480, 640),
        device='cuda',
        mask_downsample_factor=8,
        rotate_img=None,
        use_pointcloud=False,
    ):
        self.device = device
        self.mask_downsample_factor = mask_downsample_factor
        self.imgsz = imgsz
        self.rotate_img = rotate_img
        self.use_pointcloud = use_pointcloud
        self.conf = conf
        self.multimask_iou_thresh = 0.2
        self.multimask_contain_thresh = 0.5
        
        # 加载 SAM 2 模型
        self.model = build_sam2(model_cfg, weights, device=self.device)
        
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            pred_iou_thresh=self.conf,  # 对应 SAM2 的 conf
            # stability_score_thresh = 0.9,
            # stability_score_offset = 0.8,
            output_mode="binary_mask",
            multimask_output=True
        )
        
        # member variables
        self.observations = []
        # setup default filtering
        self.setup_filtering()

        assert self.device == 'cuda' or self.device == 'cpu', "Device should be 'cuda' or 'cpu'."
        assert self.rotate_img is None or self.rotate_img == 'CW' or self.rotate_img == 'CCW' \
            or self.rotate_img == '180', "Invalid rotate_img option."
        
            
    def setup_filtering(self,
        ignore_labels = [],
        use_keep_labels=False,
        keep_labels = [],
        keep_labels_option='intersect',          
        yolo_weights=None,
        yolo_det_img_size=None,
        area_bounds=np.array([0, np.inf]),
        allow_tblr_edges = [True, True, True, True],
        keep_mask_minimal_intersection=0.3,
        multimask_iou_thresh=0.2,
        multimask_contain_thresh=0.5,
        semantics: str = None,
        frame_descriptor: str = None,
        triangle_ignore_masks=None
    ):
        """
        Filtering setup function

        Args:
            ignore_labels (list, optional): List of yolo labels to ignore masks. Defaults to [].
            use_keep_labels (bool, optional): Use list of labels to only keep masks within keep mask. Defaults to False.
            keep_labels (list, optional): List of yolo labels to keep masks. Defaults to [].
            keep_labels_option (str, optional): 'intersect' or 'contain'. Defaults to 'intersect'.
            yolo_det_img_size (List[int], optional): Two-item list denoting yolo image size. Defaults to None.
            area_bounds (np.array, shape=(2,), optional): Two element array indicating min and max number of pixels. Defaults to np.array([0, np.inf]).
            allow_tblr_edges (list, optional): Allow masks touching top, bottom, left, and right edge. Defaults to [True, True, True, True].
            keep_mask_minimal_intersection (float, optional): Minimal intersection of mask within keep mask to be kept. Defaults to 0.3.
        """
        assert not use_keep_labels or keep_labels_option == 'intersect' or keep_labels_option == 'contain', "Keep labels option should be one of: intersect, contain"
        self.ignore_labels = ignore_labels
        self.use_keep_labels = use_keep_labels
        self.keep_labels = keep_labels
        self.keep_labels_option=keep_labels_option
        if len(ignore_labels) > 0 or use_keep_labels:
            if yolo_det_img_size is None:
                yolo_det_img_size=self.imgsz
            self.yolov7_det = Yolov7Detector(traced=False, img_size=yolo_det_img_size, weights=yolo_weights)
        
        self.area_bounds = area_bounds
        self.allow_tblr_edges= allow_tblr_edges
        self.keep_mask_minimal_intersection = keep_mask_minimal_intersection
        self.multimask_iou_thresh = float(multimask_iou_thresh)
        self.multimask_contain_thresh = float(multimask_contain_thresh)
        self.run_yolo = len(ignore_labels) > 0 or use_keep_labels
        self.semantics = semantics
        if semantics is None or semantics.lower() == 'none':
            self.semantics_model = None
            self.semantics_preprocess = None
        elif semantics.lower() == 'clip':
            clip_model = 'ViT-L/14'
            self.semantics_model, self.semantics_preprocess = clip.load(clip_model, device=self.device)
        elif semantics.lower() == 'dino':
            self.semantics_preprocess = AutoImageProcessor.from_pretrained('facebook/dinov2-base', do_center_crop=False)
            self.semantics_model = AutoModel.from_pretrained('facebook/dinov2-base')
            self.semantics_model.eval()
            self.semantics_model.to(self.device)
        else:
            raise ValueError(f"Invalid semantics option: {semantics}. Choose from 'clip', 'dino', or 'none'.")
        self.semantic_patches_shape = None
        self.frame_descriptor_type = frame_descriptor
        if frame_descriptor is not None:
            assert self.semantics is not None and self.semantics.lower() == 'dino', \
                "Frame descriptor only supported with DINO semantics."
        
        if triangle_ignore_masks is not None:
            self.constant_ignore_mask = np.zeros((self.depth_cam_params.height, self.depth_cam_params.width), dtype=np.uint8)
            for triangle in triangle_ignore_masks:
                assert len(triangle) == 3, "Triangle must have 3 points."
                for pt in triangle:
                    assert len(pt) == 2, "Each point must have 2 coordinates."
                    assert all([isinstance(x, int) for x in pt]), "Coordinates must be integers."
                cv.fillPoly(self.constant_ignore_mask, [np.array(triangle)], 1)
            self.constant_ignore_mask = self.apply_rotation(self.constant_ignore_mask)
        else:
            self.constant_ignore_mask = None
            
    def setup_rgbd_params(
        self, 
        depth_cam_params, 
        max_depth, 
        depth_scale=1e3,
        voxel_size=0.05, 
        within_depth_frac=0.5, 
        pcd_stride=4,
        erosion_size=3,
        plane_filter_ratio=None,
    ):
        """Setup params for processing RGB-D depth measurements

        Args:
            depth_cam_params (CameraParams): parameters of depth camera
            max_depth (float): maximum depth to be included in point cloud
            depth_scale (float, optional): scale of depth image. Defaults to 1e3.
            voxel_size (float, optional): Voxel size when downsampling point cloud. Defaults to 0.05.
            within_depth_frac(float, optional): Fraction of points that must be within max_depth. Defaults to 0.5.
            pcd_stride (int, optional): Stride for downsampling point cloud. Defaults to 4.
            plane_filter_ratio (List[float], optional): If an object's oriented bounding box's extent from max to min is > > <, mask is rejected. Defaults to None.
        """
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
            # see: https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
            erosion_shape = cv.MORPH_ELLIPSE
            self.erosion_element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                (erosion_size, erosion_size))
        else:
            self.erosion_element = None
        self.plane_filter_ratio = plane_filter_ratio

    def run(self, t, pose, img, depth_data=None):
        """
        Takes and image and returns filtered SAM2 masks as Observations.

        Args:
            img (cv image): camera image

        Returns:
            self.observations (list): list of Observations
            frame_descriptor (np.ndarray): semantic descriptor of the frame if frame_descriptor is not None, else None
        """
        self.observations = []
        
        # rotate image
        img_orig = img
        img = self.apply_rotation(img)

        if self.use_pointcloud:
            pcl, pcl_proj = depth_data

        if self.run_yolo:
            ignore_mask, keep_mask = self._create_mask(img)
        else:
            ignore_mask = None
            keep_mask = None

        if self.constant_ignore_mask is not None:
            ignore_mask = np.bitwise_or(ignore_mask, self.constant_ignore_mask) \
                if ignore_mask is not None else self.constant_ignore_mask  
        
        # run SAM2
        masks = self._process_img(img, ignore_mask=ignore_mask, keep_mask=keep_mask)

        if len(masks) > 1:
            # sort masks by area
            areas = [m.sum() for m in masks]
            indices = np.argsort(areas)[::-1]
            sorted_masks = masks[indices]
            
            keep_indices = []
            for i in range(len(sorted_masks)):
                m_i = sorted_masks[i].astype(bool)
                is_duplicate = False
                
                for idx in keep_indices:
                    m_kept = sorted_masks[idx].astype(bool)
                    # calculate IoU
                    intersection = np.logical_and(m_i, m_kept).sum()
                    union = np.logical_or(m_i, m_kept).sum()
                    iou = intersection / union

                    # if two masks have IoU > 0.2, they are considered duplicates of the same object
                    if iou > 0.2:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    keep_indices.append(i)
            
            # update masks
            masks = sorted_masks[keep_indices]
        
        if self.semantics == 'dino':
            # Process the image for DINO
            dino_shape = 768
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            preprocessed = self.semantics_preprocess(images=img_rgb, return_tensors="pt").to(self.device)
            dino_output = self.semantics_model(**preprocessed)
            dino_output_patches = self.get_output_patches(
                model_output=dino_output.last_hidden_state, 
                img_shape=img.shape, 
                feature_dim=dino_shape
            )
            dino_features = self.get_per_pixel_features(
                model_output_patches=dino_output_patches,
                img_shape=img.shape
            )
            dino_features = self.unapply_rotation(dino_features)
            
        frame_descriptor = None
        if self.frame_descriptor_type is not None:
            frame_descriptor = self.get_frame_descriptor(dino_output_patches)

        # Precompute strided depth samples and 3D points once per frame.
        depth_sample_rows = None
        depth_sample_cols = None
        depth_sample_xyz = None
        if depth_data is not None and not self.use_pointcloud:
            depth_img = np.asarray(depth_data)
            if depth_img.ndim == 3 and depth_img.shape[2] == 1:
                depth_img = depth_img[:, :, 0]
            if depth_img.ndim != 2:
                raise ValueError(f"Expected single-channel depth image, got shape {depth_img.shape}")

            h, w = depth_img.shape
            stride = max(1, int(self.pcd_stride))
            rows = np.arange(0, h, stride, dtype=np.int32)
            cols = np.arange(0, w, stride, dtype=np.int32)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            rr = rr.reshape(-1)
            cc = cc.reshape(-1)

            z = depth_img[rr, cc].astype(np.float32) / float(self.depth_scale)
            valid = np.isfinite(z) & (z > 0.0)
            depth_sample_rows = rr[valid]
            depth_sample_cols = cc[valid]
            z = z[valid]

            fx = float(self.depth_cam_params.fx)
            fy = float(self.depth_cam_params.fy)
            cx = float(self.depth_cam_params.cx)
            cy = float(self.depth_cam_params.cy)
            x = (depth_sample_cols.astype(np.float32) - cx) * z / fx
            y = (depth_sample_rows.astype(np.float32) - cy) * z / fy
            depth_sample_xyz = np.stack((x, y, z), axis=1)
        
        for mask in masks:
            
            mask = self.unapply_rotation(mask)
            # OpenCV morphology requires uint8, while SAM2 masks can be bool.
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            if mask.max() > 1:
                mask = (mask > 0).astype(np.uint8)

            # Extract point cloud of object from RGBD
            ptcld = None
            if depth_data is not None:
                if self.use_pointcloud:

                    # get 3D points that project within the mask
                    inside_mask = mask[pcl_proj[:, 1], pcl_proj[:, 0]] == 1
                    inside_mask_points = pcl[inside_mask]
                    pre_truncate_len = len(inside_mask_points)
                    ptcld_test = inside_mask_points[inside_mask_points[:, 2] < self.max_depth]

                    if len(ptcld_test) < self.within_depth_frac*pre_truncate_len:
                        continue

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(inside_mask_points)
                    
                else:
                    if depth_sample_xyz is None or depth_sample_xyz.shape[0] == 0:
                        continue
                    if self.erosion_element is not None:
                        sample_mask = cv.erode(mask.astype(np.uint8), self.erosion_element)
                    else:
                        sample_mask = mask
                    sample_mask = (sample_mask > 0)

                    inside_mask = sample_mask[depth_sample_rows, depth_sample_cols]
                    ptcld_test = depth_sample_xyz[inside_mask]
                    pre_truncate_len = len(ptcld_test)
                    if pre_truncate_len == 0:
                        continue

                    ptcld_test = ptcld_test[ptcld_test[:, 2] < self.max_depth]
                    # require some fraction of the points to be within the max depth
                    if len(ptcld_test) < self.within_depth_frac*pre_truncate_len:
                        continue

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(ptcld_test)

                # shared for depth & rangesens, once PointCloud object is created

                pcd.remove_non_finite_points()
                pcd_sampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
                if not pcd_sampled.is_empty():
                    ptcld = np.asarray(pcd_sampled.points)
                if ptcld is None:
                    continue
                
                if self.plane_filter_ratio is not None:
                    # Create oriented bounding box
                    try:
                        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                                o3d.utility.Vector3dVector(ptcld))
                        extent = np.sort(obb.extent)[::-1] # in descending order
                        if extent[0] / extent[2] > self.plane_filter_ratio or \
                            (extent[1] > 4.0 and \
                            extent[2] < 0.2):
                                continue
                    except:
                        continue
            
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Generate downsampled mask
            mask_downsampled = np.array(cv.resize(
                mask_uint8,
                (mask.shape[1]//self.mask_downsample_factor, mask.shape[0]//self.mask_downsample_factor), 
                interpolation=cv.INTER_NEAREST
            )).astype('uint8')

            if self.semantics == 'clip':
                ### Use bounding box
                bbox = self.mask_bounding_box(mask_uint8)
                if bbox is None:
                    assert False, "Bounding box is None."
                    self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld))
                else:
                    min_col, min_row, max_col, max_row = bbox
                    img_bbox = self.apply_rotation(img_orig[min_row:max_row, min_col:max_col])
                    img_bbox = cv.cvtColor(img_bbox, cv.COLOR_BGR2RGB)
                    processed_img = self.semantics_preprocess(Image.fromarray(img_bbox, mode='RGB')).to(self.device)
                    clip_embedding = self.semantics_model.encode_image(processed_img.unsqueeze(dim=0))
                    clip_embedding = clip_embedding.squeeze().cpu().detach().numpy()
                    self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld, semantic_descriptor=clip_embedding))
            elif self.semantics == 'dino':
                dino_mask_tensor = dino_features[mask] # (N, 768)
                # N = dino_mask_tensor.shape[0]

                variances = torch.var(dino_mask_tensor, dim=0, unbiased=True)
                total_variance = torch.sum(variances).item()
                mask_area = float(mask.sum())
                image_area = float(mask.size)
                area_ratio = (mask_area / image_area) if image_area > 0 else 0.0

                mean_feat = torch.mean(dino_mask_tensor, dim=0).detach().cpu().numpy()

                semantic_data = {
                    'mean': mean_feat.astype(np.float32),
                    'score': float(total_variance * area_ratio)
                }

                self.observations.append(Observation(
                    t, pose, mask, mask_downsampled, ptcld, 
                    semantic_descriptor=semantic_data
                ))
            else:
                self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld))
                
        return self.observations, frame_descriptor
    
    def apply_rotation(self, img, unrotate=False):
        if self.rotate_img is None:
            return img
        elif self.rotate_img == 'CW':
            k = 3 if not unrotate else 1
        elif self.rotate_img == 'CCW':
            k = 1 if not unrotate else 3
        elif self.rotate_img == '180':
            k = 2
        else:
            raise Exception("Invalid rotate_img option.")
        if type(img) == np.ndarray:
            result = np.rot90(img, k)
        else:
            result = torch.rot90(img, k)
        return result
        
    def unapply_rotation(self, img):
        return self.apply_rotation(img, unrotate=True)

    def _create_mask(self, img):
        
        if len(img.shape) == 2: # image is mono
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        classes, boxes, scores = self.yolov7_det.detect(img)
        ignore_boxes = []
        keep_boxes = []
        for i, cl in enumerate(classes[0]):
            if self.yolov7_det.names[cl] in self.ignore_labels:
                ignore_boxes.append(boxes[0][i])

            if self.yolov7_det.names[cl] in self.keep_labels:
                keep_boxes.append(boxes[0][i])

        ignore_mask = np.zeros(img.shape[:2]).astype(np.int8)
        for box in ignore_boxes:
            x0, y0, x1, y1 = np.array(box).astype(np.int64).reshape(-1).tolist()
            box_before_truncation = np.array([x0, y0, x1, y1])
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, ignore_mask.shape[1])
            y1 = min(y1, ignore_mask.shape[0])

            if x1 <= x0 or y1 <= y0:
                print("Ignore box: ", box_before_truncation)
                print("Ignore box after truncating: ", x0, y0, x1, y1)
                print("Ignore mask shape: ", ignore_mask.shape)
                continue
            ignore_mask[y0:y1, x0:x1] = np.ones((y1 - y0, x1 - x0)).astype(np.int8)
    

        if self.use_keep_labels:
            keep_mask = np.zeros(img.shape[:2]).astype(np.int8)
            for box in keep_boxes:
                x0, y0, x1, y1 = np.array(box).astype(np.int64).reshape(-1).tolist()
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x1, keep_mask.shape[1])
                y1 = min(y1, keep_mask.shape[0])
                if x1 <= x0 or y1 <= y0:
                    continue
                keep_mask[y0:y1, x0:x1] = np.ones((y1 - y0, x1 - x0)).astype(np.int8)
        else:
            keep_mask = None

        return ignore_mask, keep_mask
    
    def _delete_edge_masks(self, segmask):
        [numMasks, h, w] = segmask.shape
        contains_edge = np.zeros(numMasks).astype(np.bool_)
        for i in range(numMasks):
            mask = segmask[i,:,:]
            edge_width = 5
            # TODO: should be a parameter
            contains_edge[i] = (np.sum(mask[:,:edge_width]) > 0 and not self.allow_tblr_edges[2]) or (np.sum(mask[:,-edge_width:]) > 0 and not self.allow_tblr_edges[3]) or \
                            (np.sum(mask[:edge_width,:]) > 0 and not self.allow_tblr_edges[0]) or (np.sum(mask[-edge_width:, :]) > 0 and not self.allow_tblr_edges[1])
        return np.delete(segmask, contains_edge, axis=0)

    def _process_img(self, image_bgr, ignore_mask=None, keep_mask=None):
        """Process SAM2 on image, returns segment masks and center points from results

        Args:
            image_bgr ((h,w,3) np.array): color image
            SAM2Model (SAM2): SAM2 object
            device (str, optional): 'cuda' or 'cpu'. Defaults to 'cuda'.
            plot (bool, optional): Plots (slow) for visualization. Defaults to False.
            ignore_edges (bool, optional): Filters out edge-touching segments. Defaults to False.

        Returns:
            segmask ((n,h,w) np.array): n segmented masks (binary mask over image)
            blob_means ((n, 2) list): pixel means of segmasks
            blob_covs ((n, (2, 2) np.array) list): list of covariances (ellipses describing segmasks)
            (fig, ax) (Matplotlib fig, ax): fig and ax with visualization
        """

        # OpenCV uses BGR images, but SAM2 and Matplotlib require an RGB image, so convert.
        image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

        # Run SAM2
        # t1 = time.time()
        masks_info = self.mask_generator.generate(image)
        # if getattr(self.mask_generator, "multimask_output", False):
        masks_info = self._suppress_small_overlaps(masks_info)
        # t2 = time.time()
        # print("SAM2 Mask Generation Time: {:.3f} sec for image size {}".format(t2 - t1, image.shape))
        
        if len(masks_info) == 0:
            return []
        segmask = np.stack([m['segmentation'] for m in masks_info])
        [numMasks, h, w] = segmask.shape


        if not np.all(self.allow_tblr_edges):
            segmask = self._delete_edge_masks(segmask)
            [numMasks, h, w] = segmask.shape

        to_delete = []

        for maskId in range(numMasks):
            # Extract the single binary mask for this mask id
            mask_this_id = segmask[maskId,:,:]

            # filter out ignore mask
            if ignore_mask is not None and np.any(np.bitwise_and(mask_this_id.astype(np.int8), ignore_mask)):
                to_delete.append(maskId)
                continue

            # Only keep masks that are within keep_mask
            # if keep_mask is not None and not np.any(np.bitwise_and(mask_this_id.astype(np.int8), keep_mask)):
            #     print("Delete maskID: ", maskId)
            #     to_delete.append(maskId)
            #     continue
            # if keep_mask is not None and self.keep_labels_option == 'intersect' and (not np.any(np.bitwise_and(mask_this_id.astype(np.int8), keep_mask))):
            if keep_mask is not None and self.keep_labels_option == 'intersect' and (np.bitwise_and(mask_this_id.astype(np.int8), keep_mask).sum() < self.keep_mask_minimal_intersection*mask_this_id.astype(np.int8).sum()):
                to_delete.append(maskId)
                continue

            if self.area_bounds is not None:
                area = np.sum(mask_this_id)
                if area < self.area_bounds[0] or area > self.area_bounds[1]:
                    to_delete.append(maskId)
                    continue

        segmask = np.delete(segmask, to_delete, axis=0)

        return segmask

    def _suppress_small_overlaps(self, masks_info):
        if len(masks_info) <= 1:
            return masks_info

        def mask_area(m):
            if "area" in m:
                return m["area"]
            return int(np.count_nonzero(m["segmentation"]))

        sorted_masks = sorted(masks_info, key=mask_area, reverse=True)
        kept = []

        for m in sorted_masks:
            mask = m.get("segmentation")
            if mask is None:
                continue
            keep = True
            for km in kept:
                kmask = km["segmentation"]
                inter = np.logical_and(mask, kmask).sum()
                if inter == 0:
                    continue
                union = np.logical_or(mask, kmask).sum()
                if union == 0:
                    continue
                iou = inter / union
                area_mask = max(1, int(np.count_nonzero(mask)))
                area_kmask = max(1, int(np.count_nonzero(kmask)))
                contain = inter / min(area_mask, area_kmask)
                if iou >= self.multimask_iou_thresh or contain >= self.multimask_contain_thresh:
                    keep = False
                    break
            if keep:
                kept.append(m)

        return kept
    
    def mask_bounding_box(self, mask):
        # Find the indices of the True values
        true_indices = np.argwhere(mask)

        if len(true_indices) == 0:
            # No True values found, return None or an appropriate response
            return None

        # Calculate the mean of the indices
        mean_coords = np.mean(true_indices, axis=0)

        # Calculate the width and height based on the min and max indices in each dimension
        min_row, min_col = np.min(true_indices, axis=0)
        max_row, max_col = np.max(true_indices, axis=0)
        width = max_col - min_col + 1
        height = max_row - min_row + 1

        # Define a bounding box around the mean coordinates with the calculated width and height
        min_row = int(max(mean_coords[0] - height // 2, 0))
        max_row = int(min(mean_coords[0] + height // 2, mask.shape[0] - 1))
        min_col = int(max(mean_coords[1] - width // 2, 0))
        max_col = int(min(mean_coords[1] + width // 2, mask.shape[1] - 1))

        return (min_col, min_row, max_col, max_row,)

    def get_output_patches(self, model_output: ArrayLike, img_shape: ArrayLike, feature_dim: int) -> ArrayLike:
        """
        Extract (Dino) output patches

        Args:
            model_output (ArrayLike): Last hidden state of (Dino) model
            img_shape (ArrayLike): Original image shape
            feature_dim (int): Expected (Dino) feature dimension

        Returns:
            ArrayLike: Reshaped (Dino) output
        """
        model_output_flat_patches = model_output[:,1:, :]
        if self.semantic_patches_shape is None:
            ratio = img_shape[1] / img_shape[0] # width / height
            num_patches = model_output_flat_patches.shape[1]
            h = np.round(np.sqrt(num_patches / ratio)).astype(int) # number of patches along y-axis
            w = np.round(np.sqrt(num_patches * ratio)).astype(int) # number of patches along x-axis

            self.semantic_patches_shape = (1, h, w, feature_dim)
            
        model_output_patches = model_output_flat_patches.reshape(self.semantic_patches_shape)

        return model_output_patches # 1 x h x w x feature_dim

    def get_per_pixel_features(self, model_output_patches: ArrayLike, img_shape: ArrayLike) -> ArrayLike:
        """
        Extract (Dino) per-pixel features

        Args:
            model_output_patches (ArrayLike): Reshaped (Dino) output patches
            img_shape (ArrayLike): Original image shape

        Returns:
            ArrayLike: Reshaped (Dino) output
        """
        # interpolate the feature map to match the size of the original image
        per_pixel_features = torch.nn.functional.interpolate(
            model_output_patches.permute(0, 3, 1, 2), # permute to be batch, channels, height, width
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
        ) # 1 x dino_shape x h x w

        # reshape
        per_pixel_features = per_pixel_features[0].permute(1, 2, 0) # h x w x feature_dim

        return per_pixel_features # h x w x feature_dim
        
    def get_frame_descriptor(self, dino_features: torch.Tensor) -> np.ndarray:   
        with torch.no_grad(): # prevent memory leak
            dino_features_flat = dino_features.view(-1, dino_features.shape[-1])
            if self.frame_descriptor_type == 'dino-gap':
                frame_descriptor = torch.sum(dino_features_flat, dim=0)
            elif self.frame_descriptor_type == 'dino-gmp':
                frame_descriptor = torch.max(dino_features_flat, dim=0).values
            elif self.frame_descriptor_type == 'dino-gem':
                cubed_descriptor = torch.mean(dino_features_flat ** 3, dim=0)
                frame_descriptor = torch.sign(cubed_descriptor) * \
                                   (torch.abs(cubed_descriptor).clamp(min=1e-12) ** (1.0 / 3)) # avoid NaN from negative or zero root
            else:
                raise ValueError(f"frame descriptor must be one of 'dino-gap', 'dino-gmp', or 'dino-gem'.")
                
            frame_descriptor /= torch.norm(frame_descriptor)
                                            
        return frame_descriptor.cpu().detach().numpy()
            
