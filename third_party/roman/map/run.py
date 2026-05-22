###########################################################
#
# run.py
#
# A class for processing ROMAN mapping
#
# Authors: Mason Peterson, Yulun Tian, Lucas Jia, Qingyuan Li
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
import cv2 as cv

from os.path import expandvars, expanduser
from typing import Tuple
from dataclasses import dataclass
import time
from copy import deepcopy

from robotdatapy.data.img_data import ImgData
from robotdatapy.data.pose_data import PoseData
from robotdatapy.transform import transform, T_RDFFLU, T_FLURDF
from robotdatapy.data.robot_data import NoDataNearTimeException
from robotdatapy.camera import CameraParams

from roman.object.segment import Segment
from roman.viz import visualize_map_on_img, visualize_observations_on_img, visualize_3d_on_img
from roman.map.mapper import Mapper
from roman.map.observation import Observation
from roman.map.fastsam_wrapper import FastSAMWrapper
from roman.map.map import ROMANMap
from roman.map.align_pointcloud import AlignPointCloud
from roman.params.data_params import DataParams
from roman.params.mapper_params import MapperParams
from roman.params.fastsam_params import FastSAMParams

@dataclass
class ProcessingTimes:
    fastsam_times: list
    map_times: list
    total_times: list

class ROMANMapRunner:
    def __init__(self, data_params: DataParams, fastsam_params: FastSAMParams, 
                 mapper_params: MapperParams, verbose=False, viz_map=False, 
                 viz_observations=False, viz_3d=False, save_viz=False):
        self.data_params = data_params
        self.fastsam_params = fastsam_params
        self.mapper_params = mapper_params
        self.verbose = verbose

        if verbose: print("Extracting time range...")
        self.time_range = self.data_params.time_range

        if verbose: 
            print(f"Time range: {self.time_range}")
            print("Loading image data...")
        self.img_data = self.data_params.load_img_data()

        self.t0 = self.img_data.t0
        self.tf = self.img_data.tf

        if verbose: print("Loading pose data...")
        self.camera_pose_data = self.data_params.load_pose_data()

        if self.data_params.use_pointcloud:
            if verbose: print("Loading point cloud data...")
            # load PointCloudData class
            self.pointcloud_data = self.data_params.load_pointcloud_data()
            # load T_camera_rangesense (extract from tf if undefined)
            T_camera_rangesense = self.data_params.pointcloud_data_params.T_camera_rangesense
            if T_camera_rangesense is None:
                T_camera_rangesense = AlignPointCloud.extract_T_camera_rangesense(pointcloud_data=self.pointcloud_data, 
                                                                                 img_data=self.img_data, 
                                                                                 tf_bag_path=self.data_params.pointcloud_data_params.path)
            print(f"T_camera_rangesense:\n{T_camera_rangesense}")
            # create AlignPointCloud class for projecting pointcloud onto image frame
            self.align_pointcloud = AlignPointCloud(pointcloud_data=self.pointcloud_data,
                                                    img_data=self.img_data,
                                                    camera_pose_data=self.camera_pose_data,
                                                    T_camera_rangesense=T_camera_rangesense)
        else:
            if verbose: print("Loading depth data...")
            self.depth_data = self.data_params.load_depth_data()
        
        if verbose: print("Setting up FastSAM...")
        depth_cam_params = self.img_data.camera_params if self.data_params.use_pointcloud else \
            self.depth_data.camera_params
        self.fastsam = self._create_fastsam_wrapper(depth_cam_params)

        # TODO: start here
        if verbose: print("Setting up ROMAN mapper...")
        self.mapper = Mapper(self.mapper_params, self.img_data.camera_params)
        if self.data_params.pose_data_params.T_camera_flu is not None:
            self.mapper.set_T_camera_flu(self.data_params.pose_data_params.T_camera_flu)
        
        self.viz_map = viz_map
        self.viz_observations = viz_observations
        self.viz_3d = viz_3d
        self.save_viz = save_viz
        self.viz_imgs = []
        self.processing_times = ProcessingTimes([], [], [])

    def _create_fastsam_wrapper(self, depth_cam_params: CameraParams) -> FastSAMWrapper:
        fastsam = FastSAMWrapper(
            weights=expandvars(self.fastsam_params.weights_path),
            conf=self.fastsam_params.conf,
            iou=self.fastsam_params.iou,
            imgsz=tuple(self.fastsam_params.imgsz),
            device=self.fastsam_params.device,
            mask_downsample_factor=self.fastsam_params.mask_downsample_factor,
            rotate_img=self.fastsam_params.rotate_img,
            use_pointcloud=self.data_params.use_pointcloud,
            verbose=self.verbose,
        )

        img_h = int(depth_cam_params.height)
        img_w = int(depth_cam_params.width)
        min_mask_len = max(1, min(img_h, img_w) // int(self.fastsam_params.min_mask_len_div))
        max_mask_len = max(img_h, img_w) / max(float(self.fastsam_params.max_mask_len_div), 1.0)
        area_bounds = np.array(
            [
                float(min_mask_len ** 2),
                float(max_mask_len ** 2),
            ],
            dtype=np.float64,
        )

        ignore_labels = list(self.fastsam_params.ignore_labels)
        if self.fastsam_params.ignore_people and "person" not in ignore_labels:
            ignore_labels.append("person")

        fastsam.setup_filtering(
            ignore_labels=ignore_labels,
            use_keep_labels=self.fastsam_params.use_keep_labels,
            keep_labels=list(self.fastsam_params.keep_labels),
            keep_labels_option="intersect" if self.fastsam_params.keep_labels_option is None else self.fastsam_params.keep_labels_option,
            yolo_weights=expandvars(self.fastsam_params.yolo_weights_path),
            yolo_det_img_size=tuple(self.fastsam_params.yolo_imgsz),
            area_bounds=area_bounds,
            allow_tblr_edges=[True, True, True, True],
            semantics=self.fastsam_params.semantics,
            frame_descriptor=self.fastsam_params.frame_descriptor,
            triangle_ignore_masks=self.fastsam_params.triangle_ignore_masks,
        )
        fastsam.setup_rgbd_params(
            depth_cam_params=depth_cam_params,
            max_depth=self.fastsam_params.max_depth,
            depth_scale=self.fastsam_params.depth_scale,
            voxel_size=self.fastsam_params.voxel_size,
            erosion_size=self.fastsam_params.erosion_size,
            plane_filter_ratio=self.fastsam_params.plane_filter_params,
        )
        return fastsam

    def times(self):
        return np.arange(self.t0, self.tf, self.data_params.dt)

    def update(self, t: float): 
        t0 = self.img_data.t0

        if self.verbose: print(f"\nt: {t - t0:.2f} = {t}")
        img_output = None
        update_t0 = time.time()

        img_time, observations, pose_odom_camera, img, frame_descriptor = self.update_fastsam(t)
        update_t1 = time.time()
        if observations is not None and pose_odom_camera is not None and img is not None:
            img_output = self.update_segment_track(img_time, observations, pose_odom_camera, img, frame_descriptor)
        
        update_t2 = time.time()
        self.processing_times.map_times.append(update_t2 - update_t1)
        self.processing_times.fastsam_times.append(update_t1 - update_t0)
        self.processing_times.total_times.append(update_t2 - update_t0)

        return img_output

    def update_fastsam(self, t):

        try:
            img_t = self.img_data.nearest_time(t)
            img = self.img_data.img(img_t)
            T_odom_camera = self.camera_pose_data.T_WB(img_t)
            if self.data_params.use_pointcloud:
                pcl = self.align_pointcloud.aligned_pointcloud(img_t)
                pcl_proj  = self.align_pointcloud.projected_pointcloud(pcl)
                depth_data = self.align_pointcloud.filter_pointcloud_and_projection(pcl, pcl_proj)
            else:
                depth_data = self.depth_data.img(img_t)
        except NoDataNearTimeException:
            return None, None, None, None, None
        
        observations, frame_descriptor = self.fastsam.run(img_t, T_odom_camera, img, depth_data=depth_data)
        return img_t, observations, T_odom_camera, img, frame_descriptor

    def update_segment_track(self, t, observations, pose_odom_camera, img, frame_descriptor): 

        # collect reprojected masks
        reprojected_bboxs = []
        img_ret = None
        observations = self._normalize_observations_for_mapper(observations)
        if self.viz_observations:
            segment: Segment
            for i, segment in enumerate(self.mapper.segments + self.mapper.inactive_segments):
                bbox = segment.reprojected_bbox(pose_odom_camera)
                if bbox is not None:
                    reprojected_bboxs.append((segment.id, bbox))

        # update mapper with new frame
        self.mapper.update(t, pose_odom_camera, observations, frame_descriptor)
            
        if self.viz_map or self.viz_observations or self.viz_3d:
            img_ret = self.draw(t, img, pose_odom_camera, observations, reprojected_bboxs)

        if self.save_viz:
            self.viz_imgs.append(img_ret)

        return img_ret

    def _normalize_observations_for_mapper(self, observations):
        normalized = []
        for obs in observations:
            semantic_descriptor = obs.semantic_descriptor
            if isinstance(semantic_descriptor, dict):
                semantic_descriptor = semantic_descriptor.get("mean")
            normalized.append(
                Observation(
                    time=obs.time,
                    pose=obs.pose,
                    mask=obs.mask,
                    mask_downsampled=obs.mask_downsampled,
                    point_cloud=obs.point_cloud,
                    semantic_descriptor=semantic_descriptor,
                )
            )
        return normalized
        

    def draw(self, t, img, pose_odom_camera, observations, reprojected_bboxs):
        img_orig = deepcopy(img)
        
        if len(img_orig.shape) == 2:
            img = np.concatenate([img_orig[...,None]]*3, axis=2)
            
        if self.viz_map:
            img = visualize_map_on_img(t, pose_odom_camera, img, self.mapper)

        if self.viz_observations:
            img_fastsam = visualize_observations_on_img(t, img_orig, self.mapper, observations, reprojected_bboxs)
        
        if self.viz_3d:
            img_3d = visualize_3d_on_img(t, pose_odom_camera @ self.mapper.T_camera_flu, self.mapper)
                
        # rotate images
        img = self.fastsam.apply_rotation(img)
        if self.viz_observations:
            img_fastsam = self.fastsam.apply_rotation(img_fastsam)
                
        # concatenate images
        img_ret = np.zeros((img.shape[0], 0, img.shape[2]), dtype=np.uint8)
        if self.viz_map:
            img_ret = np.hstack([img_ret, img])
        if self.viz_3d:
            img_ret = np.hstack([img_ret, img_3d])
        if self.viz_observations:
            img_ret = np.hstack([img_ret, img_fastsam])

        return img_ret
