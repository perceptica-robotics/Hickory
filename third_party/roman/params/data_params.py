###########################################################
#
# data_params.py
#
# Parameter class for data loading
#
# Authors: Mason Peterson, Qingyuan Li
#
# Dec. 21, 2024
#
###########################################################

import json
import sqlite3
from dataclasses import dataclass
import numpy as np
import cv2
import yaml
from typing import List, Tuple
from functools import cached_property, partial
from pathlib import Path

from robotdatapy.data import ImgData, PoseData, PointCloudData
from robotdatapy.camera import CameraParams
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.highlevel import AnyReaderError
from robotdatapy.transform import T_FLURDF, T_RDFFLU

from roman.utils import expandvars_recursive, combinedicts_recursive


def _patch_rosbags_sqlite_temp_store():
    """Force SQLite temp storage to memory for rosbags readers.

    Some large ROS2 bags trigger sqlite temp-file open errors during grouped
    queries in rosbags; setting temp_store=MEMORY avoids those failures.
    """
    try:
        import rosbags.rosbag2.storage_sqlite3 as storage_sqlite3
    except Exception:
        return

    if getattr(storage_sqlite3, "_roman_sqlite_temp_store_patched", False):
        return

    original_connect = storage_sqlite3.sqlite3.connect

    def _connect_with_temp_store(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        try:
            conn.execute("PRAGMA temp_store=MEMORY")
        except Exception:
            pass
        return conn

    storage_sqlite3.sqlite3.connect = _connect_with_temp_store
    storage_sqlite3._roman_sqlite_temp_store_patched = True


_patch_rosbags_sqlite_temp_store()


class _FileCameraParams:
    def __init__(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float, frame_id: str = "camera"):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.frame_id = frame_id
        self.K = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.D = np.zeros((5,), dtype=np.float64)


class _FileImgData:
    def __init__(self, times, paths, loader, camera_params, time_tol):
        self.times = np.asarray(times, dtype=float)
        self.paths = [str(p) for p in paths]
        self._loader = loader
        self.camera_params = camera_params
        self.time_tol = float(time_tol)
        self.K = camera_params.K
        self.D = camera_params.D
        self.t0 = float(self.times[0])
        self.tf = float(self.times[-1])

    def nearest_time(self, t):
        idx = int(np.argmin(np.abs(self.times - t)))
        nearest_t = float(self.times[idx])
        if abs(nearest_t - t) > self.time_tol:
            try:
                from robotdatapy.data.robot_data import NoDataNearTimeException
                raise NoDataNearTimeException(f"No image within {self.time_tol} sec of {t}.")
            except ImportError:
                raise RuntimeError(f"No image within {self.time_tol} sec of {t}.")
        return nearest_t

    def img(self, t):
        idx = int(np.argmin(np.abs(self.times - t)))
        return self._loader(self.paths[idx])


class _FilePoseData:
    def __init__(self, times, poses, time_tol):
        self.times = np.asarray(times, dtype=float)
        self._poses = [np.asarray(p, dtype=np.float64) for p in poses]
        self.time_tol = float(time_tol)
        self.t0 = float(self.times[0])
        self.tf = float(self.times[-1])

    def pose(self, t):
        idx = int(np.argmin(np.abs(self.times - t)))
        nearest_t = float(self.times[idx])
        if abs(nearest_t - t) > self.time_tol:
            try:
                from robotdatapy.data.robot_data import NoDataNearTimeException
                raise NoDataNearTimeException(f"No pose within {self.time_tol} sec of {t}.")
            except ImportError:
                raise RuntimeError(f"No pose within {self.time_tol} sec of {t}.")
        return self._poses[idx]

    def T_WB(self, t):
        return self.pose(t)


class _TimeShiftedPoseData:
    def __init__(self, pose_data, time_offset: float):
        self._pose_data = pose_data
        self.time_offset = float(time_offset)
        base_times = getattr(pose_data, "times", None)
        if base_times is not None:
            self.times = np.asarray(base_times, dtype=float) - self.time_offset
            if len(self.times) > 0:
                self.t0 = float(self.times[0])
                self.tf = float(self.times[-1])
        elif hasattr(pose_data, "t0") and hasattr(pose_data, "tf"):
            self.t0 = float(pose_data.t0) - self.time_offset
            self.tf = float(pose_data.tf) - self.time_offset

    def pose(self, t):
        return self._pose_data.pose(float(t) + self.time_offset)

    def T_WB(self, t, *args, **kwargs):
        return self._pose_data.T_WB(float(t) + self.time_offset, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._pose_data, name)


def _load_color_png(path: str):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Color image not found: {path}")
    return img


def _load_depth_png(path: str):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    return depth


def _parse_homogeneous_matrix(path: str) -> np.ndarray:
    mat = np.loadtxt(str(path))
    if mat.size == 12:
        mat = mat.reshape(3, 4)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
    if mat.shape != (4, 4):
        raise ValueError(f"Unexpected 4x4 pose shape in {path}: {mat.shape}")
    return mat.astype(np.float64)


def _resolve_replicacad_path(sequence_root: str, rel_path: str, default_name: str) -> Path:
    root = Path(expandvars_recursive(sequence_root))
    if rel_path is not None:
        return root / rel_path
    return root / default_name


def _load_replicacad_camera_params(sequence_root: str, camera_params_path: str = None):
    root = Path(expandvars_recursive(sequence_root))
    candidate_paths = []
    if camera_params_path is not None:
        candidate_paths.append(Path(expandvars_recursive(camera_params_path)))
    candidate_paths.extend([root / "cam_params.json", root.parent / "cam_params.json"])
    cam_path = next((p for p in candidate_paths if p.exists()), None)
    if cam_path is None:
        raise FileNotFoundError(f"Could not find cam_params.json for ReplicaCAD sequence: {root}")
    cam_json = json.loads(cam_path.read_text())
    cam = cam_json.get("camera", cam_json)
    return _FileCameraParams(
        width=cam["w"],
        height=cam["h"],
        fx=cam["fx"],
        fy=cam["fy"],
        cx=cam["cx"],
        cy=cam["cy"],
        frame_id=cam.get("pose_frame", "camera"),
    )


def _parse_replicacad_pose_rows(poses_path: str):
    records = []
    for line in Path(poses_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 22:
            frame_idx = int(parts[0])
            rgb_time = float(parts[1])
            rgb_name = parts[2]
            depth_name = parts[4]
            T_wc = np.asarray([float(x) for x in parts[-16:]], dtype=np.float64).reshape(4, 4)
        elif len(parts) >= 19:
            frame_idx = int(parts[0])
            rgb_time = float(parts[1])
            rgb_name = parts[2]
            # Reconstruction odom dumps only store a frame name plus the 4x4 pose.
            # Reuse the RGB stem and map it onto the dataset depth naming convention.
            depth_name = Path(rgb_name).with_suffix(".png").name
            T_wc = np.asarray([float(x) for x in parts[-16:]], dtype=np.float64).reshape(4, 4)
        else:
            continue
        records.append((frame_idx, rgb_time, rgb_name, depth_name, T_wc))
    if len(records) == 0:
        raise ValueError(f"No valid ReplicaCAD pose rows found in {poses_path}")
    records.sort(key=lambda x: x[0])
    return records


def _replicacad_time_range(records, time_params):
    if time_params is None:
        return None
    if time_params["relative"]:
        t0 = records[0][1] + time_params["t0"]
        tf = records[0][1] + time_params["tf"]
    else:
        t0 = time_params["t0"]
        tf = time_params["tf"]
    return (t0, tf)


def _load_replicacad_records(sequence_root: str, poses_path: str = None):
    root = Path(expandvars_recursive(sequence_root))
    if poses_path is None:
        poses_file = root / "camera_poses.txt"
    else:
        poses_file = Path(expandvars_recursive(poses_path))
    if not poses_file.exists():
        raise FileNotFoundError(f"ReplicaCAD pose file not found: {poses_file}")
    return _parse_replicacad_pose_rows(str(poses_file))


def _filter_replicacad_records(records, time_range):
    if time_range is None:
        return records
    t0, tf = time_range
    filtered = [record for record in records if t0 <= record[1] <= tf]
    if len(filtered) == 0:
        raise ValueError(f"No ReplicaCAD frames found inside time range [{t0}, {tf}]")
    return filtered


def _iter_typestores():
    preferred = [
        "LATEST",
        "ROS2_HUMBLE",
        "ROS2_FOXY",
        "ROS2_GALACTIC",
        "ROS2_IRON",
        "ROS2_JAZZY",
        "ROS2_KILTED",
        "ROS2_DASHING",
        "ROS2_ELOQUENT",
        "ROS1_NOETIC",
    ]
    seen = set()
    for name in preferred:
        if hasattr(Stores, name):
            store = getattr(Stores, name)
            seen.add(store)
            yield store
    for store in Stores:
        if store not in seen:
            yield store


def _bag_metadata(path: str) -> dict:
    bag_path = Path(expandvars_recursive(path))
    metadata_path = bag_path / "metadata.yaml" if bag_path.is_dir() else bag_path.with_name("metadata.yaml")
    if not metadata_path.exists():
        return {}
    try:
        metadata = yaml.safe_load(metadata_path.read_text()) or {}
    except Exception:
        return {}
    return metadata.get("rosbag2_bagfile_information", metadata)


def _bag_db_paths(path: str) -> List[Path]:
    bag_path = Path(expandvars_recursive(path))
    if bag_path.is_file() and bag_path.suffix == ".db3":
        return [bag_path]

    metadata = _bag_metadata(path)
    relative_paths = metadata.get("relative_file_paths") or []
    if bag_path.is_dir() and relative_paths:
        return [bag_path / rel_path for rel_path in relative_paths]
    if bag_path.is_dir():
        return sorted(bag_path.glob("*.db3"))
    return [bag_path]


def _bag_typestore(path: str):
    metadata = _bag_metadata(path)
    ros_distro = str(metadata.get("ros_distro", "")).strip().upper()
    candidates = []
    if ros_distro:
        candidates.extend([
            f"ROS2_{ros_distro}",
            ros_distro,
        ])
    candidates.extend(["LATEST", "ROS2_JAZZY", "ROS2_HUMBLE", "ROS2_FOXY"])

    seen = set()
    for name in candidates:
        if not hasattr(Stores, name):
            continue
        store = getattr(Stores, name)
        if store in seen:
            continue
        seen.add(store)
        try:
            return get_typestore(store)
        except Exception:
            continue

    for store in _iter_typestores():
        try:
            return get_typestore(store)
        except Exception:
            continue
    return None


def _iter_sqlite_messages(path: str, topic: str):
    bag_paths = _bag_db_paths(path)
    if len(bag_paths) == 0:
        return

    topic = expandvars_recursive(topic)
    rows = []
    for db_path in bag_paths:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cursor = conn.execute(
                """
                SELECT topics.type, messages.timestamp, messages.data
                FROM messages
                JOIN topics ON topics.id = messages.topic_id
                WHERE topics.name = ?
                ORDER BY messages.timestamp
                """,
                (topic,),
            )
            rows.extend(cursor.fetchall())
        finally:
            conn.close()

    rows.sort(key=lambda row: row[1])
    for row in rows:
        yield row


def _deserialize_sqlite_topic(path: str, topic: str):
    typestore = _bag_typestore(path)
    if typestore is None:
        return None

    found = False
    for topic_type, timestamp, rawdata in _iter_sqlite_messages(path, topic):
        found = True
        yield topic_type, timestamp, typestore.deserialize_cdr(rawdata, topic_type)

    if not found:
        return


def _topic_time(msg, timestamp: int) -> float:
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return float(timestamp) * 1e-9


def _select_default_typestore(path: str):
    bag_path = expandvars_recursive(path)
    for store in _iter_typestores():
        try:
            typestore = get_typestore(store)
            with AnyReader([Path(bag_path)], default_typestore=typestore):
                return typestore
        except AnyReaderError:
            continue
        except Exception:
            continue
    return None


def _tf_tree_from_bag_tolerant(path: str, typestore):
    tf_tree = {}
    bag_path = Path(expandvars_recursive(path))
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        for tf_type in ["tf", "tf_static"]:
            connections = [x for x in reader.connections if x.topic == f"/{tf_type}"]
            for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if type(msg).__name__ != "tf2_msgs__msg__TFMessage":
                    continue
                for transform_msg in msg.transforms:
                    child = transform_msg.child_frame_id
                    parent = transform_msg.header.frame_id
                    if child in tf_tree:
                        existing_type, existing_parent = tf_tree[child]
                        if existing_parent != parent:
                            raise AssertionError(
                                f"child frame {child} has multiple parents"
                            )
                        if existing_type == tf_type:
                            continue
                        # Prefer dynamic /tf over /tf_static when both exist.
                        if existing_type == "tf_static" and tf_type == "tf":
                            tf_tree[child] = (tf_type, parent)
                        continue
                    tf_tree[child] = (tf_type, parent)
    return tf_tree


def _read_camera_params_from_bag(bag_path: str, topic: str) -> CameraParams:
    bag_path = expandvars_recursive(bag_path)
    topic = expandvars_recursive(topic)
    for store in _iter_typestores():
        try:
            typestore = get_typestore(store)
            with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
                connections = [x for x in reader.connections if x.topic == topic]
                if len(connections) == 0:
                    return None
                for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    return CameraParams.from_msg(msg)
        except AnyReaderError:
            continue
        except Exception:
            continue
    try:
        for _, _, msg in _deserialize_sqlite_topic(bag_path, topic):
            return CameraParams.from_msg(msg)
    except Exception:
        return None
    return None


def _img_data_from_bag_with_typestore(path, topic, time_range, time_tol, compressed, color_space, compressed_rvl):
    times = []
    img_msgs = []
    bag_path = expandvars_recursive(path)
    topic = expandvars_recursive(topic)
    for store in _iter_typestores():
        try:
            typestore = get_typestore(store)
            with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
                connections = [x for x in reader.connections if x.topic == topic]
                if len(connections) == 0:
                    return None
                for (connection, timestamp, rawdata) in reader.messages(connections=connections):
                    if connection.topic != topic:
                        continue
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    try:
                        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    except Exception:
                        t = timestamp
                    if time_range is not None and t < time_range[0]:
                        continue
                    elif time_range is not None and t > time_range[1]:
                        break
                    times.append(t)
                    img_msgs.append(msg)
                break
        except AnyReaderError:
            continue
        except AssertionError:
            continue
    if not times:
        try:
            for _, timestamp, msg in _deserialize_sqlite_topic(bag_path, topic):
                t = _topic_time(msg, timestamp)
                if time_range is not None and t < time_range[0]:
                    continue
                if time_range is not None and t > time_range[1]:
                    break
                times.append(t)
                img_msgs.append(msg)
        except Exception:
            return None
    if not times:
        return None
    img_msgs = [msg for _, msg in sorted(zip(times, img_msgs), key=lambda zipped: zipped[0])]
    times = sorted(times)
    return ImgData(
        times=times,
        imgs=img_msgs,
        data_type="bag",
        data_path=bag_path,
        time_tol=time_tol,
        compressed=compressed,
        color_space=color_space,
        compressed_rvl=compressed_rvl,
    )


def _pose_data_from_sqlite_topic(params_dict: dict) -> PoseData:
    path = expandvars_recursive(params_dict["path"])
    topic = expandvars_recursive(params_dict["topic"])
    interp = params_dict.get("interp", True)
    causal = params_dict.get("causal", False)
    time_tol = params_dict.get("time_tol", 0.1)
    T_premultiply = params_dict.get("T_premultiply")
    T_postmultiply = params_dict.get("T_postmultiply")

    times = []
    positions = []
    orientations = []
    t0 = None
    last_path_msg = None

    for topic_type, timestamp, msg in _deserialize_sqlite_topic(path, topic) or []:
        if t0 is None:
            t0 = _topic_time(msg, timestamp)
        if topic_type == "geometry_msgs/msg/PoseStamped":
            pose = msg.pose
            msg_time = _topic_time(msg, timestamp)
        elif topic_type == "nav_msgs/msg/Odometry":
            pose = msg.pose.pose
            msg_time = _topic_time(msg, timestamp)
        elif topic_type == "nav_msgs/msg/Path":
            last_path_msg = msg
            continue
        else:
            raise AssertionError("invalid msg type (not PoseStamped or Odometry)")

        times.append(msg_time)
        positions.append([pose.position.x, pose.position.y, pose.position.z])
        orientations.append([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ])

    if last_path_msg is not None:
        for pose_stamped in last_path_msg.poses:
            times.append(_topic_time(pose_stamped, 0))
            positions.append([
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z,
            ])
            orientations.append([
                pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w,
            ])

    if len(times) == 0:
        raise AssertionError(f"topic {topic} not found in bag file {path}")

    return PoseData(
        times,
        positions,
        orientations,
        interp=interp,
        causal=causal,
        time_tol=time_tol,
        t0=t0,
        T_premultiply=T_premultiply,
        T_postmultiply=T_postmultiply,
    )


def _topic_t_range_from_sqlite(path: str, topic: str):
    first_time, last_time = float("inf"), float("-inf")
    found = False
    for _, timestamp, msg in _deserialize_sqlite_topic(path, topic) or []:
        t = _topic_time(msg, timestamp)
        first_time = min(first_time, t)
        last_time = max(last_time, t)
        found = True
    if not found:
        return None
    return (first_time, last_time)

def find_transformation(bag_path, param_dict) -> np.array:
    """
    Converts a transform parameter dictionary into a transformation matrix.

    Returns:
        np.array: Transformation matrix.
    """
    if param_dict['input_type'] == 'string':
        if param_dict['string'] == 'T_FLURDF':
            return T_FLURDF
        elif param_dict['string'] == 'T_RDFFLU':
            return T_RDFFLU
        else:
            raise ValueError("Invalid string.")
    elif param_dict['input_type'] == 'tf':
        bag_path = expandvars_recursive(bag_path)
        # by default looks for a static tf, but if the user wants to reference a tf that is
        # theoretically static, but is published under /tf, then 'include_non_static_tf' can be set.
        if param_dict.get('include_non_static_tf', False):
            tf_data = PoseData.from_bag_tf(
                expandvars_recursive(bag_path),
                expandvars_recursive(param_dict['parent']),
                expandvars_recursive(param_dict['child'])
            )
            T = tf_data.pose(tf_data.t0)
        else:
            T = PoseData.any_static_tf_from_bag(
                expandvars_recursive(bag_path),
                expandvars_recursive(param_dict['parent']),
                expandvars_recursive(param_dict['child'])
            )
        if 'inv' in param_dict.keys() and param_dict['inv']:
            T = np.linalg.inv(T)
        return T
    elif param_dict['input_type'] == 'matrix':
        return np.array(param_dict['matrix']).reshape((4, 4))
    else:
        raise ValueError("Invalid input type.")

@dataclass
class ImgDataParams:
    
    path: str
    topic: str = None
    camera_info_topic: str = None
    compressed: bool = True
    compressed_rvl: bool = False
    color_space: str = None
    type: str = "bag"
    relative_path: str = None
    camera_params_path: str = None
    poses_path: str = None
    
    @classmethod
    def from_dict(cls, params_dict: dict):
        return cls(**params_dict)
    

@dataclass
class PointCloudDataParams:
    path: str
    topic: str
    T_camera_rangesense: None

    @classmethod
    def from_dict(cls, params_dict: dict):
        if 'T_camera_rangesense' in params_dict:
            params_dict['T_camera_rangesense'] = find_transformation(bag_path=params_dict['path'],
                                                                     param_dict=params_dict['T_camera_rangesense'])
        else:
            params_dict['T_camera_rangesense'] = None
        return cls(**params_dict)

@dataclass
class PoseDataParams:
    
    params_dict: dict
    T_camera_flu_dict: dict
    T_odombase_camera_dict: dict = None
    
    @classmethod
    def from_dict(cls, params_dict: dict):
        params_dict_subset = {k: v for k, v in params_dict.items() 
                       if k != 'T_camera_flu' and k != 'T_odombase_camera'}
        T_camera_flu_dict = params_dict['T_camera_flu']
        T_odombase_camera_dict = params_dict['T_odombase_camera'] \
            if 'T_odombase_camera' in params_dict else None
        return cls(params_dict=params_dict_subset, T_camera_flu_dict=T_camera_flu_dict, 
                   T_odombase_camera_dict=T_odombase_camera_dict)
        
    @property
    def T_camera_flu(self) -> np.array:
        return self._find_transformation(self.T_camera_flu_dict)
    
    @property
    def T_odombase_camera(self) -> np.array:
        if self.T_odombase_camera_dict is not None:
            return self._find_transformation(self.T_odombase_camera_dict)
        else:
            return np.eye(4)
        
    @property
    def odombase_frame(self) -> str:
        return self.T_odombase_camera_dict['parent'] if not self.T_odombase_camera_dict['inv'] else self.T_odombase_camera_dict['child']
        
    def load_pose_data(self, extra_key_vals: dict) -> PoseData:
        """
        Loads pose data.

        Returns:
            PoseData: Pose data object.
        """
        params_dict = {k: v for k, v in self.params_dict.items()}
        for k, v in extra_key_vals.items():
            params_dict[k] = v

        pose_type = params_dict.get("type")
        if pose_type == "replicacad":
            time_offset = float(params_dict.pop("time_offset", 0.0) or 0.0)
            records = _load_replicacad_records(
                sequence_root=params_dict["path"],
                poses_path=params_dict.get("poses_path"),
            )
            time_params = extra_key_vals.get("time_params")
            records = _filter_replicacad_records(records, _replicacad_time_range(records, time_params))
            times = [record[1] for record in records]
            poses = [record[4] for record in records]
            pose_data = _FilePoseData(times=times, poses=poses, time_tol=params_dict.get("time_tol", np.inf))
            if time_offset != 0.0:
                pose_data = _TimeShiftedPoseData(pose_data, time_offset)
            return pose_data
            
        # expand variables
        for k, v in params_dict.items():
            if type(params_dict[k]) == str:
                params_dict[k] = expandvars_recursive(params_dict[k])
        time_offset = float(params_dict.pop("time_offset", 0.0) or 0.0)
        try:
            pose_data = PoseData.from_dict(params_dict)
            if time_offset != 0.0:
                pose_data = _TimeShiftedPoseData(pose_data, time_offset)
            return pose_data
        except (AnyReaderError, AssertionError) as exc:
            pose_type = params_dict.get("type")
            if pose_type not in ("bag", "bag_tf"):
                raise
            err_text = str(exc).lower()
            needs_typestore = "no type definitions" in err_text
            multiple_tf_types = "multiple tf types" in err_text
            failed_type_parse = "failed to parse" in err_text
            if pose_type == "bag" and failed_type_parse:
                pose_data = _pose_data_from_sqlite_topic(params_dict)
                if time_offset != 0.0:
                    pose_data = _TimeShiftedPoseData(pose_data, time_offset)
                return pose_data
            if not needs_typestore and not multiple_tf_types and not failed_type_parse:
                raise
            typestore = _select_default_typestore(params_dict["path"])
            if typestore is None:
                if pose_type == "bag" and failed_type_parse:
                    pose_data = _pose_data_from_sqlite_topic(params_dict)
                    if time_offset != 0.0:
                        pose_data = _TimeShiftedPoseData(pose_data, time_offset)
                    return pose_data
                raise
            import robotdatapy.data.pose_data as pose_data_mod
            original_anyreader = pose_data_mod.AnyReader
            original_tf_tree = pose_data_mod.PoseData._tf_tree_from_bag
            try:
                if needs_typestore:
                    pose_data_mod.AnyReader = partial(AnyReader, default_typestore=typestore)
                try:
                    if multiple_tf_types and pose_type == "bag_tf":
                        pose_data_mod.PoseData._tf_tree_from_bag = (
                            lambda path: _tf_tree_from_bag_tolerant(path, typestore)
                        )
                    pose_data = PoseData.from_dict(params_dict)
                except AssertionError as inner_exc:
                    if (
                        "multiple tf types" not in str(inner_exc).lower()
                        or pose_type != "bag_tf"
                    ):
                        raise
                    pose_data_mod.PoseData._tf_tree_from_bag = (
                        lambda path: _tf_tree_from_bag_tolerant(path, typestore)
                    )
                    pose_data = PoseData.from_dict(params_dict)
            finally:
                pose_data_mod.AnyReader = original_anyreader
                pose_data_mod.PoseData._tf_tree_from_bag = original_tf_tree
            if time_offset != 0.0:
                pose_data = _TimeShiftedPoseData(pose_data, time_offset)
            return pose_data
    
    def _find_transformation(self, tf_dict) -> np.array:
        return find_transformation(self.params_dict["path"], tf_dict)
    
@dataclass
class DataParams:
    
    img_data_params: ImgDataParams
    depth_data_params: ImgDataParams
    pointcloud_data_params: PointCloudDataParams
    pose_data_params: PoseDataParams
    use_pointcloud: bool = False
    dt: float = 1/6
    runs: list = None
    run_env: str = None
    time_params: dict = None
    max_time: float = None
    kitti: bool = False
    
    def __post_init__(self):
        if self.time_params is not None:
            assert 'relative' in self.time_params, "relative must be specified in params"
            assert 't0' in self.time_params, "t0 must be specified in params"
            assert 'tf' in self.time_params, "tf must be specified in params"
        
    @classmethod
    def from_yaml(cls, yaml_path: str, run: str = None):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if run is None:
            if 'img_data' in data or 'pose_data' in data:
                run_data = data
            else:
                return cls(
                    None, None, None, None,
                    use_pointcloud=data['depth_source'] == 'pointcloud' if 'depth_source' in data else \
                                (data['use_pointcloud'] if 'use_pointcloud' in data else False),
                    dt=data['dt'] if 'dt' in data else 1/6,
                    runs=data['runs'] if 'runs' in data else None,
                    run_env=data['run_env'] if 'run_env' in data else None,
                    max_time=data['max_time'] if 'max_time' in data else None
                )
            return cls(
                ImgDataParams.from_dict(run_data['img_data']),
                ImgDataParams.from_dict(run_data['depth_data']) if 'depth_data' in run_data else None,
                PointCloudDataParams.from_dict(run_data['pointcloud_data']) if 'pointcloud_data' in run_data else None,
                PoseDataParams.from_dict(run_data['pose_data']),
                use_pointcloud=run_data['depth_source'] == 'pointcloud' if 'depth_source' in run_data else \
                              (run_data['use_pointcloud'] if 'use_pointcloud' in run_data else False),
                dt=run_data['dt'] if 'dt' in run_data else 1/6,
                runs=data['runs'] if 'runs' in data else None,
                run_env=data['run_env'] if 'run_env' in data else None,
                time_params=run_data['time'] if 'time' in run_data else None,
                max_time=run_data['max_time'] if 'max_time' in run_data else None,
                kitti=run_data['kitti'] if 'kitti' in run_data else False
            )
        elif run in data:
            run_data = combinedicts_recursive(data, data[run])
        else:
            run_data = data
        return cls(
            ImgDataParams.from_dict(run_data['img_data']),
            ImgDataParams.from_dict(run_data['depth_data']) if 'depth_data' in run_data else None,
            PointCloudDataParams.from_dict(run_data['pointcloud_data']) if 'pointcloud_data' in run_data else None,
            PoseDataParams.from_dict(run_data['pose_data']),
            use_pointcloud=run_data['depth_source'] == 'pointcloud' if 'depth_source' in run_data else \
                          (run_data['use_pointcloud'] if 'use_pointcloud' in run_data else False),
            dt=run_data['dt'] if 'dt' in run_data else 1/6,
            runs=data['runs'] if 'runs' in data else None,
            run_env=data['run_env'] if 'run_env' in data else None,
            time_params=run_data['time'] if 'time' in run_data else None,
            max_time=run_data['max_time'] if 'max_time' in run_data else None,
            kitti=run_data['kitti'] if 'kitti' in run_data else False
        )
        
    @property
    def time_range(self) -> Tuple[float, float]:
        return self._extract_time_range()
    
    def load_pose_data(self) -> PoseData:
        """
        Loads pose data.

        Returns:
            PoseData: Pose data object.
        """        
        if self.pose_data_params.T_odombase_camera is not None:
            T_postmultiply = self.pose_data_params.T_odombase_camera
        else:
            T_postmultiply = None
            
        extra_key_vals = {'T_postmultiply': T_postmultiply, 'interp': True}
        if self.time_params is not None:
            extra_key_vals['time_params'] = self.time_params
            
        return self.pose_data_params.load_pose_data(extra_key_vals)
        
    def load_pointcloud_data(self) -> PointCloudData:
        """
        Loads point cloud data.

        Returns:
            PointCloudData: PointCloud 
        """
        return PointCloudData.from_bag(
            path=expandvars_recursive(self.pointcloud_data_params.path),
            topic=expandvars_recursive(self.pointcloud_data_params.topic),
            time_tol=self.dt / 2.0,
            time_range=self.time_range
        )

    def load_img_data(self) -> ImgData:
        """
        Loads image data.
        
        Args:
            time_range (List[float, float]): Time range to load image data.

        Returns:
            ImgData: Image data object.
        """
        return self._load_img_data(color=True)
    
    def load_depth_data(self) -> ImgData:
        """
        Loads depth data.
        
        Args:
            time_range (List[float, float]): Time range to load depth data.

        Returns:
            ImgData: Depth data object.
        """
        return self._load_img_data(color=False)
        
    def _load_img_data(self, color=True) -> ImgData:
        """
        Loads color or depth image data.

        Args:
            color (bool, optional): True if color, False if depth. Defaults to True.

        Returns:
            ImgData: Image data object.
        """
        if self.kitti:
            img_data = ImgData.from_kitti(self.img_data_params.path, 'rgb' if color else 'depth')
            img_data.extract_params()
        else:
            if color:
                img_data_params = self.img_data_params
            else:
                img_data_params = self.depth_data_params
            if img_data_params.type == "replicacad":
                records = _load_replicacad_records(
                    sequence_root=img_data_params.path,
                    poses_path=img_data_params.poses_path,
                )
                records = _filter_replicacad_records(records, self.time_range)
                camera_params = _load_replicacad_camera_params(
                    sequence_root=img_data_params.path,
                    camera_params_path=img_data_params.camera_params_path,
                )
                if color:
                    frame_dir = _resolve_replicacad_path(img_data_params.path, img_data_params.relative_path, "rgb")
                    paths = [frame_dir / record[2] for record in records]
                    loader = _load_color_png
                else:
                    frame_dir = _resolve_replicacad_path(img_data_params.path, img_data_params.relative_path, "depth")
                    paths = [frame_dir / record[3] for record in records]
                    loader = _load_depth_png
                missing_paths = [str(p) for p in paths if not Path(p).exists()]
                if len(missing_paths) > 0:
                    raise FileNotFoundError(f"Missing ReplicaCAD frame files: {missing_paths[:3]}")
                img_data = _FileImgData(
                    times=[record[1] for record in records],
                    paths=paths,
                    loader=loader,
                    camera_params=camera_params,
                    time_tol=self.dt / 2.0,
                )
                return img_data
            img_file_path = expandvars_recursive(img_data_params.path)
            try:
                img_data = ImgData.from_bag(
                    path=img_file_path,
                    topic=expandvars_recursive(img_data_params.topic),
                    time_tol=self.dt / 2.0,
                    time_range=self.time_range,
                    compressed=img_data_params.compressed,
                    compressed_rvl=img_data_params.compressed_rvl,
                    color_space=img_data_params.color_space
                )
            except (AnyReaderError, AssertionError):
                img_data = _img_data_from_bag_with_typestore(
                    path=img_file_path,
                    topic=img_data_params.topic,
                    time_range=self.time_range,
                    time_tol=self.dt / 2.0,
                    compressed=img_data_params.compressed,
                    color_space=img_data_params.color_space,
                    compressed_rvl=img_data_params.compressed_rvl,
                )
                if img_data is None:
                    raise
            try:
                img_data.extract_params(expandvars_recursive(img_data_params.camera_info_topic))
            except (AnyReaderError, AssertionError):
                cam_params = _read_camera_params_from_bag(
                    img_file_path, img_data_params.camera_info_topic
                )
                if cam_params is None:
                    raise
                img_data.camera_params = cam_params
        return img_data
    
    def _extract_time_range(self) -> Tuple[float, float]:
        """
        Uses the params dictionary and image data to set an absolute time range for the data.

        Args:
            params (dict): Params dict.

        Returns:
            Tuple[float, float]: Beginning and ending time (or none).
        """
        if self.kitti:
            time_range = [self.time_params['t0'], self.time_params['tf']]
        else:
            if self.time_params is not None:
                if self.time_params['relative']:
                    topic_t0 = self.data_t0
                    time_range = [topic_t0 + self.time_params['t0'], 
                                  topic_t0 + self.time_params['tf']]
                else:
                    time_range = [self.time_params['t0'], 
                                  self.time_params['tf']]
            else:
                time_range = None
        return time_range
    
    @cached_property
    def data_t0(self) -> float:
        return self.data_t_range[0]
    
    @cached_property
    def data_tf(self) -> float:
        return self.data_t_range[1]
    
    @cached_property
    def data_t_range(self) -> float:
        if self.img_data_params.type == "replicacad":
            records = _load_replicacad_records(
                sequence_root=self.img_data_params.path,
                poses_path=self.img_data_params.poses_path,
            )
            return (records[0][1], records[-1][1])
        try:
            return ImgData.topic_t_range(expandvars_recursive(self.img_data_params.path), 
                                         expandvars_recursive(self.img_data_params.topic))
        except (AnyReaderError, AssertionError):
            time_range = _topic_t_range_from_sqlite(
                expandvars_recursive(self.img_data_params.path),
                expandvars_recursive(self.img_data_params.topic),
            )
            if time_range is None:
                raise
            return time_range
        
