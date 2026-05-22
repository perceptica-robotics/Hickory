from pathlib import Path

import numpy as np


class CameraPoseRecorder:
    def __init__(self):
        self._world_from_first = None
        self.pose_records = []
        self.world_pose_records = []

    def add(self, frame_idx: int, timestamp: float, frame_name: str, T_WC_world) -> np.ndarray:
        T_WC_world = np.asarray(T_WC_world, dtype=np.float64)
        if T_WC_world.shape != (4, 4):
            raise ValueError(f"Expected 4x4 camera pose, got {T_WC_world.shape}")

        if self._world_from_first is None:
            self._world_from_first = np.linalg.inv(T_WC_world)

        T_WC_first_view_relative = self._world_from_first @ T_WC_world
        self.pose_records.append(
            (int(frame_idx), float(timestamp), str(frame_name), T_WC_first_view_relative.copy())
        )
        self.world_pose_records.append(
            (int(frame_idx), float(timestamp), str(frame_name), T_WC_world.copy())
        )
        return T_WC_first_view_relative

    def write(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        poses_path = output_dir / "camera_poses.txt"
        with open(poses_path, "w", encoding="utf-8") as f:
            f.write("# frame_index timestamp frame_name T_WC_first_view_relative (row-major 4x4)\n")
            for frame_idx, ts, fname, T_WC in self.pose_records:
                flat = " ".join(f"{v:.8f}" for v in np.asarray(T_WC).reshape(-1))
                f.write(f"{frame_idx} {ts:.6f} {fname} {flat}\n")

        world_poses_path = output_dir / "camera_poses_world.txt"
        with open(world_poses_path, "w", encoding="utf-8") as f:
            f.write("# frame_index timestamp frame_name T_WC_dataset_world (row-major 4x4)\n")
            for frame_idx, ts, fname, T_WC_world in self.world_pose_records:
                flat = " ".join(f"{v:.8f}" for v in np.asarray(T_WC_world).reshape(-1))
                f.write(f"{frame_idx} {ts:.6f} {fname} {flat}\n")
