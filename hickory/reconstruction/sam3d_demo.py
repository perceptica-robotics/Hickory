"""Hickory adapter for the vendored SAM 3D Objects inference pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch3d.transforms import Transform3d, quaternion_to_matrix

from hickory.utils.third_party import SAM3D_ROOT, add_third_party_paths


def _resolve_pipeline_config(config_path: Path | None, tag: str) -> Path:
    candidates: list[Path] = []
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.name == "pipeline.yaml":
            candidates.append(config_path)

    checkpoints = SAM3D_ROOT / "checkpoints"
    candidates.extend(
        [
            checkpoints / tag / "pipeline.yaml",
            checkpoints / f"{tag}-download" / "checkpoints" / "pipeline.yaml",
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the SAM3D pipeline checkpoint configuration. "
        f"Searched:\n  - {searched}"
    )


def _load_intrinsics(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Camera intrinsics not found: {path}")
    values = np.load(path) if path.suffix.lower() == ".npy" else np.loadtxt(path)
    values = np.asarray(values, dtype=np.float32)
    if values.size == 9:
        return values.reshape(3, 3)
    if values.size == 16:
        return values.reshape(4, 4)[:3, :3]
    raise ValueError(f"Expected a 3x3 or 4x4 intrinsics matrix in {path}, got {values.shape}")


def _metric_pointmap(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
) -> torch.Tensor:
    if depth_scale <= 0:
        raise ValueError(f"depth_scale must be positive, got {depth_scale}")

    z = np.asarray(depth, dtype=np.float32) / float(depth_scale)
    height, width = z.shape
    v, u = np.indices((height, width), dtype=np.float32)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid focal lengths: fx={fx}, fy={fy}")

    pointmap = np.stack(
        ((u - cx) * z / fx, (v - cy) * z / fy, z),
        axis=-1,
    )
    pointmap[~np.isfinite(z) | (z <= 0)] = np.nan

    # SAM3D consumes point maps in PyTorch3D camera coordinates. Apply the
    # same R3-camera conversion used by its own MoGe depth path.
    from sam3d_objects.pipeline.inference_pipeline_pointmap import (
        camera_to_pytorch3d_camera,
    )

    points = torch.from_numpy(pointmap)
    rotation = camera_to_pytorch3d_camera(device="cpu").rotation
    return Transform3d().rotate(rotation).transform_points(points)


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


class Sam3DRunner:
    """Load SAM3D once and reconstruct Hickory object directories one by one."""

    def __init__(
        self,
        config_path: Path | None,
        *,
        run: str | None = None,
        tag: str = "hf",
        compile_model: bool = False,
        intrinsics_path: Path | None = None,
        depth_scale: float = 1000.0,
        verbose: bool = False,
    ) -> None:
        del run, verbose  # Retained for compatibility with Hickory's CLI.
        add_third_party_paths()
        pipeline_config = _resolve_pipeline_config(config_path, tag)
        self.intrinsics = _load_intrinsics(intrinsics_path)
        self.depth_scale = float(depth_scale)

        from hickory.reconstruction.sam3d_inference import Inference

        self.inference = Inference(str(pipeline_config), compile=compile_model)

    def run_on_object(self, obj_dir: Path) -> dict:
        obj_dir = Path(obj_dir)
        image_path = obj_dir / "view.png"
        mask_path = obj_dir / "mask.png"
        depth_path = obj_dir / "depth.png"
        for path in (image_path, mask_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing SAM3D input: {path}")

        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if not np.any(mask):
            raise ValueError(f"Object mask is empty: {mask_path}")

        pointmap = None
        if self.intrinsics is not None and depth_path.is_file():
            depth = np.asarray(Image.open(depth_path))
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.shape != image.shape[:2]:
                raise ValueError(
                    f"Image and depth sizes differ: {image.shape[:2]} vs {depth.shape}"
                )
            pointmap = _metric_pointmap(depth, self.intrinsics, self.depth_scale)

        output = self.inference(image, mask, seed=42, pointmap=pointmap)
        glb = output.get("glb")
        if glb is None:
            raise RuntimeError("SAM3D returned no mesh/GLB output")

        scale = _as_numpy(output["scale"]).reshape(-1)
        metric_scale = float(np.mean(scale))
        if not np.isfinite(metric_scale) or metric_scale <= 0:
            raise ValueError(f"SAM3D returned invalid scale: {scale}")
        glb.apply_scale(metric_scale)
        mesh_path = obj_dir / "obj_mesh.glb"
        glb.export(str(mesh_path))

        translation = _as_numpy(output["translation"]).reshape(-1)[:3]
        quaternion = torch.as_tensor(
            _as_numpy(output["rotation"]).reshape(-1)[:4], dtype=torch.float32
        )
        rotation = quaternion_to_matrix(quaternion).cpu().numpy()
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        pose_path = obj_dir / "pose_sam3d.txt"
        np.savetxt(pose_path, pose, fmt="%.10f")

        return {
            "translation": translation.tolist(),
            "rotation": rotation.tolist(),
            "scale": scale.tolist(),
            "pose_path": str(pose_path),
            "mesh_path": str(mesh_path),
        }

