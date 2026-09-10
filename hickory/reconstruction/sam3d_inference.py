"""Small, stable SAM 3D Objects inference interface used by Hickory.

This module deliberately contains only the public inference surface Hickory
needs.  It can also replace the upstream ``notebook/inference.py`` during a
Docker build without pulling notebook and visualization dependencies into the
runtime path.
"""

from __future__ import annotations

import builtins
import os
from pathlib import Path
import numpy as np
import torch
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from hickory.utils.third_party import add_third_party_paths


add_third_party_paths()
if "CUDA_HOME" not in os.environ and os.environ.get("CONDA_PREFIX"):
    os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

# Importing the package registers the classes referenced by the Hydra config.
import sam3d_objects  # noqa: E402,F401
from sam3d_objects.pipeline.inference_pipeline_pointmap import (  # noqa: E402
    InferencePipelinePointMap,
)


__all__ = ["Inference", "load_image", "load_mask", "load_masks", "load_single_mask"]


_ALLOWED_TARGET_ROOTS = {"sam3d_objects", "torch", "torchvision", "moge"}
_BLOCKED_TARGETS = {
    builtins.exec,
    builtins.eval,
    builtins.__import__,
    os.kill,
    os.system,
    os.putenv,
    os.remove,
    os.removedirs,
    os.rmdir,
    os.fchdir,
    os.setuid,
    os.fork,
    os.forkpty,
    os.killpg,
    os.rename,
    os.renames,
    os.truncate,
    os.replace,
    os.unlink,
    os.fchmod,
    os.fchown,
    os.chmod,
    os.chown,
    os.chroot,
    os.lchown,
    builtins.help,
}


def _check_target(target: str) -> None:
    if target.split(".", 1)[0] not in _ALLOWED_TARGET_ROOTS:
        raise RuntimeError(f"Hydra target is not allowed: {target!r}")
    if get_method(target) in _BLOCKED_TARGETS:
        raise RuntimeError(f"Hydra target is blocked: {target!r}")


def _check_hydra_safety(config: DictConfig) -> None:
    pending: list[object] = [config]
    while pending:
        node = pending.pop()
        if isinstance(node, DictConfig):
            pending.extend(node.values())
            if "_target_" in node:
                _check_target(str(node["_target_"]))
        elif isinstance(node, ListConfig):
            pending.extend(node)


class Inference:
    """Load the pinned SAM3D point-map pipeline and expose one-image inference."""

    def __init__(self, config_file: str | Path, compile: bool = False):
        config_file = str(Path(config_file).expanduser().resolve())
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"
        config.compile_model = bool(compile)
        config.workspace_dir = os.path.dirname(config_file)
        _check_hydra_safety(config)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    @staticmethod
    def _merge_mask_to_rgba(image, mask) -> np.ndarray:
        image_array = np.asarray(image, dtype=np.uint8)
        if image_array.ndim != 3 or image_array.shape[-1] < 3:
            raise ValueError(f"Expected an RGB/RGBA image, got {image_array.shape}")
        if mask is None:
            if image_array.shape[-1] == 4:
                return image_array
            alpha = np.full(image_array.shape[:2] + (1,), 255, dtype=np.uint8)
        else:
            mask_array = np.asarray(mask)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., -1]
            if mask_array.shape != image_array.shape[:2]:
                raise ValueError(
                    f"Image and mask sizes differ: {image_array.shape[:2]} vs {mask_array.shape}"
                )
            alpha = ((mask_array > 0).astype(np.uint8) * 255)[..., None]
        return np.concatenate([image_array[..., :3], alpha], axis=-1)

    def __call__(
        self,
        image,
        mask=None,
        seed: int | None = None,
        pointmap: torch.Tensor | None = None,
    ) -> dict:
        rgba_image = self._merge_mask_to_rgba(image, mask)
        return self._pipeline.run(
            rgba_image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )


def load_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint8)


def load_mask(path: str | Path) -> np.ndarray:
    mask = load_image(path)
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask > 0


def load_masks(
    folder_path: str | Path,
    indices_list: list[int] | None = None,
    extension: str = ".png",
) -> list[np.ndarray]:
    folder = Path(folder_path)
    indices = [] if indices_list is None else list(indices_list)
    if not indices:
        index = 0
        while (folder / f"{index}{extension}").is_file():
            indices.append(index)
            index += 1
    masks = []
    for index in indices:
        path = folder / f"{index}{extension}"
        if not path.is_file():
            raise FileNotFoundError(f"Mask path does not exist: {path}")
        masks.append(load_mask(path))
    return masks


def load_single_mask(
    folder_path: str | Path,
    index: int = 0,
    extension: str = ".png",
) -> np.ndarray:
    return load_masks(folder_path, [index], extension)[0]
