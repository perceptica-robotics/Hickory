# Hickory

This is the code base for Hierarchical Object Representation for Spatial Robot Perception: Points, Meshes, and Superquadrics.

## Repository Layout

- `main.py`: end-to-end pipeline entry point.
- `hickory/`: project-owned Python package.
- `params/`: small YAML configuration files.
- `third_party/roman/` and `third_party/mps/`: lightweight vendored source used
  by Hickory.

## Third-Party Source Setup

Clone the upstream FoundationPose and SAM3D repositories into `third_party/`
before building Docker or doing a local install. The Dockerfile copies these
host checkouts into the image, applies small compatibility patches, and then
copies Hickory's local wrappers over the upstream entry points inside the image.

```bash
git clone https://github.com/facebookresearch/sam-3d-objects.git third_party/sam-3d-objects
git -C third_party/sam-3d-objects checkout 81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd

git clone https://github.com/NVlabs/FoundationPose.git third_party/FoundationPose
git -C third_party/FoundationPose checkout e3d597b8c6b851d053094ebd6fa240191c5238f8
```

## Docker Install

Prerequisites:

- Linux host with NVIDIA driver.
- Docker and Docker Compose.
- NVIDIA Container Toolkit for GPU access inside Docker.

For Docker, build the image after cloning the third-party source above.
Checkpoints and weights are not needed for the build.

```bash
docker compose build hickory
```

This can take a long time on the first build because CUDA extensions are
compiled for NATTEN, PyTorch3D, FoundationPose, nvdiffrast, and CLIPPER. Later
builds should reuse Docker cache unless dependency files change.

After the image is built, download checkpoints, weights, and datasets into the
host folders below before running examples. `docker-compose.yml` mounts these
host folders into the container at runtime.

## Runtime Assets

Create the expected local asset folders. For a local/non-Docker install, run the
third-party clone commands above before creating the FoundationPose or SAM3D
asset subdirectories.

```bash
mkdir -p weights dataset \
  third_party/FoundationPose/weights \
  third_party/sam-3d-objects/checkpoints
```

The dataset used for this work can be downloaded from
[Google Drive](https://drive.google.com/drive/folders/1VTURxeCEqp3sND1iDEv0pL_SdmlJU0hS).
Place the downloaded data under `dataset/`.

Manual model downloads:

### SAM3D checkpoints

[SAM3D](https://github.com/facebookresearch/sam-3d-objects) checkpoints are
hosted on [Hugging Face](https://huggingface.co/facebook/sam-3d-objects). You
must request access on that page first. After access is approved, generate a
Hugging Face access token and log in:

```bash
python -m pip install "huggingface-hub[cli]<1.0"
hf auth login
```

Then download the checkpoints into Hickory's mounted checkpoint folder:

```bash
TAG=hf
mkdir -p third_party/sam-3d-objects/checkpoints

hf download \
  --repo-type model \
  --local-dir third_party/sam-3d-objects/checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects

mv third_party/sam-3d-objects/checkpoints/${TAG}-download/checkpoints \
  third_party/sam-3d-objects/checkpoints/${TAG}

rm -rf third_party/sam-3d-objects/checkpoints/${TAG}-download
```

After this, the file below should exist:

```text
third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml
```

### FoundationPose weights

[FoundationPose](https://github.com/NVlabs/FoundationPose#data-prepare)
weights are linked from the upstream repository's Data prepare section.

Download all network weights from the upstream Google Drive folder and place
them under:

```text
third_party/FoundationPose/weights/
```

The upstream repository notes that the refiner should include
`2023-10-28-18-33-37` and the scorer should include `2024-01-11-20-02-45`.
After download, the expected layout is:

```text
third_party/FoundationPose/weights/2023-10-28-18-33-37/
third_party/FoundationPose/weights/2024-01-11-20-02-45/
```

*FastSAM and OneFormer will be downloaded automatically on
first run of reconstruction.

## Running Examples

### Hierarchical Reconstruction Pipeline:

```bash
# For image sequence datasets, e.g. HOPE and REPLICA:
docker compose run --rm hickory python main.py \
  --scene dataset/REPLICA/apt0/1/ \
  --param params/REPLICA/ \
  --output-dir reconstruction/REPLICA/apt_0_1_test

# For ROS bag datasets, e.g. NUS-CLB:
docker compose run --rm hickory python main.py params/b2_zed.yaml
```

### Hierarchical Representation Visualization:

```bash
xhost +local:root

docker compose run --rm hickory python hickory/visualization/scene.py \
  --objects-dir reconstruction/REPLICA/apt_0_1_test/

xhost -local:root
```

### Superquadrics for Map Alignment
Calculate the relative transform for two reconstructed scenes:

```bash
docker compose run --rm hickory python hickory/align/clipper_solve.py \
  --root reconstruction/REPLICA \
  --scene-a apt_0_1_test \
  --scene-b apt_0_2_test \
```

Visualize the alignment result:

```bash
xhost +local:root

docker compose run --rm hickory python hickory/visualization/alignment.py \
  --root reconstruction/REPLICA \
  --scene-a apt_0_1_test \
  --scene-b apt_0_2_test

xhost -local:root
```

### Superquadrics for Robot Navigation:

```bash
xhost +local:root

docker compose run --rm hickory python hickory/navigation/interactive_scene_rrt_3d.py \
  --objects-dir-a reconstruction/REPLICA/apt_0_1_test/ \
  --show-3d

xhost -local:root
```

## Preparing Custom Data

Hickory can run on either an RGB-D image folder or ROS bag data.

### RGB-D Image Folder

Prepare a calibrated RGB-D sequence with this layout:

```text
dataset/CUSTOM/scene_001/
  cam_params.json
  rgb/
    frame_00000.png
    frame_00001.png
  depth/
    frame_00000.png
    frame_00001.png
  camera_poses.txt
```

`cam_params.json` stores camera intrinsics and depth scale:

```json
{
  "camera": {
    "w": 1280,
    "h": 720,
    "fx": 910.0,
    "fy": 910.0,
    "cx": 640.0,
    "cy": 360.0,
    "scale": 1000.0
  }
}
```

Use `scale: 1000.0` for `uint16` millimeter depth PNGs and `scale: 1.0` for
meter depth. RGB and depth should have the same resolution, and depth should be
registered to the RGB camera.

`camera_poses.txt` has one row per frame. The last 16 numbers are the row-major
4x4 camera pose `T_WC`:

```text
# frame_index rgb_time rgb_name depth_time depth_name pose_time T_WC(row-major)
0 0.000000 frame_00000.png 0.000000 frame_00000.png 0.000000 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

Run image-sequence data with `--scene`, `--param`, and `--output-dir` as shown
in the reconstruction example above.

### ROS Bag Data

For ROS bag input, create a YAML config like `params/b2_zed.yaml`. It should
specify RGB image, depth image, camera info, and pose topics:

```yaml
dt: 0.25

img_data:
  path: "~/Hickory/dataset/NUS/CLB_1"
  topic: /zed/zed_node/rgb/color/rect/image
  camera_info_topic: /zed/zed_node/rgb/color/rect/camera_info
  compressed: True

depth_data:
  path: "~/Hickory/dataset/NUS/CLB_1"
  topic: /zed/zed_node/depth/depth_registered
  camera_info_topic: /zed/zed_node/depth/camera_info
  compressed: False

pose_data:
  type: bag
  path: "~/Hickory/dataset/NUS/CLB_1"
  topic: /lio_sam_ros2/mapping/odometry
  time_tol: 10.0
```

Run ROS bag data by passing the config file to `main.py`:

```bash
docker compose run --rm hickory python main.py params/b2_zed.yaml
```

Before running custom data, check that the RGB/depth streams are synchronized,
camera intrinsics match the image resolution, depth is in the expected units,
and poses are in meters.

## Acknowledgments

Hickory uses and builds on several third-party projects:
- Segmentation uses [FastSAM](https://github.com/CASIA-LMC-Lab/FastSAM) and [OneFormer](https://github.com/SHI-Labs/OneFormer).
- The mapping module is based on
  [ROMAN](https://github.com/mit-acl/ROMAN).
- Mesh reconstruction uses
  [SAM3D](https://github.com/facebookresearch/sam-3d-objects).
- Pose Estimation uses
  [FoundationPose](https://github.com/NVlabs/FoundationPose).
- Object association for map alignment uses
  [CLIPPER](https://github.com/mit-acl/clipper).
