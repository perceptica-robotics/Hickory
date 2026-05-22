FROM condaforge/miniforge3:24.11.3-0

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_EXTERNAL_DEPS=0
ARG INSTALL_SAM3D_RUNTIME_DEPS=1
ARG INSTALL_FOUNDATIONPOSE_RUNTIME_DEPS=0
ARG INSTALL_CLIPPER=1
ARG INSTALL_SAM2=0
ARG SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4
ARG CLIPPER_COMMIT=0666d38db21c9e30ea666de6d503efe32a017daa
ARG OPENAI_CLIP_COMMIT=a1d071733d7111c9c014f024669f959182114e33
ARG FASTSAM_COMMIT=4d153e909f0ad9c8ecd7632566e5a24e21cf0071
ARG NVDIFFRAST_REF=v0.4.0

ENV TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ROMAN_WEIGHTS=/workspace/weights \
    TORCH_HOME=/workspace/weights/torch \
    HF_HOME=/workspace/weights/huggingface \
    TRANSFORMERS_CACHE=/workspace/weights/huggingface/transformers \
    XDG_CACHE_HOME=/workspace/weights/cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libeigen3-dev \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxkbcommon-x11-0 \
    libxrender1 \
    lsb-release \
    mesa-utils \
    ninja-build \
    pkg-config \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:/opt/conda/envs/hickory/bin:$PATH \
    CONDA_DEFAULT_ENV=hickory \
    CUDA_HOME=/opt/conda/envs/hickory \
    FORCE_CUDA=1 \
    CUDA_ARCH="7.0;7.5;8.0;8.6;8.9;9.0" \
    NATTEN_CUDA_ARCH="7.0;7.5;8.0;8.6;8.9;9.0" \
    NATTEN_N_WORKERS=4 \
    NATTEN_WITH_CUDA=1 \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    LD_LIBRARY_PATH=/opt/conda/envs/hickory/lib:/usr/local/cuda/lib64

WORKDIR /workspace

COPY environment.yml requirements.txt pyproject.toml README.md main.py ./
COPY hickory ./hickory
COPY params ./params
COPY third_party/mps ./third_party/mps
COPY third_party/roman ./third_party/roman
COPY third_party/FoundationPose ./third_party/FoundationPose
COPY third_party/sam-3d-objects ./third_party/sam-3d-objects

RUN mamba env create -f environment.yml \
    && mamba clean -afy \
    && /opt/conda/envs/hickory/bin/python -m pip install -e .

RUN CUDA_HOME=/opt/conda/envs/hickory \
      FORCE_CUDA=1 \
      CUDA_ARCH="7.0;7.5;8.0;8.6;8.9;9.0" \
      NATTEN_CUDA_ARCH="7.0;7.5;8.0;8.6;8.9;9.0" \
      NATTEN_N_WORKERS=4 \
      NATTEN_WITH_CUDA=1 \
      MAX_JOBS=4 \
      /opt/conda/envs/hickory/bin/python -m pip install \
      --no-build-isolation --no-cache-dir --force-reinstall natten==0.14.6

RUN /opt/conda/envs/hickory/bin/python -m pip install --no-build-isolation \
      "clip @ git+https://github.com/openai/CLIP.git@${OPENAI_CLIP_COMMIT}" \
    && /opt/conda/envs/hickory/bin/python -m pip install --no-deps \
      "fastsam @ git+https://github.com/CASIA-IVA-Lab/FastSAM.git@${FASTSAM_COMMIT}" \
    && /opt/conda/envs/hickory/bin/python -c "import natten, clip; from fastsam import FastSAM; print('natten, clip, and fastsam installed')"

RUN if [ "${INSTALL_SAM3D_RUNTIME_DEPS}" = "1" ]; then \
      /opt/conda/envs/hickory/bin/python -m pip install --no-cache-dir --no-deps \
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
        gradio_client==0.2.9 \
        optree==0.14.1 \
        astor==0.8.1 \
        easydict==1.13 \
        plyfile==1.1.3 \
        pymeshfix==0.17.0 ; \
    fi

RUN if [ "${INSTALL_SAM3D_RUNTIME_DEPS}" = "1" ]; then \
      /opt/conda/envs/hickory/bin/python -m pip install --no-cache-dir \
        spconv-cu121==2.3.8 \
        lightning==2.3.3 \
        igraph==0.11.8 \
        kaolin==0.17.0 \
        -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu121.html ; \
    fi

RUN if [ "${INSTALL_SAM3D_RUNTIME_DEPS}" = "1" ]; then \
      CUDA_HOME=/opt/conda/envs/hickory \
      C_INCLUDE_PATH=/opt/conda/envs/hickory/targets/x86_64-linux/include:${C_INCLUDE_PATH} \
      CPLUS_INCLUDE_PATH=/opt/conda/envs/hickory/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH} \
      LIBRARY_PATH=/opt/conda/envs/hickory/targets/x86_64-linux/lib:${LIBRARY_PATH} \
      LD_LIBRARY_PATH=/opt/conda/envs/hickory/targets/x86_64-linux/lib:${LD_LIBRARY_PATH} \
      MAX_JOBS=4 \
      FORCE_CUDA=1 \
      /opt/conda/envs/hickory/bin/python -m pip install --no-cache-dir --no-build-isolation \
        "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" ; \
    fi

RUN if [ "${INSTALL_SAM3D_RUNTIME_DEPS}" = "1" ]; then \
      /opt/conda/envs/hickory/bin/python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        --extra-index-url https://pypi.ngc.nvidia.com \
        "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" ; \
    fi

RUN sed -i "s/find_package(Boost REQUIRED COMPONENTS system program_options)/find_package(Boost REQUIRED)/" \
        third_party/FoundationPose/mycpp/CMakeLists.txt \
      && sed -i "s/ \${Boost_LIBRARIES}//" \
        third_party/FoundationPose/mycpp/CMakeLists.txt \
      && sed -i "/include_directories(/a\\  /usr/include" \
        third_party/FoundationPose/mycpp/CMakeLists.txt \
      && sed -i "/find_package(OpenMP REQUIRED)/d; s/ \${OpenMP_CXX_FLAGS}//" \
        third_party/FoundationPose/mycpp/CMakeLists.txt \
      && sed -i "/#include <omp.h>/d" \
        third_party/FoundationPose/mycpp/include/Utils.h \
      && sed -i "s/^from gsplat import rasterization$/try:\\n    from gsplat import rasterization\\nexcept ImportError:\\n    rasterization = None/" \
        third_party/sam-3d-objects/sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py \
      && sed -i "/See reference code to convert from gsplat to inria:/i\\        if rasterization is None:\\n            raise ImportError('gsplat is required for the SAM3D Gaussian renderer backend.')" \
        third_party/sam-3d-objects/sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py

COPY hickory/reconstruction/fp_demo.py /tmp/hickory_foundationpose_run_demo.py
COPY hickory/reconstruction/sam3d_inference.py /tmp/hickory_sam3d_inference.py
RUN cp /tmp/hickory_foundationpose_run_demo.py third_party/FoundationPose/run_demo.py \
      && cp /tmp/hickory_sam3d_inference.py third_party/sam-3d-objects/notebook/inference.py

RUN if [ "${INSTALL_FOUNDATIONPOSE_RUNTIME_DEPS}" = "1" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends \
        libegl1-mesa-dev \
        libgles2-mesa-dev \
        libgl1-mesa-dev \
        libglvnd-dev \
        libboost-dev \
      && rm -rf /var/lib/apt/lists/* \
      && /opt/conda/envs/hickory/bin/python -m pip install --no-cache-dir \
        pybind11==2.12.0 \
        ruamel.yaml==0.18.6 \
        kornia==0.7.2 \
        transformations==2024.6.1 \
        warp-lang==1.0.2 \
        h5py==3.10.0 \
        psutil \
        joblib \
        "pyOpenGL>=3.1.0" \
        "pyOpenGL_accelerate>=3.1.0" \
      && CUDA_HOME=/opt/conda/envs/hickory \
        C_INCLUDE_PATH=/opt/conda/envs/hickory/targets/x86_64-linux/include:${C_INCLUDE_PATH} \
        CPLUS_INCLUDE_PATH=/opt/conda/envs/hickory/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH} \
        LIBRARY_PATH=/opt/conda/envs/hickory/targets/x86_64-linux/lib:${LIBRARY_PATH} \
        LD_LIBRARY_PATH=/opt/conda/envs/hickory/targets/x86_64-linux/lib:${LD_LIBRARY_PATH} \
        MAX_JOBS=4 \
        /opt/conda/envs/hickory/bin/python -m pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git@${NVDIFFRAST_REF}" \
      && CC=/usr/bin/gcc \
        CXX=/usr/bin/g++ \
        CMAKE_PREFIX_PATH=/opt/conda/envs/hickory/lib/python3.11/site-packages/pybind11/share/cmake/pybind11 \
        cmake -S third_party/FoundationPose/mycpp -B third_party/FoundationPose/mycpp/build \
          -DPYTHON_EXECUTABLE=/opt/conda/envs/hickory/bin/python \
          -DPYTHON_LIBRARY=/opt/conda/envs/hickory/lib/libpython3.11.so \
          -DPYTHON_INCLUDE_DIR=/opt/conda/envs/hickory/include/python3.11 \
      && CC=/usr/bin/gcc \
        CXX=/usr/bin/g++ \
        cmake --build third_party/FoundationPose/mycpp/build --parallel 4 \
      && /opt/conda/envs/hickory/bin/python -m pip install --no-cache-dir warp-lang==1.13.0 \
      && /opt/conda/envs/hickory/bin/python -c "import nvdiffrast.torch; import pytorch3d; import kornia; import ruamel.yaml; print('FoundationPose runtime deps OK')" ; \
    fi

RUN if [ "${INSTALL_SAM2}" = "1" ]; then \
      git clone https://github.com/facebookresearch/sam2.git third_party/sam2 \
      && git -C third_party/sam2 checkout "${SAM2_COMMIT}" \
      && /opt/conda/envs/hickory/bin/python -m pip install -e third_party/sam2 ; \
    fi

RUN if [ "${INSTALL_CLIPPER}" = "1" ]; then \
      git clone https://github.com/mit-acl/clipper.git /opt/clipper \
      && git -C /opt/clipper checkout "${CLIPPER_COMMIT}" \
      && cmake -S /opt/clipper -B /opt/clipper/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLIPPER_BUILD_BINDINGS_PYTHON=ON \
        -DCLIPPER_BUILD_BINDINGS_MATLAB=OFF \
        -DCLIPPER_BUILD_TESTS=OFF \
        -DCLIPPER_BUILD_BENCHMARKS=OFF \
        -DCLIPPER_ENABLE_MKL=OFF \
        -DCLIPPER_ENABLE_BLAS=OFF \
        -DCLIPPER_ENABLE_SCS_SDR=OFF \
        -DPYTHON_EXECUTABLE=/opt/conda/envs/hickory/bin/python \
      && cmake --build /opt/clipper/build --target pip-install --parallel "$(nproc)" \
      && /opt/conda/envs/hickory/bin/python -c "import clipperpy; print('clipperpy OK')" ; \
    fi

RUN if [ "${INSTALL_EXTERNAL_DEPS}" = "1" ]; then \
      /opt/conda/envs/hickory/bin/python -m pip install -r third_party/FoundationPose/requirements.txt \
      && /opt/conda/envs/hickory/bin/python -m pip install -e third_party/sam-3d-objects \
      && /opt/conda/envs/hickory/bin/python -m pip install -r third_party/sam-3d-objects/requirements.inference.txt ; \
    fi

COPY docker/entrypoint.sh /usr/local/bin/hickory-entrypoint
RUN chmod +x /usr/local/bin/hickory-entrypoint \
    && mkdir -p /workspace/weights /workspace/dataset /workspace/reconstruction

ENTRYPOINT ["hickory-entrypoint"]
CMD ["python", "main.py", "--help"]
