#!/usr/bin/env bash
set -e

export PATH="/opt/conda/envs/hickory/bin:/opt/conda/bin:${PATH}"
export CONDA_DEFAULT_ENV="hickory"
export CONDA_PREFIX="${CONDA_PREFIX:-/opt/conda/envs/hickory}"
export CUDA_HOME="${CUDA_HOME:-/opt/conda/envs/hickory}"
export LD_LIBRARY_PATH="/opt/conda/envs/hickory/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

exec "$@"
