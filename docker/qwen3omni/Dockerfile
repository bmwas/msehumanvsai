# =============================================================================
# Qwen3-Omni for NVIDIA Blackwell (RTX PRO 6000, 96GB)
# =============================================================================
#
# Uses CUDA 12.8 and compiles vLLM from the Qwen fork (no pre-built base for
# this fork + Blackwell; official vllm/vllm-openai is upstream, different code).
#
# FASTER BUILD (only for RTX PRO 6000 Blackwell on this machine):
#   docker build --build-arg TORCH_CUDA_ARCH_LIST="12.0" .
#   Builds only sm_120 kernels (~1/4 the time). Image will NOT run on Ada/Hopper.
#
# More parallelism if you have RAM/CPU (e.g. 16 cores, 32GB+ RAM):
#   --build-arg MAX_JOBS=16 --build-arg NVCC_THREADS=4
#
# Build time: full arch list ~60-90 min; 12.0 only ~15-25 min.
# =============================================================================

ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y && apt upgrade -y && apt install -y --no-install-recommends \
    git git-lfs python3 python3-pip python3-dev wget vim \
    libsndfile1 ccache software-properties-common ffmpeg \
  && rm -rf /var/lib/apt/lists/* \
  && ln -s /usr/bin/python3 /usr/bin/python \
  && git lfs install

WORKDIR /data/shared/Qwen3-Omni

# PyTorch with CUDA 12.8 and Blackwell sm_120 support.
# Stable 2.8.0 cu128 wheels may not include sm_120; nightly does (per PyTorch issue #157549).
RUN pip3 install --no-cache-dir --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# Build vLLM from Qwen fork with Blackwell support.
# TORCH_CUDA_ARCH_LIST: "12.0" = fast build, RTX PRO 6000 only; "8.0;9.0;10.0;12.0" = full.
ARG TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;12.0"
ARG MAX_JOBS=8
ARG NVCC_THREADS=4
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN pip3 install --no-cache-dir wheel setuptools packaging ninja cmake \
  && cd /tmp \
  && git clone -b qwen3_omni --depth 1 https://github.com/wangxiongts/vllm.git \
  && cd vllm \
  && pip3 install --no-cache-dir -r requirements/build.txt \
  && pip3 install --no-cache-dir -r requirements/cuda.txt \
  && pip3 install --no-cache-dir . -v --no-build-isolation \
  && cd / && rm -rf /tmp/vllm

# Transformers from main (Qwen3OmniMoeProcessor etc.)
RUN pip3 install --no-cache-dir "git+https://github.com/huggingface/transformers@main"

# Runtime dependencies
COPY requirements.runtime.txt /data/shared/Qwen3-Omni/requirements.runtime.txt
RUN pip3 install --no-cache-dir \
      flask==3.0.0 --no-deps \
  && pip3 install --no-cache-dir \
      werkzeug==3.0.1 --no-deps \
  && pip3 install --no-cache-dir \
      itsdangerous==2.1.2 jinja2==3.1.4 click==8.1.7 \
  && pip3 install --no-cache-dir -r /data/shared/Qwen3-Omni/requirements.runtime.txt

# App files (sitecustomize provides tokenizer compat patches for Blackwell)
COPY app.py entrypoint.sh wandb_logger.py /data/shared/Qwen3-Omni/
COPY sitecustomize.py /data/shared/Qwen3-Omni/sitecustomize.py

ENV PYTHONPATH=/data/shared/Qwen3-Omni
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV VLLM_USE_V1=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN chmod +x /data/shared/Qwen3-Omni/entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["/data/shared/Qwen3-Omni/entrypoint.sh"]
