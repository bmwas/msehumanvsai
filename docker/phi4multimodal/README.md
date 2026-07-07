# Phi-4 Multimodal vLLM Server

A production-ready vLLM-based serving infrastructure for Microsoft's Phi-4-multimodal-instruct model, designed for full compatibility with the qwen3omni `run_all_surveys_longitudinal.py` clinical analysis pipeline.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [GPU Setup](#gpu-setup)
- [Usage with qwen3omni](#usage-with-qwen3omni)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Model Information](#model-information)

## Overview

This repository provides a Docker-based deployment of the [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) model from Microsoft. The model is a lightweight (5.6B parameters) multimodal foundation model that processes text, images, and audio inputs.

### Key Features

- **vLLM Backend**: High-performance inference using vLLM with LoRA adapters for vision and speech
- **Compatibility Proxy**: Automatic conversion of video input to image frames
- **OpenAI-Compatible API**: Standard `/v1/chat/completions` endpoint
- **Production Ready**: Docker Compose deployment with health checks and restart policies
- **GPU Optimized**: Configured for the Ada RTX 6000 (48GB) GPU

### Why a Proxy?

The qwen3omni client (`run_all_surveys_longitudinal.py`) sends video data as `video_url` content parts. However, Phi-4-multimodal-instruct only supports images (`image_url`) and audio (`audio_url`), not video. The compatibility proxy:

1. Intercepts incoming requests with `video_url` content
2. Extracts frames from the video (default: 8 evenly-spaced frames)
3. Converts frames to JPEG images encoded as `image_url` parts
4. Forwards the transformed request to the vLLM backend
5. Returns the response unchanged to the client

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  HOST MACHINE                                │
│                                                                             │
│  ┌─────────────────────────┐      ┌─────────────────────────────────────┐  │
│  │    Client Application   │      │        GPU 1: Ada RTX 6000 48GB     │  │
│  │  run_all_surveys_       │      │                                     │  │
│  │  longitudinal.py        │      │  ┌─────────────────────────────┐   │  │
│  └────────────┬────────────┘      │  │      vLLM Backend           │   │  │
│               │                    │  │  Phi-4-multimodal-instruct  │   │  │
│               │ POST /v1/chat/     │  │  + speech-lora              │   │  │
│               │ completions        │  │  + vision-lora              │   │  │
│               │ (video_url,        │  │                             │   │  │
│               │  audio_url,        │  │  Internal Port: 8000        │   │  │
│               │  text)             │  └──────────────▲──────────────┘   │  │
│               ▼                    │                 │                   │  │
│  ┌─────────────────────────┐      │                 │                   │  │
│  │     phi4-api Proxy      │──────┼─────────────────┘                   │  │
│  │                         │      │   POST /v1/chat/completions         │  │
│  │  - video_url → images   │      │   (image_url[], audio_url, text)    │  │
│  │  - audio passthrough    │      │                                     │  │
│  │  - /health proxy        │      └─────────────────────────────────────┘  │
│  │                         │                                               │
│  │  External Port: 5200    │      ┌─────────────────────────────────────┐  │
│  └─────────────────────────┘      │    GPU 0: Blackwell RTX PRO 6000    │  │
│                                    │              (Reserved)             │  │
│                                    └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| `vllm-backend` | `phi4-vllm-backend` | 8000 (internal) | vLLM server with Phi-4-multimodal + LoRA |
| `phi4-api` | `phi4-api` | 5200 (external) | Compatibility proxy for video→image conversion |

## Requirements

### Hardware

- **GPU**: NVIDIA Ada Lovelace or newer (tested on RTX 6000 Ada 48GB)
- **VRAM**: Minimum 24GB recommended, 48GB optimal
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ for model weights (stored in `./models/`)

### Software

- Docker 24.0+ with NVIDIA Container Toolkit
- Docker Compose v2.20+
- NVIDIA Driver 535+ (for Ada GPUs)
- CUDA 12.x runtime

### Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# List available GPUs
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

## Quick Start

### 1. Clone and Configure

```bash
cd docker/phi4multimodal

# The .env file is pre-configured, but verify your settings
cat .env

# Key settings to verify:
# - HUGGINGFACE_TOKEN (for model download)
# - PHI4_GPU_DEVICE=1 (Ada GPU index)
# - PHI4_PORT=5200 (external port)
```

### 2. Build and Start

```bash
# Build both containers
docker compose build

# Start services (detached)
docker compose up -d

# Watch logs (model loading takes 3-5 minutes)
# On first run, the model (~12GB) will be downloaded to ./models/
docker compose logs -f
```

**Note:** On first startup, the Phi-4-multimodal-instruct model (~12GB) will be automatically downloaded to the `./models/` directory. This requires a valid `HUGGINGFACE_TOKEN` in your `.env` file. Subsequent startups will use the cached model.

### 3. Verify Deployment

```bash
# Check service status
docker compose ps

# Test health endpoint
curl http://localhost:5200/health

# Test models endpoint
curl http://localhost:5200/v1/models

# Simple text completion test
curl -X POST http://localhost:5200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-4-multimodal-instruct",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100
  }'
```

### 4. Stop Services

```bash
docker compose down
```

## Configuration

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | - | Required for model download |
| `API_TOKEN` | - | API authentication token |
| `MODEL_PATH` | `microsoft/Phi-4-multimodal-instruct` | HuggingFace model ID |
| `PHI4_GPU_DEVICE` | `1` | Host GPU device index |
| `PHI4_PORT` | `5200` | External API port |
| `VLLM_INTERNAL_PORT` | `8000` | Internal vLLM port |
| `GPU_MEMORY_UTILIZATION` | `0.90` | GPU memory fraction (0.0-1.0) |
| `MAX_MODEL_LEN` | `32768` | Maximum context length |
| `MAX_NUM_SEQS` | `8` | Maximum concurrent sequences |

### Model Storage

Models are stored locally in the `./models/` directory (mounted as `/app/models` in the container). This directory:

- Is **gitignored** (not committed to version control)
- Is **dockerignored** (not included in Docker builds)
- Contains the Phi-4-multimodal-instruct model (~12GB)
- Is automatically created on first startup
- Persists across container restarts

To clear the model cache and force re-download:
```bash
rm -rf ./models/hub
docker compose up -d
```
| `VIDEO_MAX_FRAMES` | `8` | Frames to extract from video |
| `FRAME_JPEG_QUALITY` | `85` | JPEG quality (0-100) |
| `VLLM_REQUEST_TIMEOUT` | `600` | Request timeout in seconds |

### Memory Tuning for Different GPUs

| GPU | VRAM | `GPU_MEMORY_UTILIZATION` | `MAX_MODEL_LEN` | `MAX_NUM_SEQS` |
|-----|------|--------------------------|-----------------|----------------|
| RTX 4090 | 24GB | 0.85 | 16384 | 4 |
| RTX 6000 Ada | 48GB | 0.90 | 32768 | 8 |
| A100 | 80GB | 0.92 | 65536 | 16 |

## GPU Setup

### Multi-GPU Systems

This deployment is designed for systems with multiple GPUs of different architectures:

- **GPU 0**: Blackwell RTX PRO 6000 96GB (reserved for other workloads)
- **GPU 1**: Ada RTX 6000 48GB (used by this service)

The configuration uses `device_ids: ["1"]` in docker-compose.yml to pin to GPU 1.

### Changing GPU

To use a different GPU, modify `.env`:

```bash
# Use GPU 0 instead
PHI4_GPU_DEVICE=0
```

Or override at runtime:

```bash
PHI4_GPU_DEVICE=0 docker compose up -d
```

### Verify GPU Allocation

```bash
# Check which GPU vLLM is using
docker compose exec vllm-backend nvidia-smi

# Should show only the allocated GPU
```

## Usage with qwen3omni

### Prerequisites

Ensure the qwen3omni environment is set up:

```bash
cd docker/phi4multimodal
```

### Running Longitudinal Analysis

```bash
# Start the Phi-4 server (from phi4multimodal directory)
cd docker/phi4multimodal
docker compose up

# Wait for model to load (check logs)
docker compose logs -f vllm-backend

# Once "Application startup complete" appears, run analysis from repo root
cd ../..

python run_all_surveys_longitudinal.py \
    --patients patient_name \
    --all-timepoints \
    --api-url http://localhost:5200 \
    --model microsoft/Phi-4-multimodal-instruct \
    --parallel-workers 2
```

### Important Flags

| Flag | Value | Description |
|------|-------|-------------|
| `--api-url` | `http://localhost:5200` | Must point to the proxy |
| `--model` | `microsoft/Phi-4-multimodal-instruct` | Must match served model |
| `--parallel-workers` | `2` | Adjust based on `MAX_NUM_SEQS` |

### Parallel Workers Recommendation

The number of parallel workers should not exceed `MAX_NUM_SEQS`:

| `MAX_NUM_SEQS` | Recommended `--parallel-workers` |
|----------------|----------------------------------|
| 4 | 2 |
| 8 | 4 |
| 16 | 5 (max supported by client) |

## API Reference

### Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "proxy": "healthy",
  "backend": {"status": "healthy"},
  "backend_url": "http://vllm-backend:8000"
}
```

#### `GET /v1/models`

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "microsoft/Phi-4-multimodal-instruct",
      "object": "model",
      "owned_by": "vllm"
    }
  ]
}
```

#### `POST /v1/chat/completions`

OpenAI-compatible chat completions.

**Request:**
```json
{
  "model": "microsoft/Phi-4-multimodal-instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see in this video?"},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}},
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}
      ]
    }
  ],
  "max_tokens": 1000,
  "temperature": 0.1
}
```

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "microsoft/Phi-4-multimodal-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The video shows..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 100,
    "total_tokens": 1334
  }
}
```

### Multimodal Input Types

| Type | Support | Notes |
|------|---------|-------|
| `text` | Native | Passed through unchanged |
| `image_url` | Native | Passed through unchanged |
| `audio_url` | Native | Passed through unchanged |
| `video_url` | Converted | Transformed to multiple `image_url` |

## Troubleshooting

### Common Issues

#### 1. Model Loading Timeout

**Symptom:** Container restarts repeatedly during startup.

**Solution:** First startup downloads ~12GB of model weights. Increase health check start period:

```yaml
# In docker-compose.yml
healthcheck:
  start_period: 600s  # 10 minutes
```

#### 2. Out of Memory (OOM)

**Symptom:** CUDA out of memory errors.

**Solution:** Reduce memory usage:

```bash
# In .env
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=16384
MAX_NUM_SEQS=4
```

#### 3. LoRA Adapters Not Found

**Symptom:** Warning about missing speech-lora or vision-lora.

**Solution:** The model may not have finished downloading. Check:

```bash
# Check HuggingFace cache
ls ~/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/

# Force re-download
docker compose exec vllm-backend python -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/Phi-4-multimodal-instruct')
"
```

#### 4. Connection Refused

**Symptom:** Proxy returns 502 errors.

**Solution:** vLLM backend not ready yet:

```bash
# Check backend logs
docker compose logs vllm-backend

# Wait for "Application startup complete"
```

#### 5. Slow Video Processing

**Symptom:** Requests with video take very long.

**Solution:** Reduce frames or image quality:

```bash
# In .env
VIDEO_MAX_FRAMES=4
FRAME_JPEG_QUALITY=70
FRAME_MAX_DIMENSION=512
```

### Debug Commands

```bash
# View all logs
docker compose logs -f

# Check container status
docker compose ps

# Inspect vLLM backend
docker compose exec vllm-backend nvidia-smi
docker compose exec vllm-backend python -c "import torch; print(torch.cuda.is_available())"

# Test backend directly (bypassing proxy)
docker compose exec phi4-api curl http://vllm-backend:8000/health

# Restart specific service
docker compose restart phi4-api
```

### Log Locations

| Component | Log Access |
|-----------|------------|
| vLLM Backend | `docker compose logs vllm-backend` |
| Proxy | `docker compose logs phi4-api` |
| All | `docker compose logs -f` |

## Model Information

### Phi-4-multimodal-instruct

- **Parameters**: 5.6B
- **Architecture**: Multimodal transformer with Phi-4-Mini backbone
- **Context Length**: 128K tokens (128,000)
- **Modalities**: Text, Image, Audio
- **Languages (Text)**: 23 languages including English, Chinese, French, German, etc.
- **Languages (Audio)**: 8 languages including English, Chinese, French, German, etc.
- **License**: MIT

### Capabilities

| Capability | Support |
|------------|---------|
| Text Generation | Yes |
| Image Understanding | Yes (via vision-lora) |
| Speech Recognition | Yes (via speech-lora) |
| Speech Translation | Yes |
| Visual Question Answering | Yes |
| Document Understanding | Yes |
| Chart/Table Understanding | Yes |

### Limitations

- **Video**: Not directly supported (this proxy converts to frames)
- **Audio Length**: Maximum ~40s recommended, 30min for summarization
- **Image Resolution**: Higher resolution = more tokens = more VRAM

## License

This deployment infrastructure is provided as-is. The Phi-4-multimodal-instruct model is licensed under the MIT License by Microsoft.

## References

- [Phi-4-multimodal-instruct on HuggingFace](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Phi-4 Technical Report](https://arxiv.org/abs/2503.01743)
- [Qwen3-Omni Server](../qwen3omni/)
