# Multimodal LLM Longitudinal Clinical Video Analysis

This code accompanies the preprint: **"Human vs AI Clinical Assessment: Benchmarking a Multimodal Foundation Model Against Multi-Center Expert Judgment on the Mental Status Examination"** ([medRxiv 2026.04.17.26351105](https://www.medrxiv.org/content/10.64898/2026.04.17.26351105v1)). It contains the inference pipeline, structured clinical prompts, and model serving infrastructure used to generate the multimodal foundation model predictions reported in the study.

This repository supports two models for the human-vs-AI manuscript:

| Model | Parameters | Modalities | GPU | Port |
|-------|-----------|------------|-----|------|
| **Qwen3-Omni-30B** (full BF16) | 30B (A3B MoE) | Audio + Video + Text | Blackwell RTX PRO 6000 (96GB) | 5100 |
| **Phi-4-multimodal-instruct** | 5.6B | Audio + Images + Text | Ada RTX 6000 (48GB) | 5200 |

## Structure

```
run_all_surveys_longitudinal.py       — Batch orchestrator: runs all MSE questions across patients/timepoints
engines/
  clinician_audio_video_segment_analysis_client_longitudinal.py
                                      — Core engine: segments video, queries the model, produces diagnosis
prompts/martin_et_al_another/
  prompt.yml                          — Clinical prompt templates (segment + meta-analysis)
mse_questions.csv                     — List of 45 MSE survey items to assess

docker/
  qwen3omni/                          — Qwen3-Omni server (Blackwell GPU, port 5100)
    example_env.txt                   — Configuration (cp to .env)
    app.py                            — FastAPI vLLM server (OpenAI-compatible)
    Dockerfile                        — CUDA 12.8 image, builds vLLM from Qwen fork
    docker-compose.yml                — Compose service definition
    entrypoint.sh                     — Container startup
    sitecustomize.py                  — Blackwell tokenizer patches
    requirements.runtime.txt          — Server runtime dependencies

  phi4multimodal/                     — Phi-4 Multimodal server (Ada GPU, port 5200)
    example_env.txt                   — Configuration (cp to .env)
    proxy/app.py                      — Compatibility proxy (video → image frames)
    Dockerfile.vllm                   — vLLM backend image
    Dockerfile.proxy                  — Proxy image
    docker-compose.yml                — Two-service compose (vLLM + proxy)
    entrypoint.sh                     — Backend startup
    README.md                         — Detailed Phi-4 documentation
```

## Requirements

- Python 3.10+
- `pandas`, `requests`, `pyyaml`
- `ffmpeg` / `ffprobe` (for audio/video segmentation)
- Docker 24.0+ with NVIDIA Container Toolkit
- NVIDIA GPU with sufficient VRAM (see table above)

## Model Server Setup

You only need to run **one** of the two models at a time. Both expose an OpenAI-compatible `/v1/chat/completions` endpoint.

### Option A: Qwen3-Omni (Blackwell, port 5100)

```bash
cd docker/qwen3omni
cp example_env.txt .env
# Edit .env: set HUGGINGFACE_TOKEN, adjust GPU parameters
docker compose up
```

Server available at `http://localhost:5100/v1/chat/completions`.

For faster builds targeting only RTX PRO 6000 (sm_120):
```bash
docker compose build --build-arg TORCH_CUDA_ARCH_LIST="12.0"
docker compose up
```

### Option B: Phi-4 Multimodal (Ada, port 5200)

```bash
cd docker/phi4multimodal
cp example_env.txt .env
# Edit .env: set HUGGINGFACE_TOKEN, adjust GPU device
docker compose up
```

Server available at `http://localhost:5200/v1/chat/completions`.

The Phi-4 deployment uses a compatibility proxy that converts video input into image frames, since Phi-4 does not natively support video. See `docker/phi4multimodal/README.md` for details.

## Running the Analysis Pipeline

```bash
# Using Qwen3-Omni (default port 5100)
python run_all_surveys_longitudinal.py \
  --patients patient_name \
  --all-timepoints \
  --mse-file mse_questions.csv \
  --prompt prompts/martin_et_al_another/prompt.yml \
  --api-url http://localhost:5100

# Using Phi-4 Multimodal (port 5200)
python run_all_surveys_longitudinal.py \
  --patients patient_name \
  --all-timepoints \
  --mse-file mse_questions.csv \
  --prompt prompts/martin_et_al_another/prompt.yml \
  --api-url http://localhost:5200
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--patients` | Patient name(s) to process |
| `--timepoint N` | Single timepoint (0, 1, 2...) |
| `--all-timepoints` | Auto-detect and run all timepoints sequentially |
| `--parallel-workers` | Concurrent survey questions (default: 5, max: 5) |
| `--api-url` | Model server endpoint (5100 for Qwen3, 5200 for Phi-4) |
| `--mse-file` | Path to MSE questions CSV |
| `--prompt` | Path to prompt YAML template |

## Configuration

Both models use `TEMPERATURE=0.1` for near-deterministic clinical assessment. This is set in the respective `example_env.txt` files and as the default in the engine client code.

## Data Layout (expected)

The script expects video files at:
```
<base-dir>/<patient>/video_<timepoint>/video_<timepoint>.mov
```
