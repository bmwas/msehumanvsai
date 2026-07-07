"""
Qwen3-Omni API Server — FastAPI edition (Blackwell GPU builds)

Drop-in replacement for app.py that uses FastAPI + python-multipart instead of
Flask + werkzeug.  python-multipart streams uploads directly to disk via
SpooledTemporaryFile — there is NO form-memory-size limit and therefore NO
upload truncation bug.

Every endpoint has the exact same path, method, request schema and JSON response
format as the Flask version so existing clients work without changes.
"""
import os
import sys
import torch
import warnings
import tempfile
import logging
import threading
import queue
import time
import math
import copy
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

dotenv_path = Path(__file__).with_name('.env')
if dotenv_path.exists():
    # override=False so Docker-injected vars (e.g. from --env-file .env.blackwell) are not overwritten
    load_dotenv(dotenv_path=dotenv_path, override=False)
# Load baked-in .env.blackwell (from env-blackwell-example.txt at build time) so container works without --env-file.
# override=False so docker run --env-file .env.blackwell still overrides any value.
_env_blackwell = Path(__file__).with_name('.env.blackwell')
if _env_blackwell.exists():
    load_dotenv(dotenv_path=_env_blackwell, override=False)

_hf_token = os.getenv('HUGGINGFACE_TOKEN', '').strip()
_hf_user = os.getenv('HUGGINGFACE_USER', '').strip()

if _hf_token:
    os.environ['HF_TOKEN'] = _hf_token
    os.environ['HUGGING_FACE_HUB_TOKEN'] = _hf_token
    print("[Startup] Hugging Face token configured from HUGGINGFACE_TOKEN", file=sys.stderr)
    if _hf_user:
        print(f"[Startup] Hugging Face user: {_hf_user}", file=sys.stderr)
else:
    if not os.getenv('HF_TOKEN') and not os.getenv('HUGGING_FACE_HUB_TOKEN'):
        print("[Startup] Warning: No HUGGINGFACE_TOKEN found in .env. Private/gated models may fail to load.", file=sys.stderr)

try:
    from transformers import AutoConfig as _HF_AutoConfig, AutoModel as _HF_AutoModel

    _orig_ac_register = getattr(_HF_AutoConfig, "register", None)
    if callable(_orig_ac_register):
        def _safe_ac_register(model_type, config, exist_ok=False):
            return _orig_ac_register(model_type, config, exist_ok=True)
        _HF_AutoConfig.register = _safe_ac_register

    _orig_am_register = getattr(_HF_AutoModel, "register", None)
    if callable(_orig_am_register):
        def _safe_am_register(config, model, exist_ok=False):
            return _orig_am_register(config, model, exist_ok=True)
        _HF_AutoModel.register = _safe_am_register
except Exception:
    pass


def _preflight_check():
    try:
        import transformers as _t
        from packaging.version import Version
        required_min = Version("4.45.0")
        ver = Version(getattr(_t, "__version__", "0.0.0"))
        if ver < required_min:
            print(f"[Startup] Detected transformers=={ver}, upgrading recommended >= {required_min}.", file=sys.stderr)
        from transformers import Qwen3OmniMoeProcessor  # noqa: F401
        import transformers.modeling_rope_utils  # noqa: F401
    except Exception as e:
        print(f"[Startup] Transformers preflight failed: {e}", file=sys.stderr)
        print("[Startup] Ensure image built with --build-arg TRANSFORMERS_REF=main (or >=4.45)", file=sys.stderr)
        raise

_preflight_check()

from vllm import LLM, SamplingParams
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor

try:
    from wandb_logger import get_wandb_logger
except Exception:
    get_wandb_logger = None  # no-op if unavailable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Qwen3-Omni API Server (Blackwell)")

# ---------------------------------------------------------------------------
# Global model state & batching (identical to app.py)
# ---------------------------------------------------------------------------
model = None
processor = None

_request_queue: queue.Queue = queue.Queue()
_queue_thread = None
_queue_lock = threading.Lock()
_inference_lock = threading.Lock()
_BATCH_WAIT_TIME = 1.0
_MAX_BATCH_SIZE = 16

MODEL_PATH = os.getenv('MODEL_PATH', 'Qwen/Qwen3-Omni-30B-A3B-Thinking')
GPU_MEMORY_UTILIZATION = float(os.getenv('GPU_MEMORY_UTILIZATION', '0.92'))
MAX_MODEL_LEN = int(os.getenv('MAX_MODEL_LEN', '65536'))
_QUANTIZATION_RAW = os.getenv('QUANTIZATION', '').strip()
if _QUANTIZATION_RAW.lower() in ('', 'none', 'null', 'false', '0', 'no'):
    QUANTIZATION = None
else:
    QUANTIZATION = _QUANTIZATION_RAW

SEGMENT_DURATION = int(os.getenv('SEGMENT_DURATION', '30'))
SEGMENT_OVERLAP = int(os.getenv('SEGMENT_OVERLAP', '10'))

MAX_AUDIO_PLACEHOLDERS = int(os.getenv('AUDIO_PLACEHOLDER_LIMIT', '3'))
TOKEN_SAFETY_MARGIN = int(os.getenv('TOKEN_SAFETY_MARGIN', '1500'))
# Per-request default when client does not send max_tokens. Higher values reduce "chopped" responses.
# When running with docker run --env-file .env.blackwell, this (and all vars below) come from .env.blackwell.
try:
    DEFAULT_MAX_TOKENS = int(os.getenv('DEFAULT_MAX_TOKENS', '40960'))
except (ValueError, TypeError):
    DEFAULT_MAX_TOKENS = 40960
# Floor: never use less than this per request (avoids "Empty answer (hit max_tokens before closing </think>)")
try:
    MIN_MAX_TOKENS = int(os.getenv('MIN_MAX_TOKENS', '32768'))
except (ValueError, TypeError):
    MIN_MAX_TOKENS = 32768
MIN_MAX_TOKENS = max(256, min(MIN_MAX_TOKENS, 65536))
DEFAULT_MAX_TOKENS = max(MIN_MAX_TOKENS, min(DEFAULT_MAX_TOKENS, 65536))

_VIDEO_DOWNSAMPLE_ENABLED_RAW = os.getenv('ENABLE_VIDEO_DOWNSAMPLE', 'true').strip().lower()
ENABLE_VIDEO_DOWNSAMPLE = _VIDEO_DOWNSAMPLE_ENABLED_RAW not in ('false', '0', 'no')
try:
    VIDEO_MAX_FPS = int(os.getenv('VIDEO_MAX_FPS', '1'))
except ValueError:
    VIDEO_MAX_FPS = 1
try:
    VIDEO_MAX_FRAMES = int(os.getenv('VIDEO_MAX_FRAMES', '16'))
except ValueError:
    VIDEO_MAX_FRAMES = 16
try:
    VIDEO_MAX_WIDTH = int(os.getenv('VIDEO_MAX_WIDTH', '192'))
except ValueError:
    VIDEO_MAX_WIDTH = 192
try:
    VIDEO_MAX_HEIGHT = int(os.getenv('VIDEO_MAX_HEIGHT', '192'))
except ValueError:
    VIDEO_MAX_HEIGHT = 192
try:
    VIDEO_MIN_FRAMES = int(os.getenv('VIDEO_MIN_FRAMES', '12'))
except ValueError:
    VIDEO_MIN_FRAMES = 12
if VIDEO_MIN_FRAMES is not None and VIDEO_MIN_FRAMES < 0:
    VIDEO_MIN_FRAMES = 0
if VIDEO_MIN_FRAMES and VIDEO_MIN_FRAMES > 0:
    if VIDEO_MAX_FRAMES and VIDEO_MAX_FRAMES > 0 and VIDEO_MAX_FRAMES < VIDEO_MIN_FRAMES:
        logger.warning(
            "VIDEO_MAX_FRAMES (%s) < VIDEO_MIN_FRAMES (%s); raising max.",
            VIDEO_MAX_FRAMES, VIDEO_MIN_FRAMES,
        )
        VIDEO_MAX_FRAMES = VIDEO_MIN_FRAMES

_MAX_NUM_SEQS_RAW = os.getenv('MAX_NUM_SEQS', '').strip()
MAX_NUM_SEQS = int(_MAX_NUM_SEQS_RAW) if _MAX_NUM_SEQS_RAW else None

_MAX_NUM_BATCHED_TOKENS_RAW = os.getenv('MAX_NUM_BATCHED_TOKENS', '').strip()
MAX_NUM_BATCHED_TOKENS = int(_MAX_NUM_BATCHED_TOKENS_RAW) if _MAX_NUM_BATCHED_TOKENS_RAW else None

_ENABLE_CHUNKED_PREFILL_RAW = os.getenv('ENABLE_CHUNKED_PREFILL', '').strip()
if _ENABLE_CHUNKED_PREFILL_RAW.lower() in ('true', '1', 'yes', 't', 'y'):
    ENABLE_CHUNKED_PREFILL = True
elif _ENABLE_CHUNKED_PREFILL_RAW.lower() in ('false', '0', 'no', 'f', 'n'):
    ENABLE_CHUNKED_PREFILL = False
else:
    ENABLE_CHUNKED_PREFILL = True


# ---------------------------------------------------------------------------
# Helper functions (copied verbatim from app.py)
# ---------------------------------------------------------------------------

def _contains_audio_entries(messages):
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'audio':
                return True
    return False


def _probe_video_duration(video_path):
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        value = result.stdout.strip()
        if value:
            return float(value)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else '(no stderr)'
        logger.warning("Unable to probe duration for %s (exit %s): %s", video_path, exc.returncode, stderr)
    except Exception as exc:
        logger.warning("Unable to probe duration for %s: %s", video_path, exc)
    return None


def _probe_video_frame_count(video_path):
    probes = [
        ['-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames'],
        ['-select_streams', 'v:0', '-show_entries', 'stream=nb_frames'],
    ]
    for extra_args in probes:
        try:
            cmd = ['ffprobe', '-v', 'error', *extra_args, '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            value = result.stdout.strip()
            if value and value.upper() != 'N/A':
                return int(float(value))
        except Exception:
            continue
    return None


def _is_valid_local_video(video_path: str) -> bool:
    """Quick sanity check that a local file is a decodable video stream."""
    if not video_path or not os.path.isfile(video_path):
        return False
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return 'video' in (result.stdout or '').lower()
    except Exception:
        return False


def _maybe_downsample_video(video_path):
    if not ENABLE_VIDEO_DOWNSAMPLE or not video_path:
        return video_path, None
    try:
        if not os.path.isfile(video_path):
            return video_path, None
    except Exception:
        return video_path, None

    duration = _probe_video_duration(video_path)
    min_frames = VIDEO_MIN_FRAMES if VIDEO_MIN_FRAMES and VIDEO_MIN_FRAMES > 0 else None
    target_fps = VIDEO_MAX_FPS if VIDEO_MAX_FPS and VIDEO_MAX_FPS > 0 else None
    if target_fps is None:
        target_fps = 1.0
    if duration and min_frames:
        required_fps = min_frames / max(duration, 1e-3)
        if required_fps > target_fps:
            logger.info("Increasing fps from %.3f to %.3f for %s to satisfy VIDEO_MIN_FRAMES=%s (duration=%.2fs)",
                        target_fps, required_fps, video_path, min_frames, duration)
            target_fps = required_fps

    max_frames = VIDEO_MAX_FRAMES if VIDEO_MAX_FRAMES and VIDEO_MAX_FRAMES > 0 else None
    if min_frames and (max_frames is None or max_frames < min_frames):
        max_frames = min_frames
    if target_fps is None and max_frames is None:
        scale_needed = VIDEO_MAX_WIDTH and VIDEO_MAX_WIDTH > 0 or VIDEO_MAX_HEIGHT and VIDEO_MAX_HEIGHT > 0
        if not scale_needed:
            return video_path, None

    suffix = Path(video_path).suffix or '.mp4'
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_video.close()

    cmd = ['ffmpeg', '-y', '-v', 'error', '-i', video_path]
    filters = []
    scale_needed = (VIDEO_MAX_WIDTH and VIDEO_MAX_WIDTH > 0) or (VIDEO_MAX_HEIGHT and VIDEO_MAX_HEIGHT > 0)
    if scale_needed:
        target_w = VIDEO_MAX_WIDTH if VIDEO_MAX_WIDTH and VIDEO_MAX_WIDTH > 0 else 'iw'
        target_h = VIDEO_MAX_HEIGHT if VIDEO_MAX_HEIGHT and VIDEO_MAX_HEIGHT > 0 else 'ih'
        scale_expr = f"scale=w='min(iw\\,{target_w})':h='min(ih\\,{target_h})':force_original_aspect_ratio=decrease"
        filters.append(scale_expr)
    if target_fps is not None:
        fps_expr = f"{target_fps:.6f}".rstrip('0').rstrip('.')
        filters.append(f'fps={fps_expr}')
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    if max_frames is not None:
        cmd.extend(['-frames:v', str(max_frames)])
    _video_codec = os.environ.get('FFMPEG_VIDEO_CODEC', 'libx264')
    cmd.extend(['-c:v', _video_codec, '-preset', 'ultrafast', '-crf', '30', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '128k', temp_video.name])

    try:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode('utf-8', errors='replace').strip() if exc.stderr else '(no stderr)'
            logger.info("ffmpeg failed (exit %s) with codec %s: %s", exc.returncode, _video_codec, stderr)
            logger.info("Retrying with mpeg4 (minimal flags)")
            fallback_cmd = ['ffmpeg', '-y', '-v', 'error', '-i', video_path]
            if filters:
                fallback_cmd.extend(['-vf', ','.join(filters)])
            if max_frames is not None:
                fallback_cmd.extend(['-frames:v', str(max_frames)])
            fallback_cmd.extend(['-c:v', 'mpeg4', '-q:v', '5', '-an', temp_video.name])
            try:
                subprocess.run(fallback_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as exc2:
                stderr2 = exc2.stderr.decode('utf-8', errors='replace').strip() if exc2.stderr else '(no stderr)'
                logger.warning("mpeg4 fallback also failed (exit %s): %s", exc2.returncode, stderr2)
                raise
        actual_frames = _probe_video_frame_count(temp_video.name)
        logger.info("Downsampled video: original=%s, processed=%s, fps=%s, max_frames=%s, res=%sx%s",
                     video_path, temp_video.name,
                     f"{target_fps:.3f}" if target_fps is not None else 'unchanged',
                     max_frames if max_frames is not None else 'unchanged',
                     VIDEO_MAX_WIDTH if VIDEO_MAX_WIDTH and VIDEO_MAX_WIDTH > 0 else 'unchanged',
                     VIDEO_MAX_HEIGHT if VIDEO_MAX_HEIGHT and VIDEO_MAX_HEIGHT > 0 else 'unchanged')
        if min_frames:
            logger.info("Ensured min frames=%s (actual=%s)", min_frames, actual_frames if actual_frames is not None else 'unknown')
        return temp_video.name, temp_video.name
    except Exception as exc:
        logger.warning("Video downsample failed for %s: %s", video_path, str(exc))
        try:
            os.unlink(temp_video.name)
        except Exception:
            pass
        return video_path, None


def _estimate_multimodal_tokens(videos, audios, images, text_length):
    total = 200
    total += text_length // 3
    if videos:
        if isinstance(videos, list):
            for video in videos:
                if hasattr(video, 'shape'):
                    if len(video.shape) >= 1:
                        num_frames = video.shape[0] if len(video.shape) >= 4 else 1
                        total += num_frames * 600
                else:
                    total += 12000
        else:
            if hasattr(videos, 'shape') and len(videos.shape) >= 4:
                total += videos.shape[0] * 600
            else:
                total += 12000
    if audios:
        if isinstance(audios, list):
            total += len(audios) * 400
        else:
            total += 400
    if images:
        if isinstance(images, list):
            total += len(images) * 750
        else:
            total += 750
    return total


def _inject_audio_placeholders(messages, audio_count):
    if audio_count <= 0:
        return messages
    injected = []
    placeholders_remaining = audio_count
    for message in messages:
        new_message = copy.deepcopy(message)
        content = new_message.get('content')
        if not isinstance(content, list):
            injected.append(new_message)
            continue
        new_content = []
        for item in content:
            if placeholders_remaining > 0 and isinstance(item, dict) and item.get('type') == 'video':
                insertion_start = audio_count - placeholders_remaining
                for idx in range(insertion_start, audio_count):
                    new_content.append({'type': 'audio', 'audio': f'__audio_placeholder_{idx}__'})
                placeholders_remaining = 0
            new_content.append(item)
        new_message['content'] = new_content
        injected.append(new_message)
    return injected


# ---------------------------------------------------------------------------
# Model loading (identical to app.py)
# ---------------------------------------------------------------------------

def load_model_processor():
    global model, processor
    _hf_token = os.getenv('HUGGINGFACE_TOKEN', '').strip() or os.getenv('HF_TOKEN', '').strip()
    if _hf_token:
        try:
            from huggingface_hub import login
            login(token=_hf_token, add_to_git_credential=False)
            logger.info("Authenticated with Hugging Face using token")
        except ImportError:
            logger.info("Hugging Face token set via environment variables")
        except Exception as e:
            logger.warning(f"Failed to login to Hugging Face: {str(e)}")
    else:
        logger.warning("No Hugging Face token found. Private/gated models may fail to load.")

    logger.info(f"Loading model: {MODEL_PATH}")
    logger.info(f"DEFAULT_MAX_TOKENS (when client omits max_tokens): {DEFAULT_MAX_TOKENS}")
    logger.info(f"MIN_MAX_TOKENS (per-request floor to avoid </think> truncation): {MIN_MAX_TOKENS}")
    if QUANTIZATION:
        logger.info(f"Using quantization: {QUANTIZATION}")
    else:
        logger.info("Quantization: disabled")

    try:
        num_gpus = torch.cuda.device_count()
        if MAX_NUM_SEQS is not None:
            max_num_seqs = MAX_NUM_SEQS
        else:
            max_num_seqs = max(16, 32 // num_gpus) if num_gpus > 0 else 16

        if MAX_NUM_BATCHED_TOKENS is not None:
            max_num_batched_tokens = MAX_NUM_BATCHED_TOKENS
        else:
            max_num_batched_tokens = MAX_MODEL_LEN * 2

        # Qwen3-Omni-30B config has max_position_embeddings=65536; vLLM rejects higher unless VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
        effective_max_model_len = MAX_MODEL_LEN
        if MAX_MODEL_LEN > 65536 and os.getenv('VLLM_ALLOW_LONG_MAX_MODEL_LEN') != '1':
            effective_max_model_len = 65536
            logger.warning(
                "MAX_MODEL_LEN=%s exceeds model max (65536); capping to 65536. Set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 to allow longer (use with caution).",
                MAX_MODEL_LEN,
            )

        llm_kwargs = {
            "model": MODEL_PATH,
            "trust_remote_code": True,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "tensor_parallel_size": num_gpus,
            "hf_overrides": {"architectures": ["Qwen3OmniMoeForConditionalGeneration"]},
            "max_num_seqs": max_num_seqs,
            "max_model_len": effective_max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enable_chunked_prefill": ENABLE_CHUNKED_PREFILL,
            "disable_custom_all_reduce": False,
            "limit_mm_per_prompt": {'image': 1, 'video': 1, 'audio': 3},
            "seed": 1234,
        }
        if QUANTIZATION:
            llm_kwargs["quantization"] = QUANTIZATION

        if os.environ.get("BLACKWELL_PREBUILT") == "1":
            try:
                import vllm.multimodal.budget as _vllm_budget
                _orig_get = getattr(_vllm_budget, "get_mm_max_toks_per_item", None)
                if _orig_get is not None:
                    def _patched_get_mm_max_toks(*args, **kwargs):
                        out = _orig_get(*args, **kwargs)
                        if isinstance(out, dict) and "audio" not in out:
                            out = {**out, "audio": out.get("image", out.get("video", 256))}
                        return out
                    _vllm_budget.get_mm_max_toks_per_item = _patched_get_mm_max_toks
            except Exception:
                pass

        model = LLM(**llm_kwargs)
        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
        logger.info("Model and processor loaded successfully")
        logger.info(f"vLLM config: seqs={max_num_seqs}, batched_tokens={max_num_batched_tokens}, "
                     f"chunked_prefill={ENABLE_CHUNKED_PREFILL}, tp={num_gpus}")
        if get_wandb_logger is not None:
            try:
                wb = get_wandb_logger()
                wb.log_config({
                    "model_path": MODEL_PATH,
                    "max_model_len": MAX_MODEL_LEN,
                    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                    "max_num_seqs": max_num_seqs,
                    "max_num_batched_tokens": max_num_batched_tokens,
                    "enable_chunked_prefill": ENABLE_CHUNKED_PREFILL,
                    "default_max_tokens": DEFAULT_MAX_TOKENS,
                    "video_max_frames": VIDEO_MAX_FRAMES,
                    "video_min_frames": VIDEO_MIN_FRAMES,
                    "video_max_fps": VIDEO_MAX_FPS,
                    "video_max_width": VIDEO_MAX_WIDTH,
                    "video_max_height": VIDEO_MAX_HEIGHT,
                    "segment_duration": SEGMENT_DURATION,
                    "segment_overlap": SEGMENT_OVERLAP,
                    "token_safety_margin": TOKEN_SAFETY_MARGIN,
                    "audio_placeholder_limit": MAX_AUDIO_PLACEHOLDERS,
                })
                if wb.is_enabled():
                    logger.info("Wandb telemetry: enabled (metrics will be sent to Weights & Biases)")
                else:
                    logger.info(
                        "Wandb telemetry: disabled (set WANDB_API_KEY in .env.blackwell to enable)"
                    )
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# Batch processing (identical to app.py)
# ---------------------------------------------------------------------------

def _process_batch_requests():
    global _queue_thread
    while True:
        batch = []
        queue_depth = 0
        try:
            item = _request_queue.get(timeout=_BATCH_WAIT_TIME)
            batch.append(item)
            queue_depth = _request_queue.qsize()
            if queue_depth >= 20:
                target_batch_size = min(_MAX_BATCH_SIZE, 24)
                initial_collection_timeout = 3.0
            elif queue_depth >= 10:
                target_batch_size = min(_MAX_BATCH_SIZE, 16)
                initial_collection_timeout = 2.5
            else:
                target_batch_size = min(_MAX_BATCH_SIZE, 12)
                initial_collection_timeout = 2.0

            while len(batch) < target_batch_size:
                try:
                    if len(batch) < 8:
                        timeout = initial_collection_timeout
                    elif len(batch) < 16:
                        timeout = 1.0
                    else:
                        timeout = 0.3
                    item = _request_queue.get(timeout=timeout)
                    batch.append(item)
                    queue_depth = _request_queue.qsize()
                except queue.Empty:
                    if len(batch) >= 4 or queue_depth == 0:
                        break
                    time.sleep(0.3)
                    try:
                        item = _request_queue.get(timeout=0.5)
                        batch.append(item)
                        queue_depth = _request_queue.qsize()
                    except queue.Empty:
                        break
        except queue.Empty:
            with _queue_lock:
                if _request_queue.empty():
                    _queue_thread = None
                    return
            continue

        # Normalize to 6-tuple (messages, use_audio_in_video, temperature, max_tokens, result_queue, telemetry_ctx)
        batch = [item if len(item) >= 6 else (*item, {}) for item in batch]
        t_batch_start = time.perf_counter()
        queue_depth = _request_queue.qsize()

        if len(batch) < 4:
            queue_depth = _request_queue.qsize()
            if queue_depth > 0:
                try:
                    item = _request_queue.get(timeout=1.0)
                    batch.append(item)
                except queue.Empty:
                    pass

        logger.info(f"Processing batch of {len(batch)} requests (queue_depth={queue_depth})")
        if len(batch) < 8:
            logger.warning(f"Small batch ({len(batch)}) — consider increasing _BATCH_WAIT_TIME. Queue depth: {queue_depth}")

        def preprocess_request(item):
            messages, use_audio_in_video, temperature, max_tokens, result_queue = item[:5]
            telemetry_ctx = item[5] if len(item) >= 6 else {}
            t_preprocess_start = time.perf_counter()
            try:
                temp = float(temperature) if temperature is not None else float(os.getenv('TEMPERATURE', '0.1'))
                raw_max = int(max_tokens) if max_tokens is not None else DEFAULT_MAX_TOKENS
                max_toks = max(MIN_MAX_TOKENS, min(raw_max, 65536))
                sampling_params = SamplingParams(temperature=temp, top_p=0.1, top_k=1, max_tokens=max_toks)
                audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
                audio_count = 0
                if use_audio_in_video and isinstance(audios, list):
                    if MAX_AUDIO_PLACEHOLDERS and len(audios) > MAX_AUDIO_PLACEHOLDERS:
                        logger.warning("Truncating audio chunks from %s to %s", len(audios), MAX_AUDIO_PLACEHOLDERS)
                        audios = audios[:MAX_AUDIO_PLACEHOLDERS]
                    audio_count = len(audios)
                if use_audio_in_video and audio_count > 0 and not _contains_audio_entries(messages):
                    messages_for_template = _inject_audio_placeholders(messages, audio_count)
                else:
                    messages_for_template = messages
                text = processor.apply_chat_template(messages_for_template, tokenize=False, add_generation_prompt=True)
                inputs = {
                    'prompt': text,
                    'multi_modal_data': {},
                    'mm_processor_kwargs': {'use_audio_in_video': use_audio_in_video}
                }
                def _has_items(items):
                    if items is None:
                        return False
                    return len(items) > 0 if isinstance(items, list) else True
                if _has_items(images):
                    inputs['multi_modal_data']['image'] = images
                if _has_items(videos):
                    inputs['multi_modal_data']['video'] = videos
                if _has_items(audios):
                    inputs['multi_modal_data']['audio'] = audios
                preprocess_ms = (time.perf_counter() - t_preprocess_start) * 1000
                mm_data = inputs.get('multi_modal_data') or {}
                video_frames = len(mm_data.get('video') or []) if isinstance(mm_data.get('video'), list) else (1 if mm_data.get('video') else 0)
                audio_chunks = len(mm_data.get('audio') or []) if isinstance(mm_data.get('audio'), list) else (1 if mm_data.get('audio') else 0)
                has_video = _has_items(mm_data.get('video'))
                has_audio = _has_items(mm_data.get('audio'))
                has_image = _has_items(mm_data.get('image'))
                modality = "multimodal" if (has_video or has_audio or has_image) else "text_only"
                telemetry_extra = {
                    "preprocess_ms": preprocess_ms,
                    "prompt_char_length": len(text),
                    "video_frames": video_frames,
                    "audio_chunks": audio_chunks,
                    "has_video": has_video,
                    "has_audio": has_audio,
                    "has_image": has_image,
                    "modality": modality,
                    "temperature": temp,
                    "max_tokens_requested": max_toks,
                }
                return (inputs, sampling_params, result_queue, None, telemetry_extra)
            except Exception as e:
                logger.error(f"Error preprocessing request: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return (None, None, result_queue, e, None)

        preprocessing_results = []
        preprocessing_lock = threading.Lock()
        threads = []

        def preprocess_with_lock(item):
            result = preprocess_request(item)
            with preprocessing_lock:
                preprocessing_results.append(result)

        for item in batch:
            thread = threading.Thread(target=preprocess_with_lock, args=(item,), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        batch_inputs = []
        batch_sampling_params = []
        result_queues = []
        multimodal_flags = []
        telemetry_list = []
        errors = []

        for i, (item, result) in enumerate(zip(batch, preprocessing_results)):
            inputs, sampling_params, result_queue, error, telemetry_extra = result
            if error is not None:
                errors.append((result_queue, error))
            else:
                batch_inputs.append(inputs)
                batch_sampling_params.append(sampling_params)
                result_queues.append(result_queue)
                mm_data = inputs.get('multi_modal_data') or {}
                multimodal_flags.append(any(mm_data.get(k) for k in ('video', 'audio', 'image')))
                telemetry_ctx = item[5] if len(item) >= 6 else {}
                queue_wait_ms = (t_batch_start - telemetry_ctx.get("t_enqueue", t_batch_start)) * 1000
                telemetry_list.append({
                    **telemetry_ctx,
                    "queue_wait_ms": queue_wait_ms,
                    "batch_size": len(batch_inputs),
                    "queue_depth_at_batch": queue_depth,
                    **(telemetry_extra or {}),
                })

        for result_queue, error in errors:
            msg = str(error)
            if 'nframes should in interval' in msg or 'video_fps' in msg or 'moov atom' in msg.lower():
                msg = ("Video file could not be processed. "
                       "Ensure the file is a valid, complete MP4 (not streaming or truncated). "
                       "Re-encode with ffmpeg if needed: ffmpeg -i input.mp4 -c copy output.mp4")
            result_queue.put(('error', msg))
            _request_queue.task_done()

        text_only_indices = [i for i, is_mm in enumerate(multimodal_flags) if not is_mm]
        multimodal_indices = [i for i, is_mm in enumerate(multimodal_flags) if is_mm]

        def _process_batch(indices):
            if not indices:
                return
            try:
                local_inputs = [batch_inputs[i] for i in indices]
                if not local_inputs:
                    return
                params = batch_sampling_params[indices[0]] or SamplingParams()
                t_gen_start = time.perf_counter()
                outputs = model.generate(local_inputs, sampling_params=params)
                inference_ms = (time.perf_counter() - t_gen_start) * 1000
                if outputs is None or len(outputs) != len(local_inputs):
                    raise RuntimeError(f"generate() returned {len(outputs) if outputs else 0}, expected {len(local_inputs)}")
                for local_idx, output in enumerate(outputs):
                    rq = result_queues[indices[local_idx]]
                    try:
                        if output is None:
                            rq.put(('error', "generate() returned None"))
                        elif not hasattr(output, 'outputs') or output.outputs is None:
                            rq.put(('error', "generate() output has no outputs"))
                        elif len(output.outputs) == 0:
                            rq.put(('error', "generate() output.outputs is empty"))
                        elif not hasattr(output.outputs[0], 'text'):
                            rq.put(('error', "generate() output has no text"))
                        else:
                            response = output.outputs[0].text
                            rq.put(('error', "generate() returned None text") if response is None else ('success', response))
                            # Telemetry: use data vLLM already computed (zero overhead)
                            if get_wandb_logger is not None and indices[local_idx] < len(telemetry_list):
                                try:
                                    telemetry = dict(telemetry_list[indices[local_idx]])
                                    telemetry["inference_ms"] = inference_ms
                                    prompt_tokens = len(getattr(output, "prompt_token_ids", None) or [])
                                    completion_tokens = len(getattr(output.outputs[0], "token_ids", None) or [])
                                    telemetry["prompt_tokens"] = prompt_tokens
                                    telemetry["completion_tokens"] = completion_tokens
                                    telemetry["total_tokens"] = prompt_tokens + completion_tokens
                                    telemetry["finish_reason"] = str(getattr(output.outputs[0], "finish_reason", ""))
                                    if inference_ms > 0 and completion_tokens:
                                        telemetry["tokens_per_second"] = completion_tokens / (inference_ms / 1000.0)
                                    if "t_request_start" in telemetry:
                                        t_start = telemetry.pop("t_request_start", None)
                                        if t_start is not None:
                                            telemetry["e2e_ms"] = (time.perf_counter() - t_start) * 1000
                                    get_wandb_logger().log(telemetry)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.error(f"Error processing output: {str(e)}")
                        rq.put(('error', str(e)))
                    finally:
                        _request_queue.task_done()
            except Exception as e:
                logger.error(f"Batch inference error: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                for i in indices:
                    result_queues[i].put(('error', f"Batch inference failed: {str(e)}"))
                    _request_queue.task_done()

        _process_batch(text_only_indices)
        for idx in multimodal_indices:
            _process_batch([idx])


def run_model(messages, use_audio_in_video=False, temperature=None, max_tokens=None, endpoint=None, t_request_start=None):
    global _queue_thread
    telemetry_ctx = {}
    if endpoint is not None:
        telemetry_ctx["endpoint"] = endpoint
        telemetry_ctx["t_enqueue"] = time.perf_counter()
    if t_request_start is not None:
        telemetry_ctx["t_request_start"] = t_request_start
    with _queue_lock:
        if _queue_thread is None or not _queue_thread.is_alive():
            _queue_thread = threading.Thread(target=_process_batch_requests, daemon=True)
            _queue_thread.start()
            logger.info("Started batch queue processor")
    result_queue: queue.Queue = queue.Queue()
    _request_queue.put((messages, use_audio_in_video, temperature, max_tokens, result_queue, telemetry_ctx))
    status, result = result_queue.get()
    if status == 'error':
        raise RuntimeError(result)
    return result


# ---------------------------------------------------------------------------
# Upload helper: save UploadFile to a temp path (streaming, no size limit)
# ---------------------------------------------------------------------------

async def _save_upload(upload: UploadFile, suffix: str = '.mp4') -> str:
    """Stream an uploaded file to a temp path and return that path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    with open(tmp.name, 'wb') as f:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)
    return tmp.name


def _log_upload_diagnostics(request: Request, field_name: str, saved_path: str, original_name: Optional[str]) -> None:
    """
    Log request-level upload diagnostics to identify upstream truncation.

    If Content-Length is already around 256KB, truncation happened BEFORE app code
    (client/proxy/network path). If Content-Length is much larger than saved size,
    truncation happened during parsing.
    """
    try:
        saved_size = os.path.getsize(saved_path)
    except Exception:
        saved_size = -1
    cl_header = request.headers.get("content-length")
    ct_header = request.headers.get("content-type", "")
    logger.info(
        "Upload diagnostics: field=%s filename=%s saved_bytes=%s content_length=%s content_type=%s",
        field_name,
        original_name or "(unknown)",
        saved_size,
        cl_header or "(missing)",
        ct_header,
    )
    # 262144 bytes + multipart envelope ~= 262192 observed in failing requests.
    if saved_size in (262144, 262192):
        logger.error(
            "Detected classic 256KB truncation signature (saved=%s). "
            "This indicates the incoming body is already capped upstream of the app.",
            saved_size,
        )


# ---------------------------------------------------------------------------
# Endpoints (same paths / same JSON contracts as Flask app.py)
# ---------------------------------------------------------------------------

@app.get('/')
async def index():
    return {
        'service': 'Qwen3-Omni API Server',
        'version': '1.0.0',
        'engine': 'FastAPI (Blackwell)',
        'endpoints': {
            '/health': 'GET - Health check',
            '/videodescription': 'POST - Describe video content',
            '/audiodescription': 'POST - Describe audio content',
            '/audiovideodescription': 'POST - Analyze synced audio + video',
            '/audiovisualtextprocessing': 'POST - Analyze audio + video + transcription (tri-modal)',
            '/audiotextdescription': 'POST - Analyze audio with transcription text',
            '/docs': 'GET - Interactive API docs (Swagger UI)',
        },
        'model': MODEL_PATH,
        'documentation': 'Visit /docs for interactive Swagger docs',
    }


@app.get('/health')
async def health_check():
    return {'status': 'healthy', 'model': MODEL_PATH, 'model_loaded': model is not None}


@app.get('/config')
async def get_config():
    num_gpus = torch.cuda.device_count()
    actual_seqs = MAX_NUM_SEQS if MAX_NUM_SEQS is not None else (max(16, 32 // num_gpus) if num_gpus > 0 else 16)
    actual_batched = MAX_NUM_BATCHED_TOKENS if MAX_NUM_BATCHED_TOKENS is not None else (MAX_MODEL_LEN * 2)
    prompt_overhead = 1000
    usable_tokens = MAX_MODEL_LEN - prompt_overhead
    optimal_segment_seconds = int(usable_tokens / 70)
    return {
        'success': True,
        'model': MODEL_PATH,
        'gpu_memory_utilization': GPU_MEMORY_UTILIZATION,
        'max_model_len': MAX_MODEL_LEN,
        'default_max_tokens': DEFAULT_MAX_TOKENS,
        'min_max_tokens': MIN_MAX_TOKENS,
        'segment_duration': SEGMENT_DURATION,
        'segment_overlap': SEGMENT_OVERLAP,
        'quantization': QUANTIZATION,
        'max_num_seqs': actual_seqs,
        'max_num_batched_tokens': actual_batched,
        'enable_chunked_prefill': ENABLE_CHUNKED_PREFILL,
        'tensor_parallel_size': num_gpus,
        'optimal_segment_duration': optimal_segment_seconds,
        'recommended_segment_overlap': max(10, optimal_segment_seconds // 3),
    }


@app.post('/videodescription')
async def video_description(
    video: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form('Describe the video.'),
    use_audio_in_video: Optional[str] = Form('true'),
    temperature: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    request: Request = None,
):
    try:
        if model is None or processor is None:
            return JSONResponse({'error': 'Model not loaded'}, status_code=503)
        t_request_start = time.perf_counter()

        video_path = None
        cleanup_paths = []

        if video is not None and video.filename:
            suffix = Path(video.filename).suffix or '.mp4'
            video_path = await _save_upload(video, suffix=suffix)
            cleanup_paths.append(video_path)
            fsize = os.path.getsize(video_path)
            logger.info("Saved uploaded video to %s (%d bytes)", video_path, fsize)
            _log_upload_diagnostics(request, "video", video_path, video.filename)
            if fsize < 1024:
                logger.warning("Uploaded video file is very small (%d bytes) — may be truncated", fsize)
        else:
            # Try JSON body
            try:
                data = await request.json()
                video_path = data.get('video')
                prompt = data.get('prompt', prompt)
                use_audio_in_video = str(data.get('use_audio_in_video', use_audio_in_video))
                temperature = str(data.get('temperature')) if data.get('temperature') is not None else temperature
                max_tokens = str(data.get('max_tokens')) if data.get('max_tokens') is not None else max_tokens
            except Exception:
                pass
            if not video_path:
                return JSONResponse({'error': 'No video file or URL provided'}, status_code=400)

        audio_flag = use_audio_in_video.lower() != 'false' if isinstance(use_audio_in_video, str) else bool(use_audio_in_video)
        if not audio_flag:
            logger.info("Audio extraction from video DISABLED (use_audio_in_video=false)")

        temp_val = float(temperature) if temperature else None
        max_tok_val = int(max_tokens) if max_tokens else None

        if os.path.isfile(video_path) and not _is_valid_local_video(video_path):
            size = os.path.getsize(video_path)
            return JSONResponse(
                {
                    'error': (
                        f'Uploaded file is not a valid decodable video stream '
                        f'(saved_bytes={size}). Ensure you upload a complete MP4 file.'
                    ),
                    'success': False,
                },
                status_code=400,
            )

        processed_video_path, downsample_cleanup = _maybe_downsample_video(video_path)
        if downsample_cleanup:
            cleanup_paths.append(downsample_cleanup)

        messages = [{'role': 'user', 'content': [
            {'type': 'video', 'video': processed_video_path},
            {'type': 'text', 'text': prompt},
        ]}]
        logger.info("Processing video description: processed=%s source=%s", processed_video_path, video_path)

        response = run_model(messages, use_audio_in_video=audio_flag, temperature=temp_val, max_tokens=max_tok_val,
                            endpoint="videodescription", t_request_start=t_request_start)

        for path in cleanup_paths:
            try:
                os.unlink(path)
            except Exception:
                pass

        return {'description': response, 'success': True}

    except Exception as e:
        logger.error(f"Error processing video description: {str(e)}")
        return JSONResponse({'error': str(e), 'success': False}, status_code=500)


@app.post('/audiodescription')
async def audio_description(
    audio: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form('Describe the audio.'),
    temperature: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    request: Request = None,
):
    try:
        if model is None or processor is None:
            return JSONResponse({'error': 'Model not loaded'}, status_code=503)
        t_request_start = time.perf_counter()

        audio_path = None
        cleanup_path = None

        if audio is not None and audio.filename:
            suffix = Path(audio.filename).suffix or '.wav'
            audio_path = await _save_upload(audio, suffix=suffix)
            cleanup_path = audio_path
            _log_upload_diagnostics(request, "audio", audio_path, audio.filename)
        else:
            try:
                data = await request.json()
                audio_path = data.get('audio')
                prompt = data.get('prompt', prompt)
                temperature = str(data.get('temperature')) if data.get('temperature') is not None else temperature
                max_tokens = str(data.get('max_tokens')) if data.get('max_tokens') is not None else max_tokens
            except Exception:
                pass
            if not audio_path:
                return JSONResponse({'error': 'No audio file or URL provided'}, status_code=400)

        temp_val = float(temperature) if temperature else None
        max_tok_val = int(max_tokens) if max_tokens else None

        messages = [{'role': 'user', 'content': [
            {'type': 'audio', 'audio': audio_path},
            {'type': 'text', 'text': prompt},
        ]}]
        logger.info("Processing audio description: %s", audio_path)

        response = run_model(messages, use_audio_in_video=False, temperature=temp_val, max_tokens=max_tok_val,
                            endpoint="audiodescription", t_request_start=t_request_start)

        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except Exception:
                pass

        return {'description': response, 'success': True}

    except Exception as e:
        logger.error(f"Error processing audio description: {str(e)}")
        return JSONResponse({'error': str(e), 'success': False}, status_code=500)


@app.post('/audiovideodescription')
async def audio_video_description(
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form('Describe the audio and video.'),
    use_audio_in_video: Optional[str] = Form('true'),
    temperature: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    request: Request = None,
):
    try:
        if model is None or processor is None:
            return JSONResponse({'error': 'Model not loaded'}, status_code=503)
        t_request_start = time.perf_counter()

        video_path = None
        audio_path = None
        cleanup_paths = []

        if video is not None and video.filename and audio is not None and audio.filename:
            v_suffix = Path(video.filename).suffix or '.mp4'
            video_path = await _save_upload(video, suffix=v_suffix)
            cleanup_paths.append(video_path)
            logger.info("Saved uploaded video to %s (%d bytes)", video_path, os.path.getsize(video_path))
            _log_upload_diagnostics(request, "video", video_path, video.filename)

            a_suffix = Path(audio.filename).suffix or '.wav'
            audio_path = await _save_upload(audio, suffix=a_suffix)
            cleanup_paths.append(audio_path)
            _log_upload_diagnostics(request, "audio", audio_path, audio.filename)
        else:
            try:
                data = await request.json()
                video_path = data.get('video')
                audio_path = data.get('audio')
                prompt = data.get('prompt', prompt)
                use_audio_in_video = str(data.get('use_audio_in_video', use_audio_in_video))
                temperature = str(data.get('temperature')) if data.get('temperature') is not None else temperature
                max_tokens = str(data.get('max_tokens')) if data.get('max_tokens') is not None else max_tokens
            except Exception:
                pass
            if not video_path or not audio_path:
                return JSONResponse({'error': 'Both video and audio files/URLs are required'}, status_code=400)

        audio_flag = use_audio_in_video.lower() == 'true' if isinstance(use_audio_in_video, str) else bool(use_audio_in_video)
        temp_val = float(temperature) if temperature else None
        max_tok_val = int(max_tokens) if max_tokens else None

        if os.path.isfile(video_path) and not _is_valid_local_video(video_path):
            size = os.path.getsize(video_path)
            return JSONResponse(
                {
                    'error': (
                        f'Uploaded video is not a valid decodable stream '
                        f'(saved_bytes={size}). Ensure you upload a complete MP4 file.'
                    ),
                    'success': False,
                },
                status_code=400,
            )

        messages = [{'role': 'user', 'content': [
            {'type': 'video', 'video': video_path},
            {'type': 'audio', 'audio': audio_path},
            {'type': 'text', 'text': prompt},
        ]}]
        logger.info("Processing audio+video description: video=%s, audio=%s", video_path, audio_path)

        response = run_model(messages, use_audio_in_video=audio_flag, temperature=temp_val, max_tokens=max_tok_val,
                            endpoint="audiovideodescription", t_request_start=t_request_start)

        for path in cleanup_paths:
            try:
                os.unlink(path)
            except Exception:
                pass

        return {'description': response, 'success': True}

    except Exception as e:
        logger.error(f"Error processing audio+video description: {str(e)}")
        return JSONResponse({'error': str(e), 'success': False}, status_code=500)


@app.post('/audiovisualprocessing')
async def audiovisual_processing(
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form('Describe the audio and video.'),
    use_audio_in_video: Optional[str] = Form('true'),
    temperature: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    request: Request = None,
):
    return await audio_video_description(video=video, audio=audio, prompt=prompt,
                                         use_audio_in_video=use_audio_in_video,
                                         temperature=temperature, max_tokens=max_tokens, request=request)


@app.post('/audiovisualreasoning')
async def audiovisual_reasoning(
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form('Describe the audio and video.'),
    use_audio_in_video: Optional[str] = Form('true'),
    temperature: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    request: Request = None,
):
    return await audio_video_description(video=video, audio=audio, prompt=prompt,
                                         use_audio_in_video=use_audio_in_video,
                                         temperature=temperature, max_tokens=max_tokens, request=request)


@app.post('/audiovisualtextprocessing')
async def audiovisual_text_processing(
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    transcription: Optional[str] = Form(None),
    prompt: Optional[str] = Form('Analyze the audio, video, and transcription.'),
    use_audio_in_video: Optional[str] = Form('true'),
    temperature: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    request: Request = None,
):
    try:
        if model is None or processor is None:
            return JSONResponse({'error': 'Model not loaded'}, status_code=503)
        t_request_start = time.perf_counter()

        video_path = None
        audio_path = None
        cleanup_paths = []

        if video is not None and video.filename and audio is not None and audio.filename:
            v_suffix = Path(video.filename).suffix or '.mp4'
            video_path = await _save_upload(video, suffix=v_suffix)
            cleanup_paths.append(video_path)
            logger.info("Saved uploaded video to %s (%d bytes)", video_path, os.path.getsize(video_path))
            _log_upload_diagnostics(request, "video", video_path, video.filename)

            a_suffix = Path(audio.filename).suffix or '.wav'
            audio_path = await _save_upload(audio, suffix=a_suffix)
            cleanup_paths.append(audio_path)
            _log_upload_diagnostics(request, "audio", audio_path, audio.filename)
        else:
            try:
                data = await request.json()
                video_path = data.get('video')
                audio_path = data.get('audio')
                transcription = data.get('transcription', transcription)
                prompt = data.get('prompt', prompt)
                use_audio_in_video = str(data.get('use_audio_in_video', use_audio_in_video))
                temperature = str(data.get('temperature')) if data.get('temperature') is not None else temperature
                max_tokens = str(data.get('max_tokens')) if data.get('max_tokens') is not None else max_tokens
            except Exception:
                pass

        if not video_path:
            return JSONResponse({'error': 'Video file is required for tri-modal processing'}, status_code=400)
        if not audio_path:
            return JSONResponse({'error': 'Audio file is required for tri-modal processing'}, status_code=400)
        if not transcription:
            for path in cleanup_paths:
                try:
                    os.unlink(path)
                except Exception:
                    pass
            return JSONResponse({'error': 'Transcription text is required for tri-modal processing'}, status_code=400)

        audio_flag = use_audio_in_video.lower() == 'true' if isinstance(use_audio_in_video, str) else bool(use_audio_in_video)
        temp_val = float(temperature) if temperature else None
        max_tok_val = int(max_tokens) if max_tokens else None

        if os.path.isfile(video_path) and not _is_valid_local_video(video_path):
            size = os.path.getsize(video_path)
            return JSONResponse(
                {
                    'error': (
                        f'Uploaded video is not a valid decodable stream '
                        f'(saved_bytes={size}). Ensure you upload a complete MP4 file.'
                    ),
                    'success': False,
                },
                status_code=400,
            )

        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': f'Transcription: {transcription}'},
            {'type': 'video', 'video': video_path},
            {'type': 'audio', 'audio': audio_path},
            {'type': 'text', 'text': prompt},
        ]}]
        logger.info("Processing tri-modal request: video=%s, audio=%s, transcription_len=%d",
                     video_path, audio_path, len(transcription))

        response = run_model(messages, use_audio_in_video=audio_flag, temperature=temp_val, max_tokens=max_tok_val,
                            endpoint="audiovisualtextprocessing", t_request_start=t_request_start)

        for path in cleanup_paths:
            try:
                os.unlink(path)
            except Exception:
                pass

        return {'description': response, 'success': True}

    except Exception as e:
        logger.error(f"Error processing tri-modal request: {str(e)}")
        return JSONResponse({'error': str(e), 'success': False}, status_code=500)


@app.post('/audiotextdescription')
async def audio_text_description(
    audio: Optional[UploadFile] = File(None),
    transcription: Optional[str] = Form(None),
    instruction: Optional[str] = Form('Analyze the audio and transcription.'),
    temperature: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    request: Request = None,
):
    try:
        if model is None or processor is None:
            return JSONResponse({'error': 'Model not loaded'}, status_code=503)
        t_request_start = time.perf_counter()

        audio_path = None
        cleanup_path = None

        if audio is not None and audio.filename:
            suffix = Path(audio.filename).suffix or '.wav'
            audio_path = await _save_upload(audio, suffix=suffix)
            cleanup_path = audio_path
        else:
            try:
                data = await request.json()
                audio_path = data.get('audio')
                transcription = data.get('transcription', transcription)
                instruction = data.get('instruction', instruction)
                temperature = str(data.get('temperature')) if data.get('temperature') is not None else temperature
                max_tokens = str(data.get('max_tokens')) if data.get('max_tokens') is not None else max_tokens
            except Exception:
                pass
            if not audio_path:
                return JSONResponse({'error': 'No audio file or URL provided'}, status_code=400)

        if not transcription:
            if cleanup_path:
                try:
                    os.unlink(cleanup_path)
                except Exception:
                    pass
            return JSONResponse({'error': 'Transcription text is required'}, status_code=400)

        temp_val = float(temperature) if temperature else None
        max_tok_val = int(max_tokens) if max_tokens else None

        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': f'Transcription: {transcription}'},
            {'type': 'audio', 'audio': audio_path},
            {'type': 'text', 'text': instruction},
        ]}]
        logger.info("Processing audio+text description: %s", audio_path)

        response = run_model(messages, use_audio_in_video=False, temperature=temp_val, max_tokens=max_tok_val,
                            endpoint="audiotextdescription", t_request_start=t_request_start)

        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except Exception:
                pass

        return {'description': response, 'success': True}

    except Exception as e:
        logger.error(f"Error processing audio+text description: {str(e)}")
        return JSONResponse({'error': str(e), 'success': False}, status_code=500)


# ---------------------------------------------------------------------------
# Startup & main
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Qwen3-Omni API Server (FastAPI / Blackwell)...")
    load_model_processor()


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5100))
    logger.info(f"Starting uvicorn on port {port}")
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='info')
