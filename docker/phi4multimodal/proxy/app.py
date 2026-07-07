"""
Phi-4 Multimodal Compatibility Proxy

This FastAPI application provides a compatibility layer between the qwen3omni
client (run_all_surveys_longitudinal.py) and the Phi-4-multimodal-instruct
vLLM backend.

Key functionality:
- Converts video_url content to multiple image_url frames
- Passes audio_url content unchanged
- Proxies /health and /v1/models endpoints
- Full pass-through for /v1/chat/completions with transformation

The qwen3omni client sends:
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}

This proxy transforms it to:
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    ...
"""

import os
import sys
import base64
import tempfile
import logging
from typing import Any, Dict, List, Optional
from io import BytesIO

import cv2
import numpy as np
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Phi-4 Multimodal Compatibility Proxy",
    description="Converts video_url to image frames for Phi-4-multimodal-instruct",
    version="1.0.0"
)

VLLM_BACKEND_URL = os.environ.get("VLLM_BACKEND_URL", "http://localhost:8000")
VLLM_REQUEST_TIMEOUT = int(os.environ.get("VLLM_REQUEST_TIMEOUT", "600"))
VIDEO_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", "8"))
FRAME_JPEG_QUALITY = int(os.environ.get("FRAME_JPEG_QUALITY", "85"))
FRAME_MAX_DIMENSION = int(os.environ.get("FRAME_MAX_DIMENSION", "1024"))


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    """
    Decode a data URL to raw bytes and mime type.
    
    Args:
        data_url: Data URL string (e.g., "data:video/mp4;base64,...")
        
    Returns:
        Tuple of (raw_bytes, mime_type)
    """
    if not data_url.startswith("data:"):
        raise ValueError(f"Invalid data URL: does not start with 'data:'")
    
    header, encoded = data_url.split(",", 1)
    mime_type = header.split(":")[1].split(";")[0]
    raw_bytes = base64.b64decode(encoded)
    
    return raw_bytes, mime_type


def encode_image_to_data_url(image: np.ndarray, quality: int = 85) -> str:
    """
    Encode a numpy image array to a JPEG data URL.
    
    Args:
        image: OpenCV image (BGR format)
        quality: JPEG quality (0-100)
        
    Returns:
        Data URL string
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", image, encode_params)
    b64_data = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_data}"


def resize_frame_if_needed(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    """
    Resize frame if any dimension exceeds max_dimension while preserving aspect ratio.
    """
    h, w = frame.shape[:2]
    if max(h, w) <= max_dimension:
        return frame
    
    scale = max_dimension / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_frames_from_video(video_bytes: bytes, max_frames: int) -> List[np.ndarray]:
    """
    Extract evenly-spaced frames from video bytes.
    
    Args:
        video_bytes: Raw video file bytes
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of OpenCV images (BGR format)
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        if total_frames <= 0:
            raise ValueError("Video has no frames")
        
        num_frames = min(max_frames, total_frames)
        if num_frames <= 1:
            frame_indices = [0]
        else:
            frame_indices = [
                int(i * (total_frames - 1) / (num_frames - 1))
                for i in range(num_frames)
            ]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = resize_frame_if_needed(frame, FRAME_MAX_DIMENSION)
                frames.append(frame)
            else:
                logger.warning(f"Failed to read frame at index {idx}")
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
        
    finally:
        os.unlink(tmp_path)


def convert_video_to_images(video_data_url: str) -> List[Dict[str, Any]]:
    """
    Convert a video data URL to a list of image_url content parts.
    
    Args:
        video_data_url: Video data URL
        
    Returns:
        List of image_url content dictionaries
    """
    video_bytes, mime_type = decode_data_url(video_data_url)
    logger.info(f"Converting video ({len(video_bytes)} bytes, {mime_type}) to images")
    
    frames = extract_frames_from_video(video_bytes, VIDEO_MAX_FRAMES)
    
    image_parts = []
    for i, frame in enumerate(frames):
        data_url = encode_image_to_data_url(frame, FRAME_JPEG_QUALITY)
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })
        logger.debug(f"Frame {i+1}/{len(frames)}: encoded to JPEG data URL")
    
    return image_parts


def transform_content_parts(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform content parts, converting video_url to image_url frames.
    
    Args:
        content: List of content part dictionaries
        
    Returns:
        Transformed content parts
    """
    transformed = []
    
    for part in content:
        part_type = part.get("type")
        
        if part_type == "video_url":
            video_url = part.get("video_url", {}).get("url", "")
            if video_url:
                try:
                    image_parts = convert_video_to_images(video_url)
                    transformed.extend(image_parts)
                    logger.info(f"Converted video_url to {len(image_parts)} image_url parts")
                except Exception as e:
                    logger.error(f"Failed to convert video: {e}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to convert video to images: {str(e)}"
                    )
            else:
                logger.warning("Empty video_url, skipping")
                
        elif part_type == "audio_url":
            transformed.append(part)
            logger.debug("Passing audio_url unchanged")
            
        elif part_type == "image_url":
            transformed.append(part)
            logger.debug("Passing image_url unchanged")
            
        elif part_type == "text":
            transformed.append(part)
            logger.debug("Passing text unchanged")
            
        else:
            transformed.append(part)
            logger.debug(f"Passing unknown type '{part_type}' unchanged")
    
    return transformed


def transform_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform all messages, converting video content where needed.
    """
    transformed = []
    
    for msg in messages:
        new_msg = msg.copy()
        content = msg.get("content")
        
        if isinstance(content, list):
            new_msg["content"] = transform_content_parts(content)
        
        transformed.append(new_msg)
    
    return transformed


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Phi-4 Multimodal Compatibility Proxy",
        "version": "1.0.0",
        "backend": VLLM_BACKEND_URL,
        "features": [
            "video_url to image frames conversion",
            "audio_url pass-through",
            "OpenAI-compatible /v1/chat/completions"
        ]
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Proxies to vLLM backend and adds proxy status.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{VLLM_BACKEND_URL}/health")
            backend_healthy = response.status_code == 200
            if backend_healthy:
                try:
                    backend_data = response.json()
                except ValueError:
                    body = response.text.strip()
                    backend_data = {"message": body} if body else {"status": "ok"}
            else:
                backend_data = {}
    except Exception as e:
        logger.warning(f"Backend health check failed: {e}")
        backend_healthy = False
        backend_data = {"error": str(e)}
    
    return JSONResponse(
        status_code=200 if backend_healthy else 503,
        content={
            "status": "healthy" if backend_healthy else "degraded",
            "proxy": "healthy",
            "backend": backend_data,
            "backend_url": VLLM_BACKEND_URL
        }
    )


@app.get("/v1/models")
async def list_models():
    """
    List available models.
    Proxies to vLLM backend.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{VLLM_BACKEND_URL}/v1/models")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    
    Transforms video_url content parts to image_url frames before
    forwarding to the vLLM backend.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    logger.info(f"Received chat completion request with {len(messages)} messages")
    
    has_video = any(
        isinstance(msg.get("content"), list) and
        any(part.get("type") == "video_url" for part in msg.get("content", []))
        for msg in messages
    )
    
    if has_video:
        logger.info("Request contains video_url content, transforming...")
        body["messages"] = transform_messages(messages)
        logger.info("Transformation complete")
    else:
        logger.debug("No video_url content found, passing through")
    
    try:
        async with httpx.AsyncClient(timeout=VLLM_REQUEST_TIMEOUT) as client:
            logger.info(f"Forwarding request to {VLLM_BACKEND_URL}/v1/chat/completions")
            response = await client.post(
                f"{VLLM_BACKEND_URL}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"Backend returned {response.status_code}: {response.text[:500]}")
            
            return JSONResponse(
                status_code=response.status_code,
                content=response.json()
            )
            
    except httpx.TimeoutException:
        logger.error(f"Backend request timed out after {VLLM_REQUEST_TIMEOUT}s")
        raise HTTPException(status_code=504, detail="Backend request timed out")
    except httpx.HTTPStatusError as e:
        logger.error(f"Backend HTTP error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Backend error: {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_fallback(request: Request, path: str):
    """
    Fallback proxy for any other endpoints.
    Forwards requests directly to the vLLM backend.
    """
    try:
        body = await request.body()
        
        async with httpx.AsyncClient(timeout=VLLM_REQUEST_TIMEOUT) as client:
            response = await client.request(
                method=request.method,
                url=f"{VLLM_BACKEND_URL}/{path}",
                content=body,
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() not in ("host", "content-length")
                }
            )
            
            return JSONResponse(
                status_code=response.status_code,
                content=response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw": response.text}
            )
            
    except Exception as e:
        logger.error(f"Proxy fallback error for /{path}: {e}")
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", "5200"))
    logger.info(f"Starting Phi-4 Multimodal Compatibility Proxy on port {port}")
    logger.info(f"Backend URL: {VLLM_BACKEND_URL}")
    logger.info(f"Video max frames: {VIDEO_MAX_FRAMES}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info").lower()
    )
