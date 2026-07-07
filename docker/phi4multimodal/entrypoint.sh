#!/bin/bash
# =============================================================================
# Phi-4 Multimodal vLLM Entrypoint Script
# =============================================================================
# This script:
# 1. Checks if the Phi-4-multimodal-instruct model exists locally in ./models
# 2. Downloads it from HuggingFace if missing (using HUGGINGFACE_TOKEN from .env)
# 3. Auto-discovers speech-lora and vision-lora paths
# 4. Starts the vLLM server with proper configuration
#
# Models are stored in /app/models (mounted from ./models on host)
# =============================================================================

set -e

# Configure HuggingFace to use local models directory (not root cache)
export HF_HOME="${HF_HOME:-/app/models}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/app/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/app/models/hub}"

# Ensure models directory exists
mkdir -p "${HF_HOME}/hub"

echo "=============================================="
echo "Phi-4 Multimodal vLLM Server Startup"
echo "=============================================="
echo "Model: ${MODEL_PATH:-microsoft/Phi-4-multimodal-instruct}"
echo "Models Directory: ${HF_HOME}"
echo "Port: ${VLLM_INTERNAL_PORT:-8000}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION:-0.90}"
echo "Max Model Length: ${MAX_MODEL_LEN:-32768}"
echo "=============================================="

# Set HF_TOKEN from HUGGINGFACE_TOKEN if not already set
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: No HF_TOKEN or HUGGINGFACE_TOKEN set."
    echo "         Model download may fail for gated models."
    echo "         Set HUGGINGFACE_TOKEN in your .env file."
else
    echo "HuggingFace token configured."
fi

echo ""
echo "=============================================="
echo "Step 1: Checking Model Availability"
echo "=============================================="

# Check and download model, then discover LoRA paths using Python
LORA_PATHS=$(python3 << 'EOF'
import os
import sys

def check_model_exists(model_id, cache_dir=None):
    """Check if model is already cached locally in ./models directory."""
    if cache_dir is None:
        # Use local models directory, not root cache
        cache_dir = os.path.join(os.environ.get('HF_HOME', '/app/models'), 'hub')
    
    model_cache_name = f"models--{model_id.replace('/', '--')}"
    model_cache_path = os.path.join(cache_dir, model_cache_name)
    
    if os.path.isdir(model_cache_path):
        snapshots_dir = os.path.join(model_cache_path, "snapshots")
        if os.path.isdir(snapshots_dir) and os.listdir(snapshots_dir):
            return True, model_cache_path
    return False, model_cache_path

def download_model(model_id, token=None):
    """Download model from HuggingFace with progress feedback."""
    from huggingface_hub import snapshot_download
    
    print(f"Downloading model: {model_id}", file=sys.stderr)
    print("This may take several minutes for first-time download (~12GB)...", file=sys.stderr)
    print("", file=sys.stderr)
    
    model_path = snapshot_download(
        repo_id=model_id,
        token=token,
        local_files_only=False,
    )
    
    return model_path

def verify_lora_adapters(model_path):
    """Verify that LoRA adapters exist in the model directory."""
    speech_lora = os.path.join(model_path, "speech-lora")
    vision_lora = os.path.join(model_path, "vision-lora")
    
    speech_exists = os.path.isdir(speech_lora)
    vision_exists = os.path.isdir(vision_lora)
    
    return speech_lora if speech_exists else "", vision_lora if vision_exists else ""

try:
    from huggingface_hub import snapshot_download
    
    model_id = os.environ.get('MODEL_PATH', 'microsoft/Phi-4-multimodal-instruct')
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    
    # Step 1: Check if model exists locally
    exists, cache_path = check_model_exists(model_id)
    
    if exists:
        print(f"[OK] Model found in cache: {cache_path}", file=sys.stderr)
        print("Verifying model files...", file=sys.stderr)
    else:
        print(f"[INFO] Model not found locally.", file=sys.stderr)
        print(f"       Cache location: {cache_path}", file=sys.stderr)
        print("", file=sys.stderr)
        
        if not hf_token:
            print("[WARNING] No HuggingFace token provided!", file=sys.stderr)
            print("          Download may fail if model is gated.", file=sys.stderr)
            print("          Set HUGGINGFACE_TOKEN in .env file.", file=sys.stderr)
        else:
            print("[OK] Using HuggingFace token for authentication.", file=sys.stderr)
        
        print("", file=sys.stderr)
    
    # Step 2: Download/verify model (snapshot_download handles caching efficiently)
    print(f"Ensuring model is ready: {model_id}", file=sys.stderr)
    model_path = snapshot_download(
        repo_id=model_id,
        token=hf_token,
        local_files_only=False,
    )
    print(f"[OK] Model available at: {model_path}", file=sys.stderr)
    
    # Step 3: Verify LoRA adapters
    print("", file=sys.stderr)
    print("Checking LoRA adapters...", file=sys.stderr)
    speech_lora, vision_lora = verify_lora_adapters(model_path)
    
    if speech_lora:
        print(f"[OK] Speech LoRA found: {speech_lora}", file=sys.stderr)
    else:
        print("[WARNING] Speech LoRA not found in model directory", file=sys.stderr)
        
    if vision_lora:
        print(f"[OK] Vision LoRA found: {vision_lora}", file=sys.stderr)
    else:
        print("[WARNING] Vision LoRA not found in model directory", file=sys.stderr)
    
    # Output paths for bash (stdout - only this line goes to bash variable)
    print(f"{speech_lora}|{vision_lora}")
    
except ImportError as e:
    print(f"[ERROR] Missing required package: {e}", file=sys.stderr)
    print("        Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
    print("|")
    sys.exit(1)
    
except Exception as e:
    print(f"[ERROR] Failed to prepare model: {e}", file=sys.stderr)
    print("|")
    sys.exit(1)
EOF
)

# Check if the Python script succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Model preparation failed. Check the errors above."
    exit 1
fi

# Parse the output (last line contains the paths)
SPEECH_LORA_PATH=$(echo "$LORA_PATHS" | tail -1 | cut -d'|' -f1)
VISION_LORA_PATH=$(echo "$LORA_PATHS" | tail -1 | cut -d'|' -f2)

echo ""
echo "=============================================="
echo "Step 2: Configuring vLLM Server"
echo "=============================================="

echo "Speech LoRA: ${SPEECH_LORA_PATH:-NOT FOUND}"
echo "Vision LoRA: ${VISION_LORA_PATH:-NOT FOUND}"

# Build the vLLM command
VLLM_ARGS=(
    "serve"
    "${MODEL_PATH:-microsoft/Phi-4-multimodal-instruct}"
    "--host" "0.0.0.0"
    "--port" "${VLLM_INTERNAL_PORT:-8000}"
    "--dtype" "auto"
    "--trust-remote-code"
    "--max-model-len" "${MAX_MODEL_LEN:-32768}"
    "--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION:-0.90}"
    "--max-num-seqs" "${MAX_NUM_SEQS:-8}"
    "--max-num-batched-tokens" "${MAX_NUM_BATCHED_TOKENS:-65536}"
    "--seed" "1234"
)

# Add LoRA configuration if paths are found
if [ -n "$SPEECH_LORA_PATH" ] && [ -n "$VISION_LORA_PATH" ]; then
    echo ""
    echo "Enabling LoRA adapters for multimodal support..."
    VLLM_ARGS+=(
        "--enable-lora"
        "--max-lora-rank" "${MAX_LORA_RANK:-320}"
        "--max-loras" "${MAX_LORAS:-2}"
        "--lora-modules" "speech=${SPEECH_LORA_PATH}" "vision=${VISION_LORA_PATH}"
    )
else
    echo ""
    echo "WARNING: LoRA adapters not found!"
    echo "         Running without full multimodal support."
    echo "         Speech and vision capabilities may be limited."
fi

# Add multimodal limits (vLLM expects JSON format)
VLLM_ARGS+=(
    "--limit-mm-per-prompt" "{\"audio\": ${LIMIT_AUDIO_PER_PROMPT:-1}, \"image\": ${LIMIT_IMAGE_PER_PROMPT:-8}}"
)

# Add eager mode if configured
if [ "${VLLM_USE_EAGER:-true}" = "true" ]; then
    VLLM_ARGS+=("--enforce-eager")
    echo "Using eager mode (no CUDA graphs)"
fi

# Print final command
echo ""
echo "=============================================="
echo "Step 3: Starting vLLM Server"
echo "=============================================="
echo ""
echo "Command:"
echo "vllm ${VLLM_ARGS[*]}"
echo ""
echo "=============================================="

# Execute vLLM
exec vllm "${VLLM_ARGS[@]}"
