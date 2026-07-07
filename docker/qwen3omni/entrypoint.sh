#!/bin/bash
set -e

# App always listens on 5000 inside the container; host-side port mapping
# (-p 5100:5000 etc.) handles external access.
export PORT=5000

echo "Starting Qwen3-Omni API Server..."

# Check if we need to fix vLLM (when running manually in interactive mode)
if [ -t 0 ]; then
    echo "Interactive mode detected."
    echo "If you encounter vLLM CUDA errors, run: bash /data/shared/Qwen3-Omni/fix_vllm.sh"
fi

# Execute the app
exec python3 app.py

