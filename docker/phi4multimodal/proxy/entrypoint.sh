#!/bin/bash
# =============================================================================
# Phi-4 Multimodal Compatibility Proxy Entrypoint
# =============================================================================

set -e

echo "=============================================="
echo "Phi-4 Multimodal Compatibility Proxy"
echo "=============================================="
echo "Proxy Port: ${PROXY_PORT:-5200}"
echo "Backend URL: ${VLLM_BACKEND_URL:-http://localhost:8000}"
echo "Video Max Frames: ${VIDEO_MAX_FRAMES:-8}"
echo "Request Timeout: ${VLLM_REQUEST_TIMEOUT:-600}s"
echo "=============================================="

# Wait for vLLM backend to be ready (optional)
if [ "${WAIT_FOR_BACKEND:-false}" = "true" ]; then
    echo "Waiting for vLLM backend to be ready..."
    MAX_RETRIES=${BACKEND_WAIT_RETRIES:-60}
    RETRY_INTERVAL=${BACKEND_WAIT_INTERVAL:-5}
    
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -sf "${VLLM_BACKEND_URL}/health" > /dev/null; then
            echo "Backend is ready!"
            break
        fi
        
        if [ $i -eq $MAX_RETRIES ]; then
            echo "WARNING: Backend not ready after ${MAX_RETRIES} attempts, starting anyway..."
        else
            echo "Attempt $i/$MAX_RETRIES: Backend not ready, waiting ${RETRY_INTERVAL}s..."
            sleep $RETRY_INTERVAL
        fi
    done
fi

echo "Starting proxy server..."
exec python /app/app.py
