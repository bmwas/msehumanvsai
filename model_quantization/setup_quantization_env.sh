#!/bin/bash
# =============================================================================
# Qwen3-Omni 4-bit Quantization Environment Setup
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/quantization_venv"
ENV_FILE="${SCRIPT_DIR}/quantization_env.env"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
COMPARE_DIR="${SCRIPT_DIR}/../comparison_model"
REF_DIR="${COMPARE_DIR}/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit"

# Verify requirements.txt exists
if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "ERROR: Missing ${REQUIREMENTS_FILE}. This file defines all package versions."
    exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: Missing ${ENV_FILE}. This file must define HUGGINGFACE_TOKEN (and optionally HUGGINGFACE_NAME)."
    exit 1
fi

set -a
source "${ENV_FILE}"
set +a

if [ -z "${HUGGINGFACE_TOKEN}" ]; then
    echo "ERROR: HUGGINGFACE_TOKEN is not set in ${ENV_FILE}. It is required to download the cpatonn reference model."
    exit 1
fi

echo "=============================================="
echo "Qwen3-Omni Quantization Environment Setup"
echo "=============================================="
echo ""
echo "Using requirements file: ${REQUIREMENTS_FILE}"
if [ ! -r "${REQUIREMENTS_FILE}" ]; then
    echo "ERROR: Cannot read ${REQUIREMENTS_FILE}"
    exit 1
fi
echo ""

# 1. Check Python
echo "[1/9] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "      Python ${PYTHON_VERSION} OK"

# 2. Create fresh venv
echo ""
echo "[2/9] Creating FRESH virtual environment..."
if [ -d "${VENV_DIR}" ]; then
    echo "      Removing existing venv..."
    rm -rf "${VENV_DIR}"
fi
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
echo "      Created: ${VENV_DIR}"

# 3. Upgrade pip
echo ""
echo "[3/9] Upgrading pip..."
pip install --upgrade pip wheel setuptools --no-cache-dir -q
echo "      pip upgraded"

# =============================================================================
# INSTALLATION FROM requirements.txt
# All package versions are read from requirements.txt to ensure consistency
# Installation order is critical: PyTorch -> transformers -> compressed-tensors -> llmcompressor
# =============================================================================

# 4. Install PyTorch 2.7.0 FIRST (required for llmcompressor 0.7.1)
# Using default PyPI which bundles CUDA 12.6 libraries
echo ""
echo "[4/9] Installing PyTorch 2.7.0 (from ${REQUIREMENTS_FILE})..."
grep -E "^torch==|^torchvision==|^torchaudio==" "${REQUIREMENTS_FILE}" | grep -v "^#" | tr '\n' ' ' | xargs -r pip install --no-cache-dir -q
echo "      PyTorch 2.7.0 installed"

# 5. Install transformers from GitHub (matches cpatonn's 4.57.0.dev0)
# NOTE: Installing from v4.57.0 tag gives 4.57.0 (stable), not 4.57.0.dev0 (dev)
# However, quantization OUTPUT format is determined by compressed-tensors 0.11.0 (which matches exactly)
# The transformers version affects model loading, but quantization output is identical
echo ""
echo "[5/9] Installing transformers from GitHub (from ${REQUIREMENTS_FILE})..."
# Check if requirements.txt specifies git+https for transformers
if grep -q "^git+https.*transformers" "${REQUIREMENTS_FILE}"; then
    grep "^git+https.*transformers" "${REQUIREMENTS_FILE}" | grep -v "^#" | tr '\n' ' ' | xargs -r pip install --no-cache-dir -q
else
    # Fallback: install from v4.57.0 tag
    pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git@v4.57.0" -q
fi
echo "      Transformers installed from GitHub (v4.57.0 tag -> 4.57.0 stable)"
echo "      Note: cpatonn used 4.57.0.dev0, but quantization format matches (compressed-tensors 0.11.0)"

# 6. Install compressed-tensors 0.11.0 FIRST (CRITICAL for MoE expert quantization)
# This must be installed before llmcompressor - matches cpatonn's quantization_config.version
echo ""
echo "[6/9] Installing compressed-tensors==0.11.0 (from ${REQUIREMENTS_FILE})..."
grep -E "^compressed-tensors==" "${REQUIREMENTS_FILE}" | grep -v "^#" | tr '\n' ' ' | xargs -r pip install --no-cache-dir -q
echo "      compressed-tensors 0.11.0 installed (CRITICAL for MoE support - matches cpatonn)"

# 7. Install llmcompressor dependencies (from requirements.txt)
echo ""
echo "[7/9] Installing llmcompressor dependencies (from ${REQUIREMENTS_FILE})..."
grep -E "^datasets|^accelerate|^pydantic|^loguru|^pynvml|^nvidia-ml-py|^frozendict" "${REQUIREMENTS_FILE}" | grep -v "^#" | tr '\n' ' ' | xargs -r pip install --no-cache-dir -q
echo "      Dependencies installed"

# Now install llmcompressor with --no-deps to avoid transformers version conflict
echo "      Installing llmcompressor==0.7.1 (no-deps, from ${REQUIREMENTS_FILE})..."
grep -E "^llmcompressor==" "${REQUIREMENTS_FILE}" | grep -v "^#" | tr '\n' ' ' | xargs -r pip install --no-cache-dir --no-deps -q
echo "      llmcompressor 0.7.1 installed"

# 8. Install utilities (from requirements.txt)
echo ""
echo "[8/9] Installing utilities (from ${REQUIREMENTS_FILE})..."
grep -E "^python-dotenv|^sentencepiece|^protobuf|^psutil" "${REQUIREMENTS_FILE}" | grep -v "^#" | tr '\n' ' ' | xargs -r pip install --no-cache-dir -q
echo "      Utilities installed"

# 9. Final fixes - numpy (from requirements.txt)
echo ""
echo "[9/10] Final fixes - numpy (from ${REQUIREMENTS_FILE})..."
grep -E "^numpy>=" "${REQUIREMENTS_FILE}" | grep -v "^#" | tr '\n' ' ' | xargs -r pip install --no-cache-dir -q
echo "      numpy fixed"

# 10. Download cpatonn reference model (for tokenizer + config parity)
echo ""
echo "[10/10] Caching cpatonn reference model..."
mkdir -p "${COMPARE_DIR}"
if [ -d "${REF_DIR}" ] && [ -f "${REF_DIR}/config.json" ]; then
    echo "      Reference model already cached at ${REF_DIR}"
else
    echo "      Downloading cpatonn/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit (this may take a while)..."
    REF_DIR="${REF_DIR}" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HUGGINGFACE_TOKEN")
target = os.environ["REF_DIR"]

if not token:
    raise SystemExit("HUGGINGFACE_TOKEN is required to download the cpatonn reference model.")

os.makedirs(target, exist_ok=True)
snapshot_download(
    repo_id="cpatonn/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit",
    repo_type="model",
    local_dir=target,
    local_dir_use_symlinks=False,
    token=token,
    resume_download=True,
)
print(f"      Reference model downloaded to: {target}")
PY
fi

# ============================================
# VERIFICATION
# ============================================
echo ""
echo "=============================================="
echo "Verifying Installation..."
echo "=============================================="

# Check Qwen3OmniMoe import
echo -n "  Qwen3OmniMoeForConditionalGeneration: "
python3 -c "from transformers import Qwen3OmniMoeForConditionalGeneration; print('OK')" 2>/dev/null || {
    echo "FAILED"
    echo ""
    echo "DEBUG: Trying to import transformers..."
    python3 -c "import transformers; print(f'transformers version: {transformers.__version__}')"
    python3 -c "from transformers import Qwen3OmniMoeForConditionalGeneration" 2>&1 || true
    exit 1
}

# Check llmcompressor import
echo -n "  llmcompressor.oneshot: "
python3 -c "from llmcompressor import oneshot; print('OK')" 2>/dev/null || {
    echo "FAILED"
    exit 1
}

# Check compressed-tensors version (CRITICAL for MoE quantization)
echo -n "  compressed-tensors==0.11.0: "
CT_VER=$(python3 -c "import compressed_tensors; print(compressed_tensors.__version__)" 2>/dev/null)
if [ "$CT_VER" = "0.11.0" ]; then
    echo "OK"
else
    echo "FAILED (got $CT_VER, need 0.11.0 for MoE expert quantization)"
    exit 1
fi

# Show versions
echo ""
echo "  Versions:"
python3 -c "import torch; print(f'    torch:             {torch.__version__}')"
python3 -c "import transformers; print(f'    transformers:      {transformers.__version__}')"
python3 -c "import compressed_tensors; print(f'    compressed-tensors: {compressed_tensors.__version__}')"
python3 -c "import llmcompressor; print(f'    llmcompressor:     {llmcompressor.__version__}')"
python3 -c "import numpy; print(f'    numpy:             {numpy.__version__}')"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python awq_quantization.py"
echo ""
