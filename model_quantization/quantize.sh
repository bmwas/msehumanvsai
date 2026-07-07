#!/bin/bash
# =============================================================================
#  Qwen3-Omni 4-bit Quantization - One Command Magic
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/quantization_venv"
PYTHON_SCRIPT="${SCRIPT_DIR}/awq_quantization.py"
ENV_FILE="${SCRIPT_DIR}/quantization_env.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${PURPLE}=======================================================================${NC}"
    echo -e "${BOLD}${WHITE}  Qwen3-Omni 4-bit Quantization${NC}"
    echo -e "${PURPLE}=======================================================================${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${CYAN}--- [$1/$2] $3 ---${NC}"
}

print_success() { echo -e "    ${GREEN}OK${NC} $1"; }
print_warning() { echo -e "    ${YELLOW}WARNING${NC} $1"; }
print_error() { echo -e "    ${RED}FAILED${NC} $1"; }

spinner() {
    local pid=$1
    local spinstr='|/-\'
    while ps -p $pid > /dev/null 2>&1; do
        for i in 0 1 2 3; do
            printf "\r    ${CYAN}${spinstr:$i:1}${NC} $2"
            sleep 0.2
        done
    done
    printf "\r                                                    \r"
}

run_with_spinner() {
    local message=$1
    shift
    "$@" > /tmp/quantize_output.log 2>&1 &
    local pid=$!
    spinner $pid "$message"
    wait $pid
    local status=$?
    if [ $status -ne 0 ]; then
        print_error "$message"
        tail -10 /tmp/quantize_output.log
        return $status
    fi
    print_success "$message"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -n "$GPU_NAME" ] && echo "$GPU_NAME|$GPU_MEM" && return 0
    fi
    echo "none|0"
}

# ═══════════════════════════════════════════════════════════════════════════
# Pre-flight Checks
# ═══════════════════════════════════════════════════════════════════════════

print_header
echo "Pre-flight checks..."

command -v python3 &> /dev/null || { print_error "Python3 not found"; exit 1; }
print_success "Python $(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

[ -f "$ENV_FILE" ] || { print_error "Missing: ${ENV_FILE}"; exit 1; }
print_success "Environment file"

source "$ENV_FILE"
if [ -z "$HUGGINGFACE_TOKEN" ] || [ -z "$HUGGINGFACE_NAME" ]; then
    print_error "Missing HF credentials"
    exit 1
fi
COMPARE_DIR="${SCRIPT_DIR}/../comparison_model"
REF_DIR="${COMPARE_DIR}/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit"
print_success "HuggingFace: ${HUGGINGFACE_NAME}"

GPU_INFO=$(check_gpu)
GPU_NAME=$(echo "$GPU_INFO" | cut -d'|' -f1)
GPU_MEM=$(echo "$GPU_INFO" | cut -d'|' -f2)
if [ "$GPU_NAME" != "none" ]; then
    print_success "GPU: ${GPU_NAME} ($((GPU_MEM / 1024))GB)"
else
    print_warning "No GPU detected"
fi

TOTAL_RAM_GB=$(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))
[ "$TOTAL_RAM_GB" -ge 150 ] && print_success "RAM: ${TOTAL_RAM_GB}GB" || print_warning "RAM: ${TOTAL_RAM_GB}GB (need 150GB+)"

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Create Virtual Environment
# ═══════════════════════════════════════════════════════════════════════════

print_step 1 8 "Creating virtual environment"

[ -d "${VENV_DIR}" ] && rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}" 2>/dev/null
source "${VENV_DIR}/bin/activate"
print_success "Environment created"

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Upgrade pip
# ═══════════════════════════════════════════════════════════════════════════

print_step 2 8 "Upgrading pip"

run_with_spinner "pip, wheel, setuptools" \
    pip install --upgrade pip wheel setuptools --no-cache-dir -q

# ═══════════════════════════════════════════════════════════════════════════
# INSTALLATION FROM requirements.txt
# All package versions are read from requirements.txt to ensure consistency
# Installation order is critical: PyTorch -> transformers -> compressed-tensors -> llmcompressor
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Install PyTorch 2.7.0 FIRST (from requirements.txt)
# Using default PyPI which bundles CUDA 12.6 libraries
# ═══════════════════════════════════════════════════════════════════════════

print_step 3 8 "Installing PyTorch 2.7.0 (from requirements.txt)"

run_with_spinner "torch, torchvision, torchaudio" \
    grep -E "^torch==|^torchvision==|^torchaudio==" "${SCRIPT_DIR}/requirements.txt" | grep -v "^#" | \
    xargs pip install --no-cache-dir -q

# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Install transformers 4.57.3 (from requirements.txt)
# ═══════════════════════════════════════════════════════════════════════════

print_step 4 8 "Installing transformers==4.57.3 (from requirements.txt)"

run_with_spinner "transformers 4.57.3" \
    grep -E "^transformers==" "${SCRIPT_DIR}/requirements.txt" | grep -v "^#" | \
    xargs pip install --no-cache-dir -q

# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Install compressed-tensors 0.11.0 FIRST (from requirements.txt)
# CRITICAL: This must be installed before llmcompressor - matches cpatonn
# ═══════════════════════════════════════════════════════════════════════════

print_step 5 8 "Installing compressed-tensors==0.11.0 (from requirements.txt)"

run_with_spinner "compressed-tensors 0.11.0 (CRITICAL for MoE)" \
    grep -E "^compressed-tensors==" "${SCRIPT_DIR}/requirements.txt" | grep -v "^#" | \
    xargs pip install --no-cache-dir -q

# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Install llmcompressor dependencies + llmcompressor (from requirements.txt)
# ═══════════════════════════════════════════════════════════════════════════

print_step 6 8 "Installing llmcompressor==0.7.1 (from requirements.txt)"

run_with_spinner "llmcompressor dependencies" \
    grep -E "^datasets|^accelerate|^pydantic|^loguru|^pynvml|^nvidia-ml-py|^frozendict" "${SCRIPT_DIR}/requirements.txt" | grep -v "^#" | \
    xargs pip install --no-cache-dir -q

run_with_spinner "llmcompressor==0.7.1 (no-deps)" \
    grep -E "^llmcompressor==" "${SCRIPT_DIR}/requirements.txt" | grep -v "^#" | \
    xargs pip install --no-cache-dir --no-deps -q

# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Install utilities + final fixes (from requirements.txt)
# ═══════════════════════════════════════════════════════════════════════════

print_step 7 8 "Installing utilities + final fixes (from requirements.txt)"

run_with_spinner "utilities" \
    grep -E "^python-dotenv|^sentencepiece|^protobuf|^psutil" "${SCRIPT_DIR}/requirements.txt" | grep -v "^#" | \
    xargs pip install --no-cache-dir -q

run_with_spinner "numpy>=2.0" \
    grep -E "^numpy>=" "${SCRIPT_DIR}/requirements.txt" | grep -v "^#" | \
    xargs pip install --no-cache-dir -q

# ════════════════════════════════════════════════════════════════════════════
# Step 8: Cache cpatonn reference model
# ════════════════════════════════════════════════════════════════════════════

print_step 8 8 "Caching cpatonn reference model"

mkdir -p "${COMPARE_DIR}"
if [ -d "${REF_DIR}" ] && [ -f "${REF_DIR}/config.json" ]; then
    print_success "Reference already cached"
else
    echo "    Downloading cpatonn/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit (≈20GB)..."
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
print(f"    Reference model downloaded to: {target}")
PY
    print_success "Reference cached"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 9: Verify & Run
# ═══════════════════════════════════════════════════════════════════════════

print_step 9 9 "Verifying installation"

# Check imports
python3 -c "from transformers import Qwen3OmniMoeForConditionalGeneration" 2>/dev/null \
    && print_success "Qwen3OmniMoeForConditionalGeneration" \
    || { print_error "transformers import"; exit 1; }

python3 -c "from llmcompressor import oneshot" 2>/dev/null \
    && print_success "llmcompressor.oneshot" \
    || { print_error "llmcompressor import"; exit 1; }

# CRITICAL: Check compressed-tensors version (must be 0.11.0 for MoE expert quantization)
CT_VER=$(python3 -c "import compressed_tensors; print(compressed_tensors.__version__)" 2>/dev/null)
if [ "$CT_VER" = "0.11.0" ]; then
    print_success "compressed-tensors==0.11.0 (MoE support)"
else
    print_error "compressed-tensors version (got $CT_VER, need 0.11.0)"
    exit 1
fi

# Show versions
echo ""
echo "    Versions:"
python3 -c "import torch; print(f'      torch:             {torch.__version__}')"
python3 -c "import transformers; print(f'      transformers:      {transformers.__version__}')"
python3 -c "import compressed_tensors; print(f'      compressed-tensors: {compressed_tensors.__version__}')"
python3 -c "import llmcompressor; print(f'      llmcompressor:     {llmcompressor.__version__}')"
python3 -c "import numpy; print(f'      numpy:             {numpy.__version__}')"

echo ""
echo -e "${BOLD}${WHITE}Starting quantization...${NC}"
echo "(This will take 30min - 2+ hours)"
echo ""

cd "$SCRIPT_DIR"
python3 "$PYTHON_SCRIPT"

# ═══════════════════════════════════════════════════════════════════════════
# Complete!
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${GREEN}=======================================================================${NC}"
echo -e "${BOLD}${WHITE}  Quantization Complete!${NC}"
echo -e "${GREEN}=======================================================================${NC}"
echo ""
echo "  Local:  ./Qwen3-Omni-Thinking-4bit/"
echo "  Remote: https://huggingface.co/${HUGGINGFACE_NAME}/Qwen3-Omni-30B-Thinking-4bit"
echo ""
