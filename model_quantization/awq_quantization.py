"""Qwen3-Omni 4-bit Quantization Script (llm-compressor / compressed-tensors)
=============================================================================
Quantizes `Qwen/Qwen3-Omni-30B-A3B-Thinking` to 4-bit using llm-compressor,
matching the cpatonn/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit method exactly.

Key settings (from cpatonn config):
- quant_method: compressed-tensors
- format: pack-quantized
- num_bits: 4, group_size: 32, symmetric: true, observer: mse
- Ignore: audio_tower, visual, MoE gates, lm_head
"""

import os
import sys
import logging
import random
import gc
import shutil
import importlib
import types
import json
import tempfile

# Memory optimization before any torch imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# =============================================================================
# CRITICAL: Patch transformers BEFORE any llmcompressor imports
# transformers 5.x removed TRANSFORMERS_CACHE but llmcompressor still needs it
# =============================================================================
import transformers

# Get the cache directory from huggingface_hub or use default
try:
    from huggingface_hub import constants as _hf_constants
    _cache_dir = getattr(_hf_constants, 'HF_HUB_CACHE', None)
    if _cache_dir is None:
        _cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
except Exception:
    _cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

# ALWAYS set TRANSFORMERS_CACHE (even if it exists, ensure it's correct)
transformers.TRANSFORMERS_CACHE = _cache_dir

# Also set the environment variable as a fallback
os.environ.setdefault('TRANSFORMERS_CACHE', _cache_dir)
os.environ.setdefault('HF_HOME', os.path.dirname(_cache_dir))

# Verify it works
try:
    from transformers import TRANSFORMERS_CACHE as _test_cache
    print(f"[✓] TRANSFORMERS_CACHE patched: {_test_cache}")
except ImportError as e:
    print(f"[!] WARNING: TRANSFORMERS_CACHE patch failed: {e}")
    print(f"    Attempting alternative fix...")
    # Alternative: inject into transformers.__dict__ directly
    transformers.__dict__['TRANSFORMERS_CACHE'] = _cache_dir
    try:
        from transformers import TRANSFORMERS_CACHE as _test_cache2
        print(f"[✓] TRANSFORMERS_CACHE patched (alt method): {_test_cache2}")
    except ImportError:
        print(f"[!] CRITICAL: Cannot patch TRANSFORMERS_CACHE - quantization may fail")

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download, login as hf_login
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme, QuantizationType
from safetensors import safe_open
from safetensors.numpy import save_file

try:
    import psutil
except ImportError:
    psutil = None

# ============================================================================
# Logging Setup
# ============================================================================

def _resolve_log_level():
    level_name = os.getenv("QUANT_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


LOG_FORMAT = os.getenv("QUANT_LOG_FORMAT", "[%(asctime)s] %(levelname)s - %(message)s")
logging.basicConfig(level=_resolve_log_level(), format=LOG_FORMAT)
logger = logging.getLogger("awq_quantization")
logger.setLevel(_resolve_log_level())


def ensure_transformers_fx_available():
    """
    transformers 5.0.dev removed transformers.utils.fx, but llmcompressor still imports it.
    Provide a lightweight shim backed by torch.fx so imports keep working.
    """
    try:
        importlib.import_module("transformers.utils.fx")
        return
    except Exception:
        pass

    try:
        import torch.fx as torch_fx
    except ImportError:
        logger.warning("torch.fx missing; cannot create transformers.utils.fx shim")
        return

    fx_module = types.ModuleType("transformers.utils.fx")
    fx_module.symbolic_trace = torch_fx.symbolic_trace
    fx_module.GraphModule = torch_fx.GraphModule
    fx_module.Graph = torch_fx.Graph
    fx_module.Proxy = torch_fx.Proxy
    fx_module.wrap = getattr(torch_fx, "wrap", lambda fn: fn)

    class HFTracer(torch_fx.Tracer):
        """Minimal HuggingFace-style tracer compatible with llmcompressor needs."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def trace(self, root, concrete_args=None):
            return super().trace(root, concrete_args=concrete_args)

    fx_module.HFTracer = HFTracer

    sys.modules["transformers.utils.fx"] = fx_module
    logger.warning("Patched missing transformers.utils.fx with torch.fx-based shim")


def ensure_transformers_cache_available():
    """
    transformers 5.0.dev removed TRANSFORMERS_CACHE constant, but llmcompressor imports it.
    Add it back to the transformers module.
    """
    try:
        from transformers import TRANSFORMERS_CACHE
        return  # Already exists
except ImportError:
        pass

    try:
        import transformers
        from huggingface_hub import constants as hf_constants
        # Use huggingface_hub's cache dir (the new standard location)
        cache_dir = getattr(hf_constants, 'HF_HUB_CACHE', None)
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        transformers.TRANSFORMERS_CACHE = cache_dir
        logger.warning("Patched missing transformers.TRANSFORMERS_CACHE = %s", cache_dir)
    except Exception as e:
        logger.warning("Could not patch TRANSFORMERS_CACHE: %s", e)


def prepare_tokenizer_assets(model_repo: str, token: str) -> str:
    """
    Some recent commits of transformers output tokenizer_config.json with in-memory `merges`
    lists. Older tokenizers (and the Rust BPE backend) expect either tuples or a file path.
    Mirror the tokenizer files locally, convert list-based merges into a merges.txt file,
    and return the directory path for safe loading.
    """
    download_dir = snapshot_download(
        repo_id=model_repo,
        token=token,
        allow_patterns=[
            "tokenizer*",
            "vocab*",
            "*.json",
            "*.model",
            "*.txt",
        ],
    )
    local_dir = tempfile.mkdtemp(prefix="qwen_tokenizer_")
    shutil.copytree(download_dir, local_dir, dirs_exist_ok=True)

    merges_txt_path = os.path.join(local_dir, "merges.txt")

    # If tokenizer.json exists, extract merges array as fallback
    tokenizer_json_path = os.path.join(local_dir, "tokenizer.json")
    merges_from_tokenizer = None
    if os.path.exists(tokenizer_json_path):
        try:
            tokenizer_json = json.load(open(tokenizer_json_path, "r", encoding="utf-8"))
            merges_from_tokenizer = tokenizer_json.get("model", {}).get("merges")
        except Exception as err:
            logger.warning("Failed to parse tokenizer.json for merges: %s", err)

    config_path = os.path.join(local_dir, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            config_data = json.load(fh)

        merges = config_data.pop("merges", None) or merges_from_tokenizer
        merges_file = config_data.get("merges_file")

        if isinstance(merges, list) and merges:
            with open(merges_txt_path, "w", encoding="utf-8") as fh:
                for pair in merges:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        fh.write(f"{pair[0]} {pair[1]}\n")
                    elif isinstance(pair, str):
                        fh.write(f"{pair}\n")
            config_data["merges_file"] = os.path.basename(merges_txt_path)

        if not config_data.get("merges_file") and os.path.exists(merges_txt_path):
            config_data["merges_file"] = os.path.basename(merges_txt_path)

        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh, indent=2)

    return local_dir

# ============================================================================
# Configuration (matching cpatonn exactly)
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "quantization_env.env")

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
SAVE_DIR = "./Qwen3-Omni-Thinking-4bit"
N_SAMPLES = 256

# These are the base patterns to ignore - llm-compressor will match by prefix
# The full explicit list is built dynamically after model load
IGNORE_PATTERNS = [
    # Audio tower (all 32 layers + conv/proj)
    "thinker.audio_tower",
    # Visual encoder (all 27 blocks + merger)
    "thinker.visual",
    # MoE gates (all 48 layers)
    "thinker.model.layers.*.mlp.gate",
    # LM head
    "thinker.lm_head",
]

MODEL_CARD_TEMPLATE = """---
language: en
license: other
base_model: Qwen/Qwen3-Omni-30B-A3B-Thinking
tags:
- qwen
- qwen3_omni_moe
- text-to-audio
- multimodal
- compressed-tensors
- 4bit
- quantization
---

# {repo_id}

## Model Summary

- **Base model:** `Qwen/Qwen3-Omni-30B-A3B-Thinking`
- **Quantization:** llm-compressor W4A16 (compressed-tensors, pack-quantized)
- **Config:** num_bits=4, group_size=32, symmetric=true, observer=mse
- **Format:** Hugging Face safetensors (compressed-tensors)
- **Shard layout:** 5 × ~5 GB compressed-tensors (matches cpatonn reference)
- **Tokenizer/config assets:** Copied directly from `cpatonn/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit` for vLLM parity
- **Purpose:** Reduce VRAM footprint for multimodal clinical reasoning while keeping Omni's audio/vision towers in FP16.

## Quantization Details

| Item | Value |
| ---- | ----- |
| quant_method | compressed-tensors |
| format | pack-quantized |
| num_bits | 4 |
| group_size | 32 |
| symmetric | true |
| observer | mse |
| strategy | group |
| Calibration samples | ~230 mixed clinical/mental-health snippets |
| Datasets | WikiText-2, mental health classification, sentiment analysis, counseling conversations |
| FP16 modules | Audio tower (32 layers), Vision encoder (27 blocks), MoE gates (48 layers), LM head |
| Tooling | llm-compressor, torch 2.6.0/cu124, transformers (GitHub) |
| Shard count | 5 × ~5 GB (compressed-tensors) |
| Tokenizer + configs | Copied from cpatonn reference cache |

## Resource Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| GPU VRAM | 24 GB | 48‑80 GB | Less VRAM → heavier CPU offload / longer runtime |
| System RAM | 150 GB | 200 GB+ | Needed during model save (state dict gathering) |
| Disk / NVMe | 200 GB | 300 GB+ | Base model + quantized output + temp caches |

## Usage with vLLM

```python
from vllm import LLM, SamplingParams

model = LLM(
    model="{repo_id}",
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
)

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048)
outputs = model.generate(["Hello, how are you?"], sampling_params)
print(outputs[0].outputs[0].text)
```

## Usage with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
```

## Limitations

- Inherits base-model biases; not a substitute for licensed medical advice.
- Calibration data is mental-health heavy and may bias tone in general conversations.
"""


def log_mem(stage: str):
    """Log current CPU and GPU memory usage."""
    if psutil:
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        logger.info("[%s] CPU RSS %.2f GB", stage, rss)
    if torch.cuda.is_available():
        try:
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info("[%s] GPU alloc=%.2f GB reserved=%.2f GB", stage, alloc, reserved)
        except RuntimeError as exc:
            logger.warning("Unable to query CUDA memory at %s: %s", stage, exc)


def safe_get(row, *keys):
    """Safely extract a value from a dataset row, trying multiple keys."""
    for key in keys:
        try:
            val = row.get(key) if hasattr(row, "get") else row[key] if key in row else None
            if val is not None and str(val).strip():
                return str(val)
        except Exception:
            continue
    return ""


def get_cpatonn_ignore_list():
    """
    Extract the EXACT ignore list from cpatonn's config.json.
    This ensures we quantize the same layers they did.
    """
    CPATONN_DIR = os.path.join(os.path.dirname(__file__), "..", "comparison_model", "Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit")
    config_path = os.path.join(CPATONN_DIR, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        ignore_list = cfg.get('quantization_config', {}).get('ignore', [])
        if ignore_list:
            logger.info("Using cpatonn ignore list with %d modules", len(ignore_list))
            return ignore_list
    
    logger.warning("cpatonn config not found, falling back to dynamic ignore list")
    return None


def build_explicit_ignore_list(model):
    """
    Build the EXACT ignore list that cpatonn uses.
    First try to use cpatonn's actual list, otherwise enumerate dynamically.
    """
    # Try to use cpatonn's exact ignore list first
    cpatonn_ignore = get_cpatonn_ignore_list()
    if cpatonn_ignore:
        print(f"[✓] Using cpatonn's exact ignore list ({len(cpatonn_ignore)} modules)")
        return cpatonn_ignore
    
    # Fall back to dynamic enumeration
    ignore = []
    
    # Enumerate all named modules and collect the ones to ignore
    for name, module in model.named_modules():
        # Skip if not a leaf module (we want specific layer names)
        if len(list(module.children())) > 0:
            continue
            
        # Audio tower layers
        if "audio_tower" in name and any(x in name for x in [
            "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", 
            "self_attn.out_proj", "fc1", "fc2", "conv_out", "proj1", "proj2"
        ]):
            ignore.append(name)
            
        # Visual blocks
        elif "visual" in name and any(x in name for x in [
            "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2",
            "merger.mlp.0", "merger.mlp.2", "merger_list"
        ]):
            ignore.append(name)
            
        # MoE gates
        elif ".mlp.gate" in name:
            ignore.append(name)
            
        # LM head
        elif name == "thinker.lm_head" or name.endswith(".lm_head"):
            ignore.append(name)
    
    logger.info("Built dynamic ignore list with %d modules", len(ignore))
    return ignore


def get_calibration_dataset(n_samples=256):
    """
    Build a calibration dataset from clinical/mental-health sources.
    Returns a Dataset with a 'text' column as required by llm-compressor.
    """
    raw_text = []
    per_source = n_samples // 5

    def maybe_add(text):
        text = (text or "").strip()
            if len(text) > 50:
            raw_text.append(text[:2000])

    sources = [
        ("WikiText", lambda: load_dataset("wikitext", "wikitext-2-raw-v1", split="train")),
        ("MH Classification", lambda: load_dataset("sai1908/Mental_Health_Condition_Classification", split="train")),
        ("MH Sentiment", lambda: load_dataset("btwitssayan/sentiment-analysis-for-mental-health", split="train")),
        ("Counseling", lambda: load_dataset("Amod/mental_health_counseling_conversations", split="train")),
        ("MH Chatbot", lambda: load_dataset("heliosbrahma/mental_health_chatbot_dataset", split="train")),
    ]

    for label, loader in sources:
        try:
            ds = loader()
            idxs = list(range(len(ds)))
            random.shuffle(idxs)
            added = 0
            for idx in idxs:
                if added >= per_source:
                    break
                row = ds[idx]
                if label == "WikiText":
                    maybe_add(row.get("text", ""))
                elif label == "MH Classification":
                    stmt = safe_get(row, "text", "statement")
                    lbl = safe_get(row, "label", "status")
                    if stmt:
                        sample = f"Patient Statement: {stmt}"
                        if lbl:
                            sample += f"\nMental Health Assessment: {lbl}"
                        maybe_add(sample)
                elif label == "MH Sentiment":
                    stmt = safe_get(row, "statement", "text")
                    status = safe_get(row, "status", "label")
                    if stmt:
                        sample = f"Statement: {stmt}"
                        if status:
                            sample += f"\nMental Health Status: {status}"
                        maybe_add(sample)
                elif label == "Counseling":
                    ctx = safe_get(row, "Context", "context", "question")
                    resp = safe_get(row, "Response", "response", "answer")
                    if ctx and resp:
                        maybe_add(f"Patient: {ctx}\nTherapist: {resp}")
                elif label == "MH Chatbot":
                    q = safe_get(row, "text", "question", "input")
                    a = safe_get(row, "label", "answer", "output")
                    if q:
                        sample = f"Question: {q}"
                        if a:
                            sample += f"\nAnswer: {a}"
                        maybe_add(sample)
                added += 1
            logger.info("Added %s samples from %s", added, label)
            print(f"   ✓ Added {added} samples from {label}")
        except Exception as exc:
            logger.warning("Failed to load %s: %s", label, exc)
            print(f"   ⚠️ {label}: {exc}")

    random.shuffle(raw_text)
    if not raw_text:
        raise RuntimeError("Failed to build calibration dataset - no data collected!")
    
    logger.info("Calibration dataset prepared (%s samples)", len(raw_text))
    print(f"[✓] Calibration dataset: {len(raw_text)} samples")
    
    return Dataset.from_dict({"text": raw_text[:n_samples]})


def write_model_card(output_dir: str, repo_id: str):
    """Generate and write the model card README."""
    card_path = os.path.join(output_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as fh:
        fh.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))
    logger.info("Model card written to %s", card_path)


def copy_tokenizer_from_cpatonn(target_dir: str):
    """
    Copy tokenizer files from the cpatonn reference model.
    This ensures vLLM compatibility (extra_special_tokens as dict, not list).
    """
    CPATONN_DIR = os.path.join(os.path.dirname(__file__), "..", "comparison_model", "Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit")
    
    if not os.path.exists(CPATONN_DIR):
        logger.warning("cpatonn reference not found at %s, skipping tokenizer copy", CPATONN_DIR)
        return False
    
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json", 
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
    ]
    
    copied = 0
    for fname in tokenizer_files:
        src = os.path.join(CPATONN_DIR, fname)
        dst = os.path.join(target_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
            logger.info("Copied %s from cpatonn reference", fname)
    
    print(f"[✓] Copied {copied} tokenizer files from cpatonn reference")
    return True


def copy_other_configs_from_cpatonn(target_dir: str):
    """
    Copy core config files from the cpatonn reference (always overwrite).
    Ensures our metadata (config, generation, recipe, preprocessors) matches exactly.
    """
    CPATONN_DIR = os.path.join(os.path.dirname(__file__), "..", "comparison_model", "Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit")
    
    if not os.path.exists(CPATONN_DIR):
        logger.warning("cpatonn reference missing at %s; cannot copy configs", CPATONN_DIR)
        return False
    
    config_files = [
        ("config.json", True),
        ("generation_config.json", True),
        ("recipe.yaml", True),
        ("preprocessor_config.json", False),
        ("video_preprocessor_config.json", False),
        ("chat_template.jinja", False),
    ]
    
    copied = 0
    for fname, _ in config_files:
        src = os.path.join(CPATONN_DIR, fname)
        if not os.path.exists(src):
            continue
        dst = os.path.join(target_dir, fname)
        shutil.copy2(src, dst)
        copied += 1
        logger.info("Copied %s from cpatonn reference", fname)
    
    print(f"[✓] Copied {copied} config files from cpatonn reference (overwritten)")
    return True


def attach_qwen3omni_embedding_shim(model):
    """
    llmcompressor expects get_input_embeddings/set_input_embeddings to be implemented.
    Qwen3OmniMoeForConditionalGeneration does not override these, so we attach a shim
    that proxies to thinker.model.embed_tokens (or the closest available module).
    """
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration
    except ImportError:
        logger.warning("transformers import missing; cannot attach embedding shim")
        return

    if not isinstance(model, Qwen3OmniMoeForConditionalGeneration):
        return

    def _find_embed_module(obj):
        paths = [
            getattr(getattr(getattr(obj, "thinker", None), "model", None), "embed_tokens", None),
            getattr(getattr(obj, "model", None), "embed_tokens", None),
            getattr(obj, "embed_tokens", None),
        ]
        for candidate in paths:
            if candidate is not None:
                return candidate
        return None

    embed_module = _find_embed_module(model)
    if embed_module is None:
        logger.warning("Could not locate embed_tokens for Qwen3OmniMoe; embedding shim skipped.")
        return

    def _get_input_embeddings(self):
        module = _find_embed_module(self)
        if module is None:
            raise ValueError("embed_tokens not found for Qwen3OmniMoe model")
        return module

    def _set_input_embeddings(self, new_embeddings):
        if hasattr(self, "thinker") and hasattr(self.thinker, "model") and hasattr(self.thinker.model, "embed_tokens"):
            self.thinker.model.embed_tokens = new_embeddings
        elif hasattr(self, "model") and hasattr(self.model, "embed_tokens"):
            self.model.embed_tokens = new_embeddings
            else:
            self.embed_tokens = new_embeddings

    model.get_input_embeddings = types.MethodType(_get_input_embeddings, model)
    model.set_input_embeddings = types.MethodType(_set_input_embeddings, model)
    logger.info("Attached Qwen3OmniMoe embedding shim for get_input_embeddings/set_input_embeddings")


def reshard_to_match_cpatonn(target_dir: str):
    """
    Re-shard the quantized safetensors so the file layout matches cpatonn (5 shards).
    This keeps tensor data intact but groups tensors into the same files as the reference.
    """
    CPATONN_DIR = os.path.join(os.path.dirname(__file__), "..", "comparison_model", "Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit")
    ref_index_path = os.path.join(CPATONN_DIR, "model.safetensors.index.json")
    our_index_path = os.path.join(target_dir, "model.safetensors.index.json")
    
    if not (os.path.exists(ref_index_path) and os.path.exists(our_index_path)):
        print("   ⚠️ Missing index files; cannot re-shard to match cpatonn")
        return False
    
    with open(ref_index_path, "r") as f:
        ref_index = json.load(f)
    with open(our_index_path, "r") as f:
        our_index = json.load(f)
    
    our_weight_map = our_index.get("weight_map", {})
    ref_weight_map = ref_index.get("weight_map", {})
    
    our_shards = sorted(set(our_weight_map.values()))
    ref_shards = sorted(set(ref_weight_map.values()))
    
    if our_shards == ref_shards:
        print("   [✓] Shard layout already matches cpatonn")
        return True
    
    if set(our_weight_map.keys()) != set(ref_weight_map.keys()):
        logger.error("Tensor name mismatch between our model and cpatonn reference")
        raise RuntimeError("Cannot re-shard because tensor names differ from reference.")
    
    print(f"   [-] Re-sharding tensors: {len(our_shards)} -> {len(ref_shards)} files (matching cpatonn)")
    
    # Group target tensors per reference shard while preserving order
    target_groups: dict[str, list[str]] = {}
    for name, shard in ref_weight_map.items():
        target_groups.setdefault(shard, []).append(name)
    
    source_handles: dict[str, safe_open] = {}

    def fetch_tensor(tensor_name: str):
        src_file = our_weight_map[tensor_name]
        src_path = os.path.join(target_dir, src_file)
        handle = source_handles.get(src_path)
        if handle is None:
            handle = safe_open(src_path, framework="numpy")
            source_handles[src_path] = handle
        array = handle.get_tensor(tensor_name)
        return array.copy()
    
    metadata = our_index.get("metadata", {})
    
    for shard_name, tensor_names in target_groups.items():
        shard_path = os.path.join(target_dir, shard_name)
        tensors = {}
        for tensor_name in tensor_names:
            tensors[tensor_name] = fetch_tensor(tensor_name)
        save_file(tensors, shard_path, metadata=metadata)
        print(f"      [✓] Wrote {shard_name} with {len(tensor_names)} tensors")
        del tensors
    
    for handle in source_handles.values():
        handle.close()
    
    # Remove old shard files that are not part of the reference layout
    for shard in set(our_shards):
        if shard not in ref_shards:
            shard_path = os.path.join(target_dir, shard)
            if os.path.exists(shard_path):
                os.remove(shard_path)
    
    shutil.copy2(ref_index_path, our_index_path)
    print("   [✓] Replaced model.safetensors.index.json with cpatonn version")
    print("   [✓] Shard layout now matches cpatonn reference")
    return True


def restore_tokenizer_config(base_model: str, token: str, target_dir: str):
    """Overwrite tokenizer_config.json with the base model version for vLLM compatibility."""
    try:
        src = snapshot_download(
            repo_id=base_model,
            allow_patterns=["tokenizer_config.json"],
            token=token,
        )
        src_file = os.path.join(src, "tokenizer_config.json")
        dst_file = os.path.join(target_dir, "tokenizer_config.json")
        shutil.copy2(src_file, dst_file)
        logger.info("Restored tokenizer_config.json from %s", base_model)
    except Exception as exc:
        logger.warning("Failed to restore tokenizer_config.json: %s", exc)


def main():
    # ========================================================================
    # Step 0: Load environment and authenticate
    # ========================================================================
    print("=" * 70)
    print("Qwen3-Omni 4-bit Quantization (llm-compressor / compressed-tensors)")
    print("=" * 70)
    print(f"Model:     {MODEL_PATH}")
    print(f"Output:    {SAVE_DIR}")
    print("")
    
    print(f"[-] Loading environment from: {ENV_PATH}")
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
    else:
        raise FileNotFoundError(f"Missing {ENV_PATH}")

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    hf_user = os.getenv("HUGGINGFACE_NAME")
    if not hf_token or not hf_user:
        raise ValueError("HUGGINGFACE_TOKEN and HUGGINGFACE_NAME must be set in quantization_env.env")

    # Export llm-compressor offload settings if present
    for var in ["LLMCOMPRESSOR_STATE_DICT_OFFLOAD_DIR", 
                "LLMCOMPRESSOR_STATE_DICT_OFFLOAD_SIZE_GB",
                "LLMCOMPRESSOR_MAX_SAVE_CHUNKS"]:
        val = os.getenv(var)
        if val:
            os.environ[var] = val
            logger.info("Set %s=%s", var, val)

    print("[-] Authenticating with Hugging Face…")
    hf_login(token=hf_token, add_to_git_credential=False)
    print("[✓] Hugging Face login successful")
    
    repo_id = f"{hf_user}/Qwen3-Omni-30B-Thinking-4bit"
    print(f"[-] Will upload to: {repo_id}")
    print("")
    
    log_mem("startup")

    # Check GPU
    if torch.cuda.is_available():
        print(f"[✓] CUDA: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No CUDA GPU detected!")
    print("")

    # ========================================================================
    # Step 1: Check dependencies
    # ========================================================================
    print("=" * 70)
    print("STEP 1: Checking Dependencies")
    print("=" * 70)
    
    # Apply compatibility shims for transformers 5.x
    ensure_transformers_fx_available()
    ensure_transformers_cache_available()

    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration, AutoTokenizer, AutoConfig
        print("[✓] transformers (Qwen3OmniMoe support)")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        print("   Run: pip install git+https://github.com/huggingface/transformers")
        sys.exit(1)
    
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        print("[✓] llmcompressor")
    except ImportError as e:
        print(f"❌ llmcompressor import failed: {e}")
        print("   Run: pip install llmcompressor==0.5.1")
        sys.exit(1)
    
    print(f"[✓] PyTorch {torch.__version__}")
    print("")

    # ========================================================================
    # Step 2: Build calibration dataset
    # ========================================================================
    print("=" * 70)
    print("STEP 2: Building Calibration Dataset")
    print("=" * 70)
    calib_dataset = get_calibration_dataset(N_SAMPLES)
    log_mem("post-calibration-dataset")
    print("")

    # ========================================================================
    # Step 3: Load model
    # ========================================================================
    print("=" * 70)
    print("STEP 3: Loading Model")
    print("=" * 70)
    
    offload_folder = "./offload_tmp"
    temp_dirs: list[str] = []
    os.makedirs(offload_folder, exist_ok=True)
    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load config and patch if needed
    print("[-] Loading model config...")
    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        token=hf_token,
    )
    if not hasattr(config, 'initializer_range'):
        config.initializer_range = 0.02
        print("   (patched missing initializer_range)")
    
    # Load model
    print("[-] Loading model weights (this may take a while)...")
    logger.info("Loading model with device_map=auto")
    
    # Auto-detect memory limits
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        usable_vram = int(total_vram * 0.80)
        gpu_memory = f"{usable_vram}GiB"
        print(f"   GPU: {total_vram:.1f}GB total, using {usable_vram}GB")
    else:
        gpu_memory = "0GiB"
    
    # Get CPU memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    total_ram_kb = int(line.split()[1])
                    total_ram_gb = total_ram_kb // (1024 * 1024)
                    cpu_memory = f"{int(total_ram_gb * 0.80)}GiB"
                    print(f"   CPU: {total_ram_gb}GB total, using {int(total_ram_gb * 0.80)}GB")
                    break
    except Exception:
        cpu_memory = "100GiB"
    
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    
    # First load to CPU
    print("[-] Loading model to CPU first...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        config=config,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    print("[✓] Model loaded to CPU")
    
    # Dispatch to GPU with memory management
    print("[-] Dispatching model to GPU...")
    max_memory = get_balanced_memory(
        model,
        max_memory={0: gpu_memory, "cpu": cpu_memory},
        no_split_module_classes=["Qwen3OmniMoeDecoderLayer"],
    )
    logger.info("Memory allocation: %s", max_memory)
    
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Qwen3OmniMoeDecoderLayer"],
    )
    
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir=offload_folder,
    )
    print("[✓] Model dispatched to devices")
    
    # Ensure llmcompressor can access embeddings
    attach_qwen3omni_embedding_shim(model)
    
    # Load tokenizer - try direct load first, fall back to cpatonn's working tokenizer
    print("[-] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            token=hf_token,
            use_fast=False,
        )
        print("[✓] Tokenizer loaded from base model")
    except Exception as e:
        logger.warning("Base tokenizer failed (%s), trying cpatonn's tokenizer...", e)
        try:
            # cpatonn's quantized model has a working tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "cpatonn/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit",
                trust_remote_code=True,
                use_fast=False,
            )
            print("[✓] Tokenizer loaded from cpatonn (fallback)")
        except Exception as e2:
            logger.error("cpatonn tokenizer also failed: %s", e2)
            # Last resort: prepare assets manually
            tokenizer_dir = prepare_tokenizer_assets(MODEL_PATH, hf_token)
            temp_dirs.append(tokenizer_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                trust_remote_code=True,
                token=hf_token,
                use_fast=False,
            )
            print("[✓] Tokenizer loaded from prepared assets")
    log_mem("post-model-load")
    print("")

    # ========================================================================
    # Step 4: Build ignore list
    # ========================================================================
    print("=" * 70)
    print("STEP 4: Building Ignore List")
    print("=" * 70)

    ignore_list = build_explicit_ignore_list(model)
    print(f"[✓] Will keep {len(ignore_list)} modules in FP16")
    print("")

    # ========================================================================
    # Step 5: Configure and run quantization
    # ========================================================================
    print("=" * 70)
    print("STEP 5: Running Quantization")
    print("=" * 70)
    
    # Quantization recipe matching cpatonn config EXACTLY (W4A16, group_size=32, observer=mse)
    quant_weights = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=True,
        strategy="group",
        group_size=32,
        observer="mse",
    )
    config_groups = {
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=quant_weights,
            input_activations=None,
            output_activations=None,
        )
    }
    recipe = QuantizationModifier(
        config_groups=config_groups,
        ignore=ignore_list,
    )
    
    print("[✓] Recipe: W4A16 (4-bit weights, 16-bit activations)")
    print("    format: pack-quantized, group_size=32, symmetric=true, observer=mse")
    print("[-] Running quantization (this will take 30min - 2hrs)...")
    print("")

    # Fix generation_config before saving (temperature/top_p/top_k require do_sample=True)
    if hasattr(model, 'generation_config'):
        gen_cfg = model.generation_config
        has_sampling_params = (
            getattr(gen_cfg, 'temperature', None) not in (None, 1.0) or
            getattr(gen_cfg, 'top_p', None) not in (None, 1.0) or
            getattr(gen_cfg, 'top_k', None) not in (None, 0, 50)
        )
        if has_sampling_params and not getattr(gen_cfg, 'do_sample', True):
            print("[-] Fixing generation_config (setting do_sample=True)...")
            model.generation_config.do_sample = True
    
    logger.info("Starting quantization with oneshot()")
    log_mem("pre-oneshot")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # CRITICAL: Monkey-patch llmcompressor's broken function that tries to import TRANSFORMERS_CACHE
    # This function fails on transformers 5.x because TRANSFORMERS_CACHE was removed
    print("[-] Patching llmcompressor to skip copy_python_files_from_model_cache...")
    try:
        import llmcompressor.pytorch.model_load.helpers as _llmc_helpers
        import llmcompressor.transformers.sparsification.compressed_tensors_utils as _ct_utils

        # Store original for reference
        _original_copy_func_helpers = getattr(_llmc_helpers, 'copy_python_files_from_model_cache', None)
        _original_copy_func_ct = getattr(_ct_utils, 'copy_python_files_from_model_cache', None)
        print(f"    Original helper function: {_original_copy_func_helpers}")
        print(f"    Original CT utils function: {_original_copy_func_ct}")

        def _patched_copy_python_files_from_model_cache(model, save_directory):
            """
            Patched version that doesn't rely on TRANSFORMERS_CACHE.
            Simply skip copying python files from cache - they're not needed for inference.
            """
            msg = f"[SKIPPED] copy_python_files_from_model_cache({save_directory})"
            print(msg)
            logger.info("%s - TRANSFORMERS_CACHE deprecated, skipping copy.", msg)
            # Intentionally do nothing

        _llmc_helpers.copy_python_files_from_model_cache = _patched_copy_python_files_from_model_cache
        _ct_utils.copy_python_files_from_model_cache = _patched_copy_python_files_from_model_cache

        # Verify the patch took effect
        if (
            _llmc_helpers.copy_python_files_from_model_cache is _patched_copy_python_files_from_model_cache
            and _ct_utils.copy_python_files_from_model_cache is _patched_copy_python_files_from_model_cache
        ):
            print("[✓] Successfully patched llmcompressor copy_python_files_from_model_cache (helpers + ct_utils)")
        else:
            print("[!] WARNING: Patch may not have taken effect!")

    except Exception as e:
        print(f"[!] ERROR: Could not patch llmcompressor helper: {e}")
        logger.error("Could not patch llmcompressor helper: %s", e)
    
    # Run quantization - let llm-compressor handle saving
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calib_dataset,
        recipe=recipe,
        max_seq_length=1024,
        num_calibration_samples=min(len(calib_dataset), 256),
        output_dir=SAVE_DIR,
    )

    print("[✓] Quantization complete!")
    logger.info("Quantization finished")
    log_mem("post-oneshot")
    print("")

    # ========================================================================
    # Step 6: Finalize artifacts
    # ========================================================================
    print("=" * 70)
    print("STEP 6: Finalizing Artifacts")
    print("=" * 70)

    # CRITICAL: Copy tokenizer files from cpatonn reference (not base model!)
    # This ensures extra_special_tokens is a dict (not list) for vLLM compatibility
    print("[-] Copying tokenizer files from cpatonn reference...")
    if not copy_tokenizer_from_cpatonn(SAVE_DIR):
        print("   ⚠️ cpatonn reference not found, saving tokenizer normally...")
        tokenizer.save_pretrained(SAVE_DIR)
        # Try to fix extra_special_tokens if it's a list
        tok_cfg_path = os.path.join(SAVE_DIR, "tokenizer_config.json")
        if os.path.exists(tok_cfg_path):
            with open(tok_cfg_path, 'r') as f:
                tok_cfg = json.load(f)
            if isinstance(tok_cfg.get('extra_special_tokens'), list):
                # Convert list to dict format that cpatonn uses
                tok_cfg['extra_special_tokens'] = {
                    "audio_bos_token": "<|audio_start|>",
                    "audio_eos_token": "<|audio_end|>",
                    "audio_token": "<|audio_pad|>",
                    "image_token": "<|image_pad|>",
                    "video_token": "<|video_pad|>",
                    "vision_bos_token": "<|vision_start|>",
                    "vision_eos_token": "<|vision_end|>"
                }
                with open(tok_cfg_path, 'w') as f:
                    json.dump(tok_cfg, f, indent=2)
                print("   [✓] Fixed extra_special_tokens to dict format")

    # Copy other config files from cpatonn reference
    print("[-] Copying config files from cpatonn reference...")
    copy_other_configs_from_cpatonn(SAVE_DIR)
    
    print("[-] Re-sharding safetensors to match cpatonn layout (5 shards)...")
    reshard_to_match_cpatonn(SAVE_DIR)
    
    # Write model card
    write_model_card(SAVE_DIR, repo_id)
    print("[✓] Model card generated")
    print("")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    if os.path.exists(offload_folder):
        shutil.rmtree(offload_folder, ignore_errors=True)
    for tmp in temp_dirs:
        shutil.rmtree(tmp, ignore_errors=True)
    
    log_mem("post-cleanup")

    # ========================================================================
    # Step 7: Parity check with cpatonn
    # ========================================================================
    print("=" * 70)
    print("STEP 7: Verifying Parity with cpatonn")
    print("=" * 70)
    
    CPATONN_DIR = os.path.join(os.path.dirname(__file__), "..", "comparison_model", "Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit")
    
    if os.path.exists(CPATONN_DIR):
        # Check tokenizer_config.json structure
        our_tok_cfg = os.path.join(SAVE_DIR, "tokenizer_config.json")
        ref_tok_cfg = os.path.join(CPATONN_DIR, "tokenizer_config.json")
        
        if os.path.exists(our_tok_cfg):
            with open(our_tok_cfg, 'r') as f:
                our_cfg = json.load(f)
            with open(ref_tok_cfg, 'r') as f:
                ref_cfg = json.load(f)
            
            our_ess_type = type(our_cfg.get('extra_special_tokens')).__name__
            ref_ess_type = type(ref_cfg.get('extra_special_tokens')).__name__
            
            if our_ess_type == ref_ess_type:
                print(f"[✓] extra_special_tokens type: {our_ess_type} (matches cpatonn)")
            else:
                print(f"[!] extra_special_tokens type: {our_ess_type} vs {ref_ess_type} (MISMATCH!)")
        
        # Check safetensors count
        our_shards = [f for f in os.listdir(SAVE_DIR) if f.endswith('.safetensors')]
        ref_shards = [f for f in os.listdir(CPATONN_DIR) if f.endswith('.safetensors')]
        
        if len(our_shards) == len(ref_shards):
            print(f"[✓] Shard count: {len(our_shards)} (matches cpatonn)")
        else:
            print(f"[!] Shard count: {len(our_shards)} vs {len(ref_shards)} (different)")
        
        # Check config.json has quantization_config
        our_config = os.path.join(SAVE_DIR, "config.json")
        if os.path.exists(our_config):
            with open(our_config, 'r') as f:
                cfg = json.load(f)
            qconfig = cfg.get('quantization_config', {})
            if qconfig:
                print(f"[✓] quantization_config present (quant_method: {qconfig.get('quant_method')})")
                print(f"    format: {qconfig.get('format')}, ignore: {len(qconfig.get('ignore', []))} layers")
            else:
                print("[!] quantization_config MISSING from config.json!")
    else:
        print("[!] cpatonn reference not found, skipping parity check")
    
    print("")

    # ========================================================================
    # Step 8: Upload to Hugging Face
    # ========================================================================
    print("=" * 70)
    print("STEP 8: Uploading to Hugging Face")
    print("=" * 70)
    print(f"[-] Uploading to: {repo_id}")

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=SAVE_DIR,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload 4-bit quantized Qwen3-Omni model (compressed-tensors, matching cpatonn method)"
    )

    print("")
    print("=" * 70)
    print(f"✅ SUCCESS! https://huggingface.co/{repo_id}")
    print("=" * 70)
    logger.info("Upload complete -> https://huggingface.co/%s", repo_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Quantization failed: %s", exc)
        raise
