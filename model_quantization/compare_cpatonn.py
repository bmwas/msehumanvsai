#!/usr/bin/env python3
"""
Compare our quantized model against the cpatonn reference model.
Uses the local directory: ./comparison_model/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit
"""

import os
import json
import hashlib
import sys

# Paths
CPATONN_DIR = os.path.join(os.path.dirname(__file__), "..", "comparison_model", "Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit")
OUR_MODEL_DIR = os.path.join(os.path.dirname(__file__), "Qwen3-Omni-Thinking-4bit")


def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]  # Short hash for display


def compare_json(name: str, ours_path: str, ref_path: str, ignore_keys: list = None):
    """Compare two JSON files and report differences."""
    ignore_keys = ignore_keys or []
    
    print(f"\n{'='*60}")
    print(f"COMPARING: {name}")
    print(f"{'='*60}")
    
    if not os.path.exists(ours_path):
        print(f"  ❌ MISSING in our model: {name}")
        return False
    if not os.path.exists(ref_path):
        print(f"  ⚠️  Missing in reference: {name}")
        return True
    
    try:
        with open(ours_path, 'r') as f:
            ours = json.load(f)
        with open(ref_path, 'r') as f:
            ref = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON decode error: {e}")
        return False
    
    # Hash comparison
    ours_hash = calculate_file_hash(ours_path)
    ref_hash = calculate_file_hash(ref_path)
    
    if ours_hash == ref_hash:
        print(f"  ✓ IDENTICAL (hash: {ours_hash})")
        return True
    
    print(f"  ⚠️  DIFFERENT (ours: {ours_hash}, ref: {ref_hash})")
    
    # Find differences
    def compare_dicts(d1, d2, path=""):
        diffs = []
        all_keys = set(d1.keys()) | set(d2.keys())
        for key in sorted(all_keys):
            if key in ignore_keys:
                continue
            current_path = f"{path}.{key}" if path else key
            
            if key not in d1:
                diffs.append(f"  - MISSING in ours: {current_path}")
            elif key not in d2:
                diffs.append(f"  + EXTRA in ours: {current_path}")
            elif d1[key] != d2[key]:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    diffs.extend(compare_dicts(d1[key], d2[key], current_path))
                elif isinstance(d1[key], list) and isinstance(d2[key], list):
                    if len(d1[key]) != len(d2[key]):
                        diffs.append(f"  ~ {current_path}: list length {len(d1[key])} vs {len(d2[key])}")
                    # Don't expand huge lists
                else:
                    v1 = str(d1[key])[:50]
                    v2 = str(d2[key])[:50]
                    diffs.append(f"  ~ {current_path}: {v1} → {v2}")
        return diffs
    
    diffs = compare_dicts(ours, ref)
    if diffs:
        print("  Differences:")
        for d in diffs[:30]:  # Limit output
            print(f"    {d}")
        if len(diffs) > 30:
            print(f"    ... and {len(diffs) - 30} more differences")
    
    return False


def compare_safetensors():
    """Compare safetensors shards."""
    print(f"\n{'='*60}")
    print("COMPARING: Safetensors Shards")
    print(f"{'='*60}")
    
    our_shards = sorted([f for f in os.listdir(OUR_MODEL_DIR) if f.endswith('.safetensors')]) if os.path.exists(OUR_MODEL_DIR) else []
    ref_shards = sorted([f for f in os.listdir(CPATONN_DIR) if f.endswith('.safetensors')])
    
    print(f"  Reference: {len(ref_shards)} shards")
    print(f"  Ours:      {len(our_shards)} shards")
    
    if not our_shards:
        print("  ❌ No safetensors files found in our model!")
        return False
    
    # Calculate total sizes
    ref_size = sum(os.path.getsize(os.path.join(CPATONN_DIR, f)) for f in ref_shards) / (1024**3)
    our_size = sum(os.path.getsize(os.path.join(OUR_MODEL_DIR, f)) for f in our_shards) / (1024**3)
    
    print(f"  Reference size: {ref_size:.2f} GB")
    print(f"  Our size:       {our_size:.2f} GB")
    
    if abs(ref_size - our_size) > 1.0:  # More than 1GB difference
        print(f"  ⚠️  SIZE MISMATCH: {abs(ref_size - our_size):.2f} GB difference!")
    elif abs(ref_size - our_size) > 0.1:
        print(f"  ~ Minor size difference: {abs(ref_size - our_size):.2f} GB")
    else:
        print(f"  ✓ Size matches within tolerance")
    
    return len(our_shards) == len(ref_shards)


def list_files():
    """List all files in both directories."""
    print(f"\n{'='*60}")
    print("FILE LISTING")
    print(f"{'='*60}")
    
    ref_files = set(os.listdir(CPATONN_DIR)) if os.path.exists(CPATONN_DIR) else set()
    our_files = set(os.listdir(OUR_MODEL_DIR)) if os.path.exists(OUR_MODEL_DIR) else set()
    
    print("\n  Reference files (cpatonn):")
    for f in sorted(ref_files):
        if f.endswith('.safetensors'):
            continue  # Skip listing individual shards
        print(f"    - {f}")
    print(f"    + {len([f for f in ref_files if f.endswith('.safetensors')])} .safetensors shards")
    
    print("\n  Our files:")
    if not our_files:
        print("    ❌ No files found!")
    else:
        for f in sorted(our_files):
            if f.endswith('.safetensors'):
                continue
            marker = "✓" if f in ref_files else "+"
            print(f"    {marker} {f}")
        print(f"    + {len([f for f in our_files if f.endswith('.safetensors')])} .safetensors shards")
    
    # Missing files
    missing = ref_files - our_files - {f for f in ref_files if f.endswith('.safetensors')}
    if missing:
        print("\n  ❌ Missing from our model:")
        for f in sorted(missing):
            print(f"    - {f}")


def check_tokenizer_config_structure():
    """Check specific structure issues in tokenizer_config.json."""
    print(f"\n{'='*60}")
    print("TOKENIZER CONFIG STRUCTURE CHECK")
    print(f"{'='*60}")
    
    ref_path = os.path.join(CPATONN_DIR, "tokenizer_config.json")
    our_path = os.path.join(OUR_MODEL_DIR, "tokenizer_config.json")
    
    with open(ref_path, 'r') as f:
        ref = json.load(f)
    
    print("\n  Reference 'extra_special_tokens' type:", type(ref.get('extra_special_tokens')).__name__)
    if isinstance(ref.get('extra_special_tokens'), dict):
        print("    Keys:", list(ref['extra_special_tokens'].keys()))
    
    if os.path.exists(our_path):
        with open(our_path, 'r') as f:
            ours = json.load(f)
        print("\n  Ours 'extra_special_tokens' type:", type(ours.get('extra_special_tokens')).__name__)
        
        if isinstance(ours.get('extra_special_tokens'), list):
            print("  ❌ ERROR: extra_special_tokens is a LIST, should be DICT!")
            print("     This causes: AttributeError: 'list' object has no attribute 'keys'")
        elif isinstance(ours.get('extra_special_tokens'), dict):
            print("  ✓ extra_special_tokens is correctly a dict")
    else:
        print("\n  ❌ Our tokenizer_config.json not found!")


def check_quantization_config():
    """Check quantization config structure."""
    print(f"\n{'='*60}")
    print("QUANTIZATION CONFIG CHECK")
    print(f"{'='*60}")
    
    ref_path = os.path.join(CPATONN_DIR, "config.json")
    our_path = os.path.join(OUR_MODEL_DIR, "config.json")
    
    with open(ref_path, 'r') as f:
        ref = json.load(f)
    
    ref_qconfig = ref.get('quantization_config', {})
    print("\n  Reference quantization_config:")
    print(f"    quant_method: {ref_qconfig.get('quant_method')}")
    print(f"    format: {ref_qconfig.get('format')}")
    print(f"    ignore list: {len(ref_qconfig.get('ignore', []))} layers")
    
    group_0 = ref_qconfig.get('config_groups', {}).get('group_0', {})
    weights = group_0.get('weights', {})
    print(f"    weights.num_bits: {weights.get('num_bits')}")
    print(f"    weights.group_size: {weights.get('group_size')}")
    print(f"    weights.symmetric: {weights.get('symmetric')}")
    print(f"    weights.observer: {weights.get('observer')}")
    
    if os.path.exists(our_path):
        with open(our_path, 'r') as f:
            ours = json.load(f)
        our_qconfig = ours.get('quantization_config', {})
        
        if not our_qconfig:
            print("\n  ❌ Our model has NO quantization_config!")
        else:
            print("\n  Our quantization_config:")
            print(f"    quant_method: {our_qconfig.get('quant_method')}")
            print(f"    format: {our_qconfig.get('format')}")
            print(f"    ignore list: {len(our_qconfig.get('ignore', []))} layers")
            
            our_group_0 = our_qconfig.get('config_groups', {}).get('group_0', {})
            our_weights = our_group_0.get('weights', {})
            print(f"    weights.num_bits: {our_weights.get('num_bits')}")
            print(f"    weights.group_size: {our_weights.get('group_size')}")
            print(f"    weights.symmetric: {our_weights.get('symmetric')}")
            print(f"    weights.observer: {our_weights.get('observer')}")


def main():
    print("=" * 60)
    print("CPATONN PARITY CHECK")
    print("=" * 60)
    print(f"Reference: {CPATONN_DIR}")
    print(f"Our model: {OUR_MODEL_DIR}")
    
    if not os.path.exists(CPATONN_DIR):
        print(f"\n❌ Reference model not found at: {CPATONN_DIR}")
        sys.exit(1)
    
    if not os.path.exists(OUR_MODEL_DIR):
        print(f"\n⚠️  Our model not found at: {OUR_MODEL_DIR}")
        print("   Run quantization first, then run this comparison.")
        print("\n   Showing reference model structure for planning:\n")
    
    # List files
    list_files()
    
    # Check critical structures
    check_tokenizer_config_structure()
    check_quantization_config()
    
    if os.path.exists(OUR_MODEL_DIR):
        # Compare JSON configs
        compare_json("config.json", 
                     os.path.join(OUR_MODEL_DIR, "config.json"),
                     os.path.join(CPATONN_DIR, "config.json"),
                     ignore_keys=["transformers_version"])
        
        compare_json("generation_config.json",
                     os.path.join(OUR_MODEL_DIR, "generation_config.json"),
                     os.path.join(CPATONN_DIR, "generation_config.json"))
        
        compare_json("tokenizer_config.json",
                     os.path.join(OUR_MODEL_DIR, "tokenizer_config.json"),
                     os.path.join(CPATONN_DIR, "tokenizer_config.json"))
        
        compare_json("special_tokens_map.json",
                     os.path.join(OUR_MODEL_DIR, "special_tokens_map.json"),
                     os.path.join(CPATONN_DIR, "special_tokens_map.json"))
        
        # Compare safetensors
        compare_safetensors()
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
