"""
Blackwell-only sitecustomize: same as sitecustomize.py plus a tokenizer
compatibility patch for vLLM when using newer transformers where
Qwen2Tokenizer no longer has all_special_tokens_extended.
Loaded only in the Blackwell Docker image (see Dockerfile.blackwell).
"""
import os

try:
    from transformers import AutoConfig as _HF_AutoConfig, AutoModel as _HF_AutoModel  # type: ignore

    _orig_ac_register = getattr(_HF_AutoConfig, "register", None)
    if callable(_orig_ac_register):
        def _safe_ac_register(model_type, config, exist_ok=False):  # noqa: ANN001
            return _orig_ac_register(model_type, config, exist_ok=True)
        _HF_AutoConfig.register = _safe_ac_register  # type: ignore[attr-defined]

    _orig_am_register = getattr(_HF_AutoModel, "register", None)
    if callable(_orig_am_register):
        def _safe_am_register(config, model, exist_ok=False):  # noqa: ANN001
            return _orig_am_register(config, model, exist_ok=True)
        _HF_AutoModel.register = _safe_am_register  # type: ignore[attr-defined]
except Exception:
    pass

# Blackwell tokenizer compatibility: vLLM expects all_special_tokens_extended
# on the tokenizer; newer transformers removed it from Qwen2Tokenizer.
try:
    from transformers import tokenization_utils_base as _tok_base

    if not hasattr(_tok_base.PreTrainedTokenizerBase, "all_special_tokens_extended"):

        def _all_special_tokens_extended(self):
            tokens = self.all_special_tokens
            return list(zip(tokens, self.convert_tokens_to_ids(tokens)))

        _tok_base.PreTrainedTokenizerBase.all_special_tokens_extended = property(
            _all_special_tokens_extended
        )
except Exception:
    pass

# Blackwell prebuilt only: prebuilt vLLM V1 MultiModalBudget can miss 'audio' for Qwen3-Omni.
# Patch in sitecustomize so it runs in engine-core subprocesses too (app.py patch only runs in main).
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

# Blackwell prebuilt: torchvision.io.read_video may return info without 'video_fps'
# (newer torchvision or when video metadata is incomplete). Patch to provide a default
# so qwen_omni_utils doesn't raise KeyError: 'video_fps'.
try:
    import torchvision.io as _tv_io
    _orig_read_video = _tv_io.read_video

    def _patched_read_video(*args, **kwargs):
        video, audio, info = _orig_read_video(*args, **kwargs)
        if "video_fps" not in info:
            info["video_fps"] = 24.0
        return video, audio, info

    _tv_io.read_video = _patched_read_video
except Exception:
    pass
