#!/usr/bin/env python3
"""
Qwen3-Omni Clinical Audio-Video Analysis Client - LONGITUDINAL VERSION
Temporal Segmentation with Sliding Window + Rolling Context + Cross-Visit History

This implementation extends the standard audio-video analysis client for LONGITUDINAL studies:
- Segments video with overlap
- Processes each segment with rolling context from previous segments
- Supports cross-visit history via prior_visit_summary parameter
- Injects timepoint information (T0, T1, T2, etc.) into prompts
- Outputs summary_for_next_visit_context for use in subsequent visits
- Uses audio+video modalities simultaneously
- Performs meta-analysis to generate final diagnosis

LONGITUDINAL STUDY SUPPORT:
- --timepoint: The visit timepoint (0 for T0, 1 for T1, 2 for T2, etc.)
- --prior-visit-summary: Path to .txt file containing summary from previous visit
- If timepoint == 0: prior_visit_summary is NOT required
- If timepoint > 0: prior_visit_summary IS required
- Outputs: {video_stem}_next_visit_summary.txt for use in subsequent visits

API Compatibility:
- Compatible with vLLM-Omni OpenAI-compatible server (concurrency/)
- Endpoint: POST /v1/chat/completions
- Request format: JSON with base64 data URLs for video/audio
- Response format: OpenAI chat completion JSON
- Health check: GET /health (returns 200 OK)

Author: Benson Mwangi
Date: 2025-11-09
"""

import os
import sys
import subprocess
import math
import requests
import json
import time
import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import timedelta, datetime
import argparse
import logging
from pathlib import Path
import shlex
from typing import Optional


# ============================================================================
# .ENV FILE LOADING (from ./docker/.env.blackwell then ./docker/.env)
# ============================================================================

def _load_single_env_file(env_file: Path) -> int:
    """Parse a single .env file and load its values into os.environ.

    Only sets variables NOT already present in os.environ, so files loaded
    earlier (or shell exports) always win.

    Returns the number of newly loaded variables.
    """
    loaded_count = 0
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue

                key, value = line.split('=', 1)
                key = key.strip()
                if not key:
                    continue

                if ' #' in value or '\t#' in value:
                    value = value.split(' #')[0].split('\t#')[0]
                elif value.endswith('#'):
                    value = value[:-1]

                value = value.strip().strip('"').strip("'")

                if key not in os.environ:
                    os.environ[key] = value
                    loaded_count += 1
    except Exception as e:
        print(f"Warning: Could not load .env from {env_file}: {e}", file=sys.stderr)
    return loaded_count


def _load_dotenv_from_docker() -> Optional[Path]:
    """
    Load environment variables from docker .env files at startup.

    Loads files in priority order: .env.blackwell first, then .env.
    Because only NEW variables are set (existing keys are never overwritten),
    .env.blackwell values always win over .env values.

    Priority for each variable:
    1. Existing environment variables (already set in shell) - NOT overwritten
    2. Values from ./docker/.env.blackwell (hardware-specific overrides)
    3. Values from ./docker/.env (base defaults)

    Returns:
        Path to the primary loaded .env file, or None if nothing found
    """
    repo_root = Path(__file__).resolve().parent.parent

    blackwell_paths = [
        repo_root / "docker" / ".env.blackwell",
        Path.cwd() / "docker" / ".env.blackwell",
    ]
    base_paths = [
        repo_root / "docker" / ".env",
        Path.cwd() / "docker" / ".env",
    ]

    primary_file: Optional[Path] = None
    total_loaded = 0

    for path in blackwell_paths:
        if path.exists():
            count = _load_single_env_file(path)
            total_loaded += count
            if primary_file is None:
                primary_file = path
            print(f"✓ Loaded {count} environment variables from {path} (blackwell overrides)")
            break

    for path in base_paths:
        if path.exists():
            count = _load_single_env_file(path)
            total_loaded += count
            if primary_file is None:
                primary_file = path
            print(f"✓ Loaded {count} environment variables from {path} (base defaults)")
            break

    if primary_file is None:
        print("Warning: No .env file found in docker/ directory.", file=sys.stderr)
        print("Using default values for configuration.", file=sys.stderr)

    return primary_file


# Load .env files from docker/ at module import time (blackwell first, then base)
# This ensures all os.getenv() calls throughout the module will see the values
_LOADED_ENV_FILE = _load_dotenv_from_docker()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def survey_question_to_folder_name(survey_question: str) -> str:
    """
    Convert a survey question / MSE sign to a folder-safe name.
    
    Examples:
        "Flat affect" -> "flat_affect"
        "Grooming and hygiene (abnormal)" -> "grooming_and_hygiene_abnormal"
        "Psychomotor retardation / bradykinesia" -> "psychomotor_retardation_bradykinesia"
        "Eye contact (abnormal)" -> "eye_contact_abnormal"
    
    Args:
        survey_question: The MSE sign/symptom string
        
    Returns:
        A lowercase, underscore-separated folder name
    """
    if not survey_question:
        return "unknown_survey"
    
    # Convert to lowercase
    name = survey_question.lower()
    
    # Replace common separators with spaces first
    name = name.replace('/', ' ')
    name = name.replace('-', ' ')
    
    # Remove parentheses and their contents become part of the name
    # e.g., "(abnormal)" -> " abnormal"
    name = name.replace('(', ' ')
    name = name.replace(')', ' ')
    
    # Remove other special characters
    import re
    name = re.sub(r'[^\w\s]', '', name)
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Strip and replace spaces with underscores
    name = name.strip().replace(' ', '_')
    
    # Remove any double underscores
    while '__' in name:
        name = name.replace('__', '_')
    
    return name


import base64


def file_to_data_url(file_path: str, mime_type: str) -> str:
    """Read a local file and return a base64 data URL for the vLLM-Omni API."""
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def get_video_data_url(video_path: str) -> str:
    """Convert a local video path to a base64 data URL."""
    path_lower = video_path.lower()
    if path_lower.endswith(".mp4"):
        mime = "video/mp4"
    elif path_lower.endswith(".webm"):
        mime = "video/webm"
    elif path_lower.endswith(".mov"):
        mime = "video/quicktime"
    else:
        mime = "video/mp4"
    return file_to_data_url(video_path, mime)


def get_audio_data_url(audio_path: str) -> str:
    """Convert a local audio path to a base64 data URL."""
    path_lower = audio_path.lower()
    if path_lower.endswith(".wav"):
        mime = "audio/wav"
    elif path_lower.endswith((".mp3", ".mpeg")):
        mime = "audio/mpeg"
    elif path_lower.endswith(".ogg"):
        mime = "audio/ogg"
    elif path_lower.endswith(".flac"):
        mime = "audio/flac"
    else:
        mime = "audio/wav"
    return file_to_data_url(audio_path, mime)


def vllm_omni_chat_request(
    api_base_url: str,
    model: str,
    prompt: str,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    max_tokens: int = 16384,
    temperature: Optional[float] = None,
    timeout: int = 1200,
) -> str:
    """
    Send a chat completion request to the vLLM-Omni OpenAI-compatible API.

    Builds content parts in the order required by the server:
    [audio_url, video_url, text].  Audio and video are sent as separate
    base64 data URLs (never use_audio_in_video).

    Args:
        api_base_url: e.g. "http://localhost:5100"
        model: e.g. "Qwen/Qwen3-Omni-30B-A3B-Thinking"
        prompt: Text prompt / instructions
        video_path: Local path to video file (optional)
        audio_path: Local path to audio file (optional)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (omitted if None)
        timeout: HTTP request timeout in seconds

    Returns:
        The assistant's text content from the response.

    Raises:
        RuntimeError: on HTTP errors or empty responses.
    """
    url = f"{api_base_url}/v1/chat/completions"

    content_parts: list = []
    if audio_path:
        content_parts.append({
            "type": "audio_url",
            "audio_url": {"url": get_audio_data_url(audio_path)},
        })
    if video_path:
        content_parts.append({
            "type": "video_url",
            "video_url": {"url": get_video_data_url(video_path)},
        })
    content_parts.append({"type": "text", "text": prompt})

    payload: dict = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": max_tokens,
        "seed": 1234,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(
            f"vLLM-Omni returned no choices. Response: {json.dumps(data)[:500]}"
        )

    text = (choices[0].get("message", {}).get("content") or "").strip()
    if not text:
        raise RuntimeError("vLLM-Omni returned empty content")

    return text


def extract_patient_name_from_video_path(video_path: str) -> Optional[str]:
    """
    Extract patient name from input video path.
    
    Expected path structure:
        .../martin_et_al_another/<patient>/video_<N>/video_<N>.mov
        or more generally:
        .../<study>/<patient>/video_<N>/<filename>
    
    The patient name is the directory immediately before 'video_<N>'.
    
    Examples:
        "/analysis_results/martin_et_al_another/ben/video_0/video_0.mov" -> "ben"
        "./data/study/alice/video_2/video_2.mov" -> "alice"
        "/path/to/john_doe/video_1/recording.mov" -> "john_doe"
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Patient name string, or None if pattern not found
    """
    import re
    
    # Normalize the path
    path = os.path.normpath(video_path)
    parts = path.split(os.sep)
    
    # Find the 'video_<N>' directory in the path
    video_dir_pattern = re.compile(r'^video_\d+$')
    
    for i, part in enumerate(parts):
        if video_dir_pattern.match(part):
            # Patient name is the directory before 'video_<N>'
            if i > 0:
                return parts[i - 1]
    
    return None


def extract_patient_name_from_base_output_dir(base_output_dir: str) -> Optional[str]:
    """
    Extract patient name from base output directory.
    
    Expected path structure:
        .../martin_et_al_another/<patient>
        or more generally:
        .../<study>/<patient>
    
    The patient name is the last component of the base output directory.
    
    Examples:
        "./analysis_results/martin_et_al_another/ben" -> "ben"
        "/data/study/alice" -> "alice"
        "./results/john_doe" -> "john_doe"
    
    Args:
        base_output_dir: Base output directory path
        
    Returns:
        Patient name string, or None if path is empty
    """
    if not base_output_dir:
        return None
    
    # Normalize and get the last component (basename)
    path = os.path.normpath(base_output_dir)
    patient_name = os.path.basename(path)
    
    # Ensure we got something meaningful
    if patient_name and patient_name not in ('.', '..', ''):
        return patient_name
    
    return None


def validate_patient_name_consistency(video_path: str, base_output_dir: str) -> None:
    """
    Validate that the patient name in the input video path matches
    the patient name in the base output directory.
    
    This prevents accidental mismatches like:
        - Input: .../ben/video_0/video_0.mov
        - Base output: .../robbin  (WRONG! Should be .../ben)
    
    Args:
        video_path: Path to the input video file
        base_output_dir: Base output directory path
        
    Raises:
        ValueError: If patient names don't match or can't be extracted
    """
    video_patient = extract_patient_name_from_video_path(video_path)
    base_patient = extract_patient_name_from_base_output_dir(base_output_dir)
    
    # If we couldn't extract from video path, warn but continue
    if video_patient is None:
        logging.getLogger("segment_analysis").warning(
            "Could not extract patient name from video path: %s. "
            "Expected format: .../patient/video_N/filename.mov",
            video_path
        )
        return
    
    # If we couldn't extract from base output dir, that's an error
    if base_patient is None:
        raise ValueError(
            f"Could not extract patient name from --base-output-dir: {base_output_dir}\n"
            f"Expected format: .../study/patient (e.g., ./analysis_results/martin_et_al_another/ben)"
        )
    
    # Compare patient names (case-sensitive)
    if video_patient != base_patient:
        raise ValueError(
            f"\n{'='*70}\n"
            f"❌ PATIENT NAME MISMATCH DETECTED!\n"
            f"{'='*70}\n\n"
            f"The patient name in the input video path does NOT match\n"
            f"the patient name in --base-output-dir.\n\n"
            f"  Input video path:    {video_path}\n"
            f"  Patient from video:  '{video_patient}'\n\n"
            f"  Base output dir:     {base_output_dir}\n"
            f"  Patient from base:   '{base_patient}'\n\n"
            f"These MUST match to ensure correct data organization.\n\n"
            f"SOLUTION: Update --base-output-dir to use the correct patient name:\n"
            f"  --base-output-dir .../martin_et_al_another/{video_patient}\n\n"
            f"Or verify you're using the correct input video file.\n"
            f"{'='*70}"
        )
    
    # If we get here, patient names match
    logging.getLogger("segment_analysis").info(
        "Patient name validation passed: '%s' matches in both input video and base output dir",
        video_patient
    )
    print(f"\n✓ Patient name validation passed: '{video_patient}'")


def derive_prior_visit_summary_path(
    base_output_dir: str,
    current_timepoint: int,
    survey_question: str
) -> Optional[str]:
    """
    Automatically derive the prior visit summary path based on folder structure.
    
    Expected folder structure:
        <base_output_dir>/video_0/<survey_folder>/video_0_next_visit_summary.txt
        <base_output_dir>/video_1/<survey_folder>/video_1_next_visit_summary.txt
        ...
    
    Args:
        base_output_dir: The base directory containing video_N folders (e.g., /path/to/ben)
        current_timepoint: Current timepoint (1, 2, 3, ...)
        survey_question: The MSE sign/symptom being assessed
        
    Returns:
        Path to the prior visit summary file, or None if not found
    """
    if current_timepoint <= 0:
        return None
    
    previous_timepoint = current_timepoint - 1
    survey_folder = survey_question_to_folder_name(survey_question)
    
    # Construct the expected prior visit summary path
    # Pattern: <base>/video_<N-1>/<survey_folder>/video_<N-1>_next_visit_summary.txt
    prior_video_folder = f"video_{previous_timepoint}"
    prior_summary_filename = f"video_{previous_timepoint}_next_visit_summary.txt"
    
    prior_summary_path = os.path.join(
        base_output_dir,
        prior_video_folder,
        survey_folder,
        prior_summary_filename
    )
    
    return prior_summary_path


# ============================================================================
# AGENT REASONING LOGGER
# ============================================================================

class AgentReasoningLogger:
    """
    Logs ALL model reasoning steps (including <think> tags) to agent_reasoning.log.
    
    This creates a comprehensive record of the model's full chain-of-thought
    across all phases of the analysis pipeline:
    - Segment Analysis (per-segment raw responses)
    - Rolling Context Summary (if generated)
    - Meta-Analysis
    - Decision Trace Synthesis
    - Final Paragraph Synthesis
    
    The log preserves the complete reasoning including any <think>...</think>
    blocks that are stripped from the final outputs.
    """
    
    def __init__(self, output_dir: str, video_path: str):
        """
        Initialize the reasoning logger.
        
        Args:
            output_dir: Directory where agent_reasoning.log will be saved
            video_path: Path to input video (for metadata)
        """
        self.output_dir = output_dir
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.entries: List[Dict[str, str]] = []
        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    
    def log_segment_analysis(
        self, 
        segment_index: int, 
        total_segments: int,
        timeframe: str,
        prompt: str,
        raw_response: str,
        cleaned_response: str
    ):
        """Log a segment analysis step with full raw response."""
        self.entries.append({
            'section': 'SEGMENT ANALYSIS',
            'subsection': f'Segment {segment_index}/{total_segments} ({timeframe})',
            'prompt': prompt,
            'raw_response': raw_response,
            'cleaned_response': cleaned_response,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def log_rolling_summary(
        self,
        segments_range: str,
        prompt: str,
        raw_response: str,
        cleaned_response: str
    ):
        """Log rolling context summary generation."""
        self.entries.append({
            'section': 'ROLLING CONTEXT SUMMARY',
            'subsection': f'Summary for segments {segments_range}',
            'prompt': prompt,
            'raw_response': raw_response,
            'cleaned_response': cleaned_response,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def log_meta_analysis(
        self,
        prompt: str,
        raw_response: str,
        cleaned_response: str
    ):
        """Log meta-analysis synthesis step."""
        self.entries.append({
            'section': 'META-ANALYSIS',
            'subsection': 'Final Diagnostic Synthesis',
            'prompt': prompt,
            'raw_response': raw_response,
            'cleaned_response': cleaned_response,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def log_decision_trace(
        self,
        prompt: str,
        raw_response: str,
        cleaned_response: str
    ):
        """Log decision trace synthesis step."""
        self.entries.append({
            'section': 'DECISION TRACE SYNTHESIS',
            'subsection': 'Decision Trace Summary Generation',
            'prompt': prompt,
            'raw_response': raw_response,
            'cleaned_response': cleaned_response,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def log_final_paragraph(
        self,
        prompt: str,
        raw_response: str,
        cleaned_response: str
    ):
        """Log final paragraph synthesis step."""
        self.entries.append({
            'section': 'FINAL PARAGRAPH SYNTHESIS',
            'subsection': 'Final MSE Severity Rating Paragraph',
            'prompt': prompt,
            'raw_response': raw_response,
            'cleaned_response': cleaned_response,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def save(self) -> str:
        """
        Save all reasoning entries to agent_reasoning.log.
        
        Returns:
            Path to the saved log file
        """
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, "agent_reasoning.log")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("AGENT REASONING LOG - FULL MODEL CHAIN-OF-THOUGHT\n")
            f.write("Qwen3-Omni Clinical Audio-Video Analysis\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write(f"Input File: {self.video_path}\n")
            f.write(f"Video Name: {self.video_name}\n")
            f.write(f"Analysis Started: {self.start_time}\n")
            f.write(f"Log Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Reasoning Steps: {len(self.entries)}\n")
            f.write("\n")
            
            # Table of Contents
            f.write("-" * 80 + "\n")
            f.write("TABLE OF CONTENTS\n")
            f.write("-" * 80 + "\n")
            current_section = None
            for i, entry in enumerate(self.entries, 1):
                if entry['section'] != current_section:
                    current_section = entry['section']
                    f.write(f"\n[{current_section}]\n")
                f.write(f"  {i}. {entry['subsection']} ({entry['timestamp']})\n")
            f.write("\n")
            
            # Detailed entries
            for i, entry in enumerate(self.entries, 1):
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"STEP {i}: {entry['section']} - {entry['subsection']}\n")
                f.write(f"Timestamp: {entry['timestamp']}\n")
                f.write("=" * 80 + "\n\n")
                
                # Prompt (truncated for readability, full prompt available)
                f.write("-" * 40 + "\n")
                f.write("PROMPT SENT TO MODEL:\n")
                f.write("-" * 40 + "\n")
                f.write(entry['prompt'])
                f.write("\n\n")
                
                # Raw response (FULL - includes <think> tags)
                f.write("-" * 40 + "\n")
                f.write("RAW MODEL RESPONSE (FULL - includes <think> reasoning):\n")
                f.write("-" * 40 + "\n")
                f.write(entry['raw_response'])
                f.write("\n\n")
                
                # Cleaned response (for comparison)
                f.write("-" * 40 + "\n")
                f.write("CLEANED RESPONSE (thinking stripped):\n")
                f.write("-" * 40 + "\n")
                f.write(entry['cleaned_response'])
                f.write("\n\n")
                
                # Stats
                raw_len = len(entry['raw_response'])
                cleaned_len = len(entry['cleaned_response'])
                thinking_len = raw_len - cleaned_len
                f.write(f"[Stats: Raw={raw_len} chars, Cleaned={cleaned_len} chars, ")
                f.write(f"Thinking/Stripped={thinking_len} chars ({100*thinking_len/max(1,raw_len):.1f}%)]\n")
            
            # Footer
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF AGENT REASONING LOG\n")
            f.write("=" * 80 + "\n")
        
        return log_path


def strip_thinking_from_response(response_text: str) -> str:
    """
    Extract ONLY the structured JSON output from model response, stripping ALL
    chain-of-thought, preamble, and thinking process text.
    
    The TUL Study prompts request JSON output in this format:
    {
      "pass_1_visual_semiology": "...",
      "pass_2_acoustic_semiology": "...",
      "clinical_reasoning": "...",
      "rating": "..."
    }
    
    But models often output:
    "Got it, let's tackle this step by step... <thinking> ... </thinking> { JSON }"
    
    This function extracts ONLY the JSON to keep rolling context clean and compact.
    
    Args:
        response_text: Raw model response that may contain thinking/preamble
        
    Returns:
        Cleaned response with only the JSON output (or cleaned text if no JSON found)
    """
    import json as json_module
    
    if not response_text:
        return ""
    
    text = response_text
    
    # === STEP 1: Remove explicit thinking tags (AGGRESSIVE) ===
    # Pattern 1: Standard <think>...</think> blocks (with optional whitespace)
    text = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Pattern 2: Alternative format <|think|>...</|think|>
    text = re.sub(r'<\|think\|>.*?<\|/think\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Pattern 3: Catch any <think ...> with attributes
    text = re.sub(r'<think[^>]*>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Pattern 4: Any remaining stray opening/closing think tags (malformed or stray)
    text = re.sub(r'<\s*/?\s*think\s*/?>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|/?think\|>', '', text, flags=re.IGNORECASE)
    
    # === STEP 2: Try to extract JSON object ===
    # Find the first { and last } to extract potential JSON
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        potential_json = text[first_brace:last_brace + 1]
        
        # Try to parse it
        try:
            parsed = json_module.loads(potential_json)
            if isinstance(parsed, dict):
                # Check for placeholder text (model copying template)
                placeholder_indicators = [
                    "Current video signs.",
                    "Current audio/content signs.",
                    "Description of video signs.",
                    "Description of audio/content.",
                    "Stability/Change analysis.",
                    "Clinical description of",
                    "ONE_OF:",
                ]
                
                # Check if any value is placeholder text
                has_placeholder = False
                for key, value in parsed.items():
                    if isinstance(value, str):
                        for indicator in placeholder_indicators:
                            if indicator in value:
                                has_placeholder = True
                                logging.getLogger("segment_analysis").warning(
                                    "Detected placeholder text in JSON field '%s': '%s'",
                                    key, value[:100]
                                )
                                break
                
                if has_placeholder:
                    # Log warning but still return the JSON (let downstream handle it)
                    logging.getLogger("segment_analysis").warning(
                        "Model output contains placeholder text - may need prompt adjustment"
                    )
                
                # Return cleaned, formatted JSON
                return json_module.dumps(parsed, indent=2, ensure_ascii=False)
        except (json_module.JSONDecodeError, ValueError) as e:
            # Not valid JSON, log and continue to fallback
            logging.getLogger("segment_analysis").debug(
                "Failed to parse potential JSON: %s", str(e)[:100]
            )
    
    # === STEP 3: No valid JSON found, strip common preamble patterns ===
    # Remove common chain-of-thought preambles
    preamble_patterns = [
        r"^(?:Got it|Okay|Alright|Sure|Let me|I'll|Let's|Now|First|Then)[^\n]*\n",
        r"^(?:I will|I need to|I should|I can|I want to)[^\n]*\n",
        r"^(?:Looking at|Analyzing|Processing|Examining|Reviewing)[^\n]*\n",
        r"^(?:Step \d+|Pass \d+)[^\n]*\n",
        r"^(?:Here's|Here is|Below is)[^\n]*\n",
        r"^Thinking:?\s*\n",
        r"^(?:Based on|According to|Given the)[^\n]*\n",
        r"^(?:Then congruence|Then I|Then we|Then the)[^\n]*\n",  # Added for "Then congruence analysis" case
    ]
    
    # Apply preamble removal iteratively (preambles can stack)
    max_iterations = 10
    for _ in range(max_iterations):
        original = text
        for pattern in preamble_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = text.lstrip()
        if text == original:
            break  # No more changes
    
    # === STEP 4: Final cleanup ===
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    # === STEP 5: FINAL SAFETY - Remove ANY remaining think-like content ===
    # This is a catch-all for any edge cases missed above
    # Remove anything that looks like <think> or </think> with any spacing
    text = re.sub(r'<[^>]*think[^>]*>', '', text, flags=re.IGNORECASE)
    text = text.strip()
    
    # Log warning if no JSON was found (indicates model didn't follow format)
    if '{' not in text:
        logging.getLogger("segment_analysis").warning(
            "No JSON found in model response - model may not have followed output format. "
            "Response preview: %s", text[:200] if text else "(empty)"
        )
    
    return text


def retry_api_call(func: Callable, max_retries: int = 20, initial_delay: float = 1.0, 
                   backoff_factor: float = 1.5, max_delay: float = 60.0) -> any:
    """
    Retry an API call with exponential backoff
    
    Args:
        func: Function to call (should raise exception on failure)
        max_retries: Maximum number of retry attempts (default: 20)
        initial_delay: Initial delay in seconds (default: 1.0)
        backoff_factor: Multiplier for delay after each retry (default: 1.5)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        
    Returns:
        Result from successful function call
        
    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            conn_msg = str(e)
            if attempt < max_retries - 1:
                print(f"  ⚠ Connection error (attempt {attempt + 1}/{max_retries})")
                print(f"     Details: {conn_msg}")
                print(f"     Retrying in {delay:.1f}s... (backoff will increase to max {max_delay}s)")
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                print(f"  ✗ Connection failed after {max_retries} attempts")
                print(f"     Final error: {conn_msg}")
                print(f"     Suggestion: Verify API server is running and accessible")
        except requests.exceptions.Timeout as e:
            last_exception = e
            timeout_msg = str(e)
            # Extract timeout details if available
            if "timeout after" in timeout_msg.lower():
                # Message already contains detailed info
                timeout_detail = timeout_msg
                # Extract recommendation if present
                if "RECOMMENDATION:" in timeout_msg:
                    recommendation = timeout_msg.split("RECOMMENDATION:")[1].strip()
                else:
                    recommendation = None
            else:
                timeout_detail = f"Timeout occurred (timeout value may be too short)"
                recommendation = None
            
            if attempt < max_retries - 1:
                print(f"  ⚠ Request timeout (attempt {attempt + 1}/{max_retries})")
                print(f"     Details: {timeout_detail}")
                if recommendation:
                    print(f"     {recommendation}")
                print(f"     Retrying in {delay:.1f}s... (backoff will increase to max {max_delay}s)")
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                print(f"  ✗ Request timed out after {max_retries} attempts")
                print(f"     Final error: {timeout_detail}")
                if recommendation:
                    print(f"     {recommendation}")
                else:
                    print(f"     Suggestion: Check if API server is responsive, increase timeout with --timeout, or reduce segment size")
        except requests.exceptions.HTTPError as e:
            last_exception = e
            # HTTPError includes status code - check if it's retryable
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                # Retry on server errors (5xx) and some specific client errors
                if status_code >= 500 or status_code == 429:  # 429 = Too Many Requests
                    if attempt < max_retries - 1:
                        print(f"  ⚠ HTTP {status_code} error (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        print(f"  ✗ HTTP {status_code} error persisted after {max_retries} attempts")
                else:
                    # Don't retry client errors (4xx except 429)
                    print(f"  ✗ HTTP {status_code} error - not retrying (client error)")
                    raise
            else:
                raise
        except requests.exceptions.RequestException as e:
            last_exception = e
            # For other request exceptions, check if it's a server error (5xx)
            if hasattr(e, 'response') and e.response is not None and e.response.status_code >= 500:
                if attempt < max_retries - 1:
                    print(f"  ⚠ Server error {e.response.status_code} (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                else:
                    print(f"  ✗ Server error persisted after {max_retries} attempts")
            else:
                # For client errors (4xx) or other issues, don't retry
                print(f"  ✗ Request error - not retrying: {str(e)}")
                raise
        except Exception as e:
            # For non-request exceptions, don't retry
            print(f"  ✗ Unexpected error - not retrying: {type(e).__name__}: {str(e)}")
            raise
    
    # If we get here, all retries failed
    if last_exception:
        raise last_exception


# ============================================================================
# ENV LOADING (SEGMENT_* from .env or environment)
# ============================================================================

def _load_segment_env_defaults() -> Tuple[int, int]:
    """
    Load SEGMENT_DURATION and SEGMENT_OVERLAP from environment or .env files.
    Priority:
      1) Existing environment variables
      2) .env files (first match): ./docker/.env.blackwell, ./docker/.env, project .env, CWD .env, CWD/docker/.env
      3) Hardcoded defaults (ONLY if not found anywhere)
    
    Handles inline comments (e.g., VAR=30 # comment).
    """
    logger = logging.getLogger("env_loader")
    
    def _strip_inline_comment(value: str) -> str:
        """Remove inline comments from value"""
        if '#' in value:
            value = value.split('#')[0]
        return value.strip().strip('"').strip("'")
    
    def _parse_env_file(file_path: Path) -> Dict[str, Tuple[str, Path]]:
        """Returns dict of {key: (value, source_path)}"""
        values: Dict[str, Tuple[str, Path]] = {}
        try:
            if not file_path.exists():
                return values
            for line in file_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = _strip_inline_comment(v)
                if k in ("SEGMENT_DURATION", "SEGMENT_OVERLAP") and v:
                    values[k] = (v, file_path)
        except Exception:
            pass
        return values

    duration: Optional[int] = None
    overlap: Optional[int] = None
    duration_source = "default"
    overlap_source = "default"

    # 1) Environment variables (exported in shell)
    duration_env = os.getenv("SEGMENT_DURATION")
    overlap_env = os.getenv("SEGMENT_OVERLAP")
    if duration_env:
        duration_env = _strip_inline_comment(duration_env)
        try:
            duration = int(duration_env)
            duration_source = "environment variable"
        except ValueError:
            pass
    if overlap_env:
        overlap_env = _strip_inline_comment(overlap_env)
        try:
            overlap = int(overlap_env)
            overlap_source = "environment variable"
        except ValueError:
            pass

    # 2) .env file candidates (fill in any still-missing values)
    if duration is None or overlap is None:
        repo_root = Path(__file__).resolve().parent.parent
        candidates = [
            repo_root / "docker" / ".env.blackwell",
            repo_root / "docker" / ".env",
            repo_root / ".env",
            Path.cwd() / ".env",
            Path.cwd() / "docker" / ".env",
        ]
        for p in candidates:
            values = _parse_env_file(p)
            if duration is None and "SEGMENT_DURATION" in values:
                try:
                    duration = int(values["SEGMENT_DURATION"][0])
                    duration_source = str(values["SEGMENT_DURATION"][1])
                except ValueError:
                    pass
            if overlap is None and "SEGMENT_OVERLAP" in values:
                try:
                    overlap = int(values["SEGMENT_OVERLAP"][0])
                    overlap_source = str(values["SEGMENT_OVERLAP"][1])
                except ValueError:
                    pass
            if duration is not None and overlap is not None:
                break

    # 3) Hardcoded defaults (ONLY if not found anywhere)
    if duration is None:
        duration = 10
        logger.warning(f"SEGMENT_DURATION not found in environment or .env files, using default=10")
    else:
        logger.info(f"SEGMENT_DURATION={duration} (from {duration_source})")
    
    if overlap is None:
        overlap = 3
        logger.warning(f"SEGMENT_OVERLAP not found in environment or .env files, using default=3")
    else:
        logger.info(f"SEGMENT_OVERLAP={overlap} (from {overlap_source})")

    return duration, overlap


# ============================================================================
# ENV LOADING (MAX_ROLLING_CONTEXT_CHARS from .env or environment)
# ============================================================================

def _load_env_int(var_name: str, default: int, min_val: int = 0, max_val: int = 1000000) -> int:
    """
    Load an integer from environment or .env files.
    Priority:
      1) Existing environment variable
      2) .env files (first match): ./docker/.env.blackwell, ./docker/.env, project .env, CWD .env, CWD/docker/.env
      3) Hardcoded default (ONLY if not found anywhere)
    
    Handles inline comments (e.g., VAR=123 # comment).
    """
    logger = logging.getLogger("env_loader")
    
    def _strip_inline_comment(value: str) -> str:
        """Remove inline comments from value (e.g., '123 # comment' -> '123')"""
        # Handle # comments but not inside quotes
        if '#' in value:
            # Simple approach: take everything before first #
            value = value.split('#')[0]
        return value.strip().strip('"').strip("'")
    
    def _parse_env_file(file_path: Path, key: str) -> Optional[Tuple[int, Path]]:
        try:
            if not file_path.exists():
                return None
            for line in file_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                if k == key:
                    v = _strip_inline_comment(v)
                    if v:
                        try:
                            n = int(v)
                            if min_val <= n <= max_val:
                                return (n, file_path)
                        except ValueError:
                            pass
        except Exception:
            pass
        return None

    # 1) Environment variable (exported in shell)
    env_val = os.getenv(var_name)
    if env_val:
        env_val = _strip_inline_comment(env_val)
        try:
            n = int(env_val)
            if min_val <= n <= max_val:
                logger.debug(f"{var_name}={n} (from environment variable)")
                return n
        except ValueError:
            pass

    # 2) .env file candidates (priority order: .env.blackwell > .env > others)
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "docker" / ".env.blackwell",
        repo_root / "docker" / ".env",
        repo_root / ".env",
        Path.cwd() / ".env",
        Path.cwd() / "docker" / ".env",
    ]
    
    for p in candidates:
        result = _parse_env_file(p, var_name)
        if result is not None:
            val, source = result
            logger.info(f"{var_name}={val} (from {source})")
            return val

    # 3) Hardcoded default (ONLY if not found anywhere)
    logger.warning(f"{var_name} not found in environment or .env files, using default={default}")
    return default


def _load_env_str(var_name: str, default: str) -> str:
    """
    Load a string from environment.
    
    Since _load_dotenv_from_docker() already loads ./docker/.env into os.environ
    at module import time, this function simply reads from os.environ.
    
    Priority:
      1) Environment variable (includes values loaded from ./docker/.env)
      2) Hardcoded default (ONLY if not found)
    
    Handles inline comments (e.g., VAR=value # comment).
    """
    logger = logging.getLogger("env_loader")
    
    def _strip_inline_comment(value: str) -> str:
        """Remove inline comments from value"""
        if ' #' in value or '\t#' in value:
            value = value.split(' #')[0].split('\t#')[0]
        return value.strip().strip('"').strip("'")
    
    env_val = os.getenv(var_name)
    if env_val:
        result = _strip_inline_comment(env_val)
        if result:
            logger.debug(f"{var_name}={result} (from environment)")
            return result
    
    logger.warning(f"{var_name} not found in environment, using default={default}")
    return default


def _load_env_float(var_name: str, default: float, min_val: float = 0.0, max_val: float = 1e6) -> float:
    """
    Load a float from environment or .env files.
    Priority:
      1) Existing environment variable
      2) .env files (first match): ./docker/.env.blackwell, ./docker/.env, project .env, CWD .env, CWD/docker/.env
      3) Hardcoded default (ONLY if not found anywhere)
    
    Handles inline comments (e.g., VAR=1.5 # comment).
    """
    logger = logging.getLogger("env_loader")
    
    def _strip_inline_comment(value: str) -> str:
        """Remove inline comments from value"""
        if '#' in value:
            value = value.split('#')[0]
        return value.strip().strip('"').strip("'")

    # 1) Environment variable (exported in shell)
    env_val = os.getenv(var_name)
    if env_val:
        env_val = _strip_inline_comment(env_val)
        try:
            n = float(env_val)
            if min_val <= n <= max_val:
                logger.debug(f"{var_name}={n} (from environment variable)")
                return n
        except ValueError:
            pass

    # 2) .env file candidates (priority order: .env.blackwell > .env > others)
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "docker" / ".env.blackwell",
        repo_root / "docker" / ".env",
        repo_root / ".env",
        Path.cwd() / ".env",
        Path.cwd() / "docker" / ".env",
    ]

    for p in candidates:
        try:
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                if k.strip() == var_name:
                    v = _strip_inline_comment(v)
                    if v:
                        try:
                            n = float(v)
                            if min_val <= n <= max_val:
                                logger.info(f"{var_name}={n} (from {p})")
                                return n
                        except ValueError:
                            pass
        except Exception:
            continue

    # 3) Hardcoded default (ONLY if not found anywhere)
    logger.warning(f"{var_name} not found in environment or .env files, using default={default}")
    return default


def _load_max_rolling_context_chars() -> int:
    """
    Load MAX_ROLLING_CONTEXT_CHARS from environment or .env files.
    Fallback: 1500 chars (~375 tokens) - reduced from 2000 for safety.
    """
    return _load_env_int("MAX_ROLLING_CONTEXT_CHARS", default=1500, min_val=500, max_val=10000)


def _load_max_prompt_chars() -> int:
    """
    Load MAX_PROMPT_CHARS from environment or .env files.
    This is the maximum total prompt size sent to the API.
    
    For MAX_MODEL_LEN=16384:
    - Reserve ~12000 tokens for video frames (20 frames * 600 tokens)
    - Reserve ~1000 tokens for generation
    - Leaves ~3384 tokens for prompt text
    - 3384 tokens * 4 chars/token = ~13500 chars
    
    Fallback: 12000 chars (~3000 tokens) for safety margin.
    """
    return _load_env_int("MAX_PROMPT_CHARS", default=12000, min_val=2000, max_val=50000)


def _load_max_segment_output_tokens() -> int:
    """
    Load MAX_SEGMENT_OUTPUT_TOKENS from environment or .env files.
    This limits how long each segment's observations can be.
    
    Shorter observations = smaller rolling context = smaller prompts.
    Fallback: 512 tokens (was unlimited, causing bloated prompts).
    """
    return _load_env_int("MAX_SEGMENT_OUTPUT_TOKENS", default=512, min_val=128, max_val=16384)


# ============================================================================
# ENV LOADING (META_MAX_TOKENS from .env or environment)
# ============================================================================

def _load_meta_max_tokens_default() -> int:
    """
    Load META_MAX_TOKENS from environment or .env files.
    Priority:
      1) Existing environment variable META_MAX_TOKENS
      2) .env files (first match): ./docker/.env.blackwell, ./docker/.env, project .env, CWD .env, CWD/docker/.env
      3) Hardcoded default (ONLY if not found anywhere)
    
    Handles inline comments (e.g., VAR=8192 # comment).
    """
    logger = logging.getLogger("env_loader")
    
    def _strip_inline_comment(value: str) -> str:
        """Remove inline comments from value"""
        if '#' in value:
            value = value.split('#')[0]
        return value.strip().strip('"').strip("'")
    
    # 1) Environment variable (exported in shell)
    env_val = os.getenv("META_MAX_TOKENS")
    if env_val:
        env_val = _strip_inline_comment(env_val)
        try:
            n = int(env_val)
            if 1 <= n <= 32768:
                logger.debug(f"META_MAX_TOKENS={n} (from environment variable)")
                return n
        except ValueError:
            pass

    # 2) .env candidates
    def _parse_env(file_path: Path) -> Optional[Tuple[int, Path]]:
        try:
            if not file_path.exists():
                return None
            for line in file_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                if k.strip() == "META_MAX_TOKENS":
                    v = _strip_inline_comment(v)
                    try:
                        n = int(v)
                        if 1 <= n <= 32768:
                            return (n, file_path)
                    except ValueError:
                        return None
        except Exception:
            return None
        return None

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "docker" / ".env.blackwell",
        repo_root / "docker" / ".env",
        repo_root / ".env",
        Path.cwd() / ".env",
        Path.cwd() / "docker" / ".env",
    ]
    for p in candidates:
        result = _parse_env(p)
        if result is not None:
            val, source = result
            logger.info(f"META_MAX_TOKENS={val} (from {source})")
            return val

    # 3) Hardcoded default (ONLY if not found anywhere)
    logger.warning(f"META_MAX_TOKENS not found in environment or .env files, using default=1024")
    return 1024


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for audio-video analysis with longitudinal study support"""
    # API Configuration (vLLM-Omni OpenAI-compatible endpoint)
    api_url: str = "http://localhost:5100"
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    request_timeout: int = 1200  # Request timeout in seconds (default: 20 minutes per segment)
    
    # Segmentation Parameters
    # CRITICAL: Set to 10 seconds to avoid vLLM audio chunking issues with use_audio_in_video
    # 30+ second segments cause process_mm_info to split audio into multiple chunks,
    # leading to "390 + 390 + 390 = 1170 tokens to 390 placeholders" error
    segment_duration: int = 10  # seconds per segment (reduced from 30 to avoid audio chunking)
    segment_overlap: int = 3   # seconds of overlap between segments (reduced proportionally)
    
    # Audio + Transcription Mode
    use_transcription: bool = False  # If True, uses /audiotextdescription endpoint
    transcription_file: Optional[str] = None  # Path to full transcription file
    
    # Rolling Context Configuration
    # CRITICAL: Using sliding window (depth=2) instead of all segments (-1) to prevent:
    #   1. Token overflow and truncated/broken JSON in rolling context
    #   2. Model copying from prior context instead of observing current video
    #   3. Redundant information (each segment already contains prior observations)
    rolling_context_depth: int = 2  # Number of previous segments (2 = sliding window, -1 = all)
    rolling_context_summary_trigger: int = 3  # When >= this many prior segments, synthesize summary
    rolling_context_summary_recent: int = 2   # Include this many recent segments verbatim after summary
    rolling_context_summary_max_words: int = 250  # Target max words for rolling summary
    
    # Clinical Focus Areas
    clinical_focus: List[str] = None
    
    # Prompt Configuration
    prompt_file: Optional[str] = None  # REQUIRED: Path to YAML prompt file
    # survey_question: The specific MSE (Mental Status Examination) sign or symptom being assessed.
    # This is NOT a traditional survey question but rather the MSE item under evaluation,
    # e.g., "Grooming and hygiene (abnormal)", "Eye contact (abnormal)", "Flat affect", etc.
    # Variable name kept as 'survey_question' for template compatibility.
    survey_question: Optional[str] = None  # REQUIRED: MSE sign/symptom presence question
    temperature: Optional[float] = 0.1  # Decoding temperature (0.1 = near-deterministic, good for clinical assessment)
    
    # LONGITUDINAL STUDY CONFIGURATION
    timepoint: int = 0  # Visit timepoint (0 for T0, 1 for T1, 2 for T2, etc.)
    prior_visit_summary_file: Optional[str] = None  # Path to .txt file with prior visit summary
    prior_visit_summary: Optional[str] = None  # Loaded content from prior_visit_summary_file
    
    # Output Configuration
    output_dir: str = "./analysis_results"
    save_segments: bool = True  # Save individual segment files
    verbose: bool = True
    
    def __post_init__(self):
        # Initialize internal prompt templates storage
        self._prompt_templates: Optional[Dict] = None
        
        # REQUIRE prompt file - make script prompt-agnostic
        if not self.prompt_file:
            raise ValueError(
                "Prompt file is REQUIRED for analysis. "
                "Please specify --prompt <path_to_yaml> when running the script. "
                "Example prompt files: prompts/martin_et_al_another.yml, prompts/tul_study.yml"
            )
        
        # Load prompts from YAML
        self._load_prompts_from_yaml()
        
        # Validate required templates are present
        self._validate_required_templates()
        
        # LONGITUDINAL: Validate timepoint and prior_visit_summary requirements
        self._validate_longitudinal_params()
        
        # Set default clinical focus if not loaded from YAML
        if self.clinical_focus is None:
            self.clinical_focus = [
                "Speech patterns (fluency, prosody, volume, clarity, rate)",
                "Cognitive function (alertness, responsiveness, memory, comprehension)",
                "Affect and emotional state (tone, mood indicators)",
                "Language characteristics (word choice, pauses, repetitions)",
                "Communication effectiveness (coherence, organization)"
            ]
    
    def _validate_longitudinal_params(self):
        """Validate longitudinal study parameters and load prior visit summary"""
        # Validate timepoint is non-negative
        if self.timepoint < 0:
            raise ValueError(
                f"Timepoint must be a non-negative integer (0 for T0, 1 for T1, etc.). "
                f"Got: {self.timepoint}"
            )
        
        # If timepoint > 0, prior_visit_summary_file is REQUIRED
        if self.timepoint > 0:
            if not self.prior_visit_summary_file:
                raise ValueError(
                    f"For timepoint > 0 (T{self.timepoint}), --prior-visit-summary is REQUIRED. "
                    f"This should be the path to the _next_visit_summary.txt file from the previous visit (T{self.timepoint - 1})."
                )
            
            # Load prior visit summary from file
            if not os.path.exists(self.prior_visit_summary_file):
                raise FileNotFoundError(
                    f"Prior visit summary file not found: {self.prior_visit_summary_file}"
                )
            
            try:
                with open(self.prior_visit_summary_file, 'r', encoding='utf-8') as f:
                    self.prior_visit_summary = f.read().strip()
                
                if not self.prior_visit_summary:
                    raise ValueError(
                        f"Prior visit summary file is empty: {self.prior_visit_summary_file}"
                    )
                    
                logging.getLogger("segment_analysis").info(
                    "Loaded prior visit summary from %s (%d characters)",
                    self.prior_visit_summary_file, len(self.prior_visit_summary)
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load prior visit summary from {self.prior_visit_summary_file}: {e}"
                )
        else:
            # Timepoint == 0: No prior visit summary needed
            self.prior_visit_summary = "No prior visit data."
            logging.getLogger("segment_analysis").info(
                "Timepoint 0 (T0): No prior visit summary required. Using baseline placeholder."
            )
    
    def _load_prompts_from_yaml(self):
        """Load prompt templates from YAML file"""
        if not os.path.exists(self.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            self._prompt_templates = yaml.safe_load(f)
        
        # Override clinical_focus if specified in YAML
        if self._prompt_templates and 'clinical_focus' in self._prompt_templates:
            self.clinical_focus = self._prompt_templates['clinical_focus']
    
    def _validate_required_templates(self):
        """Validate that all required prompt templates are present in YAML
        
        For longitudinal prompts (like martin_et_al_another), supports:
        - 'segment_prompts.standard_segment' (unified template for all segments)
        
        For legacy prompts, requires:
        - 'segment_prompts.first_segment' AND 'segment_prompts.subsequent_segments'
        """
        # Check for segment prompt templates - either standard_segment OR first/subsequent
        has_standard = self.get_prompt_template('segment_prompts.standard_segment') is not None
        has_first = self.get_prompt_template('segment_prompts.first_segment') is not None
        has_subsequent = self.get_prompt_template('segment_prompts.subsequent_segments') is not None
        
        if not has_standard and not (has_first and has_subsequent):
            raise ValueError(
                f"Prompt file '{self.prompt_file}' is missing segment prompt templates.\n"
                f"Required: Either 'segment_prompts.standard_segment' (for longitudinal)\n"
                f"         OR both 'segment_prompts.first_segment' AND 'segment_prompts.subsequent_segments'\n"
                f"See prompts/martin_et_al_another/prompt.yml for longitudinal template example."
            )
        
        # meta_analysis_prompt is required for all prompt types
        if self.get_prompt_template('meta_analysis_prompt') is None:
            raise ValueError(
                f"Prompt file '{self.prompt_file}' is missing required template: 'meta_analysis_prompt'\n"
                f"Please ensure your YAML file contains the meta_analysis_prompt template."
            )
        
        # decision_trace_prompt is optional for longitudinal prompts (they may use a different output format)
        # Only warn if missing, don't fail
        if self.get_prompt_template('decision_trace_prompt') is None:
            logging.getLogger("segment_analysis").warning(
                "Prompt file '%s' is missing 'decision_trace_prompt' template. "
                "Decision trace synthesis will be skipped.",
                self.prompt_file
            )
    
    def get_prompt_template(self, key: str) -> Optional[str]:
        """Get a prompt template by key"""
        if self._prompt_templates:
            # Handle nested keys like 'segment_prompts.first_segment'
            keys = key.split('.')
            value = self._prompt_templates
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return None
            return value
        return None
    
    def get_modality_description(self) -> str:
        """Get modality description based on transcription mode"""
        if self._prompt_templates and 'modality_descriptions' in self._prompt_templates:
            if self.use_transcription:
                return self._prompt_templates['modality_descriptions'].get('audio_with_transcription', '')
            else:
                return self._prompt_templates['modality_descriptions'].get('audio_only', '')
        
        # Fallback to defaults
        if self.use_transcription:
            return "Use BOTH audio (prosody, tone, pauses) and text transcription (word content, structure) as evidence."
        else:
            return "Use audio evidence (prosody, tone, pauses, speech characteristics)."


@dataclass
class Segment:
    """Represents a video segment (with audio)"""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds
    file_path: str  # Path to video file (.mp4/.mov)
    audio_file_path: Optional[str] = None  # Path to separate audio file (.wav)
    transcription: Optional[str] = None  # Transcription for this segment
    observations: Optional[str] = None
    prompt_brief: Optional[str] = None
    rolling_context: Optional[str] = None  # Rolling context used for this segment
    
    @property
    def timeframe_str(self) -> str:
        """Human-readable timeframe string"""
        start = str(timedelta(seconds=int(self.start_time)))
        end = str(timedelta(seconds=int(self.end_time)))
        return f"{start} - {end}"


# ============================================================================
# VIDEO SEGMENTATION
# ============================================================================

class AudioSegmenter:
    """Handles video segmentation with overlap"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        # Target preprocessing to align with server-side expectations
        self.target_fps = _load_env_float("SEGMENT_TARGET_FPS", default=1.0, min_val=0.0, max_val=120.0)
        self.target_width = _load_env_int("SEGMENT_TARGET_WIDTH", default=192, min_val=0, max_val=4096)
        self.target_height = _load_env_int("SEGMENT_TARGET_HEIGHT", default=192, min_val=0, max_val=4096)
        self.video_crf = _load_env_int("SEGMENT_TARGET_CRF", default=30, min_val=0, max_val=51)
        self.target_sample_rate = _load_env_int("SEGMENT_TARGET_SAMPLE_RATE", default=16000, min_val=8000, max_val=192000)
        self.target_channels = _load_env_int("SEGMENT_TARGET_CHANNELS", default=1, min_val=1, max_val=2)
        # Video encoder - libx264 is common but some systems (e.g., Fedora) don't have it
        # Alternatives: mpeg4 (universal), h264_vaapi (AMD/Intel HW), h264_nvenc (NVIDIA HW)
        self.video_encoder = _load_env_str("SEGMENT_VIDEO_ENCODER", default="libx264")
        
    def get_audio_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get video duration: {e.stderr}")
        except ValueError as e:
            raise RuntimeError(f"Invalid duration format: {e}")
    
    def load_full_transcription(self, transcription_path: str) -> str:
        """Load full transcription from file"""
        try:
            with open(transcription_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise RuntimeError(f"Failed to load transcription file: {e}")
    
    def segment_transcription(self, full_transcription: str, segments: List[Segment]) -> List[Segment]:
        """
        Distribute transcription across segments proportionally
        
        Simple heuristic: split transcription by whitespace and distribute
        proportionally based on segment duration
        """
        if not full_transcription:
            return segments
        
        # Calculate total duration
        total_duration = max(seg.end_time for seg in segments)
        
        # Split transcription into words
        words = full_transcription.split()
        total_words = len(words)
        
        if total_words == 0:
            return segments
        
        # Distribute words to segments based on their time proportion
        word_index = 0
        for segment in segments:
            # Calculate proportion of total duration
            proportion = segment.duration / total_duration
            words_for_segment = max(1, int(total_words * proportion))
            
            # Extract words for this segment
            segment_words = words[word_index:word_index + words_for_segment]
            segment.transcription = " ".join(segment_words)
            
            word_index += words_for_segment
            
            # Handle remaining words for last segment
            if segment == segments[-1] and word_index < total_words:
                remaining_words = words[word_index:]
                segment.transcription += " " + " ".join(remaining_words)
        
        return segments
    
    def create_segments(self, video_path: str, output_dir: str) -> List[Segment]:
        """
        Create overlapping video segments
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save segments
            
        Returns:
            List of Segment objects
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If video is too short or segmentation fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check video file size (sanity check)
        video_size = os.path.getsize(video_path)
        if video_size == 0:
            raise RuntimeError(f"Video file is empty (0 bytes): {video_path}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video duration
        try:
            total_duration = self.get_audio_duration(video_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to get video duration. FFprobe may have failed or video is corrupted. "
                f"Video: {video_path}, Error: {e}"
            )
        
        # Log duration for debugging
        logging.getLogger("segment_analysis").info(
            "Video duration: %.2f seconds (%.1f minutes) for %s",
            total_duration, total_duration / 60, video_path
        )
        
        if self.config.verbose:
            print(f"Video duration: {timedelta(seconds=int(total_duration))} ({total_duration:.2f}s)")
            print(f"Segment duration: {self.config.segment_duration}s")
            print(f"Segment overlap: {self.config.segment_overlap}s")
        
        # CRITICAL: Check if video is too short
        min_segment_duration = 5.0  # Minimum segment length in seconds
        if total_duration < min_segment_duration:
            raise RuntimeError(
                f"Video is too short for segmentation. "
                f"Duration: {total_duration:.2f}s, Minimum required: {min_segment_duration}s. "
                f"Video: {video_path}"
            )
        
        # Calculate segments
        stride = self.config.segment_duration - self.config.segment_overlap
        
        # Validate stride is positive
        if stride <= 0:
            raise RuntimeError(
                f"Invalid segmentation parameters: stride must be positive. "
                f"segment_duration={self.config.segment_duration}s, overlap={self.config.segment_overlap}s, "
                f"resulting stride={stride}s. "
                f"Reduce SEGMENT_OVERLAP or increase SEGMENT_DURATION."
            )
        
        if self.config.verbose:
            expected_segments = max(1, int((total_duration - self.config.segment_overlap) / stride))
            print(f"Stride: {stride}s (expected ~{expected_segments} segments)")
        
        segments = []
        
        current_start = 0.0
        segment_index = 0
        
        while current_start < total_duration:
            # Calculate segment boundaries
            segment_start = current_start
            segment_end = min(current_start + self.config.segment_duration, total_duration)
            segment_duration = segment_end - segment_start
            
            # Skip if segment is too short (< 5 seconds)
            if segment_duration < 5.0:
                logging.getLogger("segment_analysis").debug(
                    "Skipping final segment (too short): start=%.2f, end=%.2f, duration=%.2f < 5.0s",
                    segment_start, segment_end, segment_duration
                )
                break
            
            # Generate segment file paths (video and audio)
            video_ext = Path(video_path).suffix
            segment_filename = f"segment_{segment_index:03d}_{int(segment_start):04d}_{int(segment_end):04d}{video_ext}"
            segment_path = os.path.join(output_dir, segment_filename)
            
            # Generate audio file path (.wav) - same name as video but .wav extension
            audio_filename = f"segment_{segment_index:03d}_{int(segment_start):04d}_{int(segment_end):04d}.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            
            # Create segment object with both video and audio paths
            segment = Segment(
                index=segment_index,
                start_time=segment_start,
                end_time=segment_end,
                duration=segment_duration,
                file_path=segment_path,
                audio_file_path=audio_path
            )
            segments.append(segment)
            
            # Extract video segment using ffmpeg (video only, no audio)
            self._extract_segment(video_path, segment_path, segment_start, segment_duration)
            
            # Extract audio segment from source audio file (expects .wav with same name as video)
            # Derive source audio path: same directory and name as video, but .wav extension
            source_audio_path = str(Path(video_path).with_suffix('.wav'))
            if os.path.exists(source_audio_path):
                self._extract_audio_segment(source_audio_path, audio_path, segment_start, segment_duration)
            else:
                # Fallback: try extracting audio from video; if the video has no audio
                # stream (e.g. skeleton-only videos from mmpose), proceed without audio.
                logging.getLogger("segment_analysis").warning(
                    f"No separate audio file found at {source_audio_path}, extracting from video"
                )
                try:
                    self._extract_audio_from_video(video_path, audio_path, segment_start, segment_duration)
                except RuntimeError:
                    logging.getLogger("segment_analysis").warning(
                        f"Video has no audio stream (video-only file). Proceeding without audio for segment {segment_index + 1}."
                    )
                    segment.audio_file_path = None
            
            if self.config.verbose:
                print(f"  Created segment {segment_index + 1}: {segment.timeframe_str}")
                print(f"    Video: {segment_filename}")
                print(f"    Audio: {audio_filename}")
            
            # Move to next segment
            current_start += stride
            segment_index += 1
        
        # Log final count
        logging.getLogger("segment_analysis").info(
            "Segmentation complete: created %d segments from %.2fs video (stride=%ds)",
            len(segments), total_duration, stride
        )
        
        if self.config.verbose:
            print(f"\nTotal segments created: {len(segments)}")
        
        # Warn if no segments were created (shouldn't happen due to earlier check, but just in case)
        if not segments:
            logging.getLogger("segment_analysis").error(
                "No segments created! Video duration: %.2fs, segment_duration: %ds, overlap: %ds, stride: %ds. "
                "This should not happen if video >= 5 seconds.",
                total_duration, self.config.segment_duration, self.config.segment_overlap, stride
            )
        
        return segments
    
    def _extract_segment(self, input_path: str, output_path: str, start_time: float, duration: float):
        """Extract a video segment using ffmpeg"""
        try:
            # Build filter chain to enforce deterministic FPS and resolution
            filter_chain = []
            if self.target_fps > 0:
                filter_chain.append(f"fps={self.target_fps}")
            if self.target_width > 0 and self.target_height > 0:
                filter_chain.append(
                    f"scale=w='min(iw\\,{self.target_width})':h='min(ih\\,{self.target_height})':force_original_aspect_ratio=decrease"
                )
            filter_args = []
            if filter_chain:
                filter_args = ['-vf', ','.join(filter_chain)]

            max_frames_arg = []
            if self.target_fps > 0:
                # Ensure consistent frame count (at least one frame)
                max_frames = max(1, int(math.ceil(duration * self.target_fps + 1e-6)))
                max_frames_arg = ['-frames:v', str(max_frames)]

            # Build encoder-specific arguments
            # libx264 supports -preset and -crf
            # mpeg4 only supports -q:v (quality, 1-31, lower=better)
            # Hardware encoders have their own options
            encoder = self.video_encoder
            if encoder == 'libx264':
                encoder_args = [
                    '-c:v', encoder,
                    '-preset', 'ultrafast',
                    '-crf', str(self.video_crf),
                ]
            elif encoder == 'mpeg4':
                # mpeg4 uses -q:v (1-31, 2-5 is good quality)
                # Map CRF roughly: CRF 20-30 -> q:v 2-8
                q_val = max(1, min(31, 2 + (self.video_crf - 18) // 3))
                encoder_args = [
                    '-c:v', encoder,
                    '-q:v', str(q_val),
                ]
            else:
                # Generic fallback for other encoders (h264_vaapi, h264_nvenc, etc.)
                encoder_args = [
                    '-c:v', encoder,
                ]
            
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-ss', f"{max(0.0, start_time):.3f}",
                '-i', input_path,
                '-t', f"{max(duration, 0.0):.3f}",
                *filter_args,
                '-vsync', '0',
                *max_frames_arg,
                *encoder_args,
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',  # Preserve audio with AAC codec
                '-b:a', '128k',  # Audio bitrate
                output_path
            ]
            
            # Run ffmpeg with suppressed output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract segment: {e.stderr}")
    
    def _extract_audio_segment(self, input_audio_path: str, output_path: str, start_time: float, duration: float):
        """Extract an audio segment from a source audio file (.wav) using ffmpeg"""
        try:
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-ss', f"{max(0.0, start_time):.3f}",
                '-i', input_audio_path,
                '-t', f"{max(duration, 0.0):.3f}",
                '-ar', str(self.target_sample_rate),  # Sample rate
                '-ac', str(self.target_channels),  # Channels
                '-c:a', 'pcm_s16le',  # PCM 16-bit Little Endian (standard WAV codec)
                '-f', 'wav',  # Explicitly set WAV container format
                output_path
            ]
            
            # Run ffmpeg with suppressed output
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract audio segment: {e.stderr}")
    
    def _extract_audio_from_video(self, input_video_path: str, output_path: str, start_time: float, duration: float):
        """Extract audio from video file as fallback when no separate .wav exists"""
        try:
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-ss', f"{max(0.0, start_time):.3f}",
                '-i', input_video_path,
                '-t', f"{max(duration, 0.0):.3f}",
                '-vn',  # No video
                '-ar', str(self.target_sample_rate),  # Sample rate
                '-ac', str(self.target_channels),  # Channels
                '-c:a', 'pcm_s16le',  # PCM 16-bit Little Endian (standard WAV codec)
                '-f', 'wav',  # Explicitly set WAV container format
                output_path
            ]
            
            # Run ffmpeg with suppressed output
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract audio from video: {e.stderr}")


# ============================================================================
# SEGMENT ANALYSIS WITH ROLLING CONTEXT
# ============================================================================

class SegmentAnalyzer:
    """Analyzes video segments (with audio) with rolling context"""
    
    def __init__(self, config: AnalysisConfig, reasoning_logger: Optional[AgentReasoningLogger] = None):
        self.config = config
        self.reasoning_logger = reasoning_logger
        self._rolling_summary_cache: Optional[str] = None
        self._rolling_summary_last_index: int = -1
        
        # MSE guide response cache (fetched once, used for all segments)
        self._mse_guide_response: Optional[str] = None
        # MSE guide raw response (includes reasoning before stripping)
        self._mse_guide_raw_response: Optional[str] = None
    
    def fetch_mse_guide_response(self, placeholder_video_path: str) -> str:
        """
        Fetch MSE guide observational questions from the API.
        
        This calls the API with the mse_guide prompt template to generate
        specific observational questions for the MSE domain being assessed.
        The response is a list of concrete, answerable questions that guide
        the model's audio+video analysis.
        
        Uses video as a placeholder (text-only inference, similar to meta-analysis).
        
        Args:
            placeholder_video_path: Path to a video file to use as placeholder
                                   (typically the first segment)
        
        Returns:
            str: The MSE guide response containing observational questions
            
        Raises:
            RuntimeError: If the API call fails or template is missing
        """
        # Check if already fetched (cache)
        if self._mse_guide_response is not None:
            return self._mse_guide_response
        
        # Get the mse_guide template from YAML
        mse_guide_template = self.config.get_prompt_template('mse_guide')
        if not mse_guide_template:
            logging.getLogger("segment_analysis").warning(
                "No 'mse_guide' template found in prompt YAML. Using survey_question directly."
            )
            # Return the survey question as fallback
            return self.config.survey_question or ""
        
        # Format the template with the survey_question (MSE domain)
        survey_question = self.config.survey_question or "General mental status"
        prompt = mse_guide_template.format(survey_question=survey_question)
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"FETCHING MSE GUIDE OBSERVATIONAL QUESTIONS")
            print(f"{'='*70}")
            print(f"Survey Question (MSE Domain): {survey_question}")
            print(f"Prompt length: {len(prompt)} characters (~{len(prompt)//4} tokens)")
        
        logging.getLogger("segment_analysis").info(
            "Fetching MSE guide for survey_question '%s' (prompt length: %d chars)",
            survey_question, len(prompt)
        )
        
        mse_guide_timeout = self.config.request_timeout  # Same timeout as segment analysis
        
        def make_mse_guide_request():
            if self.config.verbose:
                print(f"  → Sending MSE guide request to API... (started at {time.strftime('%H:%M:%S')})")
            
            return vllm_omni_chat_request(
                api_base_url=self.config.api_url,
                model=self.config.model,
                prompt=prompt,
                max_tokens=_load_meta_max_tokens_default(),
                temperature=self.config.temperature,
                timeout=mse_guide_timeout,
            )
        
        try:
            # Use retry mechanism
            raw_response = retry_api_call(make_mse_guide_request, max_retries=20)
            
            # Store the raw response (includes reasoning) for logging
            self._mse_guide_raw_response = raw_response
            
            # Extract questions from the response
            # Look for the === QUESTIONS START === marker and extract everything after it
            mse_response = raw_response
            
            # First strip any thinking tags
            mse_response = strip_thinking_from_response(mse_response)
            
            # Try to extract just the questions section if markers are present
            questions_start_marker = "=== QUESTIONS START ==="
            questions_end_marker = "=== QUESTIONS END ==="  # Optional end marker
            reasoning_end_marker = "=== REASONING END ==="
            reasoning_start_marker = "=== REASONING START ==="
            
            if questions_start_marker in mse_response:
                # Extract everything after QUESTIONS START
                questions_section = mse_response.split(questions_start_marker, 1)[1]
                # If there's an end marker, truncate there
                if questions_end_marker in questions_section:
                    questions_section = questions_section.split(questions_end_marker, 1)[0]
                mse_response = questions_section.strip()
            elif reasoning_end_marker in mse_response:
                # Fallback: extract everything after REASONING END
                questions_section = mse_response.split(reasoning_end_marker, 1)[1]
                # Also remove QUESTIONS START marker if present after REASONING END
                if questions_start_marker in questions_section:
                    questions_section = questions_section.split(questions_start_marker, 1)[1]
                mse_response = questions_section.strip()
            elif reasoning_start_marker in mse_response:
                # If only REASONING START is present, remove everything before it ends
                # This handles cases where REASONING END might be missing
                # Just take everything that looks like questions (lines starting with question words)
                pass  # Will be handled by the cleanup below
            
            # ROBUST CLEANUP: Remove any remaining reasoning artifacts
            # Remove lines that are clearly part of reasoning (not questions)
            cleaned_lines = []
            in_reasoning = False
            
            for line in mse_response.split('\n'):
                line_stripped = line.strip()
                
                # Skip empty lines
                if not line_stripped:
                    continue
                
                # Skip reasoning markers
                if '===' in line_stripped and ('REASONING' in line_stripped or 'QUESTIONS' in line_stripped):
                    if 'REASONING START' in line_stripped:
                        in_reasoning = True
                    elif 'REASONING END' in line_stripped:
                        in_reasoning = False
                    continue
                
                # Skip if we're inside a reasoning block
                if in_reasoning:
                    continue
                
                # Skip lines that are clearly reasoning headers (STEP 1, STEP 2, etc.)
                if line_stripped.startswith('STEP ') and ':' in line_stripped:
                    continue
                
                # Skip lines that look like reasoning content (bullet points with explanations)
                if line_stripped.startswith('- ') and ':' in line_stripped and not line_stripped.endswith('?'):
                    continue
                
                # Skip lines that are category headers from reasoning
                if line_stripped.endswith(':') and not line_stripped.endswith('?'):
                    # Skip headers like "Visual signs by body region:", "MOST COMMON (>70% of cases):", etc.
                    if any(keyword in line_stripped.upper() for keyword in [
                        'SIGN', 'COMMON', 'RARE', 'VISUAL', 'AUDITORY', 'BEHAVIORAL', 
                        'UNDERSTANDING', 'ENUMERATION', 'RANKING', 'SELECTION',
                        'HEAD', 'FACE', 'EYES', 'MOUTH', 'NECK', 'ARMS', 'TORSO', 'LEGS',
                        'SPEECH', 'BODY REGION', 'DEFINITION', 'MANIFESTATION'
                    ]):
                        continue
                
                # Skip numbered list items that are part of reasoning (e.g., "1. Sign description")
                if len(line_stripped) > 2 and line_stripped[0].isdigit() and line_stripped[1] in '.):' and not line_stripped.endswith('?'):
                    continue
                
                # Keep lines that look like questions (start with question words or end with ?)
                question_starters = ('do ', 'does ', 'is ', 'are ', 'can ', 'has ', 'have ', 'was ', 'were ')
                if line_stripped.lower().startswith(question_starters) or line_stripped.endswith('?'):
                    cleaned_lines.append(line_stripped)
            
            # If we extracted valid questions, use them; otherwise fall back to the original
            if cleaned_lines:
                mse_response = '\n'.join(cleaned_lines)
            else:
                # Fallback: just strip obvious reasoning markers
                mse_response = mse_response.replace(reasoning_start_marker, '')
                mse_response = mse_response.replace(reasoning_end_marker, '')
                mse_response = mse_response.replace(questions_start_marker, '')
                mse_response = mse_response.replace(questions_end_marker, '')
                mse_response = mse_response.strip()
            
            # Log the response
            logging.getLogger("segment_analysis").info(
                "MSE guide response received (%d chars raw, %d chars reasoning, %d chars questions)",
                len(raw_response), 
                len(self._mse_guide_raw_response),
                len(mse_response)
            )
            
            if self.config.verbose:
                print(f"\n  ✓ MSE Guide Response Received:")
                print(f"    - Raw response (with reasoning): {len(raw_response)} characters")
                print(f"    - Questions extracted: {len(mse_response)} characters")
                # Show first few lines as preview
                lines = mse_response.strip().split('\n')[:5]
                print(f"  Preview (first 5 questions):")
                for line in lines:
                    if line.strip():
                        print(f"    - {line.strip()[:80]}...")
            
            # Cache the cleaned questions (for use in segment prompts)
            self._mse_guide_response = mse_response
            
            return mse_response
            
        except Exception as e:
            logging.getLogger("segment_analysis").error(
                "Failed to fetch MSE guide response: %s", str(e)
            )
            raise RuntimeError(f"MSE guide fetch failed: {str(e)}")
    
    def save_mse_guide_response(self, output_dir: str, video_path: str) -> Optional[str]:
        """
        Save the MSE guide response to a .txt file.
        
        Args:
            output_dir: Directory to save the file
            video_path: Path to the original video (for naming)
            
        Returns:
            Path to the saved file, or None if no response available
        """
        if not self._mse_guide_response:
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(video_path).stem
        mse_guide_path = os.path.join(output_dir, f"{stem}_mse_guide_questions.txt")
        
        with open(mse_guide_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MSE GUIDE OBSERVATIONAL QUESTIONS\n")
            f.write("Generated by Qwen3-Omni for Mental Status Examination\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"MSE Domain: {self.config.survey_question}\n")
            f.write(f"Timepoint: T{self.config.timepoint}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video: {Path(video_path).name}\n")
            f.write("\n" + "-" * 80 + "\n")
            f.write("OBSERVATIONAL QUESTIONS:\n")
            f.write("-" * 80 + "\n\n")
            f.write(self._mse_guide_response)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("END OF MSE GUIDE\n")
            f.write("=" * 80 + "\n")
        
        logging.getLogger("segment_analysis").info(
            "MSE guide response saved to: %s (%d chars)",
            mse_guide_path, len(self._mse_guide_response)
        )
        
        return mse_guide_path
    
    def save_mse_guide_reasoning(self, output_dir: str, video_path: str) -> Optional[str]:
        """
        Save the MSE guide reasoning process to a log file.
        
        This saves the full raw response from the model, including the reasoning
        steps (domain understanding, sign enumeration, frequency ranking, selection)
        that led to the final observational questions.
        
        Args:
            output_dir: Directory to save the file
            video_path: Path to the original video (for naming)
            
        Returns:
            Path to the saved reasoning log file, or None if no raw response available
        """
        if not self._mse_guide_raw_response:
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(video_path).stem
        reasoning_log_path = os.path.join(output_dir, f"{stem}_mse_guide_reasoning.log")
        
        with open(reasoning_log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MSE GUIDE REASONING LOG\n")
            f.write("Full Model Reasoning Process for Observational Question Generation\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"MSE Domain: {self.config.survey_question}\n")
            f.write(f"Timepoint: T{self.config.timepoint}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video: {Path(video_path).name}\n")
            f.write(f"Raw Response Length: {len(self._mse_guide_raw_response)} characters\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("FULL MODEL RESPONSE (INCLUDING REASONING):\n")
            f.write("=" * 80 + "\n\n")
            f.write(self._mse_guide_raw_response)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("END OF REASONING LOG\n")
            f.write("=" * 80 + "\n")
        
        logging.getLogger("segment_analysis").info(
            "MSE guide reasoning saved to: %s (%d chars)",
            reasoning_log_path, len(self._mse_guide_raw_response)
        )
        
        return reasoning_log_path
        
    def analyze_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Analyze all segments sequentially with rolling context
        
        Args:
            segments: List of audio segments
            
        Returns:
            Updated segments with observations
        """
        total_segments = len(segments)
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"PHASE 1: SEGMENT-LEVEL ANALYSIS (with Rolling Context)")
            print(f"Mode: Audio + Video")
            print(f"{'='*70}\n")
        
        # STEP 0: Fetch MSE guide observational questions (once, before all segments)
        # This provides focused questions for the model to use during segment analysis
        if segments:
            try:
                mse_response = self.fetch_mse_guide_response(segments[0].file_path)
                if self.config.verbose:
                    print(f"\n✓ MSE Guide questions ready for segment analysis\n")
            except Exception as e:
                logging.getLogger("segment_analysis").warning(
                    "Failed to fetch MSE guide, falling back to survey_question: %s", str(e)
                )
                if self.config.verbose:
                    print(f"\n⚠ MSE Guide fetch failed, using survey_question directly\n")
                # Fallback: use survey_question as mse_response
                self._mse_guide_response = self.config.survey_question or ""
        
        for i, segment in enumerate(segments):
            # Get rolling context from previous segments
            rolling_context = self._get_rolling_context(segments, i)
            
            # Store rolling context in segment for later saving
            segment.rolling_context = rolling_context
            
            # Create prompt for this segment
            prompt = self._create_segment_prompt(segment, i + 1, total_segments, rolling_context)
            
            # Store a concise prompt summary for downstream decision-trace synthesis
            if i == 0:
                segment.prompt_brief = (
                    f"Baseline analysis; no prior context; mode: audio+video; "
                    f"focus areas: {', '.join(self.config.clinical_focus)}."
                )
            else:
                if self.config.rolling_context_depth is None or self.config.rolling_context_depth < 0:
                    context_desc = "all prior segments"
                else:
                    context_desc = f"last {self.config.rolling_context_depth} segment(s)"
                segment.prompt_brief = (
                    f"Change-focused analysis with prior context from {context_desc}; "
                    f"mode: audio+video; focus areas: {', '.join(self.config.clinical_focus)}."
                )
            
            logging.getLogger("segment_analysis").info(
                "Segment %s/%s prompt (timeframe=%s, mode=audio+video):\n%s",
                i + 1,
                total_segments,
                segment.timeframe_str,
                prompt,
            )
            
            if self.config.verbose:
                print(f"Analyzing Segment {i + 1}/{total_segments} ({segment.timeframe_str})...")
            
            # Analyze segment
            try:
                raw_observations = self._analyze_segment(segment, prompt)
                
                # CRITICAL: Strip thinking process from observations before storing
                # This prevents bloating the rolling context with internal reasoning
                # that doesn't need to be passed to subsequent segments
                observations = strip_thinking_from_response(raw_observations)
                segment.observations = observations
                
                # Log to agent_reasoning.log (captures FULL raw response with <think> tags)
                if self.reasoning_logger:
                    self.reasoning_logger.log_segment_analysis(
                        segment_index=i + 1,
                        total_segments=total_segments,
                        timeframe=segment.timeframe_str,
                        prompt=prompt,
                        raw_response=raw_observations,
                        cleaned_response=observations
                    )
                
                # Log both raw and cleaned for debugging (raw may be very long)
                logging.getLogger("segment_analysis").info(
                    "Segment %s/%s raw response length: %d chars, cleaned: %d chars",
                    i + 1,
                    total_segments,
                    len(raw_observations),
                    len(observations),
                )
                logging.getLogger("segment_analysis").info(
                    "Segment %s/%s cleaned response:\n%s",
                    i + 1,
                    total_segments,
                    observations,
                )
                
                if self.config.verbose:
                    # Report both raw and cleaned sizes for transparency
                    if len(raw_observations) != len(observations):
                        print(f"  ✓ Analysis complete ({len(raw_observations)} chars raw → {len(observations)} chars cleaned, thinking stripped)")
                    else:
                        print(f"  ✓ Analysis complete ({len(observations)} characters)")
                    if len(observations) < 500:
                        print(f"  Preview: {observations[:200]}...")
                
            except Exception as e:
                print(f"  ✗ Error analyzing segment {i + 1}: {e}")
                segment.observations = f"[ERROR] Failed to analyze segment: {str(e)}"
        
        return segments
    
    def _get_rolling_context(self, segments: List[Segment], current_index: int) -> Optional[str]:
        """
        Get rolling context from previous segments with token/character limit
        
        CRITICAL: Constrains rolling context size to prevent token count from growing
        unbounded as segments progress. This ensures consistent processing times.
        
        Args:
            segments: All segments
            current_index: Index of current segment
            
        Returns:
            Formatted context string or None if first segment
        """
        if current_index == 0:
            return None
        
        MAX_ROLLING_CONTEXT_CHARS = _load_max_rolling_context_chars()
        
        # Get previous N segments based on rolling_context_depth
        depth = self.config.rolling_context_depth
        if depth is None or depth < 0:
            start_index = 0
        else:
            start_index = max(0, current_index - depth)
        previous_segments = segments[start_index:current_index]
        
        raw_context = self._format_segment_context(previous_segments)
        use_summary = self._should_use_summary(previous_segments)
        
        if not use_summary and len(raw_context) <= MAX_ROLLING_CONTEXT_CHARS:
            return raw_context
        
        if not use_summary:
            use_summary = True
        
        if use_summary:
            summary_text = self._get_cached_rolling_summary(previous_segments)
            if not summary_text:
                return raw_context
            
            context_text = (
                f"ROLLING SUMMARY (segments {previous_segments[0].index + 1}"
                f"-{previous_segments[-1].index + 1}):\n{summary_text}"
            )
            
            tail = max(0, self.config.rolling_context_summary_recent)
            if tail:
                recent_segments = previous_segments[-tail:]
                context_text += "\n\nMOST RECENT SEGMENTS (full detail):\n"
                context_text += self._format_segment_context(recent_segments, header_prefix="Recent ")
            
            return context_text
        
        return raw_context
    
    def _create_segment_prompt(
        self, 
        segment: Segment, 
        segment_num: int, 
        total_segments: int,
        rolling_context: Optional[str]
    ) -> str:
        """
        Create analysis prompt for a segment with rolling context and longitudinal history
        
        Args:
            segment: Segment to analyze
            segment_num: Segment number (1-indexed)
            total_segments: Total number of segments
            rolling_context: Context from previous segments (within this visit)
            
        Returns:
            Formatted prompt string
        """
        # Format clinical focus areas
        focus_list = "\n".join(f"- {area}" for area in self.config.clinical_focus)
        
        # Get modality description
        modality_desc = self.config.get_modality_description()
        
        # Get MSE guide response (observational questions)
        # This is the detailed list of questions generated by the mse_guide prompt
        mse_response = (self._mse_guide_response or self.config.survey_question or "").replace('{', '{{').replace('}', '}}')
        
        # Load max prompt size from .env
        max_prompt_chars = _load_max_prompt_chars()
        
        # LONGITUDINAL: Prepare prior visit summary (escape curly braces)
        prior_visit_summary = (self.config.prior_visit_summary or "").replace('{', '{{').replace('}', '}}')
        current_timepoint = self.config.timepoint
        
        # For longitudinal prompts, check for 'standard_segment' template first (martin_et_al style)
        # Otherwise fall back to first_segment/subsequent_segments pattern
        standard_template = self.config.get_prompt_template('segment_prompts.standard_segment')
        
        if standard_template:
            # Use unified standard_segment template for all segments (longitudinal style)
            # Prepare rolling segment context
            rolling_segment_context = (rolling_context or "No prior observations in this visit yet.").replace('{', '{{').replace('}', '}}')
            
            # Calculate overhead for truncation
            try:
                template_overhead = len(standard_template.format(
                    segment_num=segment_num,
                    total_segments=total_segments,
                    timeframe=segment.timeframe_str,
                    rolling_segment_context="",  # Empty to measure overhead
                    prior_visit_summary=prior_visit_summary,
                    current_timepoint=current_timepoint,
                    focus_list=focus_list,
                    modality_desc=modality_desc,
                    survey_question=self.config.survey_question or "",
                    mse_response=mse_response
                ))
            except KeyError:
                template_overhead = len(standard_template.format(
                    segment_num=segment_num,
                    total_segments=total_segments,
                    timeframe=segment.timeframe_str,
                    rolling_segment_context="",  # Empty to measure overhead
                    prior_visit_summary=prior_visit_summary,
                    current_timepoint=current_timepoint,
                    focus_list=focus_list,
                    modality_desc=modality_desc,
                    survey_question=self.config.survey_question or ""
                ))
            
            max_context_chars = max_prompt_chars - template_overhead - 500
            
            if max_context_chars <= 0:
                logging.getLogger("segment_analysis").warning(
                    "MAX_PROMPT_CHARS (%d) is smaller than template overhead (%d). "
                    "Rolling context will NOT be truncated. Increase MAX_PROMPT_CHARS in .env.blackwell.",
                    max_prompt_chars, template_overhead
                )
                max_context_chars = _load_max_rolling_context_chars()
            
            if len(rolling_segment_context) > max_context_chars:
                keep_chars = max(max_context_chars - 50, 200)
                truncated_context = "[...earlier context truncated...]\n" + rolling_segment_context[-keep_chars:]
                logging.getLogger("segment_analysis").warning(
                    "Rolling context truncated from %d to %d chars to fit prompt limit",
                    len(rolling_segment_context), len(truncated_context)
                )
                rolling_segment_context = truncated_context
            
            # Try to format with mse_response
            try:
                prompt = standard_template.format(
                    segment_num=segment_num,
                    total_segments=total_segments,
                    timeframe=segment.timeframe_str,
                    rolling_segment_context=rolling_segment_context,
                    prior_visit_summary=prior_visit_summary,
                    current_timepoint=current_timepoint,
                    focus_list=focus_list,
                    modality_desc=modality_desc,
                    survey_question=self.config.survey_question or "",
                    mse_response=mse_response
                )
            except KeyError:
                prompt = standard_template.format(
                    segment_num=segment_num,
                    total_segments=total_segments,
                    timeframe=segment.timeframe_str,
                    rolling_segment_context=rolling_segment_context,
                    prior_visit_summary=prior_visit_summary,
                    current_timepoint=current_timepoint,
                    focus_list=focus_list,
                    modality_desc=modality_desc,
                    survey_question=self.config.survey_question or ""
                )
                # Append mse_response if template doesn't have placeholder
                if mse_response:
                    prompt = prompt + "\n\n=== MSE OBSERVATIONAL QUESTIONS ===\n" + mse_response + "\n=== END MSE QUESTIONS ===\n"
        elif segment_num == 1:
            # First segment - no previous context (legacy template style)
            template = self.config.get_prompt_template('segment_prompts.first_segment')
            if not template:
                raise RuntimeError(
                    "Missing required template 'segment_prompts.first_segment' or 'segment_prompts.standard_segment' in prompt YAML. "
                    "This should have been caught during config validation."
                )
            # Try to format with mse_response
            try:
                prompt = template.format(
                    segment_num=segment_num,
                    total_segments=total_segments,
                    timeframe=segment.timeframe_str,
                    focus_list=focus_list,
                    modality_desc=modality_desc,
                    prior_visit_summary=prior_visit_summary,
                    current_timepoint=current_timepoint,
                    survey_question=self.config.survey_question or "",
                    mse_response=mse_response
                )
            except KeyError:
                prompt = template.format(
                    segment_num=segment_num,
                    total_segments=total_segments,
                    timeframe=segment.timeframe_str,
                    focus_list=focus_list,
                    modality_desc=modality_desc,
                    prior_visit_summary=prior_visit_summary,
                    current_timepoint=current_timepoint,
                    survey_question=self.config.survey_question or ""
                )
                # Append mse_response if template doesn't have placeholder
                if mse_response:
                    prompt = prompt + "\n\n=== MSE OBSERVATIONAL QUESTIONS ===\n" + mse_response + "\n=== END MSE QUESTIONS ===\n"
        else:
            # Subsequent segments - include rolling context (legacy template style)
            template = self.config.get_prompt_template('segment_prompts.subsequent_segments')
            if not template:
                raise RuntimeError(
                    "Missing required template 'segment_prompts.subsequent_segments' or 'segment_prompts.standard_segment' in prompt YAML. "
                    "This should have been caught during config validation."
                )
            
            # CRITICAL: Validate and truncate rolling context if needed
            max_prompt_chars = _load_max_prompt_chars()
            
            # Estimate template overhead (everything except rolling_context)
            try:
                template_overhead = len(template.format(
                    segment_num=segment_num,
                    total_segments=total_segments,
                    timeframe=segment.timeframe_str,
                    rolling_context="",  # Empty to measure overhead
                    focus_list=focus_list,
                    modality_desc=modality_desc,
                    prior_visit_summary=prior_visit_summary,
                    current_timepoint=current_timepoint,
                    survey_question=self.config.survey_question or "",
                    mse_response=mse_response
                ))
            except KeyError:
                template_overhead = len(template.format(
                    segment_num=segment_num,
                    total_segments=total_segments,
                    timeframe=segment.timeframe_str,
                    rolling_context="",  # Empty to measure overhead
                    focus_list=focus_list,
                    modality_desc=modality_desc,
                    prior_visit_summary=prior_visit_summary,
                    current_timepoint=current_timepoint,
                    survey_question=self.config.survey_question or ""
                ))
            
            # Calculate max allowed for rolling context
            max_context_chars = max_prompt_chars - template_overhead - 500
            
            if max_context_chars <= 0:
                logging.getLogger("segment_analysis").warning(
                    "MAX_PROMPT_CHARS (%d) is smaller than template overhead (%d). "
                    "Rolling context will NOT be truncated. Increase MAX_PROMPT_CHARS in .env.blackwell.",
                    max_prompt_chars, template_overhead
                )
                max_context_chars = _load_max_rolling_context_chars()
            
            # Truncate rolling context if needed
            context_to_use = rolling_context or ""
            context_to_use = context_to_use.replace('{', '{{').replace('}', '}}')
            
            if len(context_to_use) > max_context_chars:
                keep_chars = max(max_context_chars - 50, 200)
                truncated_context = "[...earlier context truncated...]\n" + context_to_use[-keep_chars:]
                logging.getLogger("segment_analysis").warning(
                    "Rolling context truncated from %d to %d chars to fit prompt limit",
                    len(context_to_use), len(truncated_context)
                )
                context_to_use = truncated_context
            
            # Try to format with mse_response
            try:
                prompt = template.format(
                    segment_num=segment_num,
                    total_segments=total_segments,
                    timeframe=segment.timeframe_str,
                    rolling_context=context_to_use,
                    focus_list=focus_list,
                    modality_desc=modality_desc,
                    prior_visit_summary=prior_visit_summary,
                    current_timepoint=current_timepoint,
                    survey_question=self.config.survey_question or "",
                    mse_response=mse_response
                )
            except KeyError:
                prompt = template.format(
                    segment_num=segment_num,
                    total_segments=total_segments,
                    timeframe=segment.timeframe_str,
                    rolling_context=context_to_use,
                    focus_list=focus_list,
                    modality_desc=modality_desc,
                    prior_visit_summary=prior_visit_summary,
                    current_timepoint=current_timepoint,
                    survey_question=self.config.survey_question or ""
                )
                # Append mse_response if template doesn't have placeholder
                if mse_response:
                    prompt = prompt + "\n\n=== MSE OBSERVATIONAL QUESTIONS ===\n" + mse_response + "\n=== END MSE QUESTIONS ===\n"
        
        # Final validation - ensure prompt is within limits
        if len(prompt) > max_prompt_chars:
            logging.getLogger("segment_analysis").warning(
                "Final prompt length (%d chars) exceeds limit (%d). This may cause API truncation.",
                len(prompt), max_prompt_chars
            )
        
        return prompt
    
    def _should_use_summary(self, previous_segments: List[Segment]) -> bool:
        trigger = getattr(self.config, 'rolling_context_summary_trigger', 0)
        return trigger > 0 and len(previous_segments) >= trigger
    
    def _format_segment_context(
        self,
        segments_subset: List[Segment],
        header_prefix: str = ""
    ) -> str:
        parts = []
        for seg in segments_subset:
            if not seg.observations:
                continue
            prefix = header_prefix or ""
            parts.append(
                f"{prefix}Segment {seg.index + 1} ({seg.timeframe_str}):\n"
                f"{seg.observations}"
            )
        return "\n\n".join(parts) if parts else ""
    
    def _get_cached_rolling_summary(self, previous_segments: List[Segment]) -> Optional[str]:
        if not previous_segments:
            return None
        latest_index = previous_segments[-1].index
        if self._rolling_summary_last_index != latest_index:
            summary = self._generate_rolling_context_summary(previous_segments)
            if summary:
                self._rolling_summary_cache = summary
                self._rolling_summary_last_index = latest_index
        return self._rolling_summary_cache
    
    def _generate_rolling_context_summary(self, previous_segments: List[Segment]) -> Optional[str]:
        summary_prompt = self._build_summary_prompt(previous_segments)
        if not summary_prompt:
            return None
        
        max_tokens = min(1024, _load_meta_max_tokens_default())
        
        def make_request():
            return vllm_omni_chat_request(
                api_base_url=self.config.api_url,
                model=self.config.model,
                prompt=summary_prompt,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                timeout=900,
            )
        
        try:
            raw_summary = retry_api_call(make_request, max_retries=5)
            # CRITICAL: Strip thinking from summary just like segment observations
            summary_text = strip_thinking_from_response(raw_summary)
            
            # Log to agent_reasoning.log (captures FULL raw response with <think> tags)
            if self.reasoning_logger:
                segments_range = f"{previous_segments[0].index + 1}-{previous_segments[-1].index + 1}"
                self.reasoning_logger.log_rolling_summary(
                    segments_range=segments_range,
                    prompt=summary_prompt,
                    raw_response=raw_summary,
                    cleaned_response=summary_text
                )
            
            logging.getLogger("segment_analysis").info(
                "Rolling context summary generated for segments %s-%s (raw=%d chars, cleaned=%d chars)",
                previous_segments[0].index + 1,
                previous_segments[-1].index + 1,
                len(raw_summary),
                len(summary_text)
            )
            return summary_text
        except Exception as e:
            logging.getLogger("segment_analysis").error(
                "Failed to generate rolling context summary: %s", e
            )
            return None
    
    def _build_summary_prompt(self, previous_segments: List[Segment]) -> str:
        if not previous_segments:
            return ""
        max_words = getattr(self.config, 'rolling_context_summary_max_words', 250)
        segment_text = []
        for seg in previous_segments:
            segment_text.append(
                f"=== SEGMENT {seg.index + 1} ({seg.timeframe_str}) ===\n"
                f"{seg.observations or '[No observations recorded]'}"
            )
        prior_text = "\n\n".join(segment_text)
        modality_desc = self.config.get_modality_description()
        return (
            "You are a clinical audio-video analyst maintaining a rolling summary.\n"
            f"Summarize the cumulative clinical findings, behavioral trends, and risk modifiers from the prior segments below.\n"
            f"Use at most {max_words} words. Emphasize longitudinal patterns over per-segment minutiae.\n"
            f"Modality guidance: {modality_desc}\n\n"
            f"PRIOR SEGMENTS:\n{prior_text}\n"
            "\nReturn only the summary paragraph(s)."
        )
    
    def _analyze_segment(self, segment: Segment, prompt: str) -> str:
        """
        Send segment to API for analysis (with retry mechanism).

        Uses the vLLM-Omni OpenAI-compatible endpoint (POST /v1/chat/completions)
        with video and audio submitted as separate base64 data URLs.

        Args:
            segment: Video segment with path and audio_file_path
            prompt: Analysis prompt/instruction

        Returns:
            Analysis text from model
        """
        # Get segment file info for diagnostics
        segment_size = os.path.getsize(segment.file_path) if os.path.exists(segment.file_path) else 0
        segment_size_mb = segment_size / (1024 * 1024)
        
        # Get audio file info
        audio_size = 0
        if segment.audio_file_path and os.path.exists(segment.audio_file_path):
            audio_size = os.path.getsize(segment.audio_file_path)
        audio_size_mb = audio_size / (1024 * 1024)
        
        timeout_seconds = self.config.request_timeout
        
        # Load max output tokens to limit segment observation length
        # This prevents bloated rolling context and keeps prompts within limits
        max_segment_tokens = _load_max_segment_output_tokens()
        
        if self.config.verbose:
            print(f"  ℹ Video file: {os.path.basename(segment.file_path)} ({segment_size_mb:.2f} MB)")
            print(f"  ℹ Audio file: {os.path.basename(segment.audio_file_path) if segment.audio_file_path else 'N/A'} ({audio_size_mb:.2f} MB)")
            print(f"  ℹ API endpoint: {self.config.api_url}/v1/chat/completions")
            print(f"  ℹ Request timeout: {timeout_seconds}s ({timeout_seconds/60:.1f} minutes)")
            print(f"  ℹ Prompt length: {len(prompt)} characters (~{len(prompt)//4} tokens)")
            print(f"  ℹ Max output tokens: {max_segment_tokens}")
        
        # Validate audio file exists (video-only mode is allowed for skeleton videos)
        effective_audio_path = segment.audio_file_path
        if not effective_audio_path or not os.path.exists(effective_audio_path):
            effective_audio_path = None
            if self.config.verbose:
                print(f"  ℹ No audio available for segment {segment.index + 1} — running video-only analysis")
        
        def make_request():
            mode_str = "video+audio" if effective_audio_path else "video-only"
            if self.config.verbose:
                print(f"  → Sending request to API with {mode_str}... (started at {time.strftime('%H:%M:%S')})")
            
            return vllm_omni_chat_request(
                api_base_url=self.config.api_url,
                model=self.config.model,
                prompt=prompt,
                video_path=segment.file_path,
                audio_path=effective_audio_path,
                max_tokens=max_segment_tokens,
                temperature=self.config.temperature,
                timeout=timeout_seconds,
            )
        
        try:
            # Use retry mechanism with 20 attempts
            return retry_api_call(make_request, max_retries=20)
            
        except Exception as e:
            # All specific error messages are now in retry_api_call
            raise RuntimeError(f"Segment analysis failed after retries: {str(e)}")


# ============================================================================
# META-ANALYSIS
# ============================================================================

class MetaAnalyzer:
    """Performs meta-analysis on all segment observations"""
    
    def __init__(self, config: AnalysisConfig, reasoning_logger: Optional[AgentReasoningLogger] = None):
        self.config = config
        self.reasoning_logger = reasoning_logger
        # Meta-analysis uses text-only input via vLLM-Omni chat completions
    
    def synthesize_diagnosis(self, segments: List[Segment], video_path: str) -> str:
        """
        Synthesize final diagnosis from all segment observations
        
        Args:
            segments: List of analyzed segments
            video_path: Original video path (for reference, not re-processed)
            
        Returns:
            Final diagnostic synthesis
            
        Raises:
            RuntimeError: If no segments are available for analysis
        """
        # CRITICAL: Validate segments list is not empty
        if not segments:
            raise RuntimeError(
                f"No segments available for meta-analysis. "
                f"Video may be too short (< 5 seconds) or segmentation failed. "
                f"Input video: {video_path}"
            )
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"PHASE 2: META-ANALYSIS (Text-Based Synthesis)")
            print(f"{'='*70}\n")
            print("Compiling segment observations into temporal summary...")
        
        # Compile all observations
        temporal_summary = self._compile_temporal_summary(segments)
        
        # Create meta-analysis prompt
        meta_prompt = self._create_meta_analysis_prompt(temporal_summary, video_path)
        
        if self.config.verbose:
            prompt_len = len(meta_prompt)
            print(f"  ℹ Meta-analysis prompt: {prompt_len} characters (~{prompt_len//4} tokens)")
            print("Performing meta-analysis for final diagnosis...")
        
        logging.getLogger("segment_analysis").info(
            "Meta-analysis prompt (length=%d chars, ~%d tokens):\n%s",
            len(meta_prompt),
            len(meta_prompt)//4,
            meta_prompt[:2000] + "\n...[truncated for log]..." + meta_prompt[-1000:] if len(meta_prompt) > 3000 else meta_prompt,
        )
        
        # For meta-analysis, we use a dummy short video (just use first segment)
        # The actual analysis is based on the TEXT in the prompt
        meta_timeout = self.config.request_timeout * 2  # Meta-analysis gets 2x timeout (longer prompts)
        if self.config.verbose:
            print(f"  ℹ Meta-analysis timeout: {meta_timeout}s ({meta_timeout/60:.1f} minutes)")
        
        def make_meta_request():
            if self.config.verbose:
                print(f"  → Sending meta-analysis request to API... (started at {time.strftime('%H:%M:%S')})")
            
            return vllm_omni_chat_request(
                api_base_url=self.config.api_url,
                model=self.config.model,
                prompt=meta_prompt,
                max_tokens=_load_meta_max_tokens_default(),
                temperature=self.config.temperature,
                timeout=meta_timeout,
            )
        
        try:
            # Use retry mechanism with 20 attempts
            raw_text = retry_api_call(make_meta_request, max_retries=20)
            
            # CRITICAL: Strip thinking from meta-analysis response
            final_text = strip_thinking_from_response(raw_text)
            
            # Log to agent_reasoning.log (captures FULL raw response with <think> tags)
            if self.reasoning_logger:
                self.reasoning_logger.log_meta_analysis(
                    prompt=meta_prompt,
                    raw_response=raw_text,
                    cleaned_response=final_text
                )
            
            if self.config.verbose:
                if len(raw_text) != len(final_text):
                    print(f"  ✓ Meta-analysis complete (stripped {len(raw_text) - len(final_text)} chars of thinking)\n")
                else:
                    print("  ✓ Meta-analysis complete\n")
            
            logging.getLogger("segment_analysis").info(
                "Meta-analysis response (cleaned):\n%s",
                final_text,
            )
            return final_text
            
        except Exception as e:
            raise RuntimeError(f"Meta-analysis failed after retries: {str(e)}")
    
    def _compile_temporal_summary(self, segments: List[Segment]) -> str:
        """Compile all segment observations into chronological text
        
        Implements smart truncation to avoid exceeding context limits:
        - If total text < 50K chars: use all observations
        - If total text >= 50K chars: truncate each segment proportionally
        """
        summary_parts = []
        
        for segment in segments:
            if segment.observations:
                summary_parts.append(
                    f"=== SEGMENT {segment.index + 1} ({segment.timeframe_str}) ===\n"
                    f"{segment.observations}"
                )
            else:
                summary_parts.append(
                    f"=== SEGMENT {segment.index + 1} ({segment.timeframe_str}) ===\n"
                    f"[No observations recorded]"
                )
        
        full_summary = "\n\n".join(summary_parts)
        
        # Check if summary is too long
        # For MAX_MODEL_LEN=16384, we need to leave room for:
        # - Video frames: ~12000 tokens
        # - Meta-analysis template: ~500 tokens
        # - Response generation: ~1000 tokens
        # Leaves ~3000 tokens for summary = ~12000 chars
        # Use 8000 chars (~2000 tokens) to be safe
        MAX_SUMMARY_CHARS = _load_env_int("MAX_TEMPORAL_SUMMARY_CHARS", default=8000, min_val=2000, max_val=30000)
        
        if self.config.verbose:
            print(f"  ℹ Compiled summary length: {len(full_summary)} characters (~{len(full_summary)//4} tokens)")
        
        if len(full_summary) <= MAX_SUMMARY_CHARS:
            if self.config.verbose:
                print(f"  ℹ Summary within limit, using all observations")
            return full_summary
        
        # Need to truncate - calculate how much to keep per segment
        num_segments = len(segments)
        if num_segments == 0:
            return ""
        
        # Reserve space for headers and separators
        header_overhead = sum(len(f"=== SEGMENT {seg.index + 1} ({seg.timeframe_str}) ===\n") 
                             for seg in segments)
        separator_overhead = len("\n\n") * (num_segments - 1)
        available_chars = MAX_SUMMARY_CHARS - header_overhead - separator_overhead
        
        # Distribute available space proportionally based on original lengths
        original_lengths = [len(seg.observations or "") for seg in segments]
        total_original = sum(original_lengths)
        
        if total_original == 0:
            return "\n\n".join(f"=== SEGMENT {seg.index + 1} ({seg.timeframe_str}) ===\n[No observations recorded]" 
                              for seg in segments)
        
        # Truncate each segment proportionally
        truncated_parts = []
        for i, segment in enumerate(segments):
            header = f"=== SEGMENT {segment.index + 1} ({segment.timeframe_str}) ==="
            
            if not segment.observations:
                truncated_parts.append(f"{header}\n[No observations recorded]")
                continue
            
            # Calculate this segment's allocation
            proportion = original_lengths[i] / total_original
            allocated_chars = int(available_chars * proportion)
            
            # Ensure minimum useful length
            allocated_chars = max(allocated_chars, 200)
            
            obs = segment.observations
            if len(obs) <= allocated_chars:
                truncated_parts.append(f"{header}\n{obs}")
            else:
                # Truncate with ellipsis
                truncated = obs[:allocated_chars-20].rsplit('\n', 1)[0]  # Try to break at newline
                truncated_parts.append(f"{header}\n{truncated}\n[...truncated for length...]")
        
        result = "\n\n".join(truncated_parts)
        
        if self.config.verbose:
            print(f"  ℹ Summary truncated: {len(full_summary)} → {len(result)} chars (~{len(result)//4} tokens)")
            print(f"  ℹ Reduced by {100*(1-len(result)/len(full_summary)):.1f}%")
        
        return result
    
    def _create_meta_analysis_prompt(self, temporal_summary: str, video_path: str) -> str:
        """Create comprehensive meta-analysis prompt with longitudinal context"""
        
        video_name = Path(video_path).name
        
        # Get template from YAML (required, no fallback)
        template = self.config.get_prompt_template('meta_analysis_prompt')
        if not template:
            raise RuntimeError(
                "Missing required template 'meta_analysis_prompt' in prompt YAML. "
                "This should have been caught during config validation."
            )
        
        # CRITICAL: Escape curly braces in temporal_summary to prevent format() errors
        # JSON output contains { and } which Python's .format() interprets as placeholders
        safe_temporal_summary = temporal_summary.replace('{', '{{').replace('}', '}}')
        
        # LONGITUDINAL: Prepare prior visit summary and timepoint
        prior_visit_summary = (self.config.prior_visit_summary or "").replace('{', '{{').replace('}', '}}')
        current_timepoint = self.config.timepoint
        
        # Try to format with longitudinal parameters
        # Some templates may use {prior_visit_summary}, {current_timepoint}, {full_visit_logs}
        try:
            prompt = template.format(
                audio_name=video_name,  # Template uses audio_name but we pass video_name
                temporal_summary=safe_temporal_summary,
                full_visit_logs=safe_temporal_summary,  # Alias for longitudinal templates
                prior_visit_summary=prior_visit_summary,
                current_timepoint=current_timepoint,
                survey_question=self.config.survey_question or ""
            )
        except KeyError as e:
            # Fallback: some templates may not use all parameters
            logging.getLogger("segment_analysis").debug(
                "Meta-analysis template missing parameter %s, using basic format", e
            )
            prompt = template.format(
                audio_name=video_name,
                temporal_summary=safe_temporal_summary,
                survey_question=self.config.survey_question or ""
            )
        
        return prompt

    def synthesize_decision_trace(self, segments: List[Segment], final_diagnosis_text: str, video_path: str) -> Tuple[str, str]:
        """Create a concise, stepwise decision-making summary and a diagnosis paragraph.

        Returns:
            (decision_trace_summary, final_diagnosis_paragraph)
        """
        # Build structured trace input (not all prompts, just concise summaries and outputs)
        # IMPORTANT: Truncate observations to avoid exceeding context limits
        # Use 6000 chars (~1500 tokens) to leave room for template and video frames
        MAX_TRACE_INPUT_CHARS = _load_env_int("MAX_DECISION_TRACE_CHARS", default=6000, min_val=2000, max_val=20000)
        
        steps = []
        for seg in segments:
            brief = seg.prompt_brief or "Segment analysis"
            obs = (seg.observations or "[No observations recorded]").strip()
            
            # Truncate individual observations if too long
            if len(obs) > 1000:
                obs = obs[:1000] + "\n[...truncated...]"
            
            steps.append(
                f"SEGMENT {seg.index + 1} | Timeframe: {seg.timeframe_str}\n"
                f"Prompt summary: {brief}\n"
                f"Output summary: {obs}\n"
            )
        
        decision_trace_input = "\n\n".join(steps)
        
        # Check total length and truncate proportionally if needed
        if len(decision_trace_input) > MAX_TRACE_INPUT_CHARS:
            if self.config.verbose:
                print(f"  ℹ Decision trace input too long: {len(decision_trace_input)} chars, truncating...")
            
            # Calculate how much to keep per segment
            num_segments = len(segments)
            if num_segments > 0:
                chars_per_segment = MAX_TRACE_INPUT_CHARS // num_segments
                chars_per_segment = max(chars_per_segment, 300)  # Minimum 300 chars per segment
                
                truncated_steps = []
                for seg in segments:
                    brief = seg.prompt_brief or "Segment analysis"
                    obs = (seg.observations or "[No observations recorded]").strip()
                    
                    # Allocate space
                    available = chars_per_segment - len(f"SEGMENT {seg.index + 1} | Timeframe: {seg.timeframe_str}\nPrompt summary: {brief}\nOutput summary: \n")
                    available = max(available, 150)
                    
                    if len(obs) > available:
                        obs = obs[:available] + "\n[...truncated...]"
                    
                    truncated_steps.append(
                        f"SEGMENT {seg.index + 1} | Timeframe: {seg.timeframe_str}\n"
                        f"Prompt summary: {brief}\n"
                        f"Output summary: {obs}\n"
                    )
                
                decision_trace_input = "\n\n".join(truncated_steps)
                
                if self.config.verbose:
                    print(f"  ℹ Decision trace truncated to {len(decision_trace_input)} chars")
        
        if self.config.verbose:
            print(f"  ℹ Decision trace input: {len(decision_trace_input)} chars (~{len(decision_trace_input)//4} tokens)")

        # Get template from YAML (required, no fallback)
        template = self.config.get_prompt_template('decision_trace_prompt')
        if not template:
            raise RuntimeError(
                "Missing required template 'decision_trace_prompt' in prompt YAML. "
                "This should have been caught during config validation."
            )
        
        # CRITICAL: Escape curly braces in inputs to prevent format() errors
        # JSON output contains { and } which Python's .format() interprets as placeholders
        safe_decision_trace_input = decision_trace_input.replace('{', '{{').replace('}', '}}')
        safe_final_diagnosis_text = (final_diagnosis_text or "").replace('{', '{{').replace('}', '}}')
        
        # LONGITUDINAL: Include prior visit summary for trajectory comparison
        prior_visit_summary = (self.config.prior_visit_summary or "").replace('{', '{{').replace('}', '}}')
        
        prompt = template.format(
            decision_trace_input=safe_decision_trace_input,
            final_diagnosis_text=safe_final_diagnosis_text,
            prior_visit_summary=prior_visit_summary,
            survey_question=self.config.survey_question or ""
        )

        trace_timeout = self.config.request_timeout * 2  # Decision trace gets 2x timeout (longer prompts)
        if self.config.verbose:
            prompt_len = len(prompt)
            print(f"  ℹ Decision trace prompt: {prompt_len} chars (~{prompt_len//4} tokens)")
            print(f"  ℹ Decision trace timeout: {trace_timeout}s ({trace_timeout/60:.1f} minutes)")
            print("Performing decision trace synthesis...")
        
        def make_trace_request():
            if self.config.verbose:
                print(f"  → Sending decision trace request to API... (started at {time.strftime('%H:%M:%S')})")
            
            return vllm_omni_chat_request(
                api_base_url=self.config.api_url,
                model=self.config.model,
                prompt=prompt,
                max_tokens=_load_meta_max_tokens_default(),
                temperature=self.config.temperature,
                timeout=trace_timeout,
            )
        
        try:
            # Use retry mechanism with 20 attempts
            raw_text = retry_api_call(make_trace_request, max_retries=20)
            
            # CRITICAL: Strip thinking from decision trace response
            full_text = strip_thinking_from_response(raw_text)
            
            # Log to agent_reasoning.log (captures FULL raw response with <think> tags)
            if self.reasoning_logger:
                self.reasoning_logger.log_decision_trace(
                    prompt=prompt,
                    raw_response=raw_text,
                    cleaned_response=full_text
                )

            # Parse the two sections by headings; fallback to whole text
            decision_trace_summary = ""
            diagnosis_paragraph = ""
            marker_summary = "=== DECISION TRACE SUMMARY ==="
            possible_diag_markers = [
                "=== FINAL DIAGNOSIS PARAGRAPH ===",
                "=== FINAL LABEL AND RATIONALE PARAGRAPH ===",
            ]
            # Find which diagnosis marker is present, if any
            found_diag_marker = next((m for m in possible_diag_markers if m in full_text), None)
            if marker_summary in full_text and found_diag_marker:
                part = full_text.split(marker_summary, 1)[1]
                if found_diag_marker in part:
                    a, b = part.split(found_diag_marker, 1)
                    decision_trace_summary = a.strip()
                    diagnosis_paragraph = b.strip()
            if not decision_trace_summary:
                decision_trace_summary = full_text.strip() or "[Unable to parse decision trace summary]"
            if not diagnosis_paragraph:
                # Fallback: compress the final diagnosis text into a paragraph (simple heuristic)
                diagnosis_paragraph = final_diagnosis_text.strip().split("\n\n")[0][:2000]

            # Post-process to enforce a clean, stable final paragraph without meta text
            def _clean_final_paragraph(text: str, full_response: str) -> str:
                s = text or ""
                # Strip any think-tags or similar meta markers
                s = re.sub(r"</?think[^>]*>", "", s, flags=re.IGNORECASE).strip()
                # If headings are present, drop everything before the final section header
                if "=== FINAL DIAGNOSIS PARAGRAPH ===" in s:
                    s = s.split("=== FINAL DIAGNOSIS PARAGRAPH ===", 1)[-1].strip()
                if "=== FINAL LABEL AND RATIONALE PARAGRAPH ===" in s:
                    s = s.split("=== FINAL LABEL AND RATIONALE PARAGRAPH ===", 1)[-1].strip()
                # Remove any preceding instruction lines; prefer starting at 'MSE Severity Rating:'
                idx = s.lower().find("mse severity rating:")
                if idx == -1:
                    idx = s.lower().find("final tgi rating:")  # Legacy fallback
                if idx != -1:
                    s = s[idx:]
                # Cut off at next section header or blank line block
                s = s.split("\n=== ")[0].split("\n\n")[0].strip()
                # If still not starting with the required prefix, try to recover from the full response
                if not s.lower().startswith("mse severity rating:") and not s.lower().startswith("final tgi rating:"):
                    m = re.search(r"((?:mse\s*severity|final\s*tgi)\s*rating:\s*(?:Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)[^\n]*)", full_response, flags=re.IGNORECASE)
                    if m:
                        s = m.group(1).strip()
                return s

            diagnosis_paragraph = _clean_final_paragraph(diagnosis_paragraph, full_text)

            # If still not a clean paragraph starting with "MSE Severity Rating:", synthesize from meta-analysis text
            if not diagnosis_paragraph or (not diagnosis_paragraph.lower().startswith("mse severity rating:") and not diagnosis_paragraph.lower().startswith("final tgi rating:")):
                def _extract_rating(meta: str) -> Optional[str]:
                    for pat, flg in [
                        (r'"rating"\s*:\s*"(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)"', re.IGNORECASE),
                        (r"^\s*-\s*RATING:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b", re.IGNORECASE | re.MULTILINE),
                        (r"(?:mse\s*severity|final\s*tgi)\s*rating:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)", re.IGNORECASE),
                    ]:
                        hits = re.findall(pat, meta, flags=flg)
                        if hits:
                            return hits[-1].capitalize()
                    return None

                def _extract_list_after(header: str, meta: str, max_items: int = 2) -> list[str]:
                    items: list[str] = []
                    lines = (meta or "").splitlines()
                    start = None
                    for i, ln in enumerate(lines):
                        if ln.strip().upper().startswith(header.upper()):
                            start = i + 1
                            break
                    if start is None:
                        return items
                    for j in range(start, len(lines)):
                        line = lines[j].strip()
                        if not line:
                            break
                        if line.startswith("===") or line.endswith(":"):
                            break
                        if line.startswith("-"):
                            item = line.lstrip("-").strip()
                            if item:
                                items.append(item)
                                if len(items) >= max_items:
                                    break
                        else:
                            break
                    return items

                rating = _extract_rating(final_diagnosis_text or "")
                incr = _extract_list_after("KEY EVIDENCE INCREASING SEVERITY:", final_diagnosis_text or "", max_items=2)
                decr = _extract_list_after("KEY EVIDENCE DECREASING SEVERITY:", final_diagnosis_text or "", max_items=2)

                if rating:
                    parts = [f"MSE Severity Rating: {rating}."]
                    if incr:
                        parts.append(f"Key supporting evidence: {', '.join(incr)}.")
                    if decr:
                        parts.append(f"Countervailing factors: {', '.join(decr)}.")
                    diagnosis_paragraph = " ".join(parts).strip()

            logging.getLogger("segment_analysis").info(
                "Decision trace summary generated. Lengths: trace=%s, paragraph=%s",
                len(decision_trace_summary), len(diagnosis_paragraph)
            )
            return decision_trace_summary, diagnosis_paragraph
        except Exception as e:
            raise RuntimeError(f"Decision trace synthesis failed: {str(e)}")

    def _extract_final_paragraph_from_meta(self, meta_text: str) -> str:
        """Extract a final paragraph directly from meta-analysis text without additional API call.
        
        This is used when the prompt YAML does not include a 'final_summary_prompt' template.
        """
        # Try to extract an existing final paragraph from meta_text
        # Look for common paragraph markers that might already be in the meta-analysis output
        
        # Pattern 1: Look for "FINAL" section headings
        final_section_patterns = [
            r"FINAL\s+(?:MSE\s+)?(?:SEVERITY\s+)?RATING\s+PARAGRAPH:?\s*\n\s*(.+?)(?:\n\n|\n===|$)",
            r"FINAL\s+(?:TGI\s+)?RATING\s+PARAGRAPH:?\s*\n\s*(.+?)(?:\n\n|\n===|$)",  # Legacy fallback
            r"FINAL\s+(?:MSE\s+)?DETERMINATION:?\s*\n\s*(.+?)(?:\n\n|\n===|$)",
            r"FINAL\s+(?:DIAGNOSIS|LABEL)\s+PARAGRAPH:?\s*\n\s*(.+?)(?:\n\n|\n===|$)",
            r"DECISION\s+RATIONALE:?\s*\n\s*(.+?)(?:\n\n|\n===|$)",
        ]
        
        for pattern in final_section_patterns:
            match = re.search(pattern, meta_text or "", flags=re.IGNORECASE | re.DOTALL)
            if match:
                paragraph = match.group(1).strip()
                # Clean up any bullet points or multiple paragraphs - take first substantive paragraph
                lines = [line.strip() for line in paragraph.split('\n') if line.strip() and not line.strip().startswith('-')]
                if lines:
                    return ' '.join(lines[:3])  # Take first 3 lines max
        
        # Pattern 2: Extract from structured output (for formats like TUL or DAIC-WOZ)
        # Try to find a rating and build a simple paragraph
        rating_patterns = [
            r'"rating"\s*:\s*"(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)"',  # JSON format
            r"^\s*-\s*RATING:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b",
            r"^\s*-\s*LABEL:\s*([01])\b",
            r"MSE\s+Severity\s+rating:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b",
            r"Final\s+(?:TGI\s+)?rating:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b",  # Legacy fallback
        ]
        
        rating = None
        rating_type = None
        for pattern in rating_patterns:
            matches = re.findall(pattern, meta_text or "", flags=re.IGNORECASE | re.MULTILINE)
            if matches:
                rating = matches[-1]
                if pattern.find('LABEL') != -1:
                    rating_type = 'binary'
                    rating = 'Depressed' if rating == '1' else 'Normal'
                else:
                    rating_type = 'severity'
                break
        
        if rating:
            # Build a simple paragraph from the rating and any key evidence
            if rating_type == 'binary':
                return f"Final Classification: {rating}. Based on analysis of audio-video evidence across all segments."
            else:
                return f"MSE Severity Rating: {rating}. Based on analysis of audio-video evidence across all segments."
        
        # Fallback: return first substantive paragraph from meta_text
        paragraphs = [p.strip() for p in (meta_text or "").split('\n\n') if p.strip()]
        for para in paragraphs:
            # Skip section headings
            if para.isupper() or para.endswith(':') or para.startswith('==='):
                continue
            # Skip bullet lists
            if para.startswith('-'):
                continue
            # Found a substantive paragraph
            return para[:500]  # Limit length
        
        # Ultimate fallback
        return "Unable to extract final rating from meta-analysis output."

    def synthesize_final_paragraph(self, meta_text: str, video_path: str, segments: List[Segment]) -> str:
        """Derive a single clean 'MSE Severity Rating: …' paragraph from meta-analysis text."""
        # Pre-check: ensure meta_text contains a RATING.
        # Use findall + take LAST match so that model thinking/planning text
        # (which may mention ratings before the structured answer) doesn't win.
        rating_patterns_ordered = [
            (r'"rating"\s*:\s*"(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)"', re.IGNORECASE),
            (r"^\s*-\s*RATING:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b", re.IGNORECASE | re.MULTILINE),
            (r"Final\s+rating:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b", re.IGNORECASE),
        ]
        
        extracted_rating = None
        for pattern, flags in rating_patterns_ordered:
            matches = re.findall(pattern, meta_text or "", flags=flags)
            if matches:
                extracted_rating = matches[-1].capitalize()
                break
        
        if not extracted_rating:
            logging.getLogger("segment_analysis").error(
                "Meta-analysis output is missing the required '- RATING:' or 'Final rating:' line. Cannot extract final rating."
            )
            fallback_matches = re.findall(r"\b(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b",
                                          meta_text or "", flags=re.IGNORECASE)
            if fallback_matches:
                extracted_rating = fallback_matches[-1].capitalize()
                logging.getLogger("segment_analysis").warning(
                    f"Using fallback: extracted '{extracted_rating}' from meta-analysis text (no RATING line found)"
                )
            else:
                logging.getLogger("segment_analysis").error(
                    "No valid rating found in meta-analysis text. Defaulting to 'Normal' for safety."
                )
                extracted_rating = "Normal"
        
        # If no template, return immediately with extracted rating
        template = self.config.get_prompt_template('final_summary_prompt')
        if not template:
            # If final_summary_prompt is not in YAML, extract directly from meta_text
            # This is acceptable since some prompt configs may not need this extra synthesis step
            logging.getLogger("segment_analysis").info(
                "No 'final_summary_prompt' template in YAML; extracting directly from meta-analysis output"
            )
            # Return with the extracted rating to ensure consistency
            return f"MSE Severity Rating: {extracted_rating}."
        
        # CRITICAL: Pass the extracted rating to the prompt to enforce consistency
        # Also escape curly braces in meta_text to prevent format() errors
        safe_meta_text = (meta_text or "").replace('{', '{{').replace('}', '}}')
        
        # LONGITUDINAL: Include prior visit summary for trajectory comparison
        prior_visit_summary = (self.config.prior_visit_summary or "").replace('{', '{{').replace('}', '}}')
        
        prompt = template.format(
            meta_text=safe_meta_text,
            extracted_rating=extracted_rating,
            prior_visit_summary=prior_visit_summary,
            survey_question=self.config.survey_question or ""
        )

        final_timeout = self.config.request_timeout * 2  # Final paragraph gets 2x timeout
        if self.config.verbose:
            print(f"  ℹ Final paragraph timeout: {final_timeout}s ({final_timeout/60:.1f} minutes)")
        
        def make_request():
            if self.config.verbose:
                print(f"  → Sending final paragraph request to API... (started at {time.strftime('%H:%M:%S')})")
            
            return vllm_omni_chat_request(
                api_base_url=self.config.api_url,
                model=self.config.model,
                prompt=prompt,
                max_tokens=_load_meta_max_tokens_default(),
                temperature=self.config.temperature,
                timeout=final_timeout,
            )

        try:
            raw_response = retry_api_call(make_request, max_retries=20)
            # CRITICAL: Strip thinking from final paragraph response
            raw = strip_thinking_from_response(raw_response)
            
            # Log to agent_reasoning.log (captures FULL raw response with <think> tags)
            if self.reasoning_logger:
                self.reasoning_logger.log_final_paragraph(
                    prompt=prompt,
                    raw_response=raw_response,
                    cleaned_response=raw
                )
        except Exception as e:
            # Fallback: synthesize from meta_text if request fails
            raw = ""

        # Clean and enforce format - CRITICAL: validate against extracted_rating
        def clean_paragraph(s: str, meta: str, required_rating: str) -> str:
            if not s:
                s = ""
            
            # Strip think tags and instruction artifacts
            s = re.sub(r"</?think[^>]*>", "", s, flags=re.IGNORECASE).strip()
            # Remove common instruction leakage patterns
            s = re.sub(r'\(no brackets[^\)]*\)', "", s, flags=re.IGNORECASE).strip()
            s = re.sub(r'- Then [0-9]+-[0-9]+ sentences.*', "", s, flags=re.IGNORECASE).strip()
            s = re.sub(r'Absolutely no bullet points.*', "", s, flags=re.IGNORECASE).strip()
            s = re.sub(r'\[rating\]', "", s, flags=re.IGNORECASE).strip()
            s = re.sub(r'"?\s*followed by.*', "", s, flags=re.IGNORECASE).strip()  # Remove instruction leakage
            # Remove leading quotes/artifacts
            s = s.lstrip('"').lstrip("'").strip()
            # Remove trailing quotes/artifacts
            s = s.rstrip('"').rstrip("'").strip()
            
            # Check if model output literal placeholder text or instruction fragments
            if s:
                placeholder_patterns = [
                    r'final\s*tgi\s*rating:\s*WORD\.?',
                    r'final\s*tgi\s*rating:\s*\[that single word\]\.?',
                    r'final\s*tgi\s*rating:\s*\[Normal\|Minimal\|Mild\|Moderate\|Marked\|Severe\|Extreme\]\.?',
                    r'final\s*tgi\s*rating:\s*\[.*?\]\.?',
                    r'final\s*tgi\s*rating:\s*the rating\.?',
                    r'final\s*tgi\s*rating:\s*\[rating\]\.?',
                    r'final\s*tgi\s*rating:\s*"\s*followed by',  # Instruction leakage
                ]
                for pattern in placeholder_patterns:
                    if re.search(pattern, s, flags=re.IGNORECASE):
                        logging.getLogger("segment_analysis").warning(
                            f"Model output placeholder text, forcing extraction from meta-analysis. Output was: {s}"
                        )
                        s = ""  # Force fallback to meta extraction
                        break
            
            # Check if output already starts with "MSE Severity Rating:" or legacy "Final TGI Rating:"
            if s.lower().startswith("mse severity rating:") or s.lower().startswith("final tgi rating:"):
                idx = s.lower().find("mse severity rating:")
                if idx == -1:
                    idx = s.lower().find("final tgi rating:")
                s = s[idx:]
                # Collapse to single paragraph, stop at next heading or blank block
                s = s.split("\n=== ")[0].split("\n\n")[0].strip()
            else:
                # Model likely just output the rating word - extract it and wrap it
                valid_ratings = ["Normal", "Minimal", "Mild", "Moderate", "Marked", "Severe", "Extreme"]
                # Try to find a valid rating word in the output
                rating_found = None
                for rating in valid_ratings:
                    if re.search(r'\b' + rating + r'\b', s, flags=re.IGNORECASE):
                        rating_found = rating
                        break
                
                if rating_found:
                    # Model output just the rating - wrap it properly
                    s = f"MSE Severity Rating: {rating_found}."
                    logging.getLogger("segment_analysis").info(
                        f"Model output raw rating '{rating_found}', wrapped as: {s}"
                    )
                else:
                    # No valid rating in output - force extraction from meta
                    s = ""
            
            if not s or (not s.lower().startswith("mse severity rating:") and not s.lower().startswith("final tgi rating:")):
                # Extract rating from meta_text (use LAST match to skip thinking text)
                rating = None
                for _pat, _flg in [
                    (r'"rating"\s*:\s*"(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)"', re.IGNORECASE),
                    (r"^\s*-\s*RATING:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b", re.IGNORECASE | re.MULTILINE),
                    (r"Final\s+rating:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b", re.IGNORECASE),
                    (r"(?:mse\s*severity|final\s*tgi)\s*rating:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b", re.IGNORECASE),
                ]:
                    _hits = re.findall(_pat, meta or "", flags=_flg)
                    if _hits:
                        rating = _hits[-1].capitalize()
                        break
                incr = []
                decr = []
                # Grab up to two bullets from evidence sections
                def pick(header: str) -> list[str]:
                    lines = (meta or "").splitlines()
                    start = None
                    items = []
                    for i, ln in enumerate(lines):
                        if ln.strip().upper().startswith(header.upper()):
                            start = i + 1
                            break
                    if start is None:
                        return items
                    for j in range(start, len(lines)):
                        t = lines[j].strip()
                        if not t or t.startswith("===") or t.endswith(":"):
                            break
                        if t.startswith("-"):
                            it = t.lstrip("-").strip()
                            if it:
                                items.append(it)
                                if len(items) >= 2:
                                    break
                        else:
                            break
                    return items
                # If we need to synthesize, just output the rating sentence - no evidence
                if rating:
                    s = f"MSE Severity Rating: {rating}."
            
            # Final validation: ensure the paragraph contains a valid rating
            valid_ratings = ["Normal", "Minimal", "Mild", "Moderate", "Marked", "Severe", "Extreme"]
            if s and (s.lower().startswith("mse severity rating:") or s.lower().startswith("final tgi rating:")):
                # Extract the rating value to verify it's valid
                rating_match = re.search(r"(?:mse\s*severity|final\s*tgi)\s*rating:\s*(Normal|Minimal|Mild|Moderate|Marked|Severe|Extreme)\b", 
                                        s, flags=re.IGNORECASE)
                if rating_match:
                    # Normalize capitalization
                    output_rating = rating_match.group(1).capitalize()
                    if output_rating not in valid_ratings:
                        # This should never happen but fallback to meta if it does
                        logging.getLogger("segment_analysis").warning(
                            f"Invalid rating '{output_rating}' extracted, forcing required rating '{required_rating}'"
                        )
                        # Force required rating
                        s = re.sub(r"(?:mse\s*severity|final\s*tgi)\s*rating:\s*" + re.escape(output_rating), 
                                  f"MSE Severity Rating: {required_rating}", s, flags=re.IGNORECASE)
                    elif output_rating != required_rating:
                        # CRITICAL: Output rating doesn't match required rating - force consistency
                        logging.getLogger("segment_analysis").error(
                            f"RATING MISMATCH DETECTED: Output has '{output_rating}' but meta-analysis requires '{required_rating}'. "
                            f"Forcing consistency by replacing with required rating."
                        )
                        # Replace the rating in the output with the required rating
                        s = re.sub(r"(?:mse\s*severity|final\s*tgi)\s*rating:\s*" + re.escape(output_rating), 
                                  f"MSE Severity Rating: {required_rating}", s, flags=re.IGNORECASE)
                else:
                    # No valid rating found in output - force required rating
                    s = f"MSE Severity Rating: {required_rating}."
                    logging.getLogger("segment_analysis").warning(
                        f"No valid rating found in output, forcing required rating '{required_rating}'"
                    )
            
            # If still empty or invalid, force required rating
            if not s or (not s.lower().startswith("mse severity rating:") and not s.lower().startswith("final tgi rating:")):
                # Emergency fallback: use required rating
                s = f"MSE Severity Rating: {required_rating}."
                logging.getLogger("segment_analysis").warning(
                    f"Final paragraph was malformed, forcing required rating '{required_rating}' from meta-analysis"
                )
            
            # Final consistency check: ensure output starts with required rating
            if not s.lower().startswith(f"mse severity rating: {required_rating.lower()}") and not s.lower().startswith(f"final tgi rating: {required_rating.lower()}"):
                # Force it to start with required rating
                s = f"MSE Severity Rating: {required_rating}."
                logging.getLogger("segment_analysis").error(
                    f"Final consistency check failed, forcing output to: {s}"
                )
            
            return s

        return clean_paragraph(raw, meta_text, extracted_rating)

# ============================================================================
# RESULTS MANAGEMENT
# ============================================================================

class ResultsManager:
    """Manages analysis results and report generation"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def save_results(
        self, 
        video_path: str,
        segments: List[Segment],
        final_diagnosis: str,
        processing_time: float
    ):
        """
        Save complete analysis results
        
        Args:
            video_path: Original video path
            segments: Analyzed segments
            final_diagnosis: Final diagnostic synthesis
            processing_time: Total processing time in seconds
        """
        # Create output directory
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        
        # Save complete report
        report_path = os.path.join(
            output_dir, 
            f"analysis_report_{video_name}_{timestamp}.txt"
        )
        self._save_complete_report(
            report_path, 
            video_path, 
            segments, 
            final_diagnosis,
            processing_time
        )
        
        # Save JSON format for programmatic access
        json_path = os.path.join(
            output_dir,
            f"analysis_data_{video_name}_{timestamp}.json"
        )
        self._save_json_data(
            json_path,
            video_path,
            segments,
            final_diagnosis,
            processing_time
        )
        
        # Save rolling context as separate .txt file
        rolling_context_path = os.path.join(
            output_dir,
            f"rolling_context_{video_name}_{timestamp}.txt"
        )
        self._save_rolling_context(
            rolling_context_path,
            segments,
            video_path
        )
        
        if self.config.verbose:
            print(f"\nResults saved:")
            print(f"  Report: {report_path}")
            print(f"  Data:   {json_path}")
            print(f"  Rolling Context: {rolling_context_path}")
    
    def _save_rolling_context(
        self,
        rolling_context_path: str,
        segments: List[Segment],
        video_path: str
    ):
        """Save rolling context for all segments to a text file"""
        with open(rolling_context_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ROLLING CONTEXT FOR ALL SEGMENTS\n")
            f.write("Qwen3-Omni - Video Segmentation with Rolling Context\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Input File: {video_path}\n\n")
            
            for segment in segments:
                f.write(f"SEGMENT {segment.index + 1} ({segment.timeframe_str})\n")
                f.write("-" * 80 + "\n")
                if segment.rolling_context:
                    f.write("Rolling Context Used:\n")
                    f.write(segment.rolling_context)
                else:
                    f.write("No rolling context (first segment)\n")
                f.write("\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF ROLLING CONTEXT\n")
            f.write("=" * 80 + "\n")
    
    def save_final_summary_json(
        self,
        video_path: str,
        segments: List[Segment],
        decision_trace_summary: str,
        final_diagnosis_paragraph: str
    ) -> str:
        """Save final decision-trace JSON named as <video_stem>_final.json in output_dir."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        stem = Path(video_path).stem
        final_json_path = os.path.join(self.config.output_dir, f"{stem}_final.json")
        payload = {
            "input_file": str(video_path),
            "video_name": Path(video_path).name,
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "segments_count": len(segments),
            "mode": "audio+video",
            "timepoint": self.config.timepoint,  # LONGITUDINAL: Include timepoint
            "decision_trace_summary": decision_trace_summary,
            "final_diagnosis_paragraph": final_diagnosis_paragraph,
        }
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return final_json_path
    
    def save_next_visit_summary(
        self,
        video_path: str,
        meta_analysis_text: str
    ) -> Optional[str]:
        """
        LONGITUDINAL: Extract and save summary_for_next_visit_context from meta-analysis output.
        
        This file is used as --prior-visit-summary input for the next visit's analysis.
        
        Args:
            video_path: Path to the analyzed video
            meta_analysis_text: Raw meta-analysis output text (may contain JSON)
            
        Returns:
            Path to the saved summary file, or None if extraction failed
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        stem = Path(video_path).stem
        summary_path = os.path.join(self.config.output_dir, f"{stem}_next_visit_summary.txt")
        
        # Try to extract summary_for_next_visit_context from JSON in meta-analysis
        summary_text = None
        
        # Method 1: Look for JSON with summary_for_next_visit_context key
        try:
            # Find JSON in the output
            first_brace = meta_analysis_text.find('{')
            last_brace = meta_analysis_text.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                potential_json = meta_analysis_text[first_brace:last_brace + 1]
                parsed = json.loads(potential_json)
                
                if isinstance(parsed, dict):
                    # Look for the summary key (exact match or variations)
                    summary_keys = [
                        'summary_for_next_visit_context',
                        'summary_for_next_visit',
                        'next_visit_summary',
                        'clinical_handoff',
                        'trajectory_summary'
                    ]
                    
                    for key in summary_keys:
                        if key in parsed and parsed[key]:
                            summary_text = str(parsed[key]).strip()
                            logging.getLogger("segment_analysis").info(
                                "Extracted next visit summary from key '%s' (%d chars)",
                                key, len(summary_text)
                            )
                            break
        except (json.JSONDecodeError, ValueError) as e:
            logging.getLogger("segment_analysis").debug(
                "Could not parse JSON from meta-analysis: %s", str(e)[:100]
            )
        
        # Method 2: If no JSON found, use regex to find the summary section.
        # Use LAST match (findall[-1]) because the model's <think> reasoning may
        # mention "summary_for_next_visit_context" as a planning note before the
        # actual structured output. The real value is always the last occurrence.
        if not summary_text:
            import re
            patterns = [
                r'"summary_for_next_visit_context"\s*:\s*"([^"]+)"',
                r'"summary_for_next_visit_context"\s*:\s*\'([^\']+)\'',
                r'summary_for_next_visit_context[:\s]+([^\n]+)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, meta_analysis_text, re.IGNORECASE)
                if matches:
                    summary_text = matches[-1].strip().strip('"').strip("'")
                    logging.getLogger("segment_analysis").info(
                        "Extracted next visit summary via regex (%d chars, match %d/%d)",
                        len(summary_text), len(matches), len(matches)
                    )
                    break
        
        # Method 3: If still no summary, use a fallback - extract first 2-3 sentences
        if not summary_text:
            # Look for trajectory_analysis or diagnostic_impression as fallback
            try:
                first_brace = meta_analysis_text.find('{')
                last_brace = meta_analysis_text.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    potential_json = meta_analysis_text[first_brace:last_brace + 1]
                    parsed = json.loads(potential_json)
                    
                    if isinstance(parsed, dict):
                        parts = []
                        if 'diagnostic_impression' in parsed:
                            parts.append(f"Diagnosis: {parsed['diagnostic_impression']}")
                        if 'trajectory_analysis' in parsed:
                            parts.append(f"Trajectory: {parsed['trajectory_analysis']}")
                        if 'FINAL_MSE_CHECKLIST' in parsed and isinstance(parsed['FINAL_MSE_CHECKLIST'], list):
                            items = parsed['FINAL_MSE_CHECKLIST'][:5]  # First 5 items
                            parts.append(f"Key findings: {', '.join(items)}")
                        
                        if parts:
                            summary_text = " | ".join(parts)
                            logging.getLogger("segment_analysis").info(
                                "Generated fallback next visit summary from meta fields (%d chars)",
                                len(summary_text)
                            )
            except Exception as e:
                logging.getLogger("segment_analysis").debug(
                    "Fallback extraction failed: %s", str(e)[:100]
                )
        
        # Save the summary
        if summary_text:
            # Add metadata header
            header = (
                f"# Next Visit Summary for {Path(video_path).name}\n"
                f"# Input File: {video_path}\n"
                f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# Timepoint: T{self.config.timepoint}\n"
                f"# Use this file as --prior-visit-summary for T{self.config.timepoint + 1}\n"
                f"#\n"
                f"# === SUMMARY ===\n\n"
            )
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(header + summary_text)
            
            logging.getLogger("segment_analysis").info(
                "Saved next visit summary to %s (%d chars)",
                summary_path, len(summary_text)
            )
            
            if self.config.verbose:
                print(f"  ✓ Next visit summary saved: {summary_path}")
            
            return summary_path
        else:
            logging.getLogger("segment_analysis").warning(
                "Could not extract summary_for_next_visit_context from meta-analysis output. "
                "The next visit will need manual summary creation."
            )
            
            # Write a placeholder file with the full meta-analysis for manual extraction
            placeholder_path = os.path.join(self.config.output_dir, f"{stem}_next_visit_summary_MANUAL.txt")
            with open(placeholder_path, 'w', encoding='utf-8') as f:
                f.write(
                    f"# MANUAL EXTRACTION REQUIRED\n"
                    f"# Input File: {video_path}\n"
                    f"# Could not automatically extract summary_for_next_visit_context\n"
                    f"# Please review the meta-analysis output below and create a summary manually.\n"
                    f"# Save the summary to: {summary_path}\n"
                    f"#\n"
                    f"# === FULL META-ANALYSIS OUTPUT ===\n\n"
                    f"{meta_analysis_text}"
                )
            
            if self.config.verbose:
                print(f"  ⚠ Next visit summary could not be auto-extracted.")
                print(f"    Manual extraction needed: {placeholder_path}")
            
            return None
    
    def _save_complete_report(
        self,
        report_path: str,
        video_path: str,
        segments: List[Segment],
        final_diagnosis: str,
        processing_time: float
    ):
        """Save human-readable text report"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CLINICAL AUDIO-VIDEO ANALYSIS REPORT\n")
            f.write("Qwen3-Omni - Video Segmentation with Rolling Context\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write(f"Input File: {video_path}\n")
            f.write(f"Video: {Path(video_path).name}\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Mode: Audio + Video\n")
            f.write(f"Total Segments: {len(segments)}\n")
            f.write(f"Segment Duration: {self.config.segment_duration}s\n")
            f.write(f"Segment Overlap: {self.config.segment_overlap}s\n")
            if self.config.rolling_context_depth is None or self.config.rolling_context_depth < 0:
                context_desc = "all prior segments"
            else:
                context_desc = f"{self.config.rolling_context_depth} segment(s)"
            f.write(f"Rolling Context Depth: {context_desc}\n")
            f.write(f"Processing Time: {timedelta(seconds=int(processing_time))}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Final Diagnosis
            f.write("FINAL DIAGNOSTIC SYNTHESIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(final_diagnosis)
            f.write("\n\n" + "=" * 80 + "\n\n")
            
            # Detailed Segment Observations
            f.write("DETAILED SEGMENT OBSERVATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for segment in segments:
                f.write(f"Segment {segment.index + 1}: {segment.timeframe_str}\n")
                f.write("-" * 80 + "\n")
                if segment.observations:
                    f.write(segment.observations)
                else:
                    f.write("[No observations recorded]")
                f.write("\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
    
    def _save_json_data(
        self,
        json_path: str,
        video_path: str,
        segments: List[Segment],
        final_diagnosis: str,
        processing_time: float
    ):
        """Save structured JSON data"""
        data = {
            "input_file": str(video_path),
            "metadata": {
                "video_path": str(video_path),
                "video_name": Path(video_path).name,
                "analysis_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "processing_time_seconds": processing_time,
                "configuration": {
                    "segment_duration": self.config.segment_duration,
                    "segment_overlap": self.config.segment_overlap,
                    "rolling_context_depth": self.config.rolling_context_depth,
                    "mode": "audio+video",
                    "clinical_focus": self.config.clinical_focus
                }
            },
            "final_diagnosis": final_diagnosis,
            "segments": [
                {
                    "index": seg.index,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "timeframe": seg.timeframe_str,
                    "observations": seg.observations,
                    "rolling_context": seg.rolling_context
                }
                for seg in segments
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

class ClinicalAudioAnalyzer:
    """Main orchestrator for clinical audio-video analysis"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.segmenter = AudioSegmenter(config)
        # Analyzers will have reasoning_logger set in analyze() method
        self.analyzer = SegmentAnalyzer(config)
        self.meta_analyzer = MetaAnalyzer(config)
        self.results_manager = ResultsManager(config)
    
    def analyze(self, video_path: str) -> Dict:
        """
        Perform complete analysis of clinical video (with audio)
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        # Initialize logging (unique file per run)
        log_path = self._setup_logging(self.config.output_dir, video_path)
        
        # Initialize agent reasoning logger (captures ALL model reasoning including <think> tags)
        reasoning_logger = AgentReasoningLogger(self.config.output_dir, video_path)
        self.analyzer.reasoning_logger = reasoning_logger
        self.meta_analyzer.reasoning_logger = reasoning_logger
        
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("CLINICAL AUDIO-VIDEO ANALYSIS - LONGITUDINAL")
            print("Video Segmentation with Sliding Window + Rolling Context")
            print("Cross-Visit History via Prior Visit Summary")
            print("=" * 70 + "\n")
            # Log the exact execution command to file and console
            exec_cmd = " ".join(shlex.quote(a) for a in sys.argv)
            logging.getLogger("segment_analysis").info("Execution command: %s", exec_cmd)
            print(f"Input video: {video_path}")
            print(f"Analysis mode: Audio + Video")
            print(f"Timepoint: T{self.config.timepoint}")
            if self.config.timepoint > 0:
                print(f"Prior visit summary: {self.config.prior_visit_summary_file}")
            else:
                print(f"Prior visit summary: N/A (baseline visit)")
            print(f"API endpoint: {self.config.api_url}")
            print(f"Logs: {log_path}")
        
        logging.getLogger("segment_analysis").info(
            "Starting analysis | video=%s | api=%s | mode=audio+video | segment_duration=%ss | overlap=%ss | context_depth=%s",
            video_path,
            self.config.api_url,
            self.config.segment_duration,
            self.config.segment_overlap,
            self.config.rolling_context_depth,
        )
        
        # Check API health
        self._check_api_health()
        
        # Create segments directory (purge stale segments from prior runs first)
        segments_dir = os.path.join(self.config.output_dir, "segments")
        if os.path.isdir(segments_dir):
            import shutil
            shutil.rmtree(segments_dir)
            if self.config.verbose:
                print(f"  Cleared stale segments from prior run")
        
        # Step 1: Segment the video
        if self.config.verbose:
            print(f"\nStep 1: Segmenting video...")
        segments = self.segmenter.create_segments(video_path, segments_dir)
        
        # CRITICAL: Validate that at least one segment was created
        if not segments:
            raise RuntimeError(
                f"No segments created from video. The video may be too short (< 5 seconds) "
                f"or there was a segmentation error. "
                f"Video path: {video_path}\n"
                f"Segment duration: {self.config.segment_duration}s\n"
                f"Segment overlap: {self.config.segment_overlap}s\n"
                f"Please check that the video file is valid and has sufficient duration."
            )
        
        if self.config.verbose:
            print(f"  ✓ Created {len(segments)} segments")
        
        # Step 2: Analyze segments with rolling context
        segments = self.analyzer.analyze_segments(segments)
        
        # Step 3: Perform meta-analysis
        # CRITICAL: Add a small synchronization delay before meta-analysis to allow other surveys
        # to finish segment analysis. This ensures more surveys hit meta-analysis simultaneously,
        # creating more concurrent requests and maintaining higher GPU utilization.
        # The delay is adaptive: if surveys finish at different times, this helps synchronize them.
        if self.config.verbose:
            print(f"\nStep 2 complete. Synchronizing before meta-analysis to maximize GPU utilization...")
        time.sleep(2.0)  # 2 second delay to allow other surveys to catch up
        
        final_diagnosis = self.meta_analyzer.synthesize_diagnosis(segments, video_path)
        
        # LONGITUDINAL: Save next visit summary IMMEDIATELY after meta-analysis
        # This ensures the summary is available even if later steps fail
        next_visit_summary_path = self.results_manager.save_next_visit_summary(
            video_path, final_diagnosis
        )
        
        # Step 3b & 3c: Run final_paragraph and decision_trace in PARALLEL since they're independent
        # (Only if decision_trace_prompt is available)
        # Both only depend on final_diagnosis, so they can execute concurrently for better GPU utilization
        import threading
        
        final_paragraph_result = {'value': None, 'error': None}
        decision_trace_result = {'summary': None, 'paragraph': None, 'error': None}
        
        # Check if decision_trace_prompt is available (optional for longitudinal prompts)
        has_decision_trace = self.config.get_prompt_template('decision_trace_prompt') is not None
        
        def run_final_paragraph():
            try:
                final_paragraph_result['value'] = self.meta_analyzer.synthesize_final_paragraph(
                    final_diagnosis, video_path, segments
                )
            except Exception as e:
                final_paragraph_result['error'] = e
        
        def run_decision_trace():
            if not has_decision_trace:
                # Skip decision trace if template not available
                decision_trace_result['summary'] = "[Decision trace not available - template missing from prompt YAML]"
                decision_trace_result['paragraph'] = None
                return
            try:
                decision_trace_summary, dt_paragraph = self.meta_analyzer.synthesize_decision_trace(
                    segments, final_diagnosis, video_path
                )
                decision_trace_result['summary'] = decision_trace_summary
                decision_trace_result['paragraph'] = dt_paragraph
            except Exception as e:
                decision_trace_result['error'] = e
        
        # Run both in parallel threads
        thread1 = threading.Thread(target=run_final_paragraph, daemon=True)
        thread2 = threading.Thread(target=run_decision_trace, daemon=True)
        
        thread1.start()
        thread2.start()
        
        # Wait for both to complete
        thread1.join()
        thread2.join()
        
        # Extract results - handle errors gracefully for longitudinal prompts
        if final_paragraph_result['error']:
            logging.getLogger("segment_analysis").warning(
                "Final paragraph synthesis failed: %s. Using meta-analysis output directly.",
                final_paragraph_result['error']
            )
            # Use first paragraph of meta-analysis as fallback
            final_paragraph = final_diagnosis.split('\n\n')[0] if final_diagnosis else ""
        else:
            final_paragraph = final_paragraph_result['value']
        
        if decision_trace_result['error']:
            logging.getLogger("segment_analysis").warning(
                "Decision trace synthesis failed: %s. Skipping decision trace.",
                decision_trace_result['error']
            )
            decision_trace_summary = "[Decision trace failed - see logs]"
        else:
            decision_trace_summary = decision_trace_result['summary'] or ""
        
        # Fallback: still parse decision-trace if final_paragraph is empty
        if not final_paragraph:
            if decision_trace_result['paragraph']:
                final_paragraph = decision_trace_result['paragraph']
            elif has_decision_trace:
                # Need to call decision_trace again to get paragraph
                try:
                    _, dt_par = self.meta_analyzer.synthesize_decision_trace(
                        segments, final_diagnosis, video_path
                    )
                    final_paragraph = dt_par
                except Exception as e:
                    logging.getLogger("segment_analysis").warning(
                        "Fallback decision trace failed: %s", e
                    )
                    final_paragraph = final_diagnosis.split('\n\n')[0] if final_diagnosis else ""
        
        # Calculate processing time
        processing_time = time.time() - start_time
        # Log total processing time in seconds
        logging.getLogger("segment_analysis").info("Total processing time (seconds): %.3f", processing_time)
        
        # Step 4: Save results
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("SAVING RESULTS")
            print(f"{'='*70}\n")
        
        self.results_manager.save_results(
            video_path,
            segments,
            final_diagnosis,
            processing_time
        )
        # Save final summary JSON named <video_stem>_final.json in output_dir
        final_json = self.results_manager.save_final_summary_json(
            video_path=video_path,
            segments=segments,
            decision_trace_summary=decision_trace_summary,
            final_diagnosis_paragraph=final_paragraph,
        )
        
        # Save agent reasoning log (ALL model reasoning with <think> tags)
        reasoning_log_path = reasoning_logger.save()
        if self.config.verbose:
            print(f"  Agent Reasoning Log: {reasoning_log_path}")
        
        # Save MSE guide response (observational questions generated for this MSE domain)
        mse_guide_path = self.analyzer.save_mse_guide_response(self.config.output_dir, video_path)
        if mse_guide_path and self.config.verbose:
            print(f"  MSE Guide Questions: {mse_guide_path}")
        
        # Save MSE guide reasoning log (full model reasoning process)
        mse_reasoning_path = self.analyzer.save_mse_guide_reasoning(self.config.output_dir, video_path)
        if mse_reasoning_path and self.config.verbose:
            print(f"  MSE Guide Reasoning: {mse_reasoning_path}")

        # Segments are always preserved for auditability (no cleanup)
        
        # Final summary
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("LONGITUDINAL ANALYSIS COMPLETE")
            print(f"{'='*70}")
            print(f"Timepoint: T{self.config.timepoint}")
            print(f"Total processing time: {timedelta(seconds=int(processing_time))}")
            print(f"Segments analyzed: {len(segments)}")
            print(f"Results directory: {self.config.output_dir}")
            print(f"Final summary JSON: {final_json}")
            print(f"Agent Reasoning Log: {reasoning_log_path}")
            if next_visit_summary_path:
                print(f"Next visit summary: {next_visit_summary_path}")
                print(f"  → Use this file as --prior-visit-summary for T{self.config.timepoint + 1}")
            print("=" * 70 + "\n")
        
        return {
            'success': True,
            'video_path': video_path,
            'segments_count': len(segments),
            'processing_time': processing_time,
            'final_diagnosis': final_diagnosis,
            'output_dir': self.config.output_dir,
            'timepoint': self.config.timepoint,
            'next_visit_summary_path': next_visit_summary_path,
            'reasoning_log_path': reasoning_log_path
        }
    
    def _setup_logging(self, output_dir: str, video_path: str) -> str:
        """Configure logging to console and unique file per run.
        
        Removes stale timestamped output files (prompt logs, analysis data/reports,
        rolling context) from prior incomplete runs before creating new ones.
        """
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        
        # Remove stale output files from prior runs (they accumulate on retries)
        import glob as _glob
        stale_patterns = [
            f"qwen_prompts_{video_name}_*.log",
            f"analysis_data_{video_name}_*.json",
            f"analysis_report_{video_name}_*.txt",
            f"rolling_context_{video_name}_*.txt",
        ]
        for pattern in stale_patterns:
            for stale_file in _glob.glob(os.path.join(output_dir, pattern)):
                try:
                    os.remove(stale_file)
                except OSError:
                    pass
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(output_dir, f"qwen_prompts_{video_name}_{timestamp}.log")
        logger = logging.getLogger("segment_analysis")
        logger.setLevel(logging.INFO)
        # Reset handlers to ensure unique file per run
        for h in list(logger.handlers):
            logger.removeHandler(h)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
        return log_path
    
    def _check_api_health(self):
        """Check if vLLM-Omni API is accessible (with retry).
        
        vLLM /health returns 200 with an empty body when healthy.
        We also check /v1/models to confirm the model is loaded.
        """
        def make_health_check():
            response = requests.get(f"{self.config.api_url}/health", timeout=10)
            response.raise_for_status()
            return {"status": "healthy"}
        
        try:
            retry_api_call(make_health_check, max_retries=20, initial_delay=2.0)
            
            # Also verify model is served via /v1/models
            try:
                models_resp = requests.get(f"{self.config.api_url}/v1/models", timeout=10)
                models_resp.raise_for_status()
                models_data = models_resp.json()
                served_ids = [m.get("id") for m in models_data.get("data", [])]
                if self.config.verbose:
                    print(f"✓ API health check passed (vLLM-Omni)")
                    print(f"  Served models: {served_ids}")
            except Exception:
                if self.config.verbose:
                    print(f"✓ API health check passed (/health OK)")
            
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to API at {self.config.api_url} after multiple retries. "
                f"Please ensure the vLLM-Omni server is running:\n"
                f"  cd concurrency && ./run_server_blackwell.sh"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"API health check timeout at {self.config.api_url} after multiple retries. "
                f"The server may be starting up or overloaded."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"API health check failed after retries: {e}\n"
                f"Please verify the server is running at {self.config.api_url}"
            )
        except RuntimeError:
            raise
    
    def _cleanup_segments(self, segments_dir: str):
        """Remove segment files after processing"""
        try:
            import shutil
            shutil.rmtree(segments_dir)
            if self.config.verbose:
                print(f"  Cleaned up temporary segments directory")
        except Exception as e:
            print(f"  Warning: Could not cleanup segments: {e}")


# ============================================================================
# PARAMETER LOGGING
# ============================================================================

def _log_all_parameters(config: AnalysisConfig, segmenter: 'AudioSegmenter', output_dir: str, video_path: str) -> str:
    """
    Log ALL parameters used (client-side and server-side) to parameters_used.log in output_dir.
    
    This creates a comprehensive record of every configuration value used for reproducibility.
    
    Returns:
        Path to the parameters_used.log file
    """
    import json
    from datetime import datetime
    
    log_path = os.path.join(output_dir, "parameters_used.log")
    
    # Fetch server-side config (try /v1/models for vLLM-Omni, fallback to /config for legacy)
    server_config = {}
    server_config_error = None
    try:
        response = requests.get(f"{config.api_url}/v1/models", timeout=30)
        if response.status_code == 200:
            models_data = response.json()
            served_ids = [m.get("id") for m in models_data.get("data", [])]
            server_config = {"model": ", ".join(served_ids), "server_type": "vLLM-Omni"}
        else:
            server_config_error = f"HTTP {response.status_code}"
    except Exception as e:
        server_config_error = str(e)
    
    # Build comprehensive parameter log
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("QWEN3-OMNI LONGITUDINAL ANALYSIS - PARAMETERS USED\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Input File: {video_path}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write("\n")
        
        # =====================================================================
        # CLIENT-SIDE PARAMETERS
        # =====================================================================
        f.write("-" * 80 + "\n")
        f.write("CLIENT-SIDE PARAMETERS (from .env or defaults)\n")
        f.write("-" * 80 + "\n")
        f.write("\n")
        
        # Segmentation parameters
        f.write("[SEGMENTATION]\n")
        f.write(f"  SEGMENT_DURATION          = {config.segment_duration} seconds\n")
        f.write(f"  SEGMENT_OVERLAP           = {config.segment_overlap} seconds\n")
        f.write("\n")
        
        # Video preprocessing parameters (from segmenter)
        f.write("[VIDEO PREPROCESSING]\n")
        f.write(f"  SEGMENT_TARGET_FPS        = {segmenter.target_fps}\n")
        f.write(f"  SEGMENT_TARGET_WIDTH      = {segmenter.target_width} px\n")
        f.write(f"  SEGMENT_TARGET_HEIGHT     = {segmenter.target_height} px\n")
        f.write(f"  SEGMENT_TARGET_CRF        = {segmenter.video_crf}\n")
        f.write(f"  SEGMENT_VIDEO_ENCODER     = {segmenter.video_encoder}\n")
        f.write("\n")
        
        # Audio preprocessing parameters
        f.write("[AUDIO PREPROCESSING]\n")
        f.write(f"  SEGMENT_TARGET_SAMPLE_RATE = {segmenter.target_sample_rate} Hz\n")
        f.write(f"  SEGMENT_TARGET_CHANNELS    = {segmenter.target_channels}\n")
        f.write("\n")
        
        # Rolling context parameters
        f.write("[ROLLING CONTEXT]\n")
        f.write(f"  MAX_ROLLING_CONTEXT_CHARS = {_load_max_rolling_context_chars()}\n")
        f.write(f"  MAX_PROMPT_CHARS          = {_load_max_prompt_chars()}\n")
        f.write(f"  MAX_SEGMENT_OUTPUT_TOKENS = {_load_max_segment_output_tokens()}\n")
        f.write(f"  rolling_context_depth     = {config.rolling_context_depth}\n")
        f.write(f"  rolling_context_summary_trigger = {config.rolling_context_summary_trigger}\n")
        f.write(f"  rolling_context_summary_recent  = {config.rolling_context_summary_recent}\n")
        f.write(f"  rolling_context_summary_max_words = {config.rolling_context_summary_max_words}\n")
        f.write("\n")
        
        # API parameters
        f.write("[API CONFIGURATION]\n")
        f.write(f"  api_url                   = {config.api_url}\n")
        f.write(f"  model                     = {config.model}\n")
        f.write(f"  request_timeout           = {config.request_timeout} seconds\n")
        f.write(f"  temperature               = {config.temperature}\n")
        f.write(f"  META_MAX_TOKENS           = {_load_meta_max_tokens_default()}\n")
        f.write("\n")
        
        # Longitudinal study parameters
        f.write("[LONGITUDINAL STUDY]\n")
        f.write(f"  timepoint                 = T{config.timepoint}\n")
        f.write(f"  prior_visit_summary_file  = {config.prior_visit_summary_file or 'N/A (baseline visit)'}\n")
        f.write(f"  prior_visit_summary       = {('Loaded (' + str(len(config.prior_visit_summary or '')) + ' chars)') if config.prior_visit_summary else 'N/A'}\n")
        f.write("\n")
        
        # Prompt configuration
        f.write("[PROMPT CONFIGURATION]\n")
        f.write(f"  prompt_file               = {config.prompt_file}\n")
        f.write(f"  survey_question (MSE)     = {config.survey_question}\n")
        f.write("\n")
        
        # Output configuration
        f.write("[OUTPUT CONFIGURATION]\n")
        f.write(f"  output_dir                = {config.output_dir}\n")
        f.write(f"  save_segments             = {config.save_segments}\n")
        f.write(f"  verbose                   = {config.verbose}\n")
        f.write("\n")
        
        # Clinical focus areas
        f.write("[CLINICAL FOCUS AREAS]\n")
        if config.clinical_focus:
            for i, focus in enumerate(config.clinical_focus, 1):
                f.write(f"  {i}. {focus}\n")
        else:
            f.write("  (none specified)\n")
        f.write("\n")
        
        # =====================================================================
        # SERVER-SIDE PARAMETERS
        # =====================================================================
        f.write("-" * 80 + "\n")
        f.write("SERVER-SIDE PARAMETERS (from API /config endpoint)\n")
        f.write("-" * 80 + "\n")
        f.write("\n")
        
        if server_config_error:
            f.write(f"  [ERROR] Could not fetch server config: {server_config_error}\n")
        elif server_config:
            f.write("[MODEL]\n")
            f.write(f"  model                     = {server_config.get('model', 'N/A')}\n")
            f.write(f"  quantization              = {server_config.get('quantization', 'N/A')}\n")
            f.write("\n")
            
            f.write("[GPU/MEMORY]\n")
            f.write(f"  gpu_memory_utilization    = {server_config.get('gpu_memory_utilization', 'N/A')}\n")
            f.write(f"  tensor_parallel_size      = {server_config.get('tensor_parallel_size', 'N/A')}\n")
            f.write("\n")
            
            f.write("[VLLM ENGINE]\n")
            f.write(f"  max_model_len             = {server_config.get('max_model_len', 'N/A')}\n")
            f.write(f"  max_num_seqs              = {server_config.get('max_num_seqs', 'N/A')} ({server_config.get('max_num_seqs_source', 'N/A')})\n")
            f.write(f"  max_num_batched_tokens    = {server_config.get('max_num_batched_tokens', 'N/A')} ({server_config.get('max_num_batched_tokens_source', 'N/A')})\n")
            f.write(f"  enable_chunked_prefill    = {server_config.get('enable_chunked_prefill', 'N/A')}\n")
            f.write("\n")
            
            f.write("[SERVER SEGMENTATION DEFAULTS]\n")
            f.write(f"  segment_duration          = {server_config.get('segment_duration', 'N/A')} seconds\n")
            f.write(f"  segment_overlap           = {server_config.get('segment_overlap', 'N/A')} seconds\n")
            f.write("\n")
            
            # Log any additional server config keys not explicitly handled
            known_keys = {'model', 'quantization', 'gpu_memory_utilization', 'tensor_parallel_size',
                          'max_model_len', 'max_num_seqs', 'max_num_seqs_source',
                          'max_num_batched_tokens', 'max_num_batched_tokens_source',
                          'enable_chunked_prefill', 'segment_duration', 'segment_overlap'}
            extra_keys = set(server_config.keys()) - known_keys
            if extra_keys:
                f.write("[OTHER SERVER PARAMETERS]\n")
                for key in sorted(extra_keys):
                    f.write(f"  {key:27} = {server_config.get(key, 'N/A')}\n")
                f.write("\n")
        else:
            f.write("  (no server config available)\n")
        
        # =====================================================================
        # ENVIRONMENT VARIABLE SOURCES
        # =====================================================================
        f.write("-" * 80 + "\n")
        f.write("ENVIRONMENT VARIABLE RESOLUTION\n")
        f.write("-" * 80 + "\n")
        f.write("\n")
        f.write("Priority order for .env loading (highest wins):\n")
        f.write("  1. Shell environment variables (exported)\n")
        f.write("  2. {repo_root}/docker/.env.blackwell (Blackwell GPU overrides)\n")
        f.write("  3. {repo_root}/docker/.env (base defaults)\n")
        f.write("  4. {repo_root}/.env\n")
        f.write("  5. {cwd}/.env\n")
        f.write("  6. {cwd}/docker/.env\n")
        f.write("  7. Hardcoded defaults (ONLY if not found anywhere)\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF PARAMETERS LOG\n")
        f.write("=" * 80 + "\n")
    
    return log_path


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    # Load defaults for segment sizing from environment / .env
    _default_segment_duration, _default_segment_overlap = _load_segment_env_defaults()

    parser = argparse.ArgumentParser(
        description="Clinical Audio-Video Analysis using Qwen3-Omni - LONGITUDINAL VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
LONGITUDINAL STUDY EXAMPLES:

  =====================================================================
  RECOMMENDED: Using --base-output-dir (auto-constructs paths)
  =====================================================================
  
  # Baseline visit (T0) - auto-creates output folder structure
  python ./engines/clinician_audio_video_segment_analysis_client_longitudinal.py \\
    ./analysis_results/martin_et_al_another/ben/video_0/video_0.mov \\
    --prompt prompts/martin_et_al_another/prompt.yml \\
    --timepoint 0 \\
    --survey-question "Grooming and hygiene (abnormal)" \\
    --base-output-dir ./analysis_results/martin_et_al_another/ben
  
  # Output goes to: ./analysis_results/martin_et_al_another/ben/video_0/grooming_and_hygiene_abnormal/
  
  # Follow-up visit (T1) - auto-derives prior-visit-summary from T0
  python ./engines/clinician_audio_video_segment_analysis_client_longitudinal.py \\
    ./analysis_results/martin_et_al_another/ben/video_1/video_1.mov \\
    --prompt prompts/martin_et_al_another/prompt.yml \\
    --timepoint 1 \\
    --survey-question "Grooming and hygiene (abnormal)" \\
    --base-output-dir ./analysis_results/martin_et_al_another/ben
  
  # Output goes to: ./analysis_results/martin_et_al_another/ben/video_1/grooming_and_hygiene_abnormal/
  # Prior-visit-summary auto-derived from: .../ben/video_0/grooming_and_hygiene_abnormal/video_0_next_visit_summary.txt

  # Second follow-up (T2) - same pattern
  python ./engines/clinician_audio_video_segment_analysis_client_longitudinal.py \\
    ./analysis_results/martin_et_al_another/ben/video_2/video_2.mov \\
    --prompt prompts/martin_et_al_another/prompt.yml \\
    --timepoint 2 \\
    --survey-question "Grooming and hygiene (abnormal)" \\
    --base-output-dir ./analysis_results/martin_et_al_another/ben

  =====================================================================
  LEGACY: Manual --output-dir and --prior-visit-summary
  =====================================================================
  
  # Baseline visit (T0) - manual paths
  python ./engines/clinician_audio_video_segment_analysis_client_longitudinal.py \\
    patient_T0.mp4 \\
    --prompt prompts/martin_et_al_another/prompt.yml \\
    --timepoint 0 \\
    --survey-question "Grooming and hygiene (abnormal)" \\
    --output-dir ./results/patient_T0/grooming_and_hygiene_abnormal

  # Follow-up visit (T1) - manual paths
  python ./engines/clinician_audio_video_segment_analysis_client_longitudinal.py \\
    patient_T1.mp4 \\
    --prompt prompts/martin_et_al_another/prompt.yml \\
    --timepoint 1 \\
    --prior-visit-summary ./results/patient_T0/grooming_and_hygiene_abnormal/patient_T0_next_visit_summary.txt \\
    --survey-question "Grooming and hygiene (abnormal)" \\
    --output-dir ./results/patient_T1/grooming_and_hygiene_abnormal

FOLDER STRUCTURE (with --base-output-dir):
  <base-output-dir>/
    video_0/
      grooming_and_hygiene_abnormal/
        video_0_final.json
        video_0_next_visit_summary.txt    <-- used by T1
        segments/
        ...
      eye_contact_abnormal/
        video_0_final.json
        video_0_next_visit_summary.txt
        ...
    video_1/
      grooming_and_hygiene_abnormal/
        video_1_final.json
        video_1_next_visit_summary.txt    <-- used by T2
        ...
    video_2/
      grooming_and_hygiene_abnormal/
        video_2_final.json
        ...

SURVEY QUESTION TO FOLDER CONVERSION:
  "Grooming and hygiene (abnormal)"  --> grooming_and_hygiene_abnormal
  "Eye contact (abnormal)"           --> eye_contact_abnormal
  "Flat affect"                      --> flat_affect
  "Psychomotor retardation / bradykinesia" --> psychomotor_retardation_bradykinesia
  "Pressured speech"                 --> pressured_speech

MSE SIGN/SYMPTOM EXAMPLES (for --survey-question):
  - "Grooming and hygiene (abnormal)"
  - "Eye contact (abnormal)"
  - "Psychomotor retardation / bradykinesia"
  - "Psychomotor activation / akathisia"
  - "Pressured speech"
  - "Constricted range of affect, flat"
  - "Inappropriate and/or labile affect"
  - "Tremor"

OUTPUT FILES:
  - {video_stem}_final.json         : Final analysis with timepoint info
  - {video_stem}_next_visit_summary.txt : Summary for next visit (auto-used by next timepoint)
  - analysis_report_{video}_{timestamp}.txt : Full analysis report
  - analysis_data_{video}_{timestamp}.json  : Structured JSON data

WORKFLOW WITH --base-output-dir:
  1. Run T0 analysis with --base-output-dir (no --prior-visit-summary needed)
  2. T0 generates video_0_next_visit_summary.txt in the survey subfolder
  3. Run T1 with same --base-output-dir and --survey-question (auto-derives prior summary)
  4. Repeat for T2, T3, etc. - prior summaries are always auto-derived!
        """
    )
    
    parser.add_argument(
        'video',
        help='Path to input video file (with audio track)'
    )
    
    parser.add_argument(
        '--api-url',
        default='http://localhost:5100',
        help='vLLM-Omni API base URL (default: http://localhost:5100)'
    )
    
    parser.add_argument(
        '--model',
        default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        help='Model name served by vLLM-Omni (default: Qwen/Qwen3-Omni-30B-A3B-Thinking)'
    )
    
    # NOTE: Segment sizing is enforced from environment/.env only.
    # CLI overrides are intentionally not provided to ensure consistent configuration.
    
    parser.add_argument(
        '--context-depth',
        type=int,
        default=-1,
        help='Number of previous segments to include in rolling context (-1 = all prior segments, default: -1)'
    )
    
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for results. If --base-output-dir is used, this is ignored. '
             '(default: auto-constructed from base-output-dir/video_<timepoint>/<survey_folder>)'
    )
    
    parser.add_argument(
        '--base-output-dir',
        default=None,
        help='Base output directory for patient (e.g., ./analysis_results/martin_et_al_another/ben). '
             'When specified, output-dir is auto-constructed as: '
             '<base-output-dir>/video_<timepoint>/<survey_folder>/ '
             'and prior-visit-summary is auto-derived for timepoints > 0.'
    )
    
    parser.add_argument(
        '--keep-segments',
        action='store_true',
        help='Keep segment files after analysis (always kept; this flag is ignored)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (errors only)'
    )
    
    parser.add_argument(
        '--prompt',
        required=True,
        help='Path to YAML prompt file (REQUIRED, e.g., prompts/martin_et_al_another/prompt.yml)'
    )
    
    # LONGITUDINAL STUDY ARGUMENTS
    parser.add_argument(
        '--timepoint',
        type=int,
        required=True,
        help='Visit timepoint number (REQUIRED). 0 for T0/baseline, 1 for T1, 2 for T2, etc.'
    )
    parser.add_argument(
        '--prior-visit-summary',
        dest='prior_visit_summary',
        help='Path to .txt file containing summary from previous visit. '
             'REQUIRED if timepoint > 0. '
             'Use the _next_visit_summary.txt file from the previous timepoint analysis.'
    )
    
    parser.add_argument(
        '--survey-question',
        required=True,
        help='MSE sign/symptom presence question to assess (REQUIRED). '
             'This is the specific Mental Status Examination (MSE) item being evaluated, '
             'e.g., "Grooming and hygiene (abnormal)", "Eye contact (abnormal)", '
             '"Pressured speech", "Flat affect", "Psychomotor retardation". '
             'Injected into templates as {survey_question}.'
    )
    parser.add_argument(
        '--temperature', '--temparature',
        dest='temperature',
        type=float,
        help='Decoding temperature to send to the API (e.g., 0.0-1.0)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1200,
        help='Request timeout in seconds per segment (default: 1200 = 20 minutes). Meta-analysis uses 2x this value.'
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # AUTO-CONSTRUCT OUTPUT-DIR AND PRIOR-VISIT-SUMMARY FROM BASE-OUTPUT-DIR
    # =========================================================================
    # If base-output-dir is specified, automatically construct:
    #   - output-dir: <base-output-dir>/video_<timepoint>/<survey_folder>/
    #   - prior-visit-summary (for timepoint > 0): auto-derived from previous timepoint
    
    final_output_dir = args.output_dir or './analysis_results'
    final_prior_visit_summary = args.prior_visit_summary
    
    if args.base_output_dir:
        # =====================================================================
        # CRITICAL: Validate patient name consistency BEFORE proceeding
        # =====================================================================
        # Ensure the patient name in the input video path matches the patient
        # name in --base-output-dir to prevent data misorganization.
        try:
            validate_patient_name_consistency(args.video, args.base_output_dir)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        
        # Convert survey question to folder-safe name
        survey_folder = survey_question_to_folder_name(args.survey_question)
        
        # Construct output-dir: <base-output-dir>/video_<timepoint>/<survey_folder>/
        video_folder = f"video_{args.timepoint}"
        final_output_dir = os.path.join(args.base_output_dir, video_folder, survey_folder)
        
        print(f"\n📁 Auto-constructed output directory:")
        print(f"   Base: {args.base_output_dir}")
        print(f"   Video folder: {video_folder}")
        print(f"   Survey folder: {survey_folder}")
        print(f"   Full path: {final_output_dir}")
        
        # For timepoints > 0, auto-derive prior-visit-summary
        if args.timepoint > 0 and not args.prior_visit_summary:
            derived_prior_summary = derive_prior_visit_summary_path(
                base_output_dir=args.base_output_dir,
                current_timepoint=args.timepoint,
                survey_question=args.survey_question
            )
            
            if derived_prior_summary:
                if os.path.exists(derived_prior_summary):
                    final_prior_visit_summary = derived_prior_summary
                    print(f"\n📋 Auto-derived prior visit summary:")
                    print(f"   Path: {final_prior_visit_summary}")
                else:
                    # Fallback: use _MANUAL.txt when auto-extraction failed at prior timepoint
                    manual_path = derived_prior_summary.replace(
                        "_next_visit_summary.txt", "_next_visit_summary_MANUAL.txt"
                    )
                    if os.path.exists(manual_path):
                        final_prior_visit_summary = manual_path
                        print(f"\n📋 Using prior visit summary (MANUAL fallback):")
                        print(f"   Path: {final_prior_visit_summary}")
                    else:
                        print(f"\n⚠️  WARNING: Auto-derived prior visit summary not found!")
                        print(f"   Expected: {derived_prior_summary}")
                        print(f"   You may need to run timepoint {args.timepoint - 1} first.")
                        # Still set it so validation will fail with a clear error
                        final_prior_visit_summary = derived_prior_summary
    
    # Resolve prompt file path
    prompt_file = None
    if args.prompt:
        # If it's an absolute path or exists as-is, use it
        if os.path.isabs(args.prompt) or os.path.exists(args.prompt):
            prompt_file = args.prompt
        # Otherwise, try relative to ./prompts/ directory
        else:
            prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
            potential_path = os.path.join(prompts_dir, args.prompt)
            if os.path.exists(potential_path):
                prompt_file = potential_path
            else:
                # Try with .yml extension if not provided
                if not args.prompt.endswith('.yml') and not args.prompt.endswith('.yaml'):
                    potential_path_yml = os.path.join(prompts_dir, args.prompt + '.yml')
                    if os.path.exists(potential_path_yml):
                        prompt_file = potential_path_yml
                    else:
                        raise FileNotFoundError(
                            f"Prompt file not found: {args.prompt}\n"
                            f"Tried: {args.prompt}, {potential_path}, {potential_path_yml}"
                        )
                else:
                    raise FileNotFoundError(
                        f"Prompt file not found: {args.prompt}\n"
                        f"Tried: {args.prompt}, {potential_path}"
                    )
    
    # Create configuration
    # Enforce env-derived segment sizing
    _env_segment_duration, _env_segment_overlap = _default_segment_duration, _default_segment_overlap
    config = AnalysisConfig(
        api_url=args.api_url,
        model=args.model,
        request_timeout=args.timeout,
        segment_duration=_env_segment_duration,
        segment_overlap=_env_segment_overlap,
        rolling_context_depth=args.context_depth,
        use_transcription=False,  # Video mode doesn't use separate transcription
        transcription_file=None,
        prompt_file=prompt_file,
        survey_question=args.survey_question,
        temperature=args.temperature,
        # LONGITUDINAL PARAMETERS
        timepoint=args.timepoint,
        prior_visit_summary_file=final_prior_visit_summary,  # Use auto-derived if available
        output_dir=final_output_dir,  # Use auto-constructed if base-output-dir was used
        save_segments=args.keep_segments,
        verbose=not args.quiet
    )
    # Runtime re-assert to guarantee usage from env values
    config.segment_duration, config.segment_overlap = _load_segment_env_defaults()
    
    # Validate video file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    
    # Ensure output directory exists for parameter logging
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create segmenter to get preprocessing parameters for logging
    segmenter = AudioSegmenter(config)
    
    # Log ALL parameters to parameters_used.log BEFORE analysis starts
    params_log_path = _log_all_parameters(config, segmenter, config.output_dir, args.video)
    if not args.quiet:
        print(f"\n📋 Parameters logged to: {params_log_path}")
    
    # Run analysis
    try:
        analyzer = ClinicalAudioAnalyzer(config)
        result = analyzer.analyze(args.video)
        
        if not args.quiet:
            print("\n✓ Analysis completed successfully!")
            print(f"  See results in: {result['output_dir']}")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}", file=sys.stderr)
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

