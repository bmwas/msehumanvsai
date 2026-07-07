#!/usr/bin/env python3
"""
Run All Surveys Longitudinal - Batch Wrapper for Clinical Analysis

This script is a wrapper that executes clinician_audio_video_segment_analysis_client_longitudinal.py
across multiple patients and survey questions.

Survey questions are loaded from a CSV file with column 'mse_survey_question'.
Each row contains one survey question.

PARALLEL EXECUTION:
    By default, this script runs 5 survey questions in parallel within each timepoint.
    Survey questions within a timepoint are independent and can run concurrently.
    Timepoints are still processed sequentially (T0 → T1 → T2) as they depend on each other.
    
    Use --parallel-workers to control concurrency:
      - --parallel-workers 5  (default) Run 5 surveys in parallel
      - --parallel-workers 1  Serial mode (one at a time)

Usage:
    python run_all_surveys_longitudinal.py --patients patient1 patient2 --timepoint 0
    python run_all_surveys_longitudinal.py --patients patient1 --timepoint 0 --dry-run
    python run_all_surveys_longitudinal.py --patients patient1 patient2 --timepoint 1
    python run_all_surveys_longitudinal.py --patients patient1 patient2 --all-timepoints
    python run_all_surveys_longitudinal.py --patients patient1 --all-timepoints --parallel-workers 6

Arguments:
    --patients: List of patient names to process (required)
    --timepoint: Visit timepoint (default: 0) - mutually exclusive with --all-timepoints
    --all-timepoints: Auto-detect and run ALL timepoints sequentially (T0 → T1 → T2...)
    --parallel-workers: Number of parallel workers (default: 5, min: 1, max: 5)
    --dry-run: Print commands without executing them
    --mse-file: Path to MSE questions CSV file (default: ./analysis_results/martin_et_al_another/mse_questions.csv)
    --prompt: Path to prompt YAML file (default: prompts/martin_et_al_another/prompt.yml)
    --base-dir: Base directory for analysis results (default: ./analysis_results/martin_et_al_another)
    --delay-between-tasks: Delay in seconds between tasks (default: 2.0, only in serial mode)
    --continue-on-error: Continue processing remaining tasks if one fails

CSV File Format:
    The CSV file MUST have a column named 'mse_survey_question'.
    Example:
        mse_survey_question
        "Grooming and hygiene (abnormal)"
        "Eye contact (abnormal)"
        "Flat affect"

Author: Benson Mwangi
Date: 2025-12
"""

import argparse
import logging
import math
import os
import subprocess
import sys
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(verbose: bool = True, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging with both console and optional file handlers.
    
    Args:
        verbose: If True, set DEBUG level; otherwise INFO
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("batch_longitudinal")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Format with timestamp, level, and message
    console_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger: Optional[logging.Logger] = None

# Thread lock for thread-safe logging and file writes
_log_lock = threading.Lock()
_iteration_log_lock = threading.Lock()


# Console display: max width for single-line logs (fit typical terminal)
_LOG_LINE_WIDTH = 100
_MSE_DISPLAY_LEN = 44  # truncate long MSE/survey names for one-line display


def _mse_display(survey_question: str, max_len: int = _MSE_DISPLAY_LEN) -> str:
    """Truncate survey question for compact console display."""
    s = (survey_question or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def log_header(message: str, char: str = "=", width: int = 80):
    """Log a header message with decorative borders."""
    logger.info(char * width)
    logger.info(message.center(width))
    logger.info(char * width)


def log_subheader(message: str, char: str = "-", width: int = 70):
    """Log a subheader message."""
    logger.info(char * width)
    logger.info(message)
    logger.info(char * width)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TaskResult:
    """Result of a single analysis task"""
    patient: str
    survey_question: str
    timepoint: int
    success: bool
    duration_seconds: float
    error_message: Optional[str] = None
    output_dir: Optional[str] = None
    video_path: Optional[str] = None


@dataclass
class TimepointResult:
    """Result of processing all surveys for a single timepoint"""
    patient: str
    timepoint: int
    total_surveys: int
    successful_surveys: int
    failed_surveys: int
    duration_seconds: float
    task_results: List[TaskResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class PatientResult:
    """Result of processing all timepoints for a single patient"""
    patient: str
    timepoints_processed: List[int] = field(default_factory=list)
    timepoints_skipped: List[int] = field(default_factory=list)
    timepoint_results: List[TimepointResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0


# =============================================================================
# SURVEY QUESTION LOADING
# =============================================================================

# Required column name in the CSV file
MSE_SURVEY_QUESTION_COLUMN = "mse_survey_question"


def load_survey_questions(mse_csv_path: str) -> List[str]:
    """
    Load survey questions from MSE questions CSV file using pandas.
    
    The CSV file MUST have a column named 'mse_survey_question'.
    Each row contains one survey question.
    
    Args:
        mse_csv_path: Path to the MSE questions CSV file
        
    Returns:
        List of survey question strings
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the required column is missing or no valid questions found
    """
    if not os.path.exists(mse_csv_path):
        raise FileNotFoundError(f"MSE questions CSV file not found: {mse_csv_path}")
    
    # Read CSV using pandas
    try:
        df = pd.read_csv(mse_csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{mse_csv_path}': {e}")
    
    # Validate required column exists
    if MSE_SURVEY_QUESTION_COLUMN not in df.columns:
        available_columns = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Required column '{MSE_SURVEY_QUESTION_COLUMN}' not found in CSV file.\n"
            f"Available columns: {available_columns}"
        )
    
    # Extract survey questions from the column
    questions = []
    for idx, row in df.iterrows():
        question = row[MSE_SURVEY_QUESTION_COLUMN]
        
        # Skip NaN/None values
        if pd.isna(question):
            logger.warning(f"Row {idx + 1}: Skipping empty/NaN value in '{MSE_SURVEY_QUESTION_COLUMN}' column")
            continue
        
        # Convert to string and strip whitespace
        question_str = str(question).strip()
        
        # Skip empty strings
        if not question_str:
            logger.warning(f"Row {idx + 1}: Skipping empty string in '{MSE_SURVEY_QUESTION_COLUMN}' column")
            continue
        
        questions.append(question_str)
    
    if not questions:
        raise ValueError(
            f"No valid survey questions found in column '{MSE_SURVEY_QUESTION_COLUMN}' "
            f"of file '{mse_csv_path}'"
        )
    
    logger.info(f"Loaded {len(questions)} survey questions from {mse_csv_path}")
    logger.debug(f"CSV file had {len(df)} rows, {len(questions)} valid questions extracted")
    
    return questions


# =============================================================================
# TIMEPOINT DETECTION AND VALIDATION
# =============================================================================

def detect_patient_timepoints(base_dir: str, patient: str) -> List[int]:
    """
    Detect all available timepoints for a patient by scanning for video_N folders.
    
    Args:
        base_dir: Base directory (e.g., ./analysis_results/martin_et_al_another)
        patient: Patient name (e.g., ben)
        
    Returns:
        Sorted list of timepoint integers (e.g., [0, 1, 2])
    """
    patient_dir = os.path.join(base_dir, patient)
    
    if not os.path.isdir(patient_dir):
        logger.warning(f"Patient directory not found: {patient_dir}")
        return []
    
    timepoints = []
    
    # Scan for video_N directories
    try:
        for item in os.listdir(patient_dir):
            item_path = os.path.join(patient_dir, item)
            if os.path.isdir(item_path):
                # Match pattern video_N where N is a non-negative integer
                match = re.match(r'^video_(\d+)$', item)
                if match:
                    timepoint = int(match.group(1))
                    # Verify video file exists inside
                    video_file = os.path.join(item_path, f"video_{timepoint}.mov")
                    if os.path.isfile(video_file):
                        timepoints.append(timepoint)
                    else:
                        logger.debug(f"Skipping {item}: video file not found at {video_file}")
    except OSError as e:
        logger.error(f"Error scanning patient directory {patient_dir}: {e}")
        return []
    
    # Sort timepoints in ascending order (CRITICAL for sequential processing)
    timepoints.sort()
    
    return timepoints


def verify_prior_visit_summary_exists(
    base_dir: str,
    patient: str,
    current_timepoint: int,
    survey_question: str
) -> Tuple[bool, str, Optional[str]]:
    """
    Verify that the prior visit summary file exists for a given timepoint > 0.
    
    For T1, we need T0's summary. For T2, we need T1's summary, etc.
    
    Args:
        base_dir: Base directory
        patient: Patient name
        current_timepoint: Current timepoint (must be > 0)
        survey_question: Survey question to check
        
    Returns:
        Tuple of (exists: bool, expected_path: str, error_message: Optional[str])
    """
    if current_timepoint <= 0:
        # T0 doesn't need a prior summary
        return True, "", None
    
    prior_timepoint = current_timepoint - 1
    survey_folder = survey_question_to_folder_name(survey_question)
    
    # Expected path: <base>/<patient>/video_<N-1>/<survey_folder>/video_<N-1>_next_visit_summary.txt
    expected_path = os.path.join(
        base_dir,
        patient,
        f"video_{prior_timepoint}",
        survey_folder,
        f"video_{prior_timepoint}_next_visit_summary.txt"
    )
    
    if os.path.isfile(expected_path):
        return True, expected_path, None
    # Fallback: engine may have written _MANUAL.txt when auto-extraction failed
    manual_path = expected_path.replace("_next_visit_summary.txt", "_next_visit_summary_MANUAL.txt")
    if os.path.isfile(manual_path):
        return True, manual_path, None
    else:
        error_msg = (
            f"Prior visit summary NOT FOUND for T{current_timepoint}. "
            f"Expected: {expected_path}. "
            f"T{prior_timepoint} must complete successfully first."
        )
        return False, expected_path, error_msg


def validate_patient_video_exists(base_dir: str, patient: str, timepoint: int) -> Tuple[bool, str]:
    """
    Validate that the video file exists for a patient at a given timepoint.
    
    Args:
        base_dir: Base directory (e.g., ./analysis_results/martin_et_al_another)
        patient: Patient name (e.g., ben)
        timepoint: Timepoint number (e.g., 0)
        
    Returns:
        Tuple of (exists: bool, video_path: str)
    """
    video_path = os.path.join(
        base_dir, 
        patient, 
        f"video_{timepoint}", 
        f"video_{timepoint}.mov"
    )
    exists = os.path.exists(video_path)
    
    if exists:
        logger.debug(f"Video found: {video_path}")
    else:
        logger.warning(f"Video NOT found: {video_path}")
    
    return exists, video_path


# =============================================================================
# COMMAND BUILDING
# =============================================================================

def build_analysis_command(
    patient: str,
    survey_question: str,
    timepoint: int,
    base_dir: str,
    prompt_file: str,
    engine_script: str,
    api_url: str = "http://localhost:5100",
) -> Tuple[List[str], str, str]:
    """
    Build the command to execute the longitudinal analysis script.
    
    Args:
        patient: Patient name
        survey_question: Survey question to assess
        timepoint: Visit timepoint
        base_dir: Base directory for results
        prompt_file: Path to prompt YAML file
        engine_script: Path to the engine script
        
    Returns:
        Tuple of (command_list, video_path, base_output_dir)
    """
    # Construct paths
    video_path = os.path.join(
        base_dir,
        patient,
        f"video_{timepoint}",
        f"video_{timepoint}.mov"
    )
    
    base_output_dir = os.path.join(base_dir, patient)
    
    # Build command (API URL default 5100 to match run_server_blackwell.sh)
    cmd = [
        sys.executable,  # Use the same Python interpreter
        engine_script,
        video_path,
        "--prompt", prompt_file,
        "--timepoint", str(timepoint),
        "--survey-question", survey_question,
        "--base-output-dir", base_output_dir,
        "--api-url", api_url,
    ]
    
    return cmd, video_path, base_output_dir


def survey_question_to_folder_name(survey_question: str) -> str:
    """
    Convert survey question to folder name format.
    
    CRITICAL: This function MUST match the implementation in
    engines/clinician_audio_video_segment_analysis_client_longitudinal.py
    to ensure folder names are consistent between batch wrapper and engine.
    
    Examples:
        "Grooming and hygiene (abnormal)" -> "grooming_and_hygiene_abnormal"
        "Speech : Slowed/delayed" -> "speech_slowed_delayed"
        "Affect and Mood : Inappropriate and/or labile affect" -> "affect_and_mood_inappropriate_andor_labile_affect"
    """
    if not survey_question:
        return "unknown_survey"
    
    # Convert to lowercase
    name = survey_question.lower()
    
    # CRITICAL: Replace common separators with spaces FIRST (before removing special chars)
    # This ensures "Slowed/delayed" becomes "slowed delayed" not "sloweddelayed"
    name = name.replace('/', ' ')
    name = name.replace('-', ' ')
    
    # Remove parentheses (their contents become part of the name)
    name = name.replace('(', ' ')
    name = name.replace(')', ' ')
    
    # Remove other special characters (but NOT alphanumeric or whitespace)
    name = re.sub(r'[^\w\s]', '', name)
    
    # Replace multiple whitespace with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Strip and replace spaces with underscores
    name = name.strip().replace(' ', '_')
    
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Strip leading/trailing underscores
    name = name.strip('_')
    
    return name


# =============================================================================
# RUN COMPLETION DETECTION AND CLEANUP
# =============================================================================

def check_survey_completion(output_dir: str, timepoint: int) -> Tuple[bool, str]:
    """
    Check if a survey iteration was successfully completed.
    
    A successful completion is indicated by the presence of:
      - video_<timepoint>_final.json (THE KEY SUCCESS INDICATOR)
    
    This file is only created at the very end of the engine's processing,
    so its presence confirms the entire pipeline completed successfully.
    
    Args:
        output_dir: The survey-specific output directory
                   (e.g., <base>/patient/video_0/grooming_and_hygiene_abnormal/)
        timepoint: The timepoint number (0, 1, 2, etc.)
        
    Returns:
        Tuple of (is_complete: bool, final_json_path: str)
    """
    final_json_name = f"video_{timepoint}_final.json"
    final_json_path = os.path.join(output_dir, final_json_name)
    
    is_complete = os.path.isfile(final_json_path)
    
    return is_complete, final_json_path


def cleanup_incomplete_run(output_dir: str, timepoint: int, dry_run: bool = False) -> List[str]:
    """
    Clean up zombie files from an incomplete/failed run.
    
    SAFETY: This function ONLY removes files from the SPECIFIC output_dir.
    It NEVER touches parent directories, sibling directories, or other runs.
    
    Zombie files are partial outputs from a run that didn't complete successfully.
    They include: segments/, log files, partial reports, etc.
    
    Args:
        output_dir: The survey-specific output directory to clean
                   (e.g., <base>/patient/video_0/grooming_and_hygiene_abnormal/)
        timepoint: The timepoint number (for logging)
        dry_run: If True, only report what would be deleted without deleting
        
    Returns:
        List of files/directories that were (or would be) removed
    """
    import shutil
    
    removed_items = []
    
    if not os.path.isdir(output_dir):
        return removed_items
    
    # SAFETY CHECK: Verify the output_dir path looks correct (contains video_N pattern)
    # This prevents accidental deletion of wrong directories
    expected_pattern = f"video_{timepoint}"
    if expected_pattern not in output_dir:
        logger.error(
            f"SAFETY: Refusing to clean directory that doesn't match expected pattern. "
            f"Expected 'video_{timepoint}' in path: {output_dir}"
        )
        return removed_items
    
    # List all items in the directory
    try:
        items = os.listdir(output_dir)
    except OSError as e:
        logger.error(f"Cannot list directory {output_dir}: {e}")
        return removed_items
    
    for item in items:
        item_path = os.path.join(output_dir, item)
        
        if dry_run:
            removed_items.append(f"[DRY RUN] Would remove: {item_path}")
            continue
        
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                removed_items.append(f"Removed directory: {item_path}")
            else:
                os.remove(item_path)
                removed_items.append(f"Removed file: {item_path}")
        except OSError as e:
            logger.warning(f"Failed to remove {item_path}: {e}")
    
    # After cleanup, try to remove the empty output_dir itself
    if not dry_run:
        try:
            if os.path.isdir(output_dir) and not os.listdir(output_dir):
                os.rmdir(output_dir)
                removed_items.append(f"Removed empty directory: {output_dir}")
        except OSError:
            # Directory not empty or other issue - that's fine
            pass
    
    return removed_items


# =============================================================================
# TASK EXECUTION
# =============================================================================

def run_single_analysis(
    patient: str,
    survey_question: str,
    timepoint: int,
    base_dir: str,
    prompt_file: str,
    engine_script: str,
    task_num: int,
    total_tasks: int,
    survey_num: int,
    total_surveys: int,
    dry_run: bool = False,
    iteration_log_file: Optional[str] = None,
    skip_completed: bool = True,
    api_url: str = "http://localhost:5100",
) -> TaskResult:
    """
    Run a single analysis task with detailed logging.
    
    SMART EXECUTION:
    - If skip_completed=True and the run was already successful (video_N_final.json exists),
      skip re-running and return success immediately.
    - If the output folder exists but run was NOT successful (no final.json),
      clean up zombie files and re-run.
    - If the output folder doesn't exist, run normally.
    
    Args:
        patient: Patient name
        survey_question: Survey question to assess
        timepoint: Visit timepoint
        base_dir: Base directory for results
        prompt_file: Path to prompt YAML file
        engine_script: Path to the engine script
        task_num: Current task number (1-indexed)
        total_tasks: Total number of tasks
        survey_num: Current survey number for this patient (1-indexed)
        total_surveys: Total number of surveys per patient
        dry_run: If True, print command without executing
        iteration_log_file: Optional path to iteration log file
        skip_completed: If True, skip runs that were already successfully completed
        
    Returns:
        TaskResult with execution details
    """
    start_time = time.time()
    start_timestamp = datetime.now()
    
    # Build command
    cmd, video_path, base_output_dir = build_analysis_command(
        patient=patient,
        survey_question=survey_question,
        timepoint=timepoint,
        base_dir=base_dir,
        prompt_file=prompt_file,
        engine_script=engine_script,
        api_url=api_url,
    )
    
    # Calculate expected output directory
    folder_name = survey_question_to_folder_name(survey_question)
    expected_output_dir = os.path.join(base_output_dir, f"video_{timepoint}", folder_name)
    
    # Log task start: one compact line (fits on screen)
    mse_short = _mse_display(survey_question)
    logger.info(
        f"[TASK {task_num}/{total_tasks}] {patient} | T{timepoint} | MSE: \"{mse_short}\" | "
        f"survey {survey_num}/{total_surveys} | started {start_timestamp.strftime('%H:%M:%S')}"
    )
    
    # =========================================================================
    # CHECK FOR PREVIOUSLY COMPLETED RUN
    # =========================================================================
    if skip_completed and not dry_run:
        is_complete, final_json_path = check_survey_completion(expected_output_dir, timepoint)
        
        if is_complete:
            duration = time.time() - start_time
            logger.info(
                f"  ✓ SKIPPED {patient} T{timepoint} \"{_mse_display(survey_question)}\" "
                f"(already complete, {duration:.0f}s)"
            )
            
            # Log to iteration log (thread-safe)
            if iteration_log_file:
                try:
                    with _iteration_log_lock:
                        with open(iteration_log_file, 'a', encoding='utf-8') as f:
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            f.write(f"[{timestamp}] [SKIPPED] Patient={patient} | T{timepoint} | Survey={survey_question}\n")
                            f.write(f"    Reason: Already complete (found {os.path.basename(final_json_path)})\n")
                            f.write(f"    Output: {expected_output_dir}\n\n")
                except Exception as e:
                    logger.warning(f"Failed to write to iteration log: {e}")
            
            res = TaskResult(
                patient=patient,
                survey_question=survey_question,
                timepoint=timepoint,
                success=True,
                duration_seconds=duration,
                output_dir=expected_output_dir,
                video_path=video_path
            )
            return res
        
        # Check if folder exists but run was incomplete (zombie files)
        if os.path.isdir(expected_output_dir):
            removed_items = cleanup_incomplete_run(expected_output_dir, timepoint, dry_run=False)
            logger.warning(
                f"  ⚠ Cleaned {len(removed_items)} zombie file(s) from incomplete run, re-running."
            )
    
    def log_iteration(success: bool, error_msg: Optional[str] = None, duration: float = 0.0):
        """Log iteration result to the iteration log file (thread-safe)."""
        if not iteration_log_file:
            return
        try:
            with _iteration_log_lock:
                with open(iteration_log_file, 'a', encoding='utf-8') as f:
                    status = "SUCCESS" if success else "FAILED"
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"[{timestamp}] [{status}] Patient={patient} | T{timepoint} | Survey={survey_question}\n")
                    f.write(f"    Duration: {timedelta(seconds=int(duration))}\n")
                    f.write(f"    Video: {video_path}\n")
                    f.write(f"    Output: {expected_output_dir}\n")
                    if error_msg:
                        f.write(f"    Error: {error_msg}\n")
                    f.write("\n")
        except Exception as e:
            logger.warning(f"Failed to write to iteration log: {e}")
    
    # Validate video exists
    if not os.path.exists(video_path):
        error_msg = f"Video file not found: {video_path}"
        logger.error(f"  ✗ FAILED {patient} T{timepoint} \"{_mse_display(survey_question)}\": video not found")
        duration = time.time() - start_time
        log_iteration(success=False, error_msg=error_msg, duration=duration)
        res = TaskResult(
            patient=patient,
            survey_question=survey_question,
            timepoint=timepoint,
            success=False,
            duration_seconds=duration,
            error_message=error_msg,
            video_path=video_path
        )
        return res
    
    # For timepoint > 0, verify prior visit summary exists
    if timepoint > 0:
        exists, expected_path, error_msg = verify_prior_visit_summary_exists(
            base_dir=base_dir,
            patient=patient,
            current_timepoint=timepoint,
            survey_question=survey_question
        )
        if not exists:
            logger.error(f"  ✗ FAILED {patient} T{timepoint} \"{_mse_display(survey_question)}\": prior visit summary missing")
            duration = time.time() - start_time
            log_iteration(success=False, error_msg=error_msg, duration=duration)
            res = TaskResult(
                patient=patient,
                survey_question=survey_question,
                timepoint=timepoint,
                success=False,
                duration_seconds=duration,
                error_message=error_msg,
                video_path=video_path
            )
            return res
    
    # Dry run - just print
    if dry_run:
        logger.info(f"  [DRY RUN] {patient} T{timepoint} \"{_mse_display(survey_question)}\" (no execution)")
        log_iteration(success=True, duration=0.0)
        res = TaskResult(
            patient=patient,
            survey_question=survey_question,
            timepoint=timepoint,
            success=True,
            duration_seconds=0.0,
            output_dir=expected_output_dir,
            video_path=video_path
        )
        return res
    
    # Execute command
    logger.info(f"  → Running engine for {patient} T{timepoint} \"{_mse_display(survey_question)}\" (wait for completion)...")
    
    # Add visual separator for engine output
    print(f"\n{'─' * _LOG_LINE_WIDTH}")
    print(f"│ ENGINE: {patient} T{timepoint} | {_mse_display(survey_question, 50)}")
    print(f"{'─' * _LOG_LINE_WIDTH}\n")
    sys.stdout.flush()
    
    # Build command string for debugging
    cmd_str = ' '.join(cmd)
    
    try:
        # Run subprocess SYNCHRONOUSLY - waits for completion before returning
        # This ensures we don't spam the server with multiple concurrent requests
        # capture_output=True to capture stderr for debugging on failure
        result = subprocess.run(
            cmd,
            capture_output=True,  # Capture stdout/stderr for debugging
            text=True,
            timeout=7200  # 2 hour timeout per task (safety net)
        )
        
        duration = time.time() - start_time
        end_timestamp = datetime.now()
        
        # Print captured output (simulates real-time display)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        print(f"\n{'─' * _LOG_LINE_WIDTH}")
        
        # Check return code
        if result.returncode != 0:
            print(f"│ END (FAILED): {patient} T{timepoint}")
            print(f"{'─' * _LOG_LINE_WIDTH}\n")
            sys.stdout.flush()
            
            # Build detailed error message with stderr for debugging
            error_msg = f"Process exited with code {result.returncode}"
            if result.stderr:
                # Extract last 500 chars of stderr for error message
                stderr_snippet = result.stderr.strip()[-500:]
                error_msg = f"{error_msg}. STDERR: {stderr_snippet}"
            
            logger.error(
                f"  ✗ FAILED {patient} T{timepoint} \"{_mse_display(survey_question)}\" "
                f"exit={result.returncode} dur={timedelta(seconds=int(duration))}"
            )
            if result.stderr:
                last_lines = stderr_snippet.split("\n")[-5:]
                logger.error(f"    stderr: " + " | ".join(l.strip()[:60] for l in last_lines if l.strip()))
            
            log_iteration(success=False, error_msg=error_msg, duration=duration)
            
            res = TaskResult(
                patient=patient,
                survey_question=survey_question,
                timepoint=timepoint,
                success=False,
                duration_seconds=duration,
                error_message=error_msg,
                video_path=video_path
            )
            return res
        
        # Success case
        print(f"│ END: {patient} T{timepoint} | {_mse_display(survey_question, 50)}")
        print(f"{'─' * _LOG_LINE_WIDTH}\n")
        sys.stdout.flush()
        
        logger.info(
            f"  ✓ DONE {patient} T{timepoint} \"{_mse_display(survey_question)}\" "
            f"in {timedelta(seconds=int(duration))} → {expected_output_dir}"
        )
        
        log_iteration(success=True, duration=duration)
        
        res = TaskResult(
            patient=patient,
            survey_question=survey_question,
            timepoint=timepoint,
            success=True,
            duration_seconds=duration,
            output_dir=expected_output_dir,
            video_path=video_path
        )
        return res
        
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        error_msg = f"Process TIMEOUT after {duration:.0f}s (limit: 7200s). Task may be stuck or server unresponsive."
        
        print(f"\n{'─' * _LOG_LINE_WIDTH}")
        print(f"│ END (TIMEOUT): {patient} T{timepoint}")
        print(f"{'─' * _LOG_LINE_WIDTH}\n")
        sys.stdout.flush()
        
        logger.error(
            f"  ✗ TIMEOUT {patient} T{timepoint} \"{_mse_display(survey_question)}\" "
            f"after {timedelta(seconds=int(duration))}"
        )
        
        log_iteration(success=False, error_msg=error_msg, duration=duration)
        
        res = TaskResult(
            patient=patient,
            survey_question=survey_question,
            timepoint=timepoint,
            success=False,
            duration_seconds=duration,
            error_message=error_msg,
            video_path=video_path
        )
        return res
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        error_msg = f"Process exited with code {e.returncode}"
        if e.stderr:
            stderr_snippet = e.stderr.strip()[-500:]
            error_msg = f"{error_msg}. STDERR: {stderr_snippet}"
        
        print(f"\n{'─' * _LOG_LINE_WIDTH}")
        print(f"│ END (FAILED): {patient} T{timepoint}")
        print(f"{'─' * _LOG_LINE_WIDTH}\n")
        sys.stdout.flush()
        
        logger.error(
            f"  ✗ FAILED {patient} T{timepoint} \"{_mse_display(survey_question)}\" "
            f"exit={e.returncode} {(error_msg or '')[:80]}"
        )
        
        log_iteration(success=False, error_msg=error_msg, duration=duration)
        
        res = TaskResult(
            patient=patient,
            survey_question=survey_question,
            timepoint=timepoint,
            success=False,
            duration_seconds=duration,
            error_message=error_msg,
            video_path=video_path
        )
        return res
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        
        logger.error(
            f"  ✗ EXCEPTION {patient} T{timepoint} \"{_mse_display(survey_question)}\" "
            f"{error_msg[:70]}"
        )
        
        log_iteration(success=False, error_msg=error_msg, duration=duration)
        
        res = TaskResult(
            patient=patient,
            survey_question=survey_question,
            timepoint=timepoint,
            success=False,
            duration_seconds=duration,
            error_message=error_msg,
            video_path=video_path
        )
        return res


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================

def run_surveys_parallel(
    patient: str,
    survey_questions: List[str],
    timepoint: int,
    base_dir: str,
    prompt_file: str,
    engine_script: str,
    base_task_num: int,
    total_tasks: int,
    total_surveys: int,
    dry_run: bool = False,
    iteration_log_file: Optional[str] = None,
    skip_completed: bool = True,
    parallel_workers: int = 5,
    api_url: str = "http://localhost:5100",
) -> List[TaskResult]:
    """
    Run survey questions in parallel using a thread pool.
    
    This function submits multiple survey analysis tasks to a thread pool,
    maintaining up to `parallel_workers` concurrent executions at any time.
    When one task completes, another is immediately started until all are done.
    
    Args:
        patient: Patient name
        survey_questions: List of survey questions to process
        timepoint: Visit timepoint
        base_dir: Base directory for results
        prompt_file: Path to prompt YAML file
        engine_script: Path to the engine script
        base_task_num: Starting task number for global progress tracking
        total_tasks: Total number of tasks across all patients/timepoints
        total_surveys: Total number of surveys for this timepoint
        dry_run: If True, print commands without executing
        iteration_log_file: Optional path to iteration log file
        skip_completed: If True, skip runs that were already successfully completed
        parallel_workers: Number of parallel workers (default: 5)
        
    Returns:
        List of TaskResult objects for all completed tasks
    """
    results: List[TaskResult] = []
    
    if not survey_questions:
        return results
    
    logger.info(
        f"  🚀 PARALLEL {patient} T{timepoint}: {len(survey_questions)} MSE domains, "
        f"{parallel_workers} workers (output may interleave)"
    )
    
    # Track completed count for progress
    completed_count = 0
    completed_lock = threading.Lock()
    start_time = time.time()
    
    def run_task(survey_idx_question: Tuple[int, str]) -> TaskResult:
        """Wrapper to run a single task in the thread pool."""
        survey_idx, survey_question = survey_idx_question
        task_num = base_task_num + survey_idx
        
        result = run_single_analysis(
            patient=patient,
            survey_question=survey_question,
            timepoint=timepoint,
            base_dir=base_dir,
            prompt_file=prompt_file,
            engine_script=engine_script,
            task_num=task_num,
            total_tasks=total_tasks,
            survey_num=survey_idx + 1,
            total_surveys=total_surveys,
            dry_run=dry_run,
            iteration_log_file=iteration_log_file,
            skip_completed=skip_completed,
            api_url=api_url,
        )
        
        # Update progress (thread-safe)
        nonlocal completed_count
        with completed_lock:
            completed_count += 1
            elapsed = time.time() - start_time
            status = "✓" if result.success else "✗"
            
            avg_per_task = elapsed / completed_count
            remaining = len(survey_questions) - completed_count
            remaining_batches = math.ceil(remaining / parallel_workers)
            eta_seconds = remaining_batches * avg_per_task
            
            ok_so_far = sum(1 for r in results if r.success) + (1 if result.success else 0)
            fail_so_far = completed_count - ok_so_far
            
            pct = (completed_count / len(survey_questions) * 100)
            bar_len = 25
            filled = int(bar_len * completed_count / len(survey_questions))
            bar = "█" * filled + "░" * (bar_len - filled)
            
            with _log_lock:
                logger.info(
                    f"  {status} {patient} T{timepoint} [{completed_count}/{len(survey_questions)}] "
                    f"\"{_mse_display(survey_question)}\" {timedelta(seconds=int(result.duration_seconds))} | "
                    f"ETA this batch: {timedelta(seconds=int(eta_seconds))} | OK:{ok_so_far} Fail:{fail_so_far} [{bar}] {pct:.0f}%"
                )
        
        return result
    
    # Create list of (index, question) tuples for task submission
    tasks = list(enumerate(survey_questions))
    
    # Use ThreadPoolExecutor to run tasks in parallel
    # ThreadPoolExecutor is appropriate here because each task is I/O-bound (subprocess)
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        # Submit all tasks to the pool
        # The pool automatically maintains `parallel_workers` concurrent tasks
        futures = {executor.submit(run_task, task): task for task in tasks}
        
        # Collect results as they complete
        for future in as_completed(futures):
            task = futures[future]
            survey_idx, survey_question = task
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Handle unexpected exceptions (should be rare)
                logger.error(f"  ✗ Unexpected error for survey '{survey_question[:50]}...': {e}")
                err_result = TaskResult(
                    patient=patient,
                    survey_question=survey_question,
                    timepoint=timepoint,
                    success=False,
                    duration_seconds=0.0,
                    error_message=f"Unexpected error: {str(e)}"
                )
                results.append(err_result)
    
    # Sort results by original survey order for consistent output
    # Create a mapping from survey_question to original index
    survey_order = {q: i for i, q in enumerate(survey_questions)}
    results.sort(key=lambda r: survey_order.get(r.survey_question, 999))
    
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r.success)
    fail_count = sum(1 for r in results if not r.success)
    
    speedup_str = ""
    if total_time > 0 and results:
        actual_total = sum(r.duration_seconds for r in results)
        speedup = actual_total / total_time
        speedup_str = f" | speedup {speedup:.1f}x"
    logger.info(
        f"  📊 {patient} T{timepoint} DONE: {success_count}/{len(survey_questions)} OK, {fail_count} fail | "
        f"wall {timedelta(seconds=int(total_time))}{speedup_str}"
    )
    
    return results


# =============================================================================
# ALL-TIMEPOINTS EXECUTION
# =============================================================================

def run_all_timepoints_for_patient(
    patient: str,
    timepoints: List[int],
    survey_questions: List[str],
    base_dir: str,
    prompt_file: str,
    engine_script: str,
    continue_on_error: bool,
    delay_between_tasks: float,
    dry_run: bool,
    iteration_log_file: Optional[str],
    global_task_counter: Dict[str, int],
    parallel_workers: int = 1,
    skip_completed: bool = True,
    api_url: str = "http://localhost:5100",
) -> PatientResult:
    """
    Run all surveys for a single patient, processing BY SURVEY QUESTION FIRST.
    
    PROCESSING ORDER (SURVEY-FIRST):
    For each survey question, process all timepoints (T0 → T1 → T2) before moving
    to the next survey question. This ensures longitudinal dependencies are satisfied
    within each survey before moving on.
    
    Example:
      Survey "Grooming": T0 → T1 → T2  (T1 uses T0's output, T2 uses T1's output)
      Survey "Eye contact": T0 → T1 → T2
      Survey "Flat affect": T0 → T1 → T2
      ...
    
    This is preferred over the timepoint-first approach because:
    - Survey "Grooming" T1 depends ONLY on Survey "Grooming" T0 (not other surveys)
    - This allows completing each survey's full longitudinal chain before moving on
    - Failed surveys don't block other surveys from processing
    
    Within each survey, timepoints are processed SERIALLY (T0 must complete before T1).
    Different surveys can potentially run in parallel if parallel_workers > 1.
    
    Args:
        patient: Patient name
        timepoints: List of timepoints to process (must be sorted ascending)
        survey_questions: List of survey questions
        base_dir: Base directory
        prompt_file: Prompt file path
        engine_script: Engine script path
        continue_on_error: Whether to continue if a task fails
        delay_between_tasks: Delay between tasks in seconds (only used in serial mode)
        dry_run: Whether to do a dry run
        iteration_log_file: Path to iteration log file
        global_task_counter: Mutable dict with 'current' and 'total' keys for tracking
        parallel_workers: Number of parallel workers (1 = serial mode, default)
        
    Returns:
        PatientResult with all timepoint results
    """
    patient_result = PatientResult(patient=patient)
    patient_start_time = time.time()
    
    # CRITICAL: Ensure timepoints are sorted in ascending order
    timepoints = sorted(timepoints)
    
    total_tasks_patient = len(timepoints) * len(survey_questions)
    logger.info("")
    logger.info(
        f"  PATIENT {patient} | {len(survey_questions)} MSE domains × {len(timepoints)} timepoints "
        f"(T{','.join(str(t) for t in timepoints)}) = {total_tasks_patient} tasks"
    )
    
    # Pre-validate: Check which timepoints have valid videos
    valid_timepoints: List[int] = []
    for tp in timepoints:
        video_exists, video_path = validate_patient_video_exists(base_dir, patient, tp)
        if video_exists:
            valid_timepoints.append(tp)
        else:
            logger.warning(f"  ⚠ Video not found for T{tp}: {video_path}")
    
    if not valid_timepoints:
        logger.error(f"  ✗ No valid videos found for patient {patient}")
        patient_result.timepoints_skipped = timepoints
        patient_result.total_duration_seconds = time.time() - patient_start_time
        return patient_result
    
    logger.info(f"  Valid timepoints with videos: {', '.join(f'T{t}' for t in valid_timepoints)}")
    
    # Track results by timepoint for the PatientResult structure
    # We'll aggregate results into TimepointResult objects at the end
    all_task_results: List[TaskResult] = []
    survey_success_tracker: Dict[str, Dict[int, bool]] = {}  # {survey: {timepoint: success}}
    
    # =========================================================================
    # SURVEY-FIRST PROCESSING: For each survey, process all timepoints
    # =========================================================================
    
    if parallel_workers > 1:
        # PARALLEL MODE: Run multiple surveys concurrently
        # Each survey processes its timepoints serially (T0→T1→T2)
        # But different surveys can run in parallel
        logger.info(f"  Mode: PARALLEL ({parallel_workers} workers), survey-first (T0→T1→T2 per MSE)")
        
        # Thread-safe counter for task numbers
        task_counter_lock = threading.Lock()
        
        # Progress tracker shared across threads
        _progress = {
            'surveys_done': 0,
            'tasks_done': 0,
            'tasks_success': 0,
            'tasks_failed': 0,
            'total_surveys': len(survey_questions),
            'total_tasks': len(valid_timepoints) * len(survey_questions),
            'start': time.time(),
        }
        _progress_lock = threading.Lock()
        
        def _log_progress(survey_question: str, survey_results: List[TaskResult]):
            """Thread-safe progress summary after each survey completes."""
            with _progress_lock:
                _progress['surveys_done'] += 1
                for r in survey_results:
                    _progress['tasks_done'] += 1
                    if r.success:
                        _progress['tasks_success'] += 1
                    else:
                        _progress['tasks_failed'] += 1
                
                done_s = _progress['surveys_done']
                total_s = _progress['total_surveys']
                done_t = _progress['tasks_done']
                total_t = _progress['total_tasks']
                ok = _progress['tasks_success']
                fail = _progress['tasks_failed']
                elapsed = time.time() - _progress['start']
                
                # ETA based on surveys completed (each survey = full timepoint chain)
                if done_s > 0:
                    avg_per_survey = elapsed / done_s
                    remaining_surveys = total_s - done_s
                    # With parallelism, remaining wall-clock ~ remaining_surveys / workers * avg
                    remaining_batches = math.ceil(remaining_surveys / parallel_workers)
                    eta_seconds = remaining_batches * avg_per_survey
                else:
                    eta_seconds = 0
                
                survey_dur = sum(r.duration_seconds for r in survey_results)
                tp_summary = " ".join(
                    f"T{r.timepoint}:{'OK' if r.success else 'FAIL'}({timedelta(seconds=int(r.duration_seconds))})"
                    for r in survey_results
                )
            
            pct = (done_s / total_s * 100) if total_s > 0 else 0
            bar_len = 20
            filled = int(bar_len * done_s / total_s) if total_s > 0 else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            with _log_lock:
                logger.info(
                    f"  MSE done {done_s}/{total_s}: \"{_mse_display(survey_question)}\" → {tp_summary} | "
                    f"patient ETA: {timedelta(seconds=int(eta_seconds))} | {done_t}/{total_t} tasks [{bar}] {pct:.0f}%"
                )
        
        def process_survey_all_timepoints(survey_idx_question: Tuple[int, str]) -> List[TaskResult]:
            """Process all timepoints for a single survey question."""
            survey_idx, survey_question = survey_idx_question
            survey_results: List[TaskResult] = []
            previous_tp_success = True
            
            for tp_idx, timepoint in enumerate(valid_timepoints):
                # Get task number atomically
                with task_counter_lock:
                    global_task_counter['current'] += 1
                    current_task_num = global_task_counter['current']
                
                # Check if we should skip due to prior timepoint failure for THIS survey
                if timepoint > 0 and not previous_tp_success and not continue_on_error:
                    logger.warning(
                        f"  Skip T{timepoint} \"{_mse_display(survey_question)}\" (T{timepoint-1} failed)"
                    )
                    skip_result = TaskResult(
                        patient=patient,
                        survey_question=survey_question,
                        timepoint=timepoint,
                        success=False,
                        duration_seconds=0.0,
                        error_message=f"Skipped: T{timepoint-1} failed for this survey"
                    )
                    survey_results.append(skip_result)
                    continue
                
                # Run the analysis
                result = run_single_analysis(
                    patient=patient,
                    survey_question=survey_question,
                    timepoint=timepoint,
                    base_dir=base_dir,
                    prompt_file=prompt_file,
                    engine_script=engine_script,
                    task_num=current_task_num,
                    total_tasks=global_task_counter['total'],
                    survey_num=survey_idx,
                    total_surveys=len(survey_questions),
                    dry_run=dry_run,
                    iteration_log_file=iteration_log_file,
                    skip_completed=skip_completed,
                    api_url=api_url,
                )
                
                survey_results.append(result)
                previous_tp_success = result.success
            
            return survey_results
        
        # Submit all surveys to the thread pool
        survey_items = list(enumerate(survey_questions, 1))
        
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(process_survey_all_timepoints, item): item 
                for item in survey_items
            }
            
            for future in as_completed(futures):
                survey_idx, survey_question = futures[future]
                try:
                    results = future.result()
                    all_task_results.extend(results)
                    
                    # Track success per timepoint for this survey
                    survey_success_tracker[survey_question] = {}
                    for r in results:
                        survey_success_tracker[survey_question][r.timepoint] = r.success
                    
                    _log_progress(survey_question, results)
                except Exception as e:
                    logger.error(f"  ✗ [{survey_question}] Failed with exception: {e}")
    
    else:
        logger.info(f"  Mode: SERIAL (one task at a time)")
        
        serial_start = time.time()
        serial_tasks_done = 0
        serial_total_tasks = len(survey_questions) * len(valid_timepoints)
        
        for survey_idx, survey_question in enumerate(survey_questions, 1):
            logger.info(
                f"  --- MSE {survey_idx}/{len(survey_questions)}: \"{_mse_display(survey_question)}\" ---"
            )
            
            survey_success_tracker[survey_question] = {}
            previous_tp_success = True
            
            for tp_idx, timepoint in enumerate(valid_timepoints):
                global_task_counter['current'] += 1
                
                # Check if we should skip due to prior timepoint failure for THIS survey
                if timepoint > 0 and not previous_tp_success and not continue_on_error:
                    skip_msg = f"Skipping T{timepoint} - T{timepoint-1} failed for this survey"
                    logger.warning(f"  ⊘ {skip_msg}")
                    
                    result = TaskResult(
                        patient=patient,
                        survey_question=survey_question,
                        timepoint=timepoint,
                        success=False,
                        duration_seconds=0.0,
                        error_message=skip_msg
                    )
                    all_task_results.append(result)
                    survey_success_tracker[survey_question][timepoint] = False
                    serial_tasks_done += 1
                    continue
                
                _elapsed_so_far = time.time() - serial_start
                _remaining = serial_total_tasks - serial_tasks_done
                _eta_str = timedelta(seconds=int(_elapsed_so_far / serial_tasks_done * _remaining)) if serial_tasks_done > 0 else "?"
                logger.info(
                    f"  ▶ {patient} T{timepoint} \"{_mse_display(survey_question)}\" "
                    f"task {serial_tasks_done + 1}/{serial_total_tasks} | ETA this patient: {_eta_str}"
                )
                # Run the analysis
                result = run_single_analysis(
                    patient=patient,
                    survey_question=survey_question,
                    timepoint=timepoint,
                    base_dir=base_dir,
                    prompt_file=prompt_file,
                    engine_script=engine_script,
                    task_num=global_task_counter['current'],
                    total_tasks=global_task_counter['total'],
                    survey_num=survey_idx,
                    total_surveys=len(survey_questions),
                    dry_run=dry_run,
                    iteration_log_file=iteration_log_file,
                    skip_completed=skip_completed,
                    api_url=api_url,
                )
                
                all_task_results.append(result)
                survey_success_tracker[survey_question][timepoint] = result.success
                previous_tp_success = result.success
                
                serial_tasks_done += 1
                elapsed = time.time() - serial_start
                avg_per_task = elapsed / serial_tasks_done if serial_tasks_done > 0 else 0
                eta_seconds = avg_per_task * (serial_total_tasks - serial_tasks_done)
                pct = (serial_tasks_done / serial_total_tasks * 100) if serial_total_tasks > 0 else 0
                bar_len = 20
                filled = int(bar_len * serial_tasks_done / serial_total_tasks) if serial_total_tasks > 0 else 0
                bar = "█" * filled + "░" * (bar_len - filled)
                ok_count = sum(1 for r in all_task_results if r.success)
                fail_count = serial_tasks_done - ok_count
                status = "✓" if result.success else "✗"
                logger.info(
                    f"  {status} {serial_tasks_done}/{serial_total_tasks} T{timepoint} "
                    f"{timedelta(seconds=int(result.duration_seconds))} | "
                    f"ETA: {timedelta(seconds=int(eta_seconds))} | OK:{ok_count} Fail:{fail_count} [{bar}] {pct:.0f}%"
                )
                
                # Delay between tasks (only if more tasks remain and not dry-run)
                remaining_tps = len(valid_timepoints) - tp_idx - 1
                remaining_surveys = len(survey_questions) - survey_idx
                if (remaining_tps > 0 or remaining_surveys > 0) and not dry_run and delay_between_tasks > 0:
                    logger.info(f"  Waiting {delay_between_tasks}s before next task...")
                    time.sleep(delay_between_tasks)
            
            success_count = sum(1 for tp in valid_timepoints if survey_success_tracker[survey_question].get(tp, False))
            logger.info(
                f"  MSE \"{_mse_display(survey_question)}\" → {success_count}/{len(valid_timepoints)} TPs OK"
            )
    
    # =========================================================================
    # AGGREGATE RESULTS INTO TIMEPOINT-BASED STRUCTURE
    # =========================================================================
    # The PatientResult expects results grouped by timepoint, so we need to reorganize
    
    for timepoint in timepoints:
        tp_results = [r for r in all_task_results if r.timepoint == timepoint]
        
        if not tp_results:
            # Timepoint was skipped entirely (no valid video)
            patient_result.timepoint_results.append(TimepointResult(
                patient=patient,
                timepoint=timepoint,
                total_surveys=len(survey_questions),
                successful_surveys=0,
                failed_surveys=0,
                duration_seconds=0.0,
                task_results=[],
                skipped=True,
                skip_reason="Video not found"
            ))
            patient_result.timepoints_skipped.append(timepoint)
        else:
            success_count = sum(1 for r in tp_results if r.success)
            fail_count = sum(1 for r in tp_results if not r.success)
            total_duration = sum(r.duration_seconds for r in tp_results)
            
            patient_result.timepoint_results.append(TimepointResult(
                patient=patient,
                timepoint=timepoint,
                total_surveys=len(survey_questions),
                successful_surveys=success_count,
                failed_surveys=fail_count,
                duration_seconds=total_duration,
                task_results=tp_results,
                skipped=False
            ))
            patient_result.timepoints_processed.append(timepoint)
    
    patient_result.total_duration_seconds = time.time() - patient_start_time
    
    total_success = sum(1 for r in all_task_results if r.success)
    total_fail = sum(1 for r in all_task_results if not r.success)
    speedup_str = ""
    if patient_result.total_duration_seconds > 0 and all_task_results:
        total_cpu = sum(r.duration_seconds for r in all_task_results if r.duration_seconds > 0)
        if total_cpu > 0:
            speedup_str = f" | speedup {total_cpu / patient_result.total_duration_seconds:.1f}x"
    tp_bits = []
    for tp_res in patient_result.timepoint_results:
        if tp_res.skipped:
            tp_bits.append(f"T{tp_res.timepoint}:skip")
        else:
            tp_bits.append(f"T{tp_res.timepoint}:{tp_res.successful_surveys}/{tp_res.total_surveys}OK")
    logger.info("")
    logger.info(
        f"  ★ {patient} SUMMARY: {total_success} OK, {total_fail} fail | "
        f"wall {timedelta(seconds=int(patient_result.total_duration_seconds))}{speedup_str}"
    )
    logger.info(f"     Per TP: {', '.join(tp_bits)}")
    
    return patient_result


# =============================================================================
# RESULTS HANDLING
# =============================================================================

def print_summary(results: List[TaskResult], total_time: float):
    """Print compact summary of all task results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    patients_processed = list(set(r.patient for r in results))
    logger.info("")
    logger.info(
        f"  ★ SUMMARY: {len(successful)} OK, {len(failed)} fail | "
        f"{timedelta(seconds=int(total_time))} | patients: {', '.join(patients_processed)}"
    )
    for patient in patients_processed:
        pr = [r for r in results if r.patient == patient]
        ok = len([r for r in pr if r.success])
        status = "✓" if len(pr) == ok else "✗"
        logger.info(f"    {status} {patient}: {ok}/{len(pr)}")
    for r in failed:
        logger.error(f"  ✗ {r.patient} T{r.timepoint} \"{_mse_display(r.survey_question)}\" | {(r.error_message or '')[:60]}")
    logger.info("=" * _LOG_LINE_WIDTH)


def print_all_timepoints_summary(patient_results: List[PatientResult], total_time: float):
    """Print compact summary for all-timepoints mode."""
    total_tasks = sum(
        tr.total_surveys for pr in patient_results for tr in pr.timepoint_results if not tr.skipped
    )
    successful_tasks = sum(
        tr.successful_surveys for pr in patient_results for tr in pr.timepoint_results
    )
    failed_tasks = sum(
        tr.failed_surveys for pr in patient_results for tr in pr.timepoint_results
    )
    skipped_tps = sum(1 for pr in patient_results for tr in pr.timepoint_results if tr.skipped)
    logger.info("")
    logger.info(
        f"  ★ ALL-TPs SUMMARY: {successful_tasks}/{total_tasks} OK, {failed_tasks} fail, "
        f"{skipped_tps} TPs skipped | {timedelta(seconds=int(total_time))}"
    )
    for pr in patient_results:
        total_patient_tasks = sum(
            tr.successful_surveys + tr.failed_surveys for tr in pr.timepoint_results if not tr.skipped
        )
        total_patient_success = sum(tr.successful_surveys for tr in pr.timepoint_results)
        total_patient_failed = sum(tr.failed_surveys for tr in pr.timepoint_results)
        status = "✓" if total_patient_failed == 0 else "⚠"
        logger.info(
            f"    {status} {pr.patient}: T{','.join(str(t) for t in pr.timepoints_processed)} | "
            f"{total_patient_success}/{total_patient_tasks} OK | {timedelta(seconds=int(pr.total_duration_seconds))}"
        )
    all_failed = [
        task for pr in patient_results for tr in pr.timepoint_results
        for task in tr.task_results if not task.success
    ]
    for task in all_failed:
        logger.error(f"  ✗ {task.patient} T{task.timepoint} \"{_mse_display(task.survey_question)}\" | {(task.error_message or '')[:55]}")
    logger.info("=" * _LOG_LINE_WIDTH)


def save_results_json(results: List[TaskResult], output_path: str):
    """Save results to JSON file for later analysis."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "total_tasks": len(results),
        "successful": len([r for r in results if r.success]),
        "failed": len([r for r in results if not r.success]),
        "patients": list(set(r.patient for r in results)),
        "results": [
            {
                "patient": r.patient,
                "survey_question": r.survey_question,
                "timepoint": r.timepoint,
                "success": r.success,
                "duration_seconds": r.duration_seconds,
                "error_message": r.error_message,
                "output_dir": r.output_dir,
                "video_path": r.video_path
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_path}")


def save_all_timepoints_results_json(patient_results: List[PatientResult], output_path: str):
    """Save all-timepoints results to JSON file."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "mode": "all-timepoints",
        "patients": [
            {
                "patient": pr.patient,
                "timepoints_processed": pr.timepoints_processed,
                "timepoints_skipped": pr.timepoints_skipped,
                "total_duration_seconds": pr.total_duration_seconds,
                "timepoint_results": [
                    {
                        "timepoint": tr.timepoint,
                        "skipped": tr.skipped,
                        "skip_reason": tr.skip_reason,
                        "total_surveys": tr.total_surveys,
                        "successful_surveys": tr.successful_surveys,
                        "failed_surveys": tr.failed_surveys,
                        "duration_seconds": tr.duration_seconds,
                        "tasks": [
                            {
                                "survey_question": task.survey_question,
                                "success": task.success,
                                "duration_seconds": task.duration_seconds,
                                "error_message": task.error_message,
                                "output_dir": task.output_dir,
                                "video_path": task.video_path
                            }
                            for task in tr.task_results
                        ]
                    }
                    for tr in pr.timepoint_results
                ]
            }
            for pr in patient_results
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    global logger
    
    parser = argparse.ArgumentParser(
        description="Batch wrapper for longitudinal clinical audio-video analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all surveys for one patient at timepoint 0 (default: 5 parallel workers)
  python run_all_surveys_longitudinal.py --patients ben --timepoint 0

  # Run for multiple patients with 5 parallel workers (default)
  python run_all_surveys_longitudinal.py --patients ben karthik robbin --timepoint 0

  # Run ALL timepoints with SURVEY-FIRST processing (recommended for longitudinal)
  # Processing order: Survey1 T0→T1→T2, then Survey2 T0→T1→T2, etc.
  python run_all_surveys_longitudinal.py --patients ben karthik --all-timepoints

  # Run with custom number of parallel workers (e.g., 5 concurrent survey chains)
  # Different surveys run in parallel, but each survey's timepoints are serial
  python run_all_surveys_longitudinal.py --patients ben --all-timepoints --parallel-workers 5

  # Run in SERIAL mode (one survey's full timepoint chain at a time)
  python run_all_surveys_longitudinal.py --patients ben --all-timepoints --parallel-workers 1

  # Dry run - print commands without executing
  python run_all_surveys_longitudinal.py --patients ben --timepoint 0 --dry-run

  # Continue processing even if some tasks fail
  # Other surveys will continue even if one survey's timepoint chain fails
  python run_all_surveys_longitudinal.py --patients ben karthik --all-timepoints --continue-on-error

  # Save results and logs
  python run_all_surveys_longitudinal.py --patients ben --all-timepoints \\
    --save-results results.json --log-file batch_run.log

  # Use custom MSE questions CSV file
  python run_all_surveys_longitudinal.py --patients ben --timepoint 0 \\
    --mse-file /path/to/custom_questions.csv

CSV File Format:
  The MSE questions CSV file MUST have a column named 'mse_survey_question'.
  Each row contains one survey question.
  
  Example CSV content:
    mse_survey_question
    "Grooming and hygiene (abnormal)"
    "Eye contact (abnormal)"
    "Flat affect"
        """
    )
    
    parser.add_argument(
        "--patients",
        nargs="+",
        required=True,
        help="List of patient names to process (e.g., --patients ben karthik robbin)"
    )
    
    # Mutually exclusive: --timepoint OR --all-timepoints
    timepoint_group = parser.add_mutually_exclusive_group()
    
    timepoint_group.add_argument(
        "--timepoint",
        type=int,
        default=None,
        help="Visit timepoint (default: 0 for baseline). Mutually exclusive with --all-timepoints."
    )
    
    timepoint_group.add_argument(
        "--all-timepoints",
        action="store_true",
        help="Auto-detect and run ALL timepoints using SURVEY-FIRST processing order. "
             "For each survey question, processes T0 → T1 → T2 before moving to next survey. "
             "This ensures longitudinal dependencies are satisfied within each survey. "
             "Example: Survey 'Grooming' T0→T1→T2, then Survey 'Eye contact' T0→T1→T2, etc. "
             "Mutually exclusive with --timepoint."
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them"
    )
    
    parser.add_argument(
        "--mse-file",
        default="./analysis_results/martin_et_al_another/mse_questions.csv",
        help="Path to MSE questions CSV file with 'mse_survey_question' column (default: ./analysis_results/martin_et_al_another/mse_questions.csv)"
    )
    
    parser.add_argument(
        "--prompt",
        default="prompts/martin_et_al_another/prompt.yml",
        help="Path to prompt YAML file (default: prompts/martin_et_al_another/prompt.yml)"
    )
    
    parser.add_argument(
        "--base-dir",
        default="./analysis_results/martin_et_al_another",
        help="Base directory for analysis results (default: ./analysis_results/martin_et_al_another)"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining tasks if one fails"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (INFO level only)"
    )
    
    parser.add_argument(
        "--save-results",
        default=None,
        help="Path to save results JSON file (optional)"
    )
    
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to save log file (optional)"
    )
    
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-run of ALL surveys, even if they were previously completed successfully. "
             "By default, completed surveys (those with a final JSON output) are skipped."
    )
    
    parser.add_argument(
        "--delay-between-tasks",
        type=float,
        default=2.0,
        help="Delay in seconds between tasks to avoid overwhelming the server (default: 2.0, only used in serial mode)"
    )
    
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=5,
        help="Number of parallel workers to run survey questions concurrently (default: 5, min: 1, max: 5). "
             "Set to 1 for serial execution. Different surveys run in parallel (their timepoint chains "
             "are independent), but timepoints WITHIN each survey are always processed serially (T0→T1→T2)."
    )
    
    parser.add_argument(
        "--api-url",
        default="http://localhost:5100",
        help="vLLM-Omni API base URL (default: http://localhost:5100, must match server port)"
    )
    
    args = parser.parse_args()
    
    # Validate parallel workers range
    if args.parallel_workers < 1:
        parser.error("--parallel-workers must be at least 1")
    if args.parallel_workers > 5:
        parser.error("--parallel-workers must be at most 5 (vLLM-Omni server max_num_seqs limit)")
    
    # Default to timepoint 0 if neither --timepoint nor --all-timepoints specified
    if args.timepoint is None and not args.all_timepoints:
        args.timepoint = 0
    
    # Generate iteration log file path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    iteration_log_file = os.path.join(
        args.base_dir,
        f"batch_iteration_log_{timestamp}.txt"
    )
    
    # Setup logging
    logger = setup_logging(
        verbose=not args.quiet,
        log_file=args.log_file
    )
    
    logger.info("")
    logger.info(
        f"  LONGITUDINAL BATCH | {'ALL-TPs' if args.all_timepoints else f'T{args.timepoint}'} | "
        f"started {datetime.now().strftime('%H:%M:%S')} | log: {os.path.basename(iteration_log_file) or '—'}"
    )
    
    # Initialize iteration log file with header
    try:
        os.makedirs(os.path.dirname(iteration_log_file), exist_ok=True)
        with open(iteration_log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LONGITUDINAL ANALYSIS - ITERATION LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {'ALL-TIMEPOINTS' if args.all_timepoints else f'SINGLE TIMEPOINT (T{args.timepoint})'}\n")
            f.write(f"Patients: {', '.join(args.patients)}\n")
            f.write(f"Continue on Error: {args.continue_on_error}\n")
            f.write("=" * 80 + "\n\n")
    except Exception as e:
        logger.warning(f"Failed to create iteration log file: {e}")
        iteration_log_file = None
    
    # Determine engine script path (relative to this script)
    script_dir = Path(__file__).resolve().parent
    engine_script = script_dir / "engines" / "clinician_audio_video_segment_analysis_client_longitudinal.py"
    
    if not engine_script.exists():
        logger.error(f"Engine script not found: {engine_script}")
        sys.exit(1)
    
    logger.info(f"  Engine Script:   {engine_script}")
    
    # Load survey questions
    try:
        survey_questions = load_survey_questions(args.mse_file)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    
    if not survey_questions:
        logger.error(f"No survey questions found in {args.mse_file}")
        sys.exit(1)
    
    # Validate prompt file exists
    if not os.path.exists(args.prompt):
        logger.error(f"Prompt file not found: {args.prompt}")
        sys.exit(1)
    
    # ==========================================================================
    # ALL-TIMEPOINTS MODE
    # ==========================================================================
    if args.all_timepoints:
        logger.info("  ALL-TIMEPOINTS: detecting timepoints per patient…")
        patient_timepoints: Dict[str, List[int]] = {}
        for patient in args.patients:
            timepoints = detect_patient_timepoints(args.base_dir, patient)
            if timepoints:
                patient_timepoints[patient] = timepoints
                logger.info(f"  ✓ {patient}: T{','.join(str(t) for t in timepoints)}")
            else:
                logger.warning(f"  ✗ {patient}: no timepoints, skip")
        
        if not patient_timepoints:
            logger.error("No valid patients with timepoints found.")
            sys.exit(1)
        
        # Calculate total tasks
        total_tasks = sum(
            len(tps) * len(survey_questions) 
            for tps in patient_timepoints.values()
        )
        
        logger.info(
            f"  PLAN: {len(patient_timepoints)} patients, {len(survey_questions)} MSE domains → {total_tasks} tasks | "
            f"{'DRY RUN' if args.dry_run else 'EXECUTE'} | workers={args.parallel_workers}"
        )
        for patient, tps in patient_timepoints.items():
            logger.info(f"    {patient}: T{','.join(str(t) for t in tps)}")
        logger.info("=" * _LOG_LINE_WIDTH)
        
        # Execute all timepoints for each patient
        patient_results: List[PatientResult] = []
        start_time = time.time()
        global_task_counter = {'current': 0, 'total': total_tasks}
        
        try:
            for patient_idx, (patient, timepoints) in enumerate(patient_timepoints.items(), 1):
                logger.info("")
                logger.info(f"  ► PATIENT {patient_idx}/{len(patient_timepoints)}: {patient} (T{','.join(str(t) for t in timepoints)})")
                
                result = run_all_timepoints_for_patient(
                    patient=patient,
                    timepoints=timepoints,
                    survey_questions=survey_questions,
                    base_dir=args.base_dir,
                    prompt_file=args.prompt,
                    engine_script=str(engine_script),
                    continue_on_error=args.continue_on_error,
                    delay_between_tasks=args.delay_between_tasks,
                    dry_run=args.dry_run,
                    iteration_log_file=iteration_log_file,
                    global_task_counter=global_task_counter,
                    parallel_workers=args.parallel_workers,
                    skip_completed=not args.force_rerun,
                    api_url=args.api_url,
                )
                
                patient_results.append(result)
                
                # Check if we should stop (patient had failures and not continuing)
                total_failures = sum(tr.failed_surveys for tr in result.timepoint_results)
                if total_failures > 0 and not args.continue_on_error and not args.dry_run:
                    logger.warning(f"Stopping due to errors in patient '{patient}'. Use --continue-on-error to continue.")
                    break
                
        except KeyboardInterrupt:
            logger.warning("\n\nInterrupted by user (Ctrl+C)")
        
        total_time = time.time() - start_time
        
        # Print summary
        if patient_results and not args.dry_run:
            print_all_timepoints_summary(patient_results, total_time)
        elif args.dry_run:
            logger.info("")
            log_header("DRY RUN COMPLETE", char="=", width=80)
            logger.info(f"  Would process {total_tasks} tasks across {len(patient_timepoints)} patients")
        
        # Save results if requested
        if args.save_results and patient_results:
            save_all_timepoints_results_json(patient_results, args.save_results)
        
        # Finalize iteration log with FAILURES SUMMARY
        if iteration_log_file:
            try:
                with open(iteration_log_file, 'a', encoding='utf-8') as f:
                    # Collect all failures for summary
                    all_failures: List[TaskResult] = []
                    for pr in patient_results:
                        for tr in pr.timepoint_results:
                            for task in tr.task_results:
                                if not task.success:
                                    all_failures.append(task)
                    
                    # Write FAILURES SUMMARY first (for easy debugging)
                    if all_failures:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("⚠️  FAILURES SUMMARY (with error messages for debugging)\n")
                        f.write("=" * 80 + "\n\n")
                        for i, task in enumerate(all_failures, 1):
                            f.write(f"FAILURE {i}/{len(all_failures)}:\n")
                            f.write(f"  Patient:        {task.patient}\n")
                            f.write(f"  Timepoint:      T{task.timepoint}\n")
                            f.write(f"  Survey:         {task.survey_question}\n")
                            f.write(f"  Video:          {task.video_path}\n")
                            f.write(f"  ERROR MESSAGE:  {task.error_message}\n")
                            f.write("\n")
                        f.write("=" * 80 + "\n\n")
                    
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("EXECUTION COMPLETE\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Duration: {timedelta(seconds=int(total_time))}\n")
                    
                    total_success = sum(
                        sum(tr.successful_surveys for tr in pr.timepoint_results) 
                        for pr in patient_results
                    )
                    total_failed = sum(
                        sum(tr.failed_surveys for tr in pr.timepoint_results) 
                        for pr in patient_results
                    )
                    f.write(f"Tasks Successful: {total_success}\n")
                    f.write(f"Tasks Failed: {total_failed}\n")
                    f.write("=" * 80 + "\n")
                logger.info(f"Iteration log saved to: {iteration_log_file}")
            except Exception as e:
                logger.warning(f"Failed to finalize iteration log: {e}")
        
        total_failures = sum(
            sum(tr.failed_surveys for tr in pr.timepoint_results) 
            for pr in patient_results
        )
        logger.info("")
        logger.info(
            f"  ★ BATCH DONE | {timedelta(seconds=int(total_time))} | "
            f"{sum(tr.successful_surveys for pr in patient_results for tr in pr.timepoint_results if not tr.skipped)} OK, "
            f"{total_failures} fail"
        )
        
        if total_failures > 0 and not args.dry_run:
            logger.warning(f"Exiting with code 1 ({total_failures} failed tasks)")
            sys.exit(1)
        
        logger.info("Exiting with code 0 (success)")
        sys.exit(0)
    
    # ==========================================================================
    # SINGLE TIMEPOINT MODE (original behavior)
    # ==========================================================================
    else:
        logger.info("  SINGLE TIMEPOINT T{}: validating patients…".format(args.timepoint))
        valid_patients = []
        for patient in args.patients:
            exists, video_path = validate_patient_video_exists(
                args.base_dir, patient, args.timepoint
            )
            if exists:
                valid_patients.append(patient)
                logger.info(f"  ✓ {patient}")
            else:
                logger.warning(f"  ✗ {patient}: no video, skip")
        
        if not valid_patients:
            logger.error("No valid patients with videos found.")
            sys.exit(1)
        
        # Calculate total tasks
        total_tasks = len(valid_patients) * len(survey_questions)
        
        logger.info(
            f"  PLAN: {', '.join(valid_patients)} | T{args.timepoint} | "
            f"{len(survey_questions)} MSE → {total_tasks} tasks | workers={args.parallel_workers}"
        )
        if not args.quiet:
            for i, q in enumerate(survey_questions[:5], 1):
                logger.info(f"    {i}. {_mse_display(q)}")
            if len(survey_questions) > 5:
                logger.info(f"    … +{len(survey_questions) - 5} more")
        logger.info("=" * _LOG_LINE_WIDTH)
        
        # Execute tasks
        results: List[TaskResult] = []
        start_time = time.time()
        task_num = 0
        
        try:
            for patient_idx, patient in enumerate(valid_patients, 1):
                logger.info("")
                logger.info(
                    f"  ► PATIENT {patient_idx}/{len(valid_patients)}: {patient} T{args.timepoint} | "
                    f"{len(survey_questions)} MSE domains | {'PARALLEL ' + str(args.parallel_workers) + 'w' if args.parallel_workers > 1 else 'SERIAL'}"
                )
                
                # =============================================================
                # PARALLEL EXECUTION MODE
                # =============================================================
                if args.parallel_workers > 1:
                    patient_results_list = run_surveys_parallel(
                        patient=patient,
                        survey_questions=survey_questions,
                        timepoint=args.timepoint,
                        base_dir=args.base_dir,
                        prompt_file=args.prompt,
                        engine_script=str(engine_script),
                        base_task_num=task_num + 1,
                        total_tasks=total_tasks,
                        total_surveys=len(survey_questions),
                        dry_run=args.dry_run,
                        iteration_log_file=iteration_log_file,
                        skip_completed=not args.force_rerun,
                        parallel_workers=args.parallel_workers,
                        api_url=args.api_url,
                    )
                    
                    results.extend(patient_results_list)
                    task_num += len(survey_questions)
                    
                    # Log patient completion
                    patient_success = sum(1 for r in patient_results_list if r.success)
                    patient_failed = sum(1 for r in patient_results_list if not r.success)
                    logger.info(f"  ★ {patient} T{args.timepoint} done: {patient_success}/{len(patient_results_list)} OK")
                    
                    # Check if we should stop due to errors
                    if patient_failed > 0 and not args.continue_on_error and not args.dry_run:
                        logger.warning("Stopping due to errors. Use --continue-on-error to continue.")
                        break
                
                # =============================================================
                # SERIAL EXECUTION MODE (original behavior)
                # =============================================================
                else:
                    for survey_idx, survey_question in enumerate(survey_questions, 1):
                        task_num += 1
                        
                        result = run_single_analysis(
                            patient=patient,
                            survey_question=survey_question,
                            timepoint=args.timepoint,
                            base_dir=args.base_dir,
                            prompt_file=args.prompt,
                            engine_script=str(engine_script),
                            task_num=task_num,
                            total_tasks=total_tasks,
                            survey_num=survey_idx,
                            total_surveys=len(survey_questions),
                            dry_run=args.dry_run,
                            iteration_log_file=iteration_log_file,
                            skip_completed=not args.force_rerun,
                            api_url=args.api_url,
                        )
                        
                        results.append(result)
                        
                        # Log progress
                        elapsed = time.time() - start_time
                        completed = len(results)
                        remaining = total_tasks - completed
                        if completed > 0 and not args.dry_run:
                            avg_per_task = elapsed / completed
                            eta_seconds = avg_per_task * remaining
                            logger.info(f"  Progress: {completed}/{total_tasks} tasks | ETA: {timedelta(seconds=int(eta_seconds))}")
                        
                        # Stop on error if not continuing
                        if not result.success and not args.continue_on_error and not args.dry_run:
                            logger.warning("Stopping due to error. Use --continue-on-error to continue.")
                            break
                        
                        # Wait between tasks to avoid overwhelming the server
                        # Only wait if there are more tasks to process and not in dry-run mode
                        if remaining > 0 and not args.dry_run and args.delay_between_tasks > 0:
                            logger.info(f"  Waiting {args.delay_between_tasks}s before next task (server cooldown)...")
                            time.sleep(args.delay_between_tasks)
                            logger.debug(f"  Cooldown complete, proceeding to next task.")
                    
                    # Check if we should stop (inner loop break)
                    if results and not results[-1].success and not args.continue_on_error and not args.dry_run:
                        break
                    
                    patient_results_filtered = [r for r in results if r.patient == patient]
                    patient_success = len([r for r in patient_results_filtered if r.success])
                    logger.info(f"  ★ {patient} T{args.timepoint} done: {patient_success}/{len(patient_results_filtered)} OK")
        
        except KeyboardInterrupt:
            logger.warning("\n\nInterrupted by user (Ctrl+C)")
        
        total_time = time.time() - start_time
        
        # Print summary
        if results and not args.dry_run:
            print_summary(results, total_time)
        elif args.dry_run:
            logger.info("")
            log_header("DRY RUN COMPLETE", char="=", width=80)
            logger.info(f"  Would process {len(results)} tasks")
            logger.info(f"  Patients: {', '.join(valid_patients)}")
            logger.info(f"  Surveys per patient: {len(survey_questions)}")
        
        # Save results if requested
        if args.save_results and results:
            save_results_json(results, args.save_results)
        
        # Finalize iteration log with FAILURES SUMMARY
        if iteration_log_file:
            try:
                with open(iteration_log_file, 'a', encoding='utf-8') as f:
                    # Collect all failures for summary
                    all_failures = [r for r in results if not r.success]
                    
                    # Write FAILURES SUMMARY first (for easy debugging)
                    if all_failures:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("⚠️  FAILURES SUMMARY (with error messages for debugging)\n")
                        f.write("=" * 80 + "\n\n")
                        for i, task in enumerate(all_failures, 1):
                            f.write(f"FAILURE {i}/{len(all_failures)}:\n")
                            f.write(f"  Patient:        {task.patient}\n")
                            f.write(f"  Timepoint:      T{task.timepoint}\n")
                            f.write(f"  Survey:         {task.survey_question}\n")
                            f.write(f"  Video:          {task.video_path}\n")
                            f.write(f"  ERROR MESSAGE:  {task.error_message}\n")
                            f.write("\n")
                        f.write("=" * 80 + "\n\n")
                    
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("EXECUTION COMPLETE\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Duration: {timedelta(seconds=int(total_time))}\n")
                    f.write(f"Tasks Successful: {len([r for r in results if r.success])}\n")
                    f.write(f"Tasks Failed: {len([r for r in results if not r.success])}\n")
                    f.write("=" * 80 + "\n")
                logger.info(f"Iteration log saved to: {iteration_log_file}")
            except Exception as e:
                logger.warning(f"Failed to finalize iteration log: {e}")
        
        failed_count = len([r for r in results if not r.success])
        logger.info("")
        logger.info(
            f"  ★ BATCH DONE | {timedelta(seconds=int(total_time))} | "
            f"{len(results) - failed_count} OK, {failed_count} fail"
        )
        if failed_count > 0 and not args.dry_run:
            logger.warning(f"  Exit 1 ({failed_count} failed)")
            sys.exit(1)
        
        logger.info("  Exit 0 (success)")
        sys.exit(0)


if __name__ == "__main__":
    main()
