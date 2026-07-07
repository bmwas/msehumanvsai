"""
Zero-overhead wandb telemetry for Blackwell API.
Fire-and-forget: metrics are put on a queue and drained by a daemon thread.
No inference path blocks on wandb. If wandb is unavailable, all methods are no-ops.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import logging
from typing import Any, Dict, Optional

# Optional dependencies: graceful no-op if missing
_wandb = None
_pynvml = None
try:
    import wandb as _wandb
except ImportError:
    pass
try:
    import pynvml as _pynvml
except ImportError:
    pass

_METRICS_QUEUE_MAXSIZE = 10000
_GPU_SAMPLE_INTERVAL_S = 30.0


def _noop(*args: Any, **kwargs: Any) -> None:
    pass


class WandbLogger:
    """
    Non-blocking wandb logger. log() and log_config() enqueue; a daemon thread drains.
    If wandb is unavailable or WANDB_API_KEY is missing, all methods are no-ops.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        mode: Optional[str] = None,
        name: Optional[str] = None,
        enable_gpu_sampler: bool = True,
    ) -> None:
        self._api_key = (api_key or os.getenv("WANDB_API_KEY", "")).strip()
        self._project = (project or os.getenv("WANDB_PROJECT", "qwen3omni-blackwell")).strip()
        self._mode = (mode or os.getenv("WANDB_MODE", "online")).strip().strip('"').strip("'").lower()
        if self._mode not in {"online", "offline", "disabled", "dryrun"}:
            self._mode = "online"
        self._name = name or os.getenv("WANDB_RUN_NAME", "blackwell-api")
        self._enable_gpu_sampler = enable_gpu_sampler and _pynvml is not None
        self._queue: queue.Queue = queue.Queue(maxsize=_METRICS_QUEUE_MAXSIZE)
        self._config_logged = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._gpu_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._enabled = bool(_wandb and self._api_key and self._mode != "disabled")

        if _wandb is None or not self._api_key:
            self.log = _noop
            self.log_config = _noop
            self.shutdown = _noop
            return

        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()
        if self._enable_gpu_sampler:
            self._gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True)
            self._gpu_thread.start()

    def log(self, metrics: Dict[str, Any]) -> None:
        """Fire-and-forget: enqueue metrics. Drops if queue full. No blocking."""
        if _wandb is None or not self._api_key:
            return
        try:
            self._queue.put_nowait(("log", metrics))
        except queue.Full:
            pass

    def log_config(self, config: Dict[str, Any]) -> None:
        """Enqueue config to be applied once at next drain. Non-blocking."""
        if _wandb is None or not self._api_key:
            return
        try:
            self._queue.put_nowait(("config", config))
        except queue.Full:
            pass

    def _drain_loop(self) -> None:
        run = None
        _log_exception_once = [True]  # mutable so inner except can set to False
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            kind, payload = item
            try:
                if kind == "config":
                    if run is None:
                        run = _wandb.init(
                            project=self._project,
                            config=payload,
                            mode=self._mode,
                            name=self._name,
                        )
                    else:
                        for k, v in payload.items():
                            _wandb.config.update({k: v}, allow_val_change=True)
                elif kind == "log":
                    if run is None:
                        run = _wandb.init(
                            project=self._project,
                            mode=self._mode,
                            name=self._name,
                        )
                    _wandb.log(payload)
            except Exception as e:
                if _log_exception_once[0]:
                    _log_exception_once[0] = False
                    logging.getLogger(__name__).warning(
                        "wandb telemetry: first drain error (wandb.init or wandb.log failed): %s", e
                    )

    def _gpu_loop(self) -> None:
        if _pynvml is None:
            return
        try:
            _pynvml.nvmlInit()
        except Exception:
            return
        try:
            device_count = _pynvml.nvmlDeviceGetCount()
        except Exception:
            device_count = 0
        while not self._shutdown.is_set():
            self._shutdown.wait(_GPU_SAMPLE_INTERVAL_S)
            if self._shutdown.is_set():
                break
            metrics: Dict[str, Any] = {}
            try:
                for i in range(device_count):
                    handle = _pynvml.nvmlDeviceGetHandleByIndex(i)
                    try:
                        mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)
                        metrics[f"gpu/{i}/memory_used_mb"] = mem.used // (1024 * 1024)
                        metrics[f"gpu/{i}/memory_total_mb"] = mem.total // (1024 * 1024)
                        metrics[f"gpu/{i}/memory_pct"] = 100.0 * mem.used / mem.total if mem.total else 0
                    except Exception:
                        pass
                    try:
                        util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics[f"gpu/{i}/utilization_pct"] = util.gpu
                    except Exception:
                        pass
                    try:
                        temp = _pynvml.nvmlDeviceGetTemperature(handle, _pynvml.NVML_TEMPERATURE_GPU)
                        metrics[f"gpu/{i}/temperature_c"] = temp
                    except Exception:
                        pass
                    try:
                        power = _pynvml.nvmlDeviceGetPowerUsage(handle)
                        metrics[f"gpu/{i}/power_w"] = power / 1000.0
                    except Exception:
                        pass
                if metrics:
                    self.log(metrics)
            except Exception:
                pass
        try:
            _pynvml.nvmlShutdown()
        except Exception:
            pass

    def shutdown(self) -> None:
        self._shutdown.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def is_enabled(self) -> bool:
        """True if wandb is available and WANDB_API_KEY is set; otherwise logging is no-op."""
        return getattr(self, "_enabled", False)


# Singleton used by app_blackwell
_wandb_logger: Optional[WandbLogger] = None
_wandb_logger_lock = threading.Lock()


def get_wandb_logger() -> WandbLogger:
    global _wandb_logger
    with _wandb_logger_lock:
        if _wandb_logger is None:
            _wandb_logger = WandbLogger()
        return _wandb_logger
