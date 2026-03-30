"""Device detection with graceful GPU → CPU fallback.

Supports ROCm (AMD), CUDA (NVIDIA), and CPU.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def get_device(override: str | None = None) -> torch.device:
    """Detect the best available compute device.

    Priority: env var FC_DEVICE > override arg > ROCm/CUDA GPU > CPU.

    Args:
        override: Optional device string (e.g., "cpu", "cuda", "cuda:0").

    Returns:
        A torch.device for the selected backend.
    """
    env_device = os.environ.get("FC_DEVICE")
    if env_device:
        device = torch.device(env_device)
        logger.info("Device from FC_DEVICE env var: %s", device)
        return device

    if override:
        device = torch.device(override)
        logger.info("Device from override: %s", device)
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        backend = "ROCm" if is_rocm else "CUDA"
        logger.info("Using GPU (%s): %s", backend, name)
        return device

    logger.info("No GPU available, using CPU")
    return torch.device("cpu")


def device_info(device: torch.device) -> dict[str, str | int | bool]:
    """Return a summary of the given device's capabilities.

    Args:
        device: The torch device to inspect.

    Returns:
        Dictionary with device type, name, memory, and backend info.
    """
    info: dict[str, str | int | bool] = {"type": device.type}

    if device.type == "cuda":
        idx = device.index or 0
        info["name"] = torch.cuda.get_device_name(idx)
        info["memory_gb"] = round(torch.cuda.get_device_properties(idx).total_mem / 1e9, 1)
        info["is_rocm"] = hasattr(torch.version, "hip") and torch.version.hip is not None
    else:
        info["name"] = "CPU"

    return info
