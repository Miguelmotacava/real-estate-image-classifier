"""Hardware detection and device-specific optimization utilities."""
from __future__ import annotations

import os
import platform

import torch


def detect_device(verbose: bool = True) -> torch.device:
    """Detect the best available compute device and apply backend tweaks.

    Returns
    -------
    torch.device
        ``cuda``, ``mps`` or ``cpu`` according to availability.
    """
    if verbose:
        print("=== Hardware Report ===")
        print(f"PyTorch: {torch.__version__}")
        print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            if verbose:
                print(f"CUDA GPU {i}: {props.name} - {props.total_memory // 1024**2} MB VRAM")
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if verbose:
            print("Apple MPS (Metal) available")
        device = torch.device("mps")
    else:
        if verbose:
            print("Only CPU available - applying CPU optimizations")
        torch.set_num_threads(os.cpu_count() or 1)
        device = torch.device("cpu")

    if verbose:
        print(f"Selected device: {device}")
    return device


def recommended_batch_size(device: torch.device) -> int:
    """Heuristic batch size based on device VRAM (or CPU default)."""
    if device.type == "cuda":
        vram_mb = torch.cuda.get_device_properties(0).total_memory // 1024**2
        if vram_mb < 4096:
            return 16
        if vram_mb < 8192:
            return 32
        if vram_mb < 16384:
            return 64
        return 128
    if device.type == "mps":
        return 32
    return 16  # CPU-friendly default


def supports_amp(device: torch.device) -> bool:
    """Whether automatic mixed precision is worth enabling."""
    return device.type in {"cuda", "mps"}


if __name__ == "__main__":
    dev = detect_device()
    print(f"Recommended batch size: {recommended_batch_size(dev)}")
    print(f"AMP supported: {supports_amp(dev)}")
