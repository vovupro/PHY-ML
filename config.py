"""Execution configuration and hardware environment probe for PHY-ML.

Supports:
    - Backend: 'auto' | 'cpu' | 'cuda'
    - Precision: 'double' (float64/complex128) | 'single' (float32/complex64)
    - Batch size, CPU threading, and output directory isolation.
"""
from dataclasses import dataclass
import os
import sys
from typing import Any, Dict, Optional, Tuple
import torch
import sionna


@dataclass
class ExecutionConfig:
    """Execution configuration parameters."""
    backend: str = "cuda"              # Frozen for production Monte Carlo: 'cuda'
    precision: str = "double"          # Frozen for production Monte Carlo: 'double' (FP64)
    batch_blocks: int = 500            # Frozen for production Monte Carlo: 500 blocks/chunk
    cpu_threads: Optional[int] = 12    # Intra-op PyTorch CPU threads
    fresh_run: bool = True             # Do not reuse previous cache
    results_dir: str = "results/l2_cuda_rtx3060"
    master_seed_a: int = 20260918
    master_seed_b: int = 20260919
    initial_blocks_per_seed: int = 5000
    block_increment: int = 5000
    max_blocks_per_seed: int = 200000
    symbols_per_block: int = 1536


def resolve_device_and_dtype(config: ExecutionConfig) -> Tuple[torch.device, torch.dtype, torch.dtype, str]:
    """Resolve active torch device, real dtype, complex dtype, and sionna precision name.

    Returns
    -------
    device : torch.device
    real_dtype : torch.dtype
    complex_dtype : torch.dtype
    precision_str : str ('double' or 'single')
    """
    # 1. Device resolution
    backend = config.backend.lower().strip()
    if backend == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    elif backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Backend 'cuda' was explicitly requested, but torch.cuda.is_available() is False. "
                "Halting to prevent silent CPU fallback."
            )
        device = torch.device("cuda:0")
    elif backend == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported: 'auto', 'cpu', 'cuda'")

    # 2. Precision resolution
    prec = config.precision.lower().strip()
    if prec == "double":
        real_dtype = torch.float64
        complex_dtype = torch.complex128
        precision_str = "double"
    elif prec == "single":
        real_dtype = torch.float32
        complex_dtype = torch.complex64
        precision_str = "single"
    else:
        raise ValueError(f"Unknown precision '{prec}'. Supported: 'double', 'single'")

    return device, real_dtype, complex_dtype, precision_str


def probe_environment(config: Optional[ExecutionConfig] = None) -> Dict[str, Any]:
    """Inspect and report the real runtime environment according to Requirement 3.

    Reports:
        torch version
        Sionna version
        torch.cuda.is_available()
        CUDA runtime version
        GPU name
        VRAM
        driver/device properties
        CPU logical threads
        selected backend
        selected precision
    """
    if config is None:
        config = ExecutionConfig()

    device, real_dtype, complex_dtype, precision_str = resolve_device_and_dtype(config)

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    cuda_runtime = torch.version.cuda if cuda_available else "N/A"
    
    vram_bytes = 0
    vram_str = "N/A"
    device_props: Dict[str, Any] = {}
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        vram_bytes = props.total_memory
        vram_str = f"{vram_bytes / (1024**3):.2f} GB ({vram_bytes / (1024**2):.0f} MiB)"
        device_props = {
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "total_memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "is_integrated": getattr(props, "is_integrated", 0),
            "is_multi_gpu_board": getattr(props, "is_multi_gpu_board", 0),
        }

    cpu_logical = os.cpu_count() or 1

    report = {
        "torch_version": torch.__version__,
        "sionna_version": getattr(sionna, "__version__", "unknown"),
        "cuda_available": cuda_available,
        "cuda_runtime_version": cuda_runtime,
        "gpu_name": gpu_name,
        "vram": vram_str,
        "vram_bytes": vram_bytes,
        "device_properties": device_props,
        "cpu_logical_threads": cpu_logical,
        "configured_backend": config.backend,
        "selected_device": str(device),
        "configured_precision": config.precision,
        "selected_real_dtype": str(real_dtype),
        "selected_complex_dtype": str(complex_dtype),
        "sionna_precision": precision_str,
    }

    return report


def print_environment_report(report: Dict[str, Any]) -> None:
    """Print formatted hardware & runtime environment report."""
    print("=" * 70)
    print("            PHY-ML HARDWARE & CUDA ENVIRONMENT REPORT")
    print("=" * 70)
    print(f"PyTorch Version:         {report['torch_version']}")
    print(f"Sionna Version:          {report['sionna_version']}")
    print(f"CUDA Available:          {report['cuda_available']}")
    print(f"CUDA Runtime Version:    {report['cuda_runtime_version']}")
    print(f"GPU Name:                {report['gpu_name']}")
    print(f"Total VRAM:              {report['vram']}")
    if report['device_properties']:
        props = report['device_properties']
        print(f"Compute Capability:      {props['major']}.{props['minor']}")
        print(f"SM Multiprocessors:      {props['multi_processor_count']}")
    print(f"CPU Logical Processors:  {report['cpu_logical_threads']}")
    print("-" * 70)
    print(f"Selected Backend:        {report['selected_device']} (configured: '{report['configured_backend']}')")
    print(f"Selected Precision:      {report['sionna_precision']} ({report['selected_real_dtype']} / {report['selected_complex_dtype']})")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    cfg = ExecutionConfig()
    env = probe_environment(cfg)
    print_environment_report(env)
