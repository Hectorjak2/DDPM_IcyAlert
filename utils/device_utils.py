"""
Utility functions for checking and managing compute devices (CPU, CUDA, MPS).
"""

import torch
import time


def get_device_info() -> dict:
    """
    Get comprehensive information about available compute devices.

    Returns:
        dict: Dictionary containing device availability and information
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "cpu_available": True,
    }

    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        info["current_cuda_device"] = torch.cuda.current_device()

    if info["mps_available"]:
        info["mps_enabled"] = True

    return info


def print_device_info() -> None:
    """Pretty print device information to console."""
    info = get_device_info()

    print("=" * 60)
    print("PyTorch Device Information")
    print("=" * 60)
    print(f"PyTorch Version: {info['pytorch_version']}")
    print()

    print("Available Devices:")
    print(f"  CPU: {'✓' if info['cpu_available'] else '✗'}")
    print(f"  CUDA: {'✓' if info['cuda_available'] else '✗'}")
    if info["cuda_available"]:
        print(f"    └─ CUDA Version: {info['cuda_version']}")
        print(f"    └─ Device Count: {info['cuda_device_count']}")
        for idx, name in enumerate(info["cuda_device_names"]):
            current = " (current)" if idx == info["current_cuda_device"] else ""
            print(f"       [{idx}] {name}{current}")

    print(f"  MPS (Metal): {'✓' if info['mps_available'] else '✗'}")
    print()
    print("=" * 60)


def get_device() -> torch.device:
    """
    Get the best available device for computation.
    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: The recommended device for training
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device (no GPU acceleration available)")

    return device


def benchmark_device(device: torch.device, size: int = 1000, iterations: int = 100) -> dict:
    """
    Simple benchmark to verify accelerated computation.

    Args:
        device: The device to benchmark
        size: Size of matrices to multiply (size x size)
        iterations: Number of iterations to run

    Returns:
        dict: Benchmark results including time and throughput
    """
    # Create test tensors
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(10):
        _ = torch.mm(a, b)

    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = torch.mm(a, b)

    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    ops = iterations * 2 * (size ** 3) / elapsed / 1e9  # GFLOPS

    return {
        "device": str(device),
        "matrix_size": size,
        "iterations": iterations,
        "time_seconds": elapsed,
        "gflops": ops,
    }


def print_benchmark(device: torch.device, size: int = 1000, iterations: int = 100) -> None:
    """
    Run and pretty print benchmark results.

    Args:
        device: The device to benchmark
        size: Size of matrices to multiply
        iterations: Number of iterations to run
    """
    print(f"\nBenchmarking {device}...")
    print(f"Matrix Size: {size}x{size}, Iterations: {iterations}")

    results = benchmark_device(device, size, iterations)

    print(f"Time Elapsed: {results['time_seconds']:.4f} seconds")
    print(f"Performance: {results['gflops']:.2f} GFLOPS")
    print()


def verify_cuda_is_working() -> bool:
    """
    Verify that CUDA is properly installed and working.

    Returns:
        bool: True if CUDA is available and working, False otherwise
    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return False

    try:
        # Create a tensor on CUDA
        x = torch.rand(10, 10, device="cuda")
        y = torch.rand(10, 10, device="cuda")

        # Perform an operation
        z = torch.mm(x, y)

        # Move back to CPU
        z_cpu = z.cpu()

        print("✓ CUDA is working correctly!")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        return True
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def device_diagnostics(device): 
    # Print device information
    print_device_info()

    # Verify it's working
    print("\nVerifying device...")
    if device.type == "cuda":
        verify_cuda_is_working()
    else:
        print("✓ Device is ready for computation")

    # Run benchmark on primary device
    print_benchmark(device)

    # If we have GPU acceleration, also benchmark CPU for comparison
    if device.type in ["cuda", "mps"]:
        print("\nBenchmarking CPU for comparison...")
        cpu_device = torch.device("cpu")
        print_benchmark(cpu_device)

        # Print speedup
        gpu_results = benchmark_device(device)
        cpu_results = benchmark_device(cpu_device)
        speedup = cpu_results["time_seconds"] / gpu_results["time_seconds"]
        print(f"GPU Speedup: {speedup:.2f}x faster than CPU")
        print("=" * 60)

if __name__ == "__main__":
    # Get the best device
    device = get_device()
    device_diagnostics(device)