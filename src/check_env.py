import subprocess, sys

results = {}

# PyTorch
try:
    import torch
    results["torch"] = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "version": torch.__version__
    }
except ImportError:
    results["torch"] = "not installed"

# PyTorch Geometric
try:
    import torch_geometric
    results["torch_geometric"] = {
        "version": torch_geometric.__version__,
        "cuda_available": torch.cuda.is_available()
    }
except ImportError:
    results["torch_geometric"] = "not installed"

# CuPy
try:
    import cupy as cp
    results["cupy"] = {
        "available": True,
        "device_count": cp.cuda.runtime.getDeviceCount(),
        "device_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    }
except Exception as e:
    results["cupy"] = f"error: {e}"

# cuDF
try:
    import cudf
    df = cudf.DataFrame({"a": [1, 2, 3]})
    results["cudf"] = {"available": True, "version": cudf.__version__}
except Exception as e:
    results["cudf"] = f"error: {e}"

# cuGraph
try:
    import cugraph
    results["cugraph"] = {"available": True, "version": cugraph.__version__}
except Exception as e:
    results["cugraph"] = f"error: {e}"

# JAX
try:
    import jax
    results["jax"] = {
        "devices": [str(d) for d in jax.devices()],
        "gpu_available": jax.default_backend() == "gpu"
    }
except Exception as e:
    results["jax"] = f"error: {e}"

# Numba CUDA
try:
    from numba import cuda
    results["numba_cuda"] = {
        "available": cuda.is_available(),
        "device": str(cuda.get_current_device()) if cuda.is_available() else None
    }
except Exception as e:
    results["numba_cuda"] = f"error: {e}"

# RAPIDS Dask / dask-cudf
try:
    import dask_cudf
    results["dask_cudf"] = {"available": True, "version": dask_cudf.__version__}
except Exception as e:
    results["dask_cudf"] = f"error: {e}"

# nvidia-ml-py (pynvml)
try:
    import pynvml
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    devices = []
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        devices.append(pynvml.nvmlDeviceGetName(handle))
    results["pynvml"] = {"device_count": count, "devices": devices}
except Exception as e:
    results["pynvml"] = f"error: {e}"

# Print summary
print("\n=== GPU Access Check ===\n")
for lib, status in results.items():
    print(f"{lib}:")
    if isinstance(status, dict):
        for k, v in status.items():
            print(f"  {k}: {v}")
    else:
        print(f"  {status}")
    print()
