from __future__ import annotations

import os
import platform
import shutil
import sqlite3
from pathlib import Path

import torch

from backend.services.analysis_store import DB_PATH
from backend.services.model_registry import list_models
from backend.services.video_model_registry import list_video_models
from scripts.training.image.experiment import PROJECT_ROOT


def system_status(video_data_root: Path | None = None) -> dict:
    gpu = None
    gpu_total = None
    gpu_free = None
    gpu_allocated = None
    gpu_reserved = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = props.name
        gpu_total = int(props.total_memory)
        try:
            gpu_free = int(torch.cuda.mem_get_info(0)[0])
        except Exception:
            gpu_free = None
        try:
            gpu_allocated = int(torch.cuda.memory_allocated(0))
            gpu_reserved = int(torch.cuda.memory_reserved(0))
        except Exception:
            pass
    ram_total = None
    ram_available = None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        info = MEMORYSTATUSEX()
        info.dwLength = ctypes.sizeof(info)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
            ram_total = int(info.ullTotalPhys)
            ram_available = int(info.ullAvailPhys)
    except Exception:
        pass
    usage = shutil.disk_usage(PROJECT_ROOT)
    database_status = "not_initialized"
    if DB_PATH.is_file():
        try:
            uri = f"file:{DB_PATH.as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=2) as connection:
                database_status = "ok" if connection.execute("PRAGMA quick_check").fetchone()[0] == "ok" else "unavailable"
        except sqlite3.Error:
            database_status = "unavailable"
    image_models = list_models()
    video_models = list_video_models(video_data_root)
    return {
        "python_version": platform.python_version(),
        "cpu_name": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "database_status": database_status,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu,
        "gpu_memory_total_bytes": gpu_total,
        "gpu_memory_free_bytes": gpu_free,
        "gpu_memory_allocated_bytes": gpu_allocated,
        "gpu_memory_reserved_bytes": gpu_reserved,
        "ram_total_bytes": ram_total,
        "ram_available_bytes": ram_available,
        "disk": {"project_root_free_bytes": usage.free, "total_bytes": usage.total},
        "image_models_available": sum(x["status"] == "available" for x in image_models),
        "video_models_available": sum(x["status"] == "available" for x in video_models),
        "video_prediction_status": (
            "available" if any(x["status"] == "available" for x in video_models) else "unavailable_no_validated_video_checkpoint"
        ),
    }
