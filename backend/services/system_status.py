from __future__ import annotations
import os, platform, shutil, sqlite3
from pathlib import Path
import torch
from scripts.training.image.experiment import PROJECT_ROOT
from backend.services.analysis_store import DB_PATH

def system_status() -> dict:
    gpu=None; free_vram=None
    if torch.cuda.is_available():
        props=torch.cuda.get_device_properties(0); gpu=props.name; free_vram=int(props.total_memory)
    ram_total=None; ram_available=None
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
        info=MEMORYSTATUSEX();info.dwLength=ctypes.sizeof(info)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):ram_total=int(info.ullTotalPhys);ram_available=int(info.ullAvailPhys)
    except Exception: pass
    usage=shutil.disk_usage(PROJECT_ROOT)
    database_status="not_initialized"
    if DB_PATH.is_file():
        try:
            uri=f"file:{DB_PATH.as_posix()}?mode=ro"
            with sqlite3.connect(uri,uri=True,timeout=2) as connection:
                database_status="ok" if connection.execute("PRAGMA quick_check").fetchone()[0]=="ok" else "unavailable"
        except sqlite3.Error:
            database_status="unavailable"
    return {"python_version":platform.python_version(),"cpu_name":platform.processor() or platform.machine(),"logical_cpu_count":os.cpu_count(),"database_status":database_status,"platform":platform.platform(),"pytorch_version":torch.__version__,"torchvision_version":__import__('torchvision').__version__,"cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda,"gpu_name":gpu,"gpu_memory_total_bytes":free_vram,"ram_total_bytes":ram_total,"ram_available_bytes":ram_available,"disk":{"project_root_free_bytes":usage.free,"total_bytes":usage.total},"image_models_available":sum(x['status']=='available' for x in __import__('backend.services.model_registry',fromlist=['list_models']).list_models()),"video_prediction_status":"unavailable_no_validated_video_checkpoint"}
