from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import numpy as np
import torch

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json_exclusive(path: str | Path, value) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2); f.write('\n')

def seed_everything(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    if hasattr(torch.backends,'cudnn'): torch.backends.cudnn.benchmark=False

