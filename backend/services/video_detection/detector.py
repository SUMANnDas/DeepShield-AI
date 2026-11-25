from __future__ import annotations
from pathlib import Path
from scripts.training.video.inference import predict_video

class VideoDetectionService:
    """Local model adapter. Caller supplies a validated temporary file and loaded model."""
    def __init__(self,model,device='cpu',max_bytes=200_000_000):self.model=model;self.device=device;self.max_bytes=max_bytes
    def detect(self,path):
        p=Path(path)
        if not p.is_file() or p.stat().st_size>self.max_bytes:raise ValueError('invalid or oversized video input')
        result=predict_video(p,self.model,self.device)
        # Do not return host paths to clients.
        if result.get('metadata'): result['metadata'].pop('file_path',None)
        if result.get('report',{}).get('metadata'): result['report']['metadata'].pop('file_path',None)
        return result
