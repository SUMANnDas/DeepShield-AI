from __future__ import annotations
from pathlib import Path
import cv2
from .utils import sha256_file

def inspect_video(path: str | Path) -> dict:
    p=Path(path); result={'file_path':str(p),'file_hash':None,'duration_seconds':None,'fps':None,'width':None,'height':None,'frame_count':None,'codec':None,'audio_present':None,'decode_status':'failed','error':None}
    if not p.is_file() or p.stat().st_size==0:
        result['error']='missing_or_empty_file'; return result
    result['file_hash']=sha256_file(p)
    cap=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened(): result['error']='decoder_open_failed'; return result
        fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result.update(fps=float(fps) if fps>0 else None,frame_count=n or None,width=w or None,height=h or None,duration_seconds=(n/fps if fps>0 and n>0 else None),audio_present=None,decode_status='metadata_only')
        fourcc=int(cap.get(cv2.CAP_PROP_FOURCC)); result['codec']=''.join(chr((fourcc>>(8*i))&255) for i in range(4)).strip('\x00') if fourcc else None
        cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,frame=cap.read()
        result['decode_status']='ok' if ok and frame is not None else 'failed'
        if result['decode_status']=='failed': result['error']='first_frame_decode_failed'
    except Exception as e: result['error']=f'{type(e).__name__}: {e}'
    finally: cap.release()
    return result
