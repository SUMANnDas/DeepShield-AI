from __future__ import annotations
import cv2, numpy as np

def uniform_frame_indices(frame_count:int, max_frames:int=12)->list[int]:
    if frame_count<0 or max_frames<1: raise ValueError('frame_count must be >=0 and max_frames >=1')
    if frame_count==0:return []
    k=min(frame_count,max_frames)
    return np.unique(np.rint(np.linspace(0,frame_count-1,k)).astype(int)).tolist()

def sample_video(path:str,max_frames:int=12)->list[dict]:
    cap=cv2.VideoCapture(path,cv2.CAP_FFMPEG); out=[]
    try:
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
        for idx in uniform_frame_indices(n,max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,frame=cap.read()
            out.append({'frame_index':idx,'timestamp_seconds':idx/fps if fps>0 else None,'frame':cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) if ok and frame is not None else None,'decode_ok':bool(ok and frame is not None)})
    finally: cap.release()
    return out
