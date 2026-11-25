from __future__ import annotations
import cv2, numpy as np
from .face_detection import valid_bbox

def crop_face(rgb,bbox,size=224,padding=.15):
    if rgb is None or rgb.ndim!=3 or rgb.shape[2]!=3: raise ValueError('Expected RGB image')
    h,w=rgb.shape[:2]
    if not valid_bbox(bbox,w,h): raise ValueError('Invalid or out-of-bounds face bbox')
    x,y,bw,bh=map(int,bbox); px,py=round(bw*padding),round(bh*padding)
    x0=max(0,x-px); y0=max(0,y-py); x1=min(w,x+bw+px); y1=min(h,y+bh+py)
    crop=rgb[y0:y1,x0:x1]
    if crop.size==0: raise ValueError('Empty face crop')
    return cv2.resize(crop,(size,size),interpolation=cv2.INTER_AREA if crop.shape[0]>size else cv2.INTER_LINEAR),(x0,y0,x1-x0,y1-y0)
