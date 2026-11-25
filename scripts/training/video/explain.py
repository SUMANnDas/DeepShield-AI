from __future__ import annotations
import numpy as np
import cv2

def frame_heatmap(rgb,score):
    """Score overlay is an evidence visualization, not a causal explanation."""
    heat=np.zeros(rgb.shape[:2],dtype=np.uint8); heat[:]=int(np.clip(score,0,1)*255); heat=cv2.applyColorMap(heat,cv2.COLORMAP_JET); heat=cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb.astype('uint8'),.65,heat,.35,0)
