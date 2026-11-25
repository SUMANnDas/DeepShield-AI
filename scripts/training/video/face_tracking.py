from __future__ import annotations
from .face_detection import FaceDetection

def associate_faces(frames:list[list[FaceDetection]],iou_threshold=.2):
    """Greedy within-video IoU matching; IDs are track IDs, not identities."""
    tracks={}; active={}; next_id=0; output=[]
    for detections in frames:
        current={}; used=set()
        for d in detections:
            x,y,w,h=d.bbox; best=None; score=iou_threshold
            for tid,prev in active.items():
                if tid in used: continue
                a,b,c,e=prev.bbox; ix=max(x,a); iy=max(y,b); ir=min(x+w,a+c); ib=min(y+h,b+e); inter=max(0,ir-ix)*max(0,ib-iy); union=w*h+c*e-inter
                value=inter/union if union else 0
                if value>score:best,score=tid,value
            if best is None: best=next_id; next_id+=1; tracks[best]=[]
            tracks[best].append(d); current[best]=d; used.add(best); output.append({'track_id':best,'detection':d,'iou_match':score if score>iou_threshold else None})
        active=current
    return output
