from __future__ import annotations
from pathlib import Path
from .video_metadata import inspect_video
from .frame_sampling import sample_video
from .face_detection import HaarFaceDetector
from .face_tracking import associate_faces
from .face_cropping import crop_face

def preprocess_video(path,detector=None,max_frames=12,max_crops=12,size=224,padding=.15,max_frame_pixels=16_000_000):
    meta=inspect_video(path,max_frame_pixels=max_frame_pixels)
    if meta['decode_status']!='ok': return {'metadata':meta,'frames':[],'error':meta['error']}
    sampled=sample_video(str(path),max_frames); detector=detector or HaarFaceDetector(); detections=[]
    for f in sampled:
        ds=detector.detect(f['frame'],f['frame_index'],f['timestamp_seconds']) if f['decode_ok'] else []
        detections.append(ds)
    tracks=associate_faces(detections); selected={}
    tracks_map=_tracks_by_id(tracks)
    # Main subject: most temporally persistent track; ties favor total face area.
    for item in tracks:
        d=item['detection']; area=d.bbox[2]*d.bbox[3]
        score=(len(tracks_map[item['track_id']]),area)
        if item['track_id'] not in selected or score>selected[item['track_id']]['score']:
            selected[item['track_id']]={'score':score,'item':item}
    if selected:
        main=max(selected,key=lambda k:selected[k]['score'])
        for item in tracks:
            if item['track_id']!=main: continue
            d=item['detection']; frame=next(f for f in sampled if f['frame_index']==d.frame_index)
            try:
                crop,box=crop_face(frame['frame'],d.bbox,size,padding)
                yield_record={'frame_index':d.frame_index,'timestamp_seconds':d.timestamp_seconds,'track_id':main,'bbox':list(d.bbox),'crop_bbox':list(box),'confidence':d.confidence,'crop':crop}
                yield_record['crop']=crop
                yield_record['error']=None
                # generator assembled below
                selected.setdefault('_crops',[]).append(yield_record)
            except ValueError as e: selected.setdefault('_errors',[]).append(str(e))
    return {'metadata':meta,'frames':selected.get('_crops',[])[:max_crops],'errors':selected.get('_errors',[]),'face_detector':type(detector).__name__,'frames_sampled':len(sampled)}

def _tracks_by_id(items):
    d={}
    for x in items:d.setdefault(x['track_id'],[]).append(x['detection'])
    return d
