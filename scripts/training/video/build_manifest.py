from __future__ import annotations
import json
from pathlib import Path
from .video_metadata import inspect_video
from .utils import sha256_file

LABELS={'REAL':0,'MANIPULATED':1}
def build_record(path,video_id,source_video_id,label,relative_path,manipulation='original',compression='c23'):
    if label not in LABELS: raise ValueError('label must be REAL or MANIPULATED')
    p=Path(path); meta=inspect_video(p)
    return {'video_id':str(video_id),'source_video_id':str(source_video_id),'relative_path':str(relative_path),'label':LABELS[label],'label_name':label,'manipulation_method':manipulation,'compression':compression,'file_hash':meta['file_hash'],'duration_seconds':meta['duration_seconds'],'fps':meta['fps'],'width':meta['width'],'height':meta['height'],'frame_count':meta['frame_count'],'codec':meta['codec'],'audio_present':meta['audio_present'],'decode_status':meta['decode_status']}

def write_manifest(records,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')

def read_manifest(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(line) for line in f if line.strip()]
