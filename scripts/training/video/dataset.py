from __future__ import annotations
import json
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from .frame_sampling import sample_video
from .face_detection import HaarFaceDetector
from .face_cropping import crop_face

class VideoSequenceDataset(Dataset):
    def __init__(self,manifest_path,root,max_frames=12,size=224,detector=None):
        self.records=[json.loads(x) for x in Path(manifest_path).read_text(encoding='utf-8').splitlines() if x.strip()]; self.root=Path(root).resolve(); self.max_frames=max_frames; self.size=size; self.detector=detector or HaarFaceDetector()
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index]; p=(self.root/r['relative_path']).resolve()
        if self.root not in p.parents:raise ValueError('manifest path escapes dataset root')
        sampled=sample_video(str(p),self.max_frames); crops=[]
        for f in sampled:
            if not f['decode_ok']:continue
            ds=self.detector.detect(f['frame'],f['frame_index'],f['timestamp_seconds'])
            if ds:
                d=max(ds,key=lambda x:x.bbox[2]*x.bbox[3]); crop,_=crop_face(f['frame'],d.bbox,self.size); crops.append(crop)
        x=torch.zeros(self.max_frames,3,self.size,self.size,dtype=torch.float32); mask=torch.zeros(self.max_frames,dtype=torch.bool)
        for i,img in enumerate(crops[:self.max_frames]):x[i]=torch.from_numpy(np.ascontiguousarray(img.transpose(2,0,1))).float()/255.; mask[i]=True
        return {'frames':x,'mask':mask,'label':torch.tensor(int(r['label']),dtype=torch.long),'video_id':r['video_id'],'source_video_id':r['source_video_id']}

def collate_sequences(batch):
    return {'frames':torch.stack([x['frames'] for x in batch]),'mask':torch.stack([x['mask'] for x in batch]),'label':torch.stack([x['label'] for x in batch]),'video_id':[x['video_id'] for x in batch],'source_video_id':[x['source_video_id'] for x in batch]}
