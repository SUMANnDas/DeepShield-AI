from __future__ import annotations
import torch
from .preprocess_video import preprocess_video
from .report import forensic_report

@torch.no_grad()
def predict_video(path,model,device='cpu',max_frames=12,size=224,detector=None):
    prepared=preprocess_video(path,detector,max_frames,size=size)
    if prepared.get('error') or not prepared['frames']:return {'status':'error','error':prepared.get('error','no_valid_face_crops'),'metadata':prepared['metadata']}
    imgs=torch.stack([torch.from_numpy(x['crop'].transpose(2,0,1).copy()).float()/255. for x in prepared['frames']]); seq=torch.zeros(1,max_frames,3,size,size);mask=torch.zeros(1,max_frames,dtype=torch.bool);n=min(max_frames,len(imgs));seq[0,:n]=imgs[:n];mask[0,:n]=True
    model.eval();out=model(seq.to(device),mask.to(device));p=torch.softmax(out['logits'],1)[0,1].item();scores=out['frame_scores'][0,:n].cpu().tolist();evidence=[{'frame_index':x['frame_index'],'timestamp_seconds':x['timestamp_seconds'],'score':s} for x,s in zip(prepared['frames'][:n],scores)]
    return {'status':'ok','report':forensic_report(prepared['metadata'].get('file_hash'),p,evidence,prepared['metadata'],type(model).__name__,{'size':size,'sampling':'uniform','face_detector':prepared['face_detector']},prepared.get('errors'))}
