from __future__ import annotations
import torch
from .metrics import binary_metrics

@torch.no_grad()
def evaluate(model,loader,device='cpu'):
    model.eval(); labels=[]; probs=[]; rows=[]
    for b in loader:
        x=b['frames'].to(device);m=b['mask'].to(device)
        if x.ndim==5 and not hasattr(model,'temporal'):
            valid=m.reshape(-1);flat=x.reshape(-1,*x.shape[2:])[valid]
            if not len(flat):continue
            frame_p=torch.softmax(model(flat),1)[:,1];p=[];offset=0
            for row in m:
                count=int(row.sum());p.append(frame_p[offset:offset+count].mean().item());offset+=count
        else:
            out=model(x,m) if x.ndim==5 else {'logits':model(x)};p=torch.softmax(out['logits'],1)[:,1].cpu().tolist()
        probs+=p;labels+=b['label'].tolist()
        for i,v in enumerate(p):rows.append({'video_id':b.get('video_id',[str(len(rows))])[i],'probability_manipulated':v})
    return {'metrics':binary_metrics(labels,probs),'videos':rows}
