from __future__ import annotations
import torch
from torch import nn
from .utils import seed_everything

def train_epoch(model,loader,optimizer,device='cpu'):
    model.train();loss_fn=nn.CrossEntropyLoss();total=0.;n=0
    for batch in loader:
        frames=batch['frames'].to(device);mask=batch['mask'].to(device);labels=batch['label'].to(device);optimizer.zero_grad(set_to_none=True)
        if frames.ndim==5 and not hasattr(model,'temporal'):
            valid=mask.reshape(-1); frames=frames.reshape(-1,*frames.shape[2:])[valid]; labels=labels[:,None].expand(-1,mask.shape[1]).reshape(-1)[valid]
            if not len(labels): continue
            out={'logits':model(frames)}
        else: out=model(frames,mask) if frames.ndim==5 else {'logits':model(frames)}
        loss=loss_fn(out['logits'],labels)
        if not torch.isfinite(loss):raise FloatingPointError('non-finite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item()*len(labels);n+=len(labels)
    return total/max(1,n)

def fit(model,train_loader,val_loader,optimizer,epochs,device,seed=42,checkpoint_dir=None):
    seed_everything(seed);history=[];best=float('inf')
    for epoch in range(1,epochs+1):
        tr=train_epoch(model,train_loader,optimizer,device);model.eval();vl=0.;n=0
        with torch.no_grad():
            for b in val_loader:
                x=b['frames'].to(device);m=b['mask'].to(device);y=b['label'].to(device)
                if x.ndim==5 and not hasattr(model,'temporal'):
                    valid=m.reshape(-1);x=x.reshape(-1,*x.shape[2:])[valid];y=y[:,None].expand(-1,m.shape[1]).reshape(-1)[valid]
                    if not len(y):continue
                    o={'logits':model(x)}
                else:o=model(x,m) if x.ndim==5 else {'logits':model(x)}
                vl+=nn.functional.cross_entropy(o['logits'],y,reduction='sum').item();n+=len(y)
        vl/=max(1,n);history.append({'epoch':epoch,'train_loss':tr,'validation_loss':vl})
        if checkpoint_dir and vl<best:
            best=vl; from .checkpoint import save_checkpoint; save_checkpoint(f'{checkpoint_dir}/best-{epoch:03}.pt',model,optimizer,epoch)
    return history
