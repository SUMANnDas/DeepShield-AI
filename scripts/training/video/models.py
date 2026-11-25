from __future__ import annotations
import torch
from torch import nn
from torchvision.models import efficientnet_b0

class EfficientNetFrameClassifier(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__(); self.encoder=efficientnet_b0(weights=None); self.encoder.classifier=nn.Identity(); self.head=nn.Linear(1280,num_classes)
    def forward(self,x):return self.head(self.encoder(x))

class TemporalAttention(nn.Module):
    def __init__(self,dim):super().__init__(); self.score=nn.Linear(dim,1)
    def forward(self,x,mask):
        logits=self.score(x).squeeze(-1).masked_fill(~mask,-torch.inf)
        # all-missing sequences return finite uniform-zero pooled features
        empty=~mask.any(1); logits=torch.where(empty[:,None],torch.zeros_like(logits),logits)
        weights=torch.softmax(logits,dim=1)*mask.float(); weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        return torch.sum(x*weights.unsqueeze(-1),dim=1),weights

class EfficientNetBiGRU(nn.Module):
    def __init__(self,hidden_dim=256,layers=2,dropout=.3,freeze_encoder=False):
        super().__init__(); self.encoder=efficientnet_b0(weights=None); self.encoder.classifier=nn.Identity(); self.temporal=nn.GRU(1280,hidden_dim,layers,batch_first=True,bidirectional=True,dropout=dropout if layers>1 else 0); self.attention=TemporalAttention(hidden_dim*2); self.dropout=nn.Dropout(dropout); self.head=nn.Linear(hidden_dim*2,2)
        if freeze_encoder:
            for p in self.encoder.parameters():p.requires_grad=False
    def encode_frames(self,frames,mask):
        b,t,c,h,w=frames.shape; flat=frames.reshape(b*t,c,h,w); valid=mask.reshape(-1); out=frames.new_zeros((b*t,1280))
        if valid.any():out[valid]=self.encoder(flat[valid])
        return out.reshape(b,t,1280)
    def forward(self,frames,mask):
        features=self.encode_frames(frames,mask); temporal,_=self.temporal(features); pooled,weights=self.attention(temporal,mask); logits=self.head(self.dropout(pooled)); return {'logits':logits,'frame_scores':torch.sigmoid(self.attention.score(temporal).squeeze(-1)).masked_fill(~mask,0),'attention':weights,'features':temporal}

class TemporalTransformer(nn.Module):
    def __init__(self,dim=1280,max_frames=12,layers=2,heads=4,dropout=.1):
        super().__init__(); self.position=nn.Parameter(torch.zeros(1,max_frames,dim)); layer=nn.TransformerEncoderLayer(dim,heads,dim*2,dropout,batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,layers); self.attention=TemporalAttention(dim); self.head=nn.Linear(dim,2)
    def forward(self,features,mask):
        x=self.encoder(features+self.position[:,:features.size(1)],src_key_padding_mask=~mask); pooled,weights=self.attention(x,mask); return {'logits':self.head(pooled),'frame_scores':torch.sigmoid(x.mean(-1)).masked_fill(~mask,0),'attention':weights,'features':x}
