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
        b,t,c,h,w=frames.shape
        flat=frames.reshape(b*t,c,h,w)
        valid=mask.reshape(-1)
        if valid.any():
            encoded=self.encoder(flat[valid])
            out=encoded.new_zeros((b*t,1280))
            out[valid]=encoded
        else:
            out=frames.new_zeros((b*t,1280))
        return out.reshape(b,t,1280)
    def forward(self,frames,mask):
        features=self.encode_frames(frames,mask); temporal,_=self.temporal(features); pooled,weights=self.attention(temporal,mask); logits=self.head(self.dropout(pooled)); return {'logits':logits,'frame_scores':torch.sigmoid(self.attention.score(temporal).squeeze(-1)).masked_fill(~mask,0),'attention':weights,'features':temporal}

class TemporalTransformer(nn.Module):
    def __init__(self,dim=1280,max_frames=12,layers=2,heads=4,dropout=.1):
        super().__init__(); self.position=nn.Parameter(torch.zeros(1,max_frames,dim)); layer=nn.TransformerEncoderLayer(dim,heads,dim*2,dropout,batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,layers); self.attention=TemporalAttention(dim); self.head=nn.Linear(dim,2)
    def forward(self,features,mask):
        if features.size(1) > self.position.size(1):
            raise ValueError(f"sequence length {features.size(1)} exceeds configured maximum {self.position.size(1)}")
        empty=~mask.any(1)
        safe_mask=mask.clone()
        safe_mask[empty,0]=True
        x=self.encoder(features+self.position[:,:features.size(1)],src_key_padding_mask=~safe_mask)
        x=x.masked_fill(empty[:,None,None],0.)
        pooled,weights=self.attention(x,mask)
        logits=self.head(pooled).masked_fill(empty[:,None],0.)
        return {'logits':logits,'frame_scores':torch.sigmoid(x.mean(-1)).masked_fill(~mask,0),'attention':weights,'features':x}


class EfficientNetTransformerVideoClassifier(nn.Module):
    """EfficientNet frame features followed by masked temporal attention."""

    def __init__(self, max_frames=12, layers=2, heads=4, dropout=0.1, freeze_encoder=False):
        super().__init__()
        self.encoder = efficientnet_b0(weights=None)
        self.encoder.classifier = nn.Identity()
        self.temporal = TemporalTransformer(
            dim=1280,
            max_frames=max_frames,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def encode_frames(self, frames, mask):
        batch, steps, channels, height, width = frames.shape
        flat = frames.reshape(batch * steps, channels, height, width)
        valid = mask.reshape(-1)
        if valid.any():
            encoded = self.encoder(flat[valid])
            features = encoded.new_zeros((batch * steps, 1280))
            features[valid] = encoded
        else:
            features = frames.new_zeros((batch * steps, 1280))
        return features.reshape(batch, steps, 1280)

    def forward(self, frames, mask):
        return self.temporal(self.encode_frames(frames, mask), mask)
