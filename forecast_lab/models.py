"""Predictors and a small, independently designed alternating-attention model.
NOT an implementation of TimesFM 3.0 and NOT a pretrained foundation model.
"""
import numpy as np
import torch
from torch import nn


def seasonal_naive(raw_windows, context, horizon, season=24, targets=3):
    history = raw_windows[:, :context, :targets]
    idx = (np.arange(horizon) % season) + context - season
    return history[:,idx,:].transpose(0,2,1)


def ridge_features(windows, context, horizon, targets=3, include_covariates=True):
    """Compact causal lags plus legally known-ahead covariates."""
    hist = windows[:,:context,:targets]
    feats = [hist[:,-1,:], hist[:,-24,:], hist[:,-48,:], hist[:,-24:,:].mean(1),
             windows[:,context-1,3:4]]
    if include_covariates:
        future_known = windows[:,context:context+horizon,4:]
        feats += [future_known.reshape(len(windows),-1)]
    return np.concatenate(feats,axis=1)

class RidgeDirect:
    def __init__(self, alpha=10.): self.alpha=alpha
    def fit(self,x,y):
        x=np.column_stack([np.ones(len(x)),x]); y=y.reshape(len(y),-1)
        regular=np.eye(x.shape[1]);regular[0,0]=0
        self.w=np.linalg.solve(x.T@x+self.alpha*regular,x.T@y)
        self.shape_y=None
        return self
    def predict(self,x,channels,horizon):
        x=np.column_stack([np.ones(len(x)),x]);return (x@self.w).reshape(len(x),channels,horizon)

class AltAttentionBlock(nn.Module):
    def __init__(self,d=24,heads=4, cross=True):
        super().__init__(); self.cross=cross
        self.temporal=nn.MultiheadAttention(d,heads,batch_first=True)
        self.variate=nn.MultiheadAttention(d,heads,batch_first=True)
        self.ln1=nn.LayerNorm(d);self.ln2=nn.LayerNorm(d);self.ln3=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(self,x):
        b,v,p,d=x.shape
        t=x.reshape(b*v,p,d)
        causal=torch.triu(torch.ones(p,p,device=x.device,dtype=torch.bool),diagonal=1)
        z,_=self.temporal(t,t,t,attn_mask=causal,need_weights=False)
        x=self.ln1(x+z.reshape(b,v,p,d))
        if self.cross:
            t=x.permute(0,2,1,3).reshape(b*p,v,d)
            z,_=self.variate(t,t,t,need_weights=False)
            x=self.ln2(x+z.reshape(b,p,v,d).permute(0,2,1,3))
        return self.ln3(x+self.ff(x))

class MiniMaskedForecaster(nn.Module):
    """Patches every channel, masks unknown future observations, outputs horizon at once."""
    def __init__(self,channels=6,targets=3,context=96,horizon=24,patch=8,d=24,heads=4,layers=2,cross=True):
        super().__init__()
        if context%patch or horizon%patch:raise ValueError('context/horizon must be multiples of patch')
        self.context=context;self.horizon=horizon;self.patch=patch;self.targets=targets
        self.channels=channels;self.cross=cross
        p=(context+horizon)//patch
        self.value=nn.Linear(2*patch,d)
        self.channel_embed=nn.Parameter(torch.randn(channels,d)*0.02)
        self.position=nn.Parameter(torch.randn(p,d)*0.02)
        self.blocks=nn.ModuleList([AltAttentionBlock(d,heads,cross=cross) for _ in range(layers)])
        self.head=nn.Linear(d,patch)
    def forward(self, values, observed_mask):
        # values and masks: [B, context + horizon, channels]
        b,l,c=values.shape
        assert l==self.context+self.horizon and c==self.channels
        inp=torch.stack([values.transpose(1,2),observed_mask.transpose(1,2)],dim=-1)
        inp=inp.reshape(b,c,l//self.patch,self.patch,2).reshape(b,c,l//self.patch,2*self.patch)
        x=self.value(inp)+self.channel_embed[None,:,None,:]+self.position[None,None,:,:]
        for block in self.blocks:x=block(x)
        future=x[:,:self.targets,self.context//self.patch:,:]
        return self.head(future).reshape(b,self.targets,self.horizon)


def predict_minimodel(model, windows, context, batch_size=128, with_future=True):
    from .data import input_visibility
    # windows already scaled; set future unknown to zero *after* scaling
    predictions=[]; model.eval()
    with torch.no_grad():
        for i in range(0,len(windows),batch_size):
            x,mask=input_visibility(windows[i:i+batch_size],context)
            if not with_future:
                x[:,context:,4:]=0;mask[:,context:,4:]=0
            predictions.append(model(torch.from_numpy(x).float(),torch.from_numpy(mask).float()).numpy())
    return np.concatenate(predictions,axis=0)
