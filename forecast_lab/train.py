import random
import numpy as np
import torch
from .data import make_windows,TrainScaler,input_visibility
from .models import MiniMaskedForecaster

def train_model(ds, seed=0, context=96,horizon=24,steps=160,batch=32,lr=0.0015, cross=True, use_future=True):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    scaler=TrainScaler().fit(ds)
    raw,y,_=make_windows(ds,context,horizon,start=context,stop=ds.train_end,stride=2)
    xs=scaler.transform(raw).astype('float32')
    ys=((y-scaler.mu[:3][None,:,None])/scaler.sd[:3][None,:,None]).astype('float32')
    x,mask=input_visibility(xs,context)
    if not use_future:
        x[:,context:,4:]=0
        mask[:,context:,4:]=0
    x=torch.from_numpy(x);mask=torch.from_numpy(mask);ys=torch.from_numpy(ys)
    m=MiniMaskedForecaster(context=context,horizon=horizon,cross=cross)
    optimizer=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    rng=np.random.default_rng(seed)
    m.train()
    for step in range(steps):
        j=torch.from_numpy(rng.integers(0,len(xs),size=batch,dtype=np.int64))
        pred=m(x[j],mask[j]);loss=(pred-ys[j]).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);optimizer.step()
    return m,scaler,float(loss.item()),len(xs)
