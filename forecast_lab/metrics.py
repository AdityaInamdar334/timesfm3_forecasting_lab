import numpy as np

def mae(y,p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

def seasonal_scale(train_targets,season=24):
    tr=np.asarray(train_targets)
    if len(tr)<=season:raise ValueError('training data too short')
    return np.maximum(np.abs(tr[season:]-tr[:-season]).mean(0),1e-6)

def mase(y,p,scale):
    e=np.mean(np.abs(y-p),axis=(0,2))
    return float(np.mean(e/scale))

def score(y,p,scale):
    return {'mae':round(mae(y,p),5),'rmse':round(rmse(y,p),5),'mase':round(mase(y,p,scale),5),
            'n_origins':int(len(y)),'n_values':int(np.size(y))}

def per_origin_mae(y,p): return np.mean(np.abs(y-p),axis=(1,2))

def paired_origin_bootstrap_delta(y, a, b, seed=0, repeats=1000):
    """Paired origin-level bootstrap for (MAE(a)-MAE(b)); nonoverlapping horizons expected."""
    d=per_origin_mae(y,a)-per_origin_mae(y,b)
    rng=np.random.default_rng(seed)
    samples=rng.choice(d,size=(repeats,len(d)),replace=True).mean(axis=1)
    return {'delta_mean':float(d.mean()),'ci95':[float(z) for z in np.quantile(samples,[.025,.975])]}
