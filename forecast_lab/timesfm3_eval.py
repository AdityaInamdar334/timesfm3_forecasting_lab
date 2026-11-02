"""OPTIONAL real TimesFM-3 inference using Google's official API.
Requires separate model checkpoint download and acceptance of license.
Never executed or claimed as part of packaged benchmark results.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from .data import make_energy, load_etth1, make_windows, origins_for
from .models import seasonal_naive
from .metrics import score,seasonal_scale


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['synthetic','etth1'],default='synthetic')
    parser.add_argument('--data-path',default='data/ETTh1.csv')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--max-origins',type=int,default=6)
    parser.add_argument('--output',default='results/timesfm3_optional.json')
    args=parser.parse_args()
    try:
        from timesfm3 import TimesFM3Evaluator, ModelConfig
    except ImportError as e:
        raise SystemExit("TimesFM-3 not installed. See the official google-research/timesfm README: pip install 'timesfm[torch]'") from e
    ds=make_energy(0) if args.dataset=='synthetic' else load_etth1(args.data_path)
    context,horizon=96,24
    print('Loading official checkpoint. Download/model loading time is EXCLUDED from inference latency.')
    cfg=ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',per_core_batch_size=8,device=args.device)
    t0=time.perf_counter()
    forecaster=TimesFM3Evaluator(cfg)
    load_seconds=time.perf_counter()-t0
    scale=seasonal_scale(ds.values[:ds.train_end,:3],24)
    results={'real_timesfm3':True,'checkpoint':'google/timesfm-3.0-pytorch',
             'licensed_for_research_only':True,'dataset':args.dataset,'device':args.device,
             'load_seconds':load_seconds,'splits':{}}
    for split,(start,stop) in origins_for(ds,horizon).items():
        raw,y,origins=make_windows(ds,context,horizon,start,stop,stride=horizon)
        raw=raw[:args.max_origins];y=y[:args.max_origins];origins=origins[:args.max_origins]
        pred=[];q=[]; timings=[];uni=[];uni_timing=[]
        for w in raw:
            targets=w[:context,:3].T.astype(np.float32)
            cov_hist=w[:context,3:4].T.astype(np.float32)
            cov_future=w[:,4:].T.astype(np.float32)
            t0=time.perf_counter()
            out=list(forecaster.predict_batch(
                contexts=[targets],horizon=horizon,past_only_covariates=[cov_hist],
                past_future_covariates=[cov_future],return_quantiles=True,
                use_symmetric_averaging=False,make_positive=False))[0]
            timings.append(time.perf_counter()-t0)
            pred.append(np.asarray(out.forecast));q.append(np.asarray(out.quantiles))
            t0=time.perf_counter()
            single=[]
            for target in targets:
                u=list(forecaster.predict_batch(contexts=[target],horizon=horizon,
                             return_quantiles=False,use_symmetric_averaging=False,make_positive=False))[0]
                single.append(np.asarray(u.forecast))
            uni_timing.append(time.perf_counter()-t0)
            uni.append(np.stack(single,axis=0))
        pred=np.stack(pred);q=np.stack(q);uni=np.stack(uni)
        if pred.shape!=y.shape or q.shape!=(*y.shape,9):
            raise ValueError(f'Unexpected model outputs: {pred.shape} / {q.shape}, expected {y.shape}')
        tau=np.arange(1,10,dtype=float)/10
        e=y[:,:,:,None]-q
        pinball=np.maximum(tau*e,(tau-1)*e)
        coverage=float(np.mean((y>=q[:,:,:,0])&(y<=q[:,:,:,-1])))
        results['splits'][split]={
          'origins':origins.tolist(),'samples':int(len(y)),
          'seasonal_naive':score(y,seasonal_naive(raw,context,horizon),scale),
          'timesfm3_multivariate':score(y,pred,scale),
          'timesfm3_univariate':score(y,uni,scale),
          'mean_pinball_multivariate':float(pinball.mean()),
          'empirical_10_90_coverage':coverage,
          'mean_multivariate_latency_s':float(np.mean(timings)),
          'mean_three_univariate_latency_s':float(np.mean(uni_timing)),
        }
        print(split,results['splits'][split],flush=True)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2)); print('Saved',out)
if __name__=='__main__': main()
