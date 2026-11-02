"""Usage: python -m forecast_lab.benchmark --seeds 2 --steps 160 --output results/benchmark.json"""
import argparse,json,time
from pathlib import Path
import numpy as np
from .data import make_energy,make_windows,origins_for
from .train import train_model
from .models import seasonal_naive,RidgeDirect,ridge_features,predict_minimodel
from .metrics import score,seasonal_scale,paired_origin_bootstrap_delta


def run(seed,steps,dataset="synthetic",data_path="data/ETTh1.csv"):
    ds=make_energy(seed) if dataset=="synthetic" else __import__("forecast_lab.data",fromlist=["load_etth1"]).load_etth1(data_path)
    con,hor=96,24
    model,scaler,loss,ntrain=train_model(ds,seed=seed,context=con,horizon=hor,steps=steps)
    no_cross,_,_,_=train_model(ds,seed=seed,context=con,horizon=hor,steps=steps,cross=False)
    no_future,_,_,_=train_model(ds,seed=seed,context=con,horizon=hor,steps=steps,use_future=False)
    raw_tr,y_tr,_=make_windows(ds,con,hor,con,ds.train_end,stride=2)
    xtr=ridge_features(raw_tr,con,hor)
    ridge=RidgeDirect(alpha=200).fit(xtr,y_tr)
    no_cov=RidgeDirect(alpha=200).fit(ridge_features(raw_tr,con,hor,include_covariates=False),y_tr)
    scale=seasonal_scale(ds.values[:ds.train_end,:3],24)
    results={'seed':seed,'train_windows':ntrain,'last_training_loss':loss,'splits':{}}
    # No in-test training or refitting, so shifts reflect genuine holdout errors.
    for name,(start,stop) in origins_for(ds,hor).items():
        raw,y,origins=make_windows(ds,con,hor,start,stop,stride=24)
        s=seasonal_naive(raw,con,hor)
        r=ridge.predict(ridge_features(raw,con,hor),3,hor)
        r_no=no_cov.predict(ridge_features(raw,con,hor,include_covariates=False),3,hor)
        x=scaler.transform(raw).astype('float32')
        z=predict_minimodel(model,x,con)
        m=scaler.inverse_targets(z)
        m_nocov=scaler.inverse_targets(predict_minimodel(no_future,x,con,with_future=False))
        m_nocross=scaler.inverse_targets(predict_minimodel(no_cross,x,con))
        predictions={'seasonal_naive':s,'ridge_multivariate':r,'ridge_no_future_cov':r_no,
                     'mini_alt_attention':m,'mini_temporal_only':m_nocross,
                     'mini_trained_no_future_cov':m_nocov}
        part={'origins':origins.tolist(),'metrics':{k:score(y,p,scale) for k,p in predictions.items()},
              'delta_ci_vs_seasonal':{k:paired_origin_bootstrap_delta(y,s,p,seed=seed) for k,p in predictions.items() if k!='seasonal_naive'}}
        results['splits'][name]=part
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--start-seed',type=int,default=0)
    ap.add_argument('--steps',type=int,default=160)
    ap.add_argument('--output',default='results/benchmark.json')
    ap.add_argument('--dataset',choices=['synthetic','etth1'],default='synthetic')
    ap.add_argument('--data-path',default='data/ETTh1.csv')
    args=ap.parse_args()
    start=time.perf_counter()
    runs=[]
    for seed in range(args.start_seed,args.start_seed+args.seeds):
        r=run(seed,args.steps,dataset=args.dataset,data_path=args.data_path);runs.append(r)
        print('seed',seed,{k:{m:round(v['mase'],3) for m,v in x['metrics'].items()} for k,x in r['splits'].items()},flush=True)
    result={'description':('Independent synthetic hourly electricity benchmark' if args.dataset=='synthetic' else 'ETTh1 real-world transformer operating metrics benchmark') + ', NOT Google TimesFM-3 results',
            'configs':{'dataset':args.dataset,'seeds':args.seeds,'steps':args.steps,'context':96,'horizon':24,'stride':24,
                       'train_cutoff': (864 if args.dataset=='synthetic' else '60%'), 'shift_start':(1152 if args.dataset=='synthetic' else '80%')},'runs':runs,'elapsed_seconds':round(time.perf_counter()-start,2)}
    for split in ('in_domain','shift'):
        result.setdefault('aggregate',{})[split]={}
        for name in runs[0]['splits'][split]['metrics']:
            vals=[r['splits'][split]['metrics'][name]['mase'] for r in runs]
            result['aggregate'][split][name]={'mean_mase':round(float(np.mean(vals)),5),
                                               'sd_mase':round(float(np.std(vals,ddof=1)),5) if len(vals)>1 else None}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print('Saved:',out,'elapsed',result['elapsed_seconds'])
if __name__=='__main__':main()
