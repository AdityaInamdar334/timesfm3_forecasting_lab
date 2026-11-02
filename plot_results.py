"""Generate two publication-oriented figures from saved experimental results."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from forecast_lab.data import make_energy
root=Path(__file__).resolve().parent
p=json.loads((root/'results/benchmark.json').read_text())
labels=['Seasonal naive','Direct ridge','Mini alternating\nattention']
keys=['seasonal_naive','ridge_multivariate','mini_alt_attention']
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
x=np.arange(len(keys));width=.35
for offset,(split,title) in zip((-.5,.5),(('in_domain','Before shift'),('shift','After shift'))):
 a=p['aggregate'][split]
 values=[a[k]['mean_mase'] for k in keys]
 errs=[a[k]['sd_mase'] for k in keys]
 ax.bar(x+offset*width,values,width,yerr=errs,label=title,capsize=3)
ax.set_xticks(x,labels);ax.set_ylabel('MASE (lower is better)')
ax.set_title('Independent synthetic electricity benchmark across five seeds')
ax.legend();ax.grid(axis='y',alpha=.2)
fig.savefig(root/'figures'/'01_multivariate_vs_shift.png',dpi=175)
plt.close(fig)
ds=make_energy(0)
fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
start=1100;stop=1210
ax.plot(np.arange(start,stop),ds.values[start:stop,0],label='Synthetic residential load')
ax.axvline(ds.shift_start,linestyle=':',label='Regime shift at hour 1152')
ax.set_xlabel('Hour index');ax.set_ylabel('Synthetic load (arbitrary units)')
ax.set_title('Forecasting beyond a structural break')
ax.legend();ax.grid(alpha=.2)
fig.savefig(root/'figures'/'02_structural_break.png',dpi=175)
plt.close(fig)
print('Saved figures in figures/')
