#!/usr/bin/env python3
"""Create all plotted tables from the saved per-realization analyses."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd

def aggregate(source,out,require_complete=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(f.read_text()) for f in Path(source).glob('*_result.json')]
    d=pd.DataFrame(records).sort_values(['E','N','rep'])
    if require_complete and len(d)!=270:raise ValueError(f'Expected270realizations,found{len(d)}')
    d.to_csv(out/'fig3_replicates.csv',index=False)
    for K,label in [(160,'fig3_summary.csv'),(5,'fig3_fixed5_summary.csv')]:
        rows=[]
        for (N,E),a in d.groupby(['N','E']):
            true=a.true_rate.mean();est=a[f'estimator_K{K}'].mean()
            qr=a[f'estimator_K{K}']/a.true_rate
            rows.append(dict(N=N,E=E,true_rate=true,true_rate_sd=a.true_rate.std(ddof=1),
                true_rate_thermal_sem=np.sqrt(np.sum(a.true_rate_sampling_se**2))/len(a),
                estimator_rate=est,estimator_rate_sd=a[f'estimator_K{K}'].std(ddof=1),
                quality_factor_Q=est/true,quality_factor_mean_of_ratios=qr.mean(),quality_factor_sd=qr.std(ddof=1),
                realizations=len(a),boxes_per_side=K,lag_min=a[f'lag_K{K}'].min(),lag_max=a[f'lag_K{K}'].max()))
        pd.DataFrame(rows).to_csv(out/label,index=False)
    cols=['name','N','E','rep','production_periods']+[c for c in d if c.startswith(('lag_K','phase_K','cap_K','estimator_K'))]
    d[cols].to_csv(out/'optimization_summary.csv',index=False)
    print(f'Wrote{len(d)}realizations and {len(rows)}plotpoints perfigure')

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser();p.add_argument('--source',default=str(here.parent/'data/analysis_runs'))
    p.add_argument('--output',default=str(here.parent/'data'));p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();aggregate(a.source,a.output,not a.allow_partial)
