#!/usr/bin/env python3
"""Aggregate steady comparison results without changing their estimates."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
def aggregate(source,output=None,allow_partial=False):
 source=Path(source);output=Path(output or source);output.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(f.read_text()) for f in source.glob('*_result.json')]
 d=pd.DataFrame(rows).sort_values(['N','E','rep'])
 if not allow_partial and len(d)!=225:raise ValueError(f'Expected225results,found{len(d)}')
 if d.duplicated(['N','E','rep']).any():raise ValueError('Duplicate realizations')
 d.to_csv(output/'steady_replicates.csv',index=False)
 summary=[]
 for (N,E),a in d.groupby(['N','E']):
  for K in [5,20]:
   tr=a.true_rate.mean();est=a[f'estimator_K{K}'];tur=a[f'tur_K{K}']
   summary.append(dict(N=N,E=E,K=K,true_rate=tr,true_rate_sd=a.true_rate.std(ddof=1),true_rate_sampling_se=np.sqrt(np.sum(a.true_rate_sampling_se**2))/len(a),
    density_rate=est.mean(),density_rate_sd=est.std(ddof=1),tur_rate=tur.mean(),tur_rate_sd=tur.std(ddof=1),Q_density=est.mean()/tr,Q_tur=tur.mean()/tr,
    Q_density_sd=(est/a.true_rate).std(ddof=1),Q_tur_sd=(tur/a.true_rate).std(ddof=1),realizations=len(a),
    density_lag_min=a[f'lag_K{K}'].min(),density_lag_max=a[f'lag_K{K}'].max(),tur_lag_min=a[f'tur_lag_K{K}'].min(),tur_lag_max=a[f'tur_lag_K{K}'].max()))
 s=pd.DataFrame(summary);s.to_csv(output/'steady_summary.csv',index=False)
 expected={(N,E,r) for N in [1,2,5,10,20,50,60,80,100] for E in [.1,.2,.5,.8,1.] for r in range(5)}
 present=set(d[['N','E','rep']].itertuples(index=False,name=None))
 if not allow_partial and present!=expected:raise ValueError('Parameter inventory differs from requested 225 realizations')
 primary=s[s.K==5];interacting=d[d.N>1]
 checks=dict(realizations=len(d),complete_inventory=present==expected,summary_points_per_grid=len(primary),
  primary_density_over_ftur_ratio_min=float((primary.density_rate/primary.tur_rate).min()),
  primary_density_over_ftur_ratio_max=float((primary.density_rate/primary.tur_rate).max()),
  primary_every_realization_exceeds_target_scan_ftur=bool((interacting.estimator_K5>interacting.tur_test_scan_max_K5).all()),
  primary_minimum_realization_advantage_over_target_scan_ftur=float((interacting.estimator_K5/interacting.tur_test_scan_max_K5).min()),
  minimum_true_rate=float(d.true_rate.min()))
 for K in [5,20]:
  q=d[f'estimator_K{K}']/d.true_rate
  checks[f'density_Q_min_K{K}']=float(q.min());checks[f'density_Q_max_K{K}']=float(q.max())
  checks[f'all_finite_K{K}']=bool(np.isfinite(d[['true_rate',f'estimator_K{K}',f'tur_K{K}']].to_numpy()).all())
 (output/'analysis_checks.json').write_text(json.dumps(checks,indent=2))
 print('Wrote',len(d),'replicates and',len(summary),'summaryrows')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=HERE/'results');p.add_argument('--output',type=Path);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();aggregate(a.source,a.output,a.allow_partial)
