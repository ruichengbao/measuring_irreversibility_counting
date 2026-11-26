#!/usr/bin/env python3
"""Cross-fitted density estimator and fixed-current fluctuation-theorem bound."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd
from steady_core import *

def analyze_group(N,E,runs,output,S=0,stride=10,qmc_power=11,qmc_origins=256):
 from moment_kernel import moments
 runs=Path(runs);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 if not S:S=128 if N<=5 else 256 if N<=20 else 512
 records=[]
 for rep in range(5):
  name=f'N{N:03d}_E{E:.1f}_r{rep}';m,t,y,stats=load_run(runs/name)
  records.append(dict(name=name,meta=m,t=t,y=y,stats=stats))
 start=time.time();best=[{K:None for K in GRIDS} for _ in records]
 scans=[[] for r in records];moment_records=[{} for r in records];tur_candidates=[{K:[] for K in GRIDS} for _ in records]
 for tau in LAGS:
  shifts=[round(tau/np.median(np.diff(r['t']))) for r in records]
  effective_stride=min(stride,max(1,round(tau/.05)))
  H=[histogram(r['y'],shift,S,stride=effective_stride)[0] for r,shift in zip(records,shifts)]
  for K in GRIDS:
   probs=np.array([[probabilities(h,r['meta'],E,tau,K) for h in H] for r in records])
   for i,r in enumerate(records):
    peers=probs[i,np.arange(5)!=i];cv=nested_cv(peers,tau);means=cv.mean(axis=1);j=int(np.argmax(means));cap=CAPS[j]
    current=best[i][K]
    if current is None or means[j]>current['selection_score']:
     train=peers.mean(axis=0);f=fit(train,cap)
     best[i][K]=dict(tau=tau,cap=cap,selection_score=float(means[j]),p_train=train,p_test=probs[i,i].copy(),f=f)
    m=r['meta'];p=probs[i,i];mu=current_mean(p)
    qm=moments(r['y'],shifts[i],K,m['Gamma'],m['Q'],E,tau,origins=qmc_origins,power=qmc_power,seed=890000+N*1000+round(E*10)*31+i)
    second=float(qm['second_moment']);tvalue=tur_rate(mu,second,tau)
    moment_records[i][K,tau]=qm
    entry=dict(tau=tau,boxes_per_side=K,current_mean=mu,current_second_moment=second,current_variance=second-mu*mu,tur_rate=tvalue,
      histogram_origin_stride=effective_stride,qmc_mean=float(qm['mean']),qmc_origins=int(qm['origins']),qmc_power=qmc_power,qmc_seed=890000+N*1000+round(E*10)*31+i,
      density_plugin_rate=plugin(p)/tau)
    tur_candidates[i][K].append(entry)
    for c in range(len(CAPS)):
     scans[i].append(dict(tau=tau,boxes_per_side=K,cap=CAPS[c],density_training_cv=float(means[c]),**{a:b for a,b in entry.items() if a not in ['tau','boxes_per_side']}))
  print(f'N={N} E={E} tau={tau} elapsed={time.time()-start:.1f}s',flush=True)
 results=[]
 for i,r in enumerate(records):
  result=work_statistics(r['meta'],r['stats'],r['y']);result.update(name=r['name'],rep=i,quadrature_grid=S,histogram_origin_stride=stride,qmc_power=qmc_power,qmc_origins=qmc_origins,
   selection_lags=LAGS,selection_grids=GRIDS,selection_caps=CAPS,box_label_order='m=iy*K+ix',field_protocol='constant')
  saved={}
  for K in GRIDS:
   b=best[i][K];value=score(b['p_test'],b['f'])/b['tau']
   peer_rates=np.array([[v['tur_rate'] for v in tur_candidates[j][K]] for j in range(5) if j!=i]);sel=int(np.argmax(peer_rates.mean(axis=0)))
   tb=tur_candidates[i][K][sel]
   result.update({f'estimator_K{K}':value,f'lag_K{K}':b['tau'],f'cap_K{K}':b['cap'],f'train_cv_K{K}':b['selection_score'],
    f'plugin_K{K}':plugin(b['p_test'])/b['tau'],f'tur_K{K}':tb['tur_rate'],f'tur_lag_K{K}':tb['tau'],f'tur_current_mean_K{K}':tb['current_mean'],
    f'tur_current_second_moment_K{K}':tb['current_second_moment'],f'tur_current_variance_K{K}':tb['current_variance'],f'tur_training_score_K{K}':float(peer_rates.mean(axis=0)[sel]),f'tur_test_scan_max_K{K}':max(v['tur_rate'] for v in tur_candidates[i][K])})
   for name in ['p_train','p_test','f']:saved[f'K{K}_{name}']=b[name]
   qm=moment_records[i][K,tb['tau']]
   for name in ['origin_indices','per_origin_mean','per_origin_second']:saved[f'K{K}_tur_{name}']=qm[name]
  result['analysis_seconds']=time.time()-start
  dest=output/(r['name']+'_result.json');tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2));os.replace(tmp,dest)
  pd.DataFrame(scans[i]).to_csv(output/(r['name']+'_scan.csv'),index=False)
  np.savez_compressed(output/(r['name']+'_selected.npz'),**saved)
  results.append(result)
 return results

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('N',type=int);p.add_argument('E',type=float);p.add_argument('--runs',type=Path,default=HERE.parent/'engine/raw');p.add_argument('--output',type=Path,default=HERE/'results');p.add_argument('--S',type=int,default=0);p.add_argument('--stride',type=int,default=10);p.add_argument('--qmc-power',type=int,default=11);p.add_argument('--qmc-origins',type=int,default=256);a=p.parse_args()
 analyze_group(a.N,a.E,a.runs,a.output,a.S,a.stride,a.qmc_power,a.qmc_origins)
