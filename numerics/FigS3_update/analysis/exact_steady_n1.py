#!/usr/bin/env python3
"""Exact density and fixed-label-current moments for five single-particle species."""
from pathlib import Path
import json,numpy as np,pandas as pd
from steady_core import *
from positive_kernel import gaussian_bin_kernel

def run(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 species=[(.02,.5,1.),(.03,1/3,.9),(.01,1.,.9),(.02,.5,1.1),(.02,.5,1.1)]
 for E in [.1,.2,.5,.8,1.]:
  for rep,(sigma,D,q) in enumerate(species):
   name=f'N001_E{E:.1f}_r{rep}';true=D*q*q*E*E;rows=[];saved={}
   r=dict(name=name,N=1,E=E,rep=rep,true_rate=true,true_rate_sampling_se=0.,collective_true_rate=true,internal_work=0.,true_rate_upper=true,
    method='exact wrapped Gaussian propagator and exact current moments',sigma=sigma,D=D,q=q,production_periods=0,correlation_frames=0,dt=0.,acceptance=1.,selection_lags=LAGS,selection_grids=GRIDS,box_label_order='m=iy*K+ix')
   for K in GRIDS:
    choices=[]
    for tau in LAGS:
     h=5./K;z=np.arange(K)*h;s=np.sqrt(2*D*tau)
     p=gaussian_bin_kernel(z-D*q*E*tau,h,s)[:,None]*gaussian_bin_kernel(z,h,s)[None,:];p/=p.sum()
     mu=current_mean(p);second=1-p[0,0];tur=tur_rate(mu,second,tau);bound=plugin(p)/tau
     choices.append(dict(tau=tau,boxes_per_side=K,density_rate=bound,tur_rate=tur,current_mean=mu,current_second_moment=second,current_variance=second-mu*mu));saved[f'K{K}_tau{tau}_p']=p
    best=max(choices,key=lambda x:x['density_rate']);bt=max(choices,key=lambda x:x['tur_rate'])
    r.update({f'estimator_K{K}':best['density_rate'],f'lag_K{K}':best['tau'],f'cap_K{K}':150.,f'plugin_K{K}':best['density_rate'],f'tur_K{K}':bt['tur_rate'],f'tur_lag_K{K}':bt['tau'],f'tur_current_mean_K{K}':bt['current_mean'],f'tur_current_second_moment_K{K}':bt['current_second_moment'],f'tur_current_variance_K{K}':bt['current_variance']})
    rows.extend(choices)
   (output/(name+'_result.json')).write_text(json.dumps(r,indent=2));pd.DataFrame(rows).to_csv(output/(name+'_scan.csv'),index=False);np.savez_compressed(output/(name+'_selected.npz'),**saved)
 print('Completed25exactN1cases')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'results');a=p.parse_args();run(a.output)
