#!/usr/bin/env python3
"""Exact one-particle control using the SAME five seeded species assignments."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,numpy as np,pandas as pd
from analyze_run import probability,kl,fit_function,variational,LAGS
from analyze_group import GRIDS

HERE=Path(__file__).resolve().parent; OUT=HERE.parent/'data/analysis_runs';OUT.mkdir(exist_ok=True)
species=[(.02,.5,1.),(.03,1/3,.9),(.01,1.,.9),(.02,.5,1.1),(.02,.5,1.1)]
for rep,(sigma,D,q) in enumerate(species):
    Gamma=1/D;n=np.arange(-int(np.ceil(np.sqrt(35*Gamma)/(2*np.pi/5))),int(np.ceil(np.sqrt(35*Gamma)/(2*np.pi/5)))+1)
    A=np.ones((len(n),len(n)),complex)
    for E in [.1,.2,.4,.5,.8,1.]:
        name=f'N001_E{E:.1f}_r{rep}'
        true=1.5*D*q*q*E*E
        result=dict(name=name,N=1,E=E,rep=rep,dt=0.,true_rate=true,true_rate_sampling_se=0.,
            collective_true_rate=true,true_rate_upper=true,internal_work=0.,internal_bulk_work=0.,
            max_ibp_discrepancy=0.,acceptance=1.,fourier_kmax=int(n[-1]),max_negative_mass=0.,
            production_periods=0,correlation_frames=0,max_relative_radius=0.,relative_rg2=0.,
            weighted_center_error=0.,method='exact wrapped Brownian propagator',
            sigma=sigma,D=D,q=q,disorder_seed=171109+47*rep)
        rows=[];saved={}
        for K in GRIDS:
            best=(-np.inf,None,None)
            for lag in LAGS:
                p,neg=probability(A,n,Gamma,q,E,lag,K)
                value=variational(p,fit_function(p))/lag
                rows.append(dict(lag=lag,boxes_per_side=K,phase='max',plugin_rate=kl(p)/lag,
                    train_0=value,heldout_0=value,train_1=value,heldout_1=value))
                if value>best[0]:best=(value,lag,p)
            result[f'estimator_K{K}']=best[0];result[f'train_K{K}']=best[0]
            result[f'lag_K{K}']=best[1];result[f'phase_K{K}']='max';result[f'cap_K{K}']=20.;result[f'plugin_K{K}']=best[0]
            for fold in range(2):
                result[f'lag{fold}_K{K}']=best[1];result[f'phase{fold}_K{K}']='max'
                if K in [5,80,160]:
                    for s in ['p_train','p_test','p_full']:saved[f'K{K}_fold{fold}_{s}']=best[2]
        result['estimator_rate']=result['estimator_K160'];result['quality_factor_Q']=result['estimator_rate']/true;result['selection_optimism']=0.
        (OUT/(name+'_result.json')).write_text(json.dumps(result,indent=2))
        pd.DataFrame(rows).to_csv(OUT/(name+'_lag_grid_scan.csv'),index=False)
        np.savez_compressed(OUT/(name+'_selected_correlations.npz'),**saved)
print('Generated exact N=1 controls:30cases')
