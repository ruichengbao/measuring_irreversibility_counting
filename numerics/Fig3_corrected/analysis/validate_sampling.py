#!/usr/bin/env python3
"""Evaluate conditional sampling errors and timestep tests at fixed learned f."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from analyze_run import load_run,reverse
from analyze_group import work_statistics,fit,score
from positive_kernel import kernel_matrices,pair_scores,pair_histograms,prob_from_kernels

HERE=Path(__file__).resolve().parent

def hac_sem(v,lag):
    v=np.asarray(v);n=len(v)
    if n<4:return float('nan')
    d=v-v.mean();m=min(n//3,max(10,2*lag));g0=d@d/n
    variance=g0+2*sum((1-j/(m+1))*float(d[:-j]@d[j:])/n for j in range(1,m+1))
    return float(np.sqrt(max(0,variance)/n))

def fixed_stat(prefix,result,selected,K=160):
    m,t,y,stats=load_run(prefix);y=y[-320:]
    lag=result[f'lag_K{K}'];phase=result[f'phase_K{K}'];S=result['quadrature_grid']
    f=selected[f'K{K}_f'];W=f-np.expm1(reverse(f))
    W=.5*(W+W[:,(-np.arange(K))%K])
    Hx,Hy=kernel_matrices(float(m['Gamma']),float(m['Q']),float(m['E']),float(lag),K,S)
    J=Hx.T@W@Hy
    v=pair_scores(y,lag,J)/lag
    if phase!='pooled':v=v[(1 if phase=='max' else 0)::2]
    w=work_statistics(m,stats,y)
    w.update(estimator_fixed_f=float(v.mean()),estimator_sampling_se=hac_sem(v,lag),lag=lag,phase=phase,dt=m['dt'],num_origins=len(v))
    return w,v

def main(runs,analysis,validation,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sampling=[];timestep=[]
    for resultpath in sorted(Path(analysis).glob('*_result.json')):
        r=json.loads(resultpath.read_text())
        if r['N']==1:continue
        name=r['name'];z=np.load(Path(analysis)/(name+'_selected_correlations.npz'))
        w,v=fixed_stat(Path(runs)/name,r,z)
        sampling.append(dict(name=name,N=r['N'],E=r['E'],
            estimator_sampling_se=w['estimator_sampling_se'],
            true_rate_sampling_se=w['true_rate_sampling_se'],
            reconstruction_error=w['estimator_fixed_f']-r['estimator_rate'],
            num_origins=w['num_origins']))
        fine=Path(validation)/name
        if Path(str(fine)+'_meta.json').exists():
            try:q,qv=fixed_stat(fine,r,z)
            except ValueError:continue
            ds=q['estimator_fixed_f']-w['estimator_fixed_f'];dt=q['true_rate']-w['true_rate']
            se_s=np.hypot(q['estimator_sampling_se'],w['estimator_sampling_se'])
            se_t=np.hypot(q['true_rate_sampling_se'],w['true_rate_sampling_se'])
            timestep.append(dict(name=name,N=r['N'],E=r['E'],coarse_dt=w['dt'],fine_dt=q['dt'],
                true_coarse=w['true_rate'],true_fine=q['true_rate'],true_relative_change=dt/w['true_rate'],
                true_difference_se=se_t,true_difference_z=dt/se_t,
                estimator_coarse=w['estimator_fixed_f'],estimator_fine=q['estimator_fixed_f'],
                estimator_relative_change=ds/w['estimator_fixed_f'],estimator_difference_se=se_s,
                estimator_difference_z=ds/se_s,lag=w['lag'],phase=w['phase']))
    pd.DataFrame(sampling).to_csv(out/'sampling_errors.csv',index=False)
    pd.DataFrame(timestep).to_csv(out/'timestep_comparison.csv',index=False)
    print(json.dumps(dict(runs=len(sampling),timestep_cases=len(timestep),
        max_score_reconstruction_error=max(abs(r['reconstruction_error']) for r in sampling)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',default=str(HERE.parent/'raw'))
    p.add_argument('--analysis',default=str(HERE.parent/'data/analysis_runs'));p.add_argument('--validation',default=str(HERE.parent/'validation/raw'))
    p.add_argument('--output',default=str(HERE.parent/'validation'))
    a=p.parse_args();main(a.runs,a.analysis,a.validation,a.output)
