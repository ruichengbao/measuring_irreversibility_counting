#!/usr/bin/env python3
"""Refine the histogram quadrature while holding fitted functions and schedules fixed."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json
from functools import lru_cache
import numpy as np
import pandas as pd
from shortlag_core import load_run,edge_histograms,kernels,prob_from_kernels,reverse,coarsen,fit,score

PACKAGE=Path(__file__).resolve().parent.parent

def validate(N,E,raw,results,output,refined=640,grids=(10,20,160)):
    names=[f'N{N:03d}_E{E:.1f}_r{i}' for i in range(5)]
    runs=[load_run(Path(raw)/n) for n in names]
    saved=[json.loads((Path(results)/(n+'_result.json')).read_text()) for n in names]
    M=runs[0]['meta']['M'];fineK=160
    @lru_cache(maxsize=48)
    def hist(rep,end,steps,S):
        return edge_histograms(runs[rep],end,steps,S)
    @lru_cache(maxsize=64)
    def probabilities(rep,target,end,steps,S):
        m=runs[target]['meta']
        kern=kernels(float(m['Gamma']),float(m['Q']),float(E),M,end,steps,fineK,S)
        hf,hr=hist(rep,end,steps,S)
        return prob_from_kernels(hf,*kern),reverse(prob_from_kernels(hr,*kern))
    rows=[]
    for target,record in enumerate(saved):
        S=record['S'];peers=[i for i in range(5) if i!=target]
        for K in grids:
            result=record['grids'][str(K)];original=0.;fine=0.
            for edge in result['edges']:
                end,steps=edge['end_phase'],edge['lag_steps']
                pp,qq=zip(*(probabilities(rep,target,end,steps,S) for rep in peers))
                f=fit(coarsen(np.mean(pp,axis=0),K),coarsen(np.mean(qq,axis=0),K),edge['cap'],edge['kind'])
                p,q=probabilities(target,target,end,steps,S)
                original+=score(coarsen(p,K),coarsen(q,K),f)
                p,q=probabilities(target,target,end,steps,refined)
                fine+=score(coarsen(p,K),coarsen(q,K),f)
            original/=result['normalizer'];fine/=result['normalizer']
            if not np.isclose(original,result['phase_resolved_rate'],rtol=2e-12,atol=2e-12):
                raise AssertionError(f'Reconstructed score differs from saved value: {original} vs {result["phase_resolved_rate"]}')
            baseline=result['baseline_rate']
            rows.append(dict(N=N,E=E,rep=target,K=K,S=S,S_refined=refined,
                baseline_rate=baseline,original_rate=original,refined_rate=fine,
                absolute_shift=fine-original,relative_shift=(fine-original)/original,
                original_gain=original/baseline-1,refined_gain=fine/baseline-1,
                expanded_selected=result['expanded_selected'],strategy=result['phase_strategy']))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/f'N{N:03d}_E{E:.1f}_fixed_function_S{refined}.csv',index=False)
    return frame

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=PACKAGE/'engine/raw')
    p.add_argument('--results',type=Path,default=PACKAGE/'analysis/results')
    p.add_argument('--output',type=Path,default=PACKAGE/'analysis/validation')
    p.add_argument('--refined',type=int,default=640)
    p.add_argument('--N',type=int,nargs='+',default=[2,5]);p.add_argument('--E',type=float,default=1.)
    a=p.parse_args()
    frame=pd.concat([validate(N,a.E,a.raw,a.results,a.output,a.refined) for N in a.N],ignore_index=True)
    summary=frame.groupby(['N','E','K']).agg(original_rate=('original_rate','mean'),
        refined_rate=('refined_rate','mean'),baseline_rate=('baseline_rate','mean'),
        maximum_absolute_relative_shift=('relative_shift',lambda x:np.max(np.abs(x))),
        minimum_refined_gain=('refined_gain','min'),maximum_refined_gain=('refined_gain','max')).reset_index()
    summary['original_mean_gain']=summary.original_rate/summary.baseline_rate-1
    summary['refined_mean_gain']=summary.refined_rate/summary.baseline_rate-1
    summary.to_csv(a.output/f'fixed_function_S{a.refined}_summary.csv',index=False)
    print(summary.to_string(index=False))
