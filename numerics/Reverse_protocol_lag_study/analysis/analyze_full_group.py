#!/usr/bin/env python3
"""Cross-fit the full phase/lag/partition optimization for one (N,E) group."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from shortlag_core import *

PACKAGE=Path(__file__).resolve().parent.parent
OLD=PACKAGE/'baseline/analysis_runs'
RAW=PACKAGE/'engine/raw'
OUT=Path(__file__).resolve().parent/'results'

def analyze(N,E,raw=RAW,old=OLD,output=OUT,S=0,grids=GRIDS,fallback_se=1.):
    start=time.time();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not S:S=320 if N<=20 else 640
    names=[f'N{N:03d}_E{E:.1f}_r{i}' for i in range(5)]
    runs=[load_run(Path(raw)/n) for n in names]
    originals=[json.loads((Path(old)/(n+'_result.json')).read_text()) for n in names]
    M=runs[0]['meta']['M'];fineK=max(grids)
    for r in runs:
        if r['meta']['M']!=M:raise ValueError('Phase grids differ')
    edge_cv=np.zeros((5,len(grids),M,M));edge_test=edge_cv.copy()
    edge_caps=edge_cv.copy();edge_kind=np.empty(edge_cv.shape,dtype='U4')
    edge_folds=np.zeros(edge_cv.shape+(4,))
    scans=[];checks=[]
    for end in range(M):
        for steps in range(1,M+1):
            hist=None if N==1 else [edge_histograms(r,end,steps,S) for r in runs]
            for target in range(5):
                meta=runs[target]['meta'];kern=kernels(float(meta['Gamma']),float(meta['Q']),float(E),M,end,steps,fineK,S)
                # Every training relative histogram is transformed with the
                # target's fixed collective parameters, as in the baseline.
                if N==1:
                    exact=np.outer(kern[0][:,0],kern[1][:,0]);exact/=exact.sum()
                    P=np.repeat(exact[None],5,axis=0);Q=reverse(P)
                else:
                    P=np.array([prob_from_kernels(h[0],*kern) for h in hist])
                    Q=np.array([reverse(prob_from_kernels(h[1],*kern)) for h in hist])
                peers=np.arange(5)!=target
                if np.any(P<0) or np.any(Q<0):raise AssertionError('Negative probability')
                if not np.allclose(P.sum(axis=(1,2)),1,atol=3e-14) or not np.allclose(Q.sum(axis=(1,2)),1,atol=3e-14):
                    raise AssertionError('Probability normalization')
                for ki,K in enumerate(grids):
                    p=coarsen(P,K);q=coarsen(Q,K)
                    if N==1:
                        f=fit(p[target],q[target],150.,'full')
                        exact_score=score(p[target],q[target],f)
                        result=dict(cv=exact_score,cv_sd=0.,test=exact_score,cap=150.,kind='full',train=exact_score,fold_scores=[exact_score]*4)
                    else:
                        result=select_function(p[peers],q[peers],p[target],q[target])
                    edge_cv[target,ki,end,steps-1]=result['cv']
                    edge_test[target,ki,end,steps-1]=result['test']
                    edge_caps[target,ki,end,steps-1]=result['cap']
                    edge_kind[target,ki,end,steps-1]=result['kind']
                    edge_folds[target,ki,end,steps-1]=result['fold_scores']
                    scans.append(dict(N=N,E=E,rep=target,K=K,end_phase=end,lag_steps=steps,
                                      lag=steps/M,**result))
                checks.append(dict(target=target,end_phase=end,lag_steps=steps,
                                   min_probability=float(min(P.min(),Q.min())),
                                   max_normalization_error=float(max(abs(P.sum(axis=(1,2))-1).max(),abs(Q.sum(axis=(1,2))-1).max()))))
    records=[]
    for target in range(5):
        original=originals[target]
        result=dict(name=names[target],N=N,E=E,rep=target,M=M,S=S,
                    true_rate=original['true_rate'],true_rate_sampling_se=original['true_rate_sampling_se'],
                    Gamma=runs[target]['meta']['Gamma'],Q=runs[target]['meta']['Q'],
                    origins_per_phase=runs[target]['meta']['origins_per_phase'],
                    training_replicas=[j for j in range(5) if j!=target],
                    analysis_seconds=time.time()-start,
                    selection_method='target_collective_transform_nested_peer_cv',
                    selection_rule='training_one_standard_error_fallback',fallback_se_multiplier=float(fallback_se),
                    grids={})
        for ki,K in enumerate(grids):
            best,candidates=optimal_schedule(edge_cv[target,ki],edge_test[target,ki],strict_short=False)
            baseline=float(original[f'estimator_K{K}'])
            # The archived N=1 benchmark is evaluated exactly, so no fitting
            # or validation score is needed for that deterministic candidate.
            baseline_cv=float(original.get(f'cv_K{K}',baseline))
            schedule_folds=sum((edge_folds[target,ki,j,k-1] for j,k in best['edges']))/best['normalizer']
            if not np.isclose(schedule_folds.mean(),best['cv'],atol=1e-10,rtol=1e-10):raise AssertionError('CV fold aggregation')
            schedule_se=float(schedule_folds.std(ddof=1)/np.sqrt(len(schedule_folds)))
            use_new=best['cv']-fallback_se*schedule_se>baseline_cv
            use_new_unpenalized=best['cv']>baseline_cv
            selected_edges=[]
            for j,k in best['edges']:
                selected_edges.append(dict(end_phase=j,lag_steps=k,lag=k/M,
                                           cap=float(edge_caps[target,ki,j,k-1]),
                                           kind=str(edge_kind[target,ki,j,k-1])))
            result['grids'][str(K)]=dict(
                baseline_rate=baseline,baseline_cv=baseline_cv,
                phase_resolved_rate=best['test'],phase_resolved_cv=best['cv'],
                phase_resolved_cv_se=schedule_se,phase_resolved_cv_folds=schedule_folds.tolist(),
                expanded_rate=best['test'] if use_new else baseline,
                expanded_selected='phase_resolved' if use_new else 'integer_period',
                expanded_unpenalized_rate=best['test'] if use_new_unpenalized else baseline,
                expanded_unpenalized_selected='phase_resolved' if use_new_unpenalized else 'integer_period',
                phase_strategy=best['strategy'],start_phase=best['start_phase'],
                common_lag_steps=best['lag_steps'],normalizer=best['normalizer'],
                edges=selected_edges,
                selected_has_subperiod=any(k<M for _,k in best['edges']),
                baseline_lag=original[f'lag_K{K}'],baseline_phase=original[f'phase_K{K}'],
                candidates=candidates)
        destination=output/(names[target]+'_result.json')
        tmp=destination.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2));os.replace(tmp,destination)
        records.append(result)
    pd.DataFrame(scans).to_csv(output/f'N{N:03d}_E{E:.1f}_edge_scores.csv',index=False)
    pd.DataFrame(checks).to_csv(output/f'N{N:03d}_E{E:.1f}_probability_checks.csv',index=False)
    return records

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('N',type=int);parser.add_argument('E',type=float)
    parser.add_argument('--raw',type=Path,default=RAW);parser.add_argument('--old',type=Path,default=OLD)
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--S',type=int,default=0)
    parser.add_argument('--fallback-se',type=float,default=1.,help='Training-only standard-error margin for switching from the archived integer strategy.')
    args=parser.parse_args()
    records=analyze(args.N,args.E,args.raw,args.old,args.output,args.S,fallback_se=args.fallback_se)
    print(json.dumps([dict(N=r['N'],E=r['E'],rep=r['rep'],seconds=r['analysis_seconds'],
                          baseline=r['grids']['160']['baseline_rate'],phase=r['grids']['160']['phase_resolved_rate'],
                          expanded=r['grids']['160']['expanded_rate'],selected=r['grids']['160']['expanded_selected']) for r in records],indent=2))
