#!/usr/bin/env python3
"""Positive-tail density bound with joint nested cross-validation.

Lag, sampling phase and variational log-ratio cap are chosen jointly using only
the four peers of each target realization. Inner leave-one-peer-out validation
fits each function to three peers. The selected function is refitted to all
four peers and evaluated on the untouched target. Fine-grid correlations are
coarsened by exact positive summation for the nested observation grids.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from functools import lru_cache
import argparse,json,time
import numpy as np
import pandas as pd
from analyze_run import load_run,reverse
from positive_kernel import pair_histograms,kernel_matrices,prob_from_kernels

GRIDS=[5,10,20,40,80,160]
LAGS=list(range(1,41))
CAPS=[8.,12.,16.,20.,24.,28.,32.,40.,50.,64.,80.,100.,150.]

def fit(p,cap=20.):
    return np.clip(np.log(np.maximum(p,1e-300))-np.log(np.maximum(reverse(p),1e-300)),-cap,cap)

def score(p,f):
    return float(np.sum(p*f)-np.sum(reverse(p)*np.expm1(f)))

def nested_cv(peers,lag):
    values=np.empty((len(CAPS),4))
    for j in range(4):
        train=peers[np.arange(4)!=j].mean(axis=0)
        raw=fit(train,CAPS[-1]); maxf=float(np.max(np.abs(raw)))
        p=peers[j];pr=reverse(p)
        for k,cap in enumerate(CAPS):
            f=np.clip(raw,-cap,cap)
            values[k,j]=(np.sum(p*f)-np.sum(pr*np.expm1(f)))/lag
            if cap>=maxf:
                values[k:,j]=values[k,j]
                break
    return values

def cap_selection(peers,lag):
    values=nested_cv(np.asarray(peers),lag)
    out=[dict(cap=c,training_cv_mean=float(values[k].mean()),training_cv_sd=float(values[k].std(ddof=1))) for k,c in enumerate(CAPS)]
    return max(out,key=lambda r:r['training_cv_mean'])['cap'],out

@lru_cache(maxsize=20)
def coarsening_matrix(fine,coarse):
    if fine%coarse:raise ValueError('Observation grids must be nested')
    r=fine//coarse
    delta=(np.arange(fine)[None,:]-r*np.arange(coarse)[:,None]+fine//2)%fine-fine//2
    W=np.maximum(0,r-np.abs(delta))/r
    if not np.allclose(W.sum(axis=0),1,rtol=0,atol=2e-15):raise AssertionError('Coarsening normalization')
    return W

def coarsen_probability(p,K):
    if p.shape[-1]==K:return p
    W=coarsening_matrix(p.shape[-1],K)
    return W@p@W.T
def work_statistics(meta,stats,y):
    work=stats.internal_work_ibp.to_numpy()
    blocks=np.array_split(work,min(50,max(2,len(work)//10)))
    blockmeans=np.array([v.mean() for v in blocks]);se=blockmeans.std(ddof=1)/np.sqrt(len(blockmeans))
    base=1.5*meta['E']**2*meta['Q']**2/meta['Gamma']
    return dict(N=meta['N'],E=meta['E'],dt=meta['dt'],true_rate=base+float(work.mean()),
        true_rate_sampling_se=float(se),collective_true_rate=base,true_rate_upper=1.5*meta['E']**2*meta['B'],
        internal_work=float(work.mean()),production_periods=meta['production_periods'],correlation_frames=len(y),
        internal_bulk_work=float(np.mean(work-2*meta['E']*(stats.p_end-stats.p_start))),
        max_ibp_discrepancy=float(np.max(np.abs(stats.internal_work-stats.internal_work_ibp))),
        acceptance=meta['acceptance'],max_relative_radius=float(np.linalg.norm(y,axis=2).max()),
        relative_rg2=float(np.mean(np.sum(y*y,axis=2))),
        weighted_center_error=float(np.max(np.abs(np.einsum('tnc,n->tc',y,1/np.array([a['D'] for a in meta['particles']])))))/meta['Gamma'])

def analyze_group(N,E,base,output,S=0,grids=GRIDS,lags=LAGS,max_frames=320):
    if not S:S=128 if N<=5 else 256 if N<=20 else 512
    base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[]
    for rep in range(5):
        name=f'N{N:03d}_E{E:.1f}_r{rep}';meta,t,y,stats=load_run(base/name)
        records.append(dict(name=name,meta=meta,y=y[-max_frames:],stats=stats))
    best=[{K:None for K in grids} for _ in records]
    scans=[[] for _ in records];start=time.time();finest=max(grids)
    for lag in lags:
        if 2*lag>=min(len(r['y']) for r in records)-2:continue
        hist=np.array([pair_histograms(r['y'],lag,S) for r in records])
        # Exact transverse-reflection symmetry, preserving positive probabilities.
        hist=.5*(hist+hist[:,:,:,(-np.arange(S))%S])
        for i,r in enumerate(records):
            m=r['meta'];kernels=kernel_matrices(float(m['Gamma']),float(m['Q']),float(E),float(lag),finest,S)
            fine=np.array([[prob_from_kernels(hist[j,p],*kernels) for p in range(2)] for j in range(5)])
            for K in grids:
                probs=coarsen_probability(fine,K)
                for phase,j in [('min',0),('max',1),('pooled',None)]:
                    phase_probs=probs.mean(axis=1) if j is None else probs[:,j]
                    peers=phase_probs[np.arange(5)!=i]
                    cv=nested_cv(peers,lag);means=cv.mean(axis=1);sds=cv.std(axis=1,ddof=1)
                    k=int(np.argmax(means));cap=CAPS[k]
                    for c in range(len(CAPS)):
                        scans[i].append(dict(lag=lag,phase=phase,boxes_per_side=K,cap=CAPS[c],selection_score=float(means[c]),training_cv_mean=float(means[c]),training_cv_sd=float(sds[c])))
                    if best[i][K] is None or means[k]>best[i][K]['selection_score']:
                        train=peers.mean(axis=0)
                        best[i][K]=dict(lag=lag,phase=phase,cap=cap,selection_score=float(means[k]),
                            p_train=train,p_test=phase_probs[i].copy(),f=fit(train,cap),
                            cv=[dict(cap=c,training_cv_mean=float(means[v]),training_cv_sd=float(sds[v])) for v,c in enumerate(CAPS)])
    results=[]
    for i,r in enumerate(records):
        result=work_statistics(r['meta'],r['stats'],r['y']);result['name']=r['name'];result['rep']=i
        result['quadrature_grid']=S;result['training_replicas']=[j for j in range(5) if j!=i]
        result['selection_method']='joint_nested_cv';result['selection_caps']=CAPS
        result['selection_lags']=list(lags);result['selection_grids']=list(grids)
        saved={};cap_rows=[]
        for K in grids:
            b=best[i][K];f=b['f'];value=score(b['p_test'],f)/b['lag']
            result[f'estimator_K{K}']=value;result[f'lag_K{K}']=b['lag'];result[f'phase_K{K}']=b['phase'];result[f'cap_K{K}']=b['cap']
            result[f'train_K{K}']=score(b['p_train'],f)/b['lag'];result[f'cv_K{K}']=b['selection_score']
            result[f'plugin_K{K}']=score(b['p_test'],fit(b['p_test'],CAPS[-1]))/b['lag']
            for row in b['cv']:cap_rows.append(dict(boxes_per_side=K,lag=b['lag'],phase=b['phase'],**row))
            saved[f'K{K}_p_train']=b['p_train'];saved[f'K{K}_p_evaluation']=b['p_test'];saved[f'K{K}_f']=f
        result['estimator_rate']=result[f'estimator_K{max(grids)}']
        result['quality_factor_Q']=result['estimator_rate']/result['true_rate']
        result['selection_optimism']=result[f'train_K{max(grids)}']-result['estimator_rate']
        result['analysis_group_seconds']=time.time()-start
        results.append(result)
        pd.DataFrame(scans[i]).to_csv(output/(r['name']+'_lag_grid_scan.csv'),index=False)
        pd.DataFrame(cap_rows).to_csv(output/(r['name']+'_cap_validation.csv'),index=False)
        np.savez_compressed(output/(r['name']+'_selected_correlations.npz'),**saved)
        dest=output/(r['name']+'_result.json');temporary=dest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result,indent=2));os.replace(temporary,dest)
    return results

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('N',type=int);a.add_argument('E',type=float)
    a.add_argument('--runs',default=str(Path(__file__).resolve().parent.parent/'raw'))
    a.add_argument('--output',default=str(Path(__file__).resolve().parent.parent/'data/analysis_runs'))
    a.add_argument('--S',type=int,default=0)
    args=a.parse_args();r=analyze_group(args.N,args.E,args.runs,args.output,args.S)
    print(json.dumps([{k:v for k,v in x.items() if k in ['name','true_rate','estimator_rate','quality_factor_Q','analysis_group_seconds','cap_K160','lag_K160']} for x in r],indent=2))
