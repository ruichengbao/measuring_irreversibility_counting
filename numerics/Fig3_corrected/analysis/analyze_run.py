#!/usr/bin/env python3
"""Analyze exact-CM / simulated-relative Brownian density correlations.

All density observables are normalized counts. Uniform initial translation and
the independent Gaussian center-of-friction increment are integrated exactly.
Fourier truncation is checked, not inferred from noisy cell frequencies.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

L=5.
GRIDS=[5,10,20,40,80]
LAGS=list(range(1,41))
F_CAP=20.

def load_run(prefix):
    prefix=Path(prefix)
    meta=json.loads(Path(str(prefix)+'_meta.json').read_text())
    binary=Path(str(prefix)+'_snapshots.bin')
    if binary.exists():
        data=np.fromfile(binary,dtype='<f8').reshape(-1,1+2*meta['N'])
        t=data[:,0]; y=data[:,1:].reshape(-1,meta['N'],2)
    else:
        df=pd.read_csv(str(prefix)+'_snapshots.csv')
        y=df[['x_relative','y_relative']].to_numpy().reshape(-1,meta['N'],2)
        t=df.time.to_numpy()[::meta['N']]
    stats=pd.read_csv(str(prefix)+'_periods.csv')
    return meta,t,y,stats

def modes(y,n):
    """Nonuniform Fourier transform evaluated directly, without particle binning."""
    q=2*np.pi*n/L
    result=np.empty((len(y),len(n),len(n)),dtype=np.complex128)
    for start in range(0,len(y),16):
        z=y[start:start+16]
        x=np.exp(1j*z[:,:,0,None]*q)
        v=np.exp(1j*z[:,:,1,None]*q)
        result[start:start+16]=np.matmul(x.transpose(0,2,1),v)/y.shape[1]
    return result

def probability(A,n,Gamma,Q,E,lag,K):
    """P[d]=K² C[0,d], a normalized distribution on spatial displacements."""
    nx=n[:,None];ny=n[None,:]
    w=np.sinc(nx/K)**2*np.sinc(ny/K)**2
    factor=np.exp(-(2*np.pi/L)**2*(nx*nx+ny*ny)*lag/Gamma)
    factor=factor*np.exp(1j*2*np.pi/L*nx*Q/Gamma*E*lag)
    coeff=A*w*factor
    folded=np.zeros((K,K),complex)
    np.add.at(folded,(np.broadcast_to(nx%K,coeff.shape).ravel(),np.broadcast_to(ny%K,coeff.shape).ravel()),coeff.ravel())
    p=np.fft.ifft2(folded).real
    negative=-float(np.minimum(p,0).sum())
    if negative>2e-11:
        raise ValueError(f'Unresolved Fourier series: total negative mass {negative}')
    # Roundoff-only repair. Its total removed mass is saved in the diagnostics.
    p=np.maximum(p,0)
    p/=p.sum()
    return p,negative

def reverse(p):
    k=len(p); idx=(-np.arange(k))%k
    return p[np.ix_(idx,idx)]

def kl(p):
    q=reverse(p)
    # The floor is below retained absolute Fourier accuracy. The plotted result
    # uses the bounded variational function below, not this plug-in diagnostic.
    return float(np.sum(p*np.log(np.maximum(p,1e-300)/np.maximum(q,1e-300))))

def fit_function(p):
    return np.clip(np.log(np.maximum(p,1e-300)/np.maximum(reverse(p),1e-300)),-F_CAP,F_CAP)

def variational(p,f):
    """Linear held-out KL bound: <f>_P - <exp(f)-1>_{P^R}."""
    return float(np.sum(p*f)-np.sum(reverse(p)*np.expm1(f)))

def lag_probabilities(F,n,meta,lag,indices,Ks=GRIDS):
    shift=2*lag
    origins=indices[indices+shift<=indices[-1]]
    if not len(origins): raise ValueError('Insufficient trajectory length')
    prod=F[origins+shift]*F[origins].conj()
    out={}
    # index0 has t=.5 (field minimum); index1 t=1 (field maximum).
    for phase in ['max','min','pooled']:
        choose=np.ones(len(origins),bool) if phase=='pooled' else origins%2==(1 if phase=='max' else 0)
        A=prod[choose].mean(axis=0)
        for K in Ks:
            out[(phase,K)]=probability(A,n,meta['Gamma'],meta['Q'],meta['E'],lag,K)
    return out

def analyze(prefix,output,grids=GRIDS,lags=LAGS,cutoff_log=35.,save_selected=True):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    meta,t,y,stats=load_run(prefix)
    # Longer low-N records primarily resolve weak-field work. The last160
    # periods suffice for the low-variance, analytically integrated correlations.
    t=t[-320:]; y=y[-320:]
    kmax=int(np.ceil(np.sqrt(cutoff_log*meta['Gamma'])/(2*np.pi/L)))
    n=np.arange(-kmax,kmax+1)
    F=modes(y,n)
    mid=len(y)//2
    # A two-period guard separates selection and evaluation records.
    halves=[np.arange(0,mid-2),np.arange(mid+2,len(y))]
    all_indices=np.arange(len(y))
    best=[{K:None for K in grids} for _ in range(2)]
    rows=[]; maxnegative=0.
    for lag in lags:
        if 2*lag>=min(len(h) for h in halves)-2:continue
        ps=[lag_probabilities(F,n,meta,lag,h,grids) for h in halves]
        full=lag_probabilities(F,n,meta,lag,all_indices,grids)
        for phase in ['max','min','pooled']:
            for K in grids:
                p0,neg0=ps[0][(phase,K)];p1,neg1=ps[1][(phase,K)]
                pall,negall=full[(phase,K)]
                maxnegative=max(maxnegative,neg0,neg1,negall)
                train=[variational(p0,fit_function(p0))/lag,variational(p1,fit_function(p1))/lag]
                hold=[variational(p1,fit_function(p0))/lag,variational(p0,fit_function(p1))/lag]
                rows.append(dict(lag=lag,boxes_per_side=K,phase=phase,plugin_rate=kl(pall)/lag,
                    train_0=train[0],heldout_0=hold[0],train_1=train[1],heldout_1=hold[1]))
                for h in range(2):
                    if best[h][K] is None or train[h]>best[h][K]['train']:
                        best[h][K]=dict(lag=lag,phase=phase,train=train[h],heldout=hold[h],p_train=[p0,p1][h],p_test=[p1,p0][h],p_full=pall)
    name=Path(prefix).name
    pd.DataFrame(rows).to_csv(output/(name+'_lag_grid_scan.csv'),index=False)
    base=1.5*meta['E']**2*meta['Q']**2/meta['Gamma']
    upper=1.5*meta['E']**2*meta['B']
    work=stats.internal_work_ibp.to_numpy()
    # Period blocks retain temporal correlations and include the boundary term.
    blocks=np.array_split(work,min(16,max(2,len(work)//10)))
    blockmeans=np.array([v.mean() for v in blocks])
    worksem=float(blockmeans.std(ddof=1)/np.sqrt(len(blockmeans)))
    result=dict(name=name,N=meta['N'],E=meta['E'],dt=meta['dt'],
        true_rate=base+float(work.mean()),true_rate_sampling_se=worksem,
        collective_true_rate=base,true_rate_upper=upper,
        internal_work=float(work.mean()),
        internal_bulk_work=float(np.mean(work-2*meta['E']*(stats.p_end-stats.p_start))),
        max_ibp_discrepancy=float(np.max(np.abs(stats.internal_work-stats.internal_work_ibp))),
        acceptance=meta['acceptance'],fourier_kmax=kmax,max_negative_mass=maxnegative,
        production_periods=meta['production_periods'],correlation_frames=len(y),
        max_relative_radius=float(np.linalg.norm(y,axis=2).max()),
        relative_rg2=float(np.mean(np.sum(y*y,axis=2))),
        weighted_center_error=float(np.max(np.abs(np.einsum('tnc,n->tc',y,1/np.array([a['D'] for a in meta['particles']])))))/meta['Gamma'])
    saved={}
    for K in grids:
        a,b=best[0][K],best[1][K]
        rate=.5*(a['heldout']+b['heldout'])
        result[f'estimator_K{K}']=rate
        result[f'train_K{K}']=.5*(a['train']+b['train'])
        result[f'lag0_K{K}']=a['lag'];result[f'lag1_K{K}']=b['lag']
        result[f'phase0_K{K}']=a['phase'];result[f'phase1_K{K}']=b['phase']
        if save_selected and K in [5,max(grids)]:
            for h,z in enumerate([a,b]):
                for key in ['p_train','p_test','p_full']:
                    saved[f'K{K}_fold{h}_{key}']=z[key]
    result['estimator_rate']=result[f'estimator_K{max(grids)}']
    result['quality_factor_Q']=result['estimator_rate']/result['true_rate'] if result['true_rate'] else None
    result['selection_optimism']=result[f'train_K{max(grids)}']-result['estimator_rate']
    (output/(name+'_result.json')).write_text(json.dumps(result,indent=2))
    if saved:np.savez_compressed(output/(name+'_selected_correlations.npz'),**saved)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prefix');p.add_argument('--output',default='analysis');p.add_argument('--cutoff-log',type=float,default=35)
    args=p.parse_args();r=analyze(args.prefix,args.output,cutoff_log=args.cutoff_log)
    print(json.dumps(r,indent=2))
