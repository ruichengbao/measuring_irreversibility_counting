#!/usr/bin/env python3
"""Phase-resolved density correlations with genuine reverse protocols.

All probability contractions are positive real-space operations. No FFT tail
clipping or replacement of the reverse distribution by the forward is used.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import ctypes,json,subprocess
from functools import lru_cache
import numpy as np
import pandas as pd
from positive_kernel import gaussian_bin_kernel,prob_from_kernels

HERE=Path(__file__).resolve().parent
L=5.
GRIDS=[5,10,20,40,80,160]
CAPS=np.array([0.,1.,2.,4.,8.,12.,20.,32.,50.,80.,150.])
_DLL=None

def reverse(p):
    ix=(-np.arange(p.shape[-1]))%p.shape[-1]
    return p[...,ix,:][...,ix]

def load_run(prefix):
    prefix=Path(prefix)
    meta=json.loads(Path(str(prefix)+'_meta.json').read_text())
    N,M=meta['N'],meta['M']
    forward=np.fromfile(str(prefix)+'_forward.bin',dtype='<f8').reshape(-1,1+2*N)
    origins=pd.read_csv(str(prefix)+'_origins.csv')
    rev=np.fromfile(str(prefix)+'_reverse.bin',dtype='<f8').reshape(len(origins),M,N,2)
    t=forward[:,0];y=forward[:,1:].reshape(-1,N,2)
    if len(forward)!=meta['n_frames'] or len(origins)!=meta['n_branches']:
        raise ValueError('Record sizes disagree with metadata')
    if not np.allclose(t,np.arange(len(t))/M,atol=1e-10):
        raise ValueError('Forward frame clock does not match phase grid')
    idx=origins.forward_end_index.to_numpy(dtype=int)
    if np.any(idx<M) or np.any(idx>=len(y)):
        raise ValueError('Forward endpoint index out of range')
    if not np.all(idx%M==origins.phase_index.to_numpy()):
        raise ValueError('Phase labels disagree with endpoint indices')
    return dict(prefix=str(prefix),meta=meta,t=t,y=y,origins=origins,reverse=rev)

def histogram(a,b,S):
    global _DLL
    if _DLL is None:
        src=HERE/'pair_endpoints.cpp';lib=HERE/'pair_endpoints.so'
        if not lib.exists() or lib.stat().st_mtime<src.stat().st_mtime:
            temp=HERE/f'.pair_endpoints_{os.getpid()}.so'
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(temp)],check=True)
            os.replace(temp,lib)
        _DLL=ctypes.CDLL(str(lib)).endpoint_histogram
        _DLL.argtypes=[ctypes.POINTER(ctypes.c_double)]*2+[ctypes.c_int]*3+[ctypes.c_double,ctypes.POINTER(ctypes.c_double)]
        _DLL.restype=ctypes.c_int
    a=np.ascontiguousarray(a,dtype=np.float64);b=np.ascontiguousarray(b,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[-1]!=2:raise ValueError('Endpoint array shapes')
    H=np.empty((S,S),dtype=np.float64)
    ptr=lambda x:x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    err=_DLL(ptr(a),ptr(b),len(a),a.shape[1],S,L,ptr(H))
    if err:raise ValueError('Endpoint histogram failed')
    # Reflection in the transverse direction is an exact model symmetry.
    H=.5*(H+H[:,(-np.arange(S))%S])
    return H

def edge_histograms(run,end_phase,steps,S):
    M=run['meta']['M'];o=run['origins'];use=np.flatnonzero(o.phase_index.to_numpy()==end_phase)
    ix=o.forward_end_index.to_numpy(dtype=int)[use]
    y=run['y']
    HF=histogram(y[ix-steps],y[ix],S)
    HR=histogram(y[ix],run['reverse'][use,steps-1],S)
    return HF,HR

@lru_cache(maxsize=20)
def kernels(Gamma,Q,E,M,end_phase,steps,K,S):
    tau=steps/M;tend=end_phase/M;tstart=tend-tau
    drift=(Q/Gamma)*E*(tau+(np.sin(2*np.pi*tend)-np.sin(2*np.pi*tstart))/(2*np.pi))
    sigma=np.sqrt(2*tau/Gamma);h=L/K
    z=np.arange(K)[:,None]*h-np.arange(S)[None,:]*L/S
    return gaussian_bin_kernel(z-drift,h,sigma),gaussian_bin_kernel(z,h,sigma)

@lru_cache(maxsize=20)
def coarsening_matrix(fine,coarse):
    if fine%coarse:raise ValueError('Grids must be nested')
    r=fine//coarse
    delta=(np.arange(fine)[None,:]-r*np.arange(coarse)[:,None]+fine//2)%fine-fine//2
    W=np.maximum(0,r-np.abs(delta))/r
    if not np.allclose(W.sum(axis=0),1,atol=2e-15,rtol=0):raise AssertionError('Coarsening normalization')
    return W

def coarsen(p,K):
    if p.shape[-1]==K:return p
    W=coarsening_matrix(p.shape[-1],K)
    return W@p@W.T

def fit(p,q,cap,kind):
    raw=np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300))
    if kind=='odd':raw=.5*(raw-reverse(raw))
    return np.clip(raw,-cap,cap)

def score(p,q,f):
    return float(np.sum(p*f)-np.sum(q*np.expm1(f)))

def select_function(peers_p,peers_q,test_p,test_q):
    """Nested leave-one-peer-out selection; test data never choose a function."""
    npers=len(peers_p)
    values=np.empty((2,len(CAPS),npers))
    sump=peers_p.sum(axis=0);sumq=peers_q.sum(axis=0)
    for j in range(npers):
        p=(sump-peers_p[j])/(npers-1);q=(sumq-peers_q[j])/(npers-1)
        raw=np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300))
        for kind in range(2):
            rawkind=raw if kind==0 else .5*(raw-reverse(raw))
            maxf=np.max(np.abs(rawkind))
            for k,cap in enumerate(CAPS):
                f=np.clip(rawkind,-cap,cap)
                values[kind,k,j]=score(peers_p[j],peers_q[j],f)
                if cap>=maxf:
                    values[kind,k:,j]=values[kind,k,j]
                    break
    means=values.mean(axis=-1);kind,idx=np.unravel_index(np.argmax(means),means.shape)
    name=['full','odd'][kind];cap=float(CAPS[idx])
    f=fit(sump/npers,sumq/npers,cap,name)
    return dict(cv=float(means[kind,idx]),cv_sd=float(values[kind,idx].std(ddof=1)),
                test=score(test_p,test_q,f),cap=cap,kind=name,
                train=score(sump/npers,sumq/npers,f),fold_scores=values[kind,idx].tolist())

def optimal_schedule(edge_cv,edge_test,strict_short=True):
    """Choose a common lag or a nonoverlapping cyclic partition on training only."""
    M=edge_cv.shape[0];maxstep=M-1 if strict_short else M
    candidates=[]
    for k in range(1,maxstep+1):
        candidates.append(dict(strategy='common_lag',cv=float(edge_cv[:,k-1].sum()/k),
                               test=float(edge_test[:,k-1].sum()/k),lag_steps=k,
                               start_phase=None,edges=[[j,k] for j in range(M)],normalizer=k))
    for start in range(M):
        dp=np.full(M+1,-np.inf);dp[0]=0.;prev=np.zeros(M+1,dtype=int)
        for pos in range(1,M+1):
            for k in range(1,min(pos,maxstep)+1):
                value=dp[pos-k]+edge_cv[(start+pos)%M,k-1]
                if value>dp[pos]:dp[pos]=value;prev[pos]=k
        pos=M;edges=[]
        while pos:
            k=int(prev[pos]);edges.append([(start+pos)%M,k]);pos-=k
        edges.reverse()
        test=sum(edge_test[j,k-1] for j,k in edges)
        candidates.append(dict(strategy='partition',cv=float(dp[M]),test=float(test),
                               lag_steps=None,start_phase=start,edges=edges,normalizer=1))
    return max(candidates,key=lambda r:r['cv']),candidates
