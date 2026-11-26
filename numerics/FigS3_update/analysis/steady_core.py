"""Numerical kernels for the constant-field density/FTUR comparison."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import ctypes,json,subprocess
from functools import lru_cache
import numpy as np
import pandas as pd
from positive_kernel import kernel_matrices,prob_from_kernels
HERE=Path(__file__).resolve().parent
LAGS=[.05,.1,.2,.5,1.,2.,5.,10.,20.]
GRIDS=[5,20]
CAPS=[8.,12.,16.,20.,24.,28.,32.,40.,50.,64.,80.,100.,150.]

def reverse(p):return p[np.ix_((-np.arange(p.shape[0]))%p.shape[0],(-np.arange(p.shape[1]))%p.shape[1])]
def fit(p,cap=150.):return np.clip(np.log(np.maximum(p,1e-300))-np.log(np.maximum(reverse(p),1e-300)),-cap,cap)
def score(p,f):return float(np.sum(p*f)-np.sum(reverse(p)*np.expm1(f)))
def plugin(p):return score(p,fit(p,150.))
def load_run(prefix):
 prefix=Path(prefix);meta=json.loads(prefix.with_name(prefix.name+'_meta.json').read_text())
 d=np.fromfile(prefix.with_name(prefix.name+'_snapshots.bin'),dtype='<f8').reshape(-1,1+2*meta['N'])
 y=np.ascontiguousarray(d[:,1:].reshape(-1,meta['N'],2));t=d[:,0]
 stats=pd.read_csv(prefix.with_name(prefix.name+'_periods.csv'))
 return meta,t,y,stats

@lru_cache(maxsize=1)
def _hist():
 src=HERE/'steady_hist.cpp';lib=HERE/'steady_hist.so'
 if not lib.exists() or lib.stat().st_mtime<src.stat().st_mtime:
  tmp=HERE/f'.steady_hist_{os.getpid()}.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(tmp)],check=True);os.replace(tmp,lib)
 dll=ctypes.CDLL(str(lib));f=dll.steady_histogram_cpp
 f.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_longlong,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_double)];f.restype=ctypes.c_int
 return f
def histogram(y,shift,S=512,stride=10,offset=0):
 y=np.ascontiguousarray(y,dtype=float);H=np.zeros((S,S));ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
 n=_hist()(ptr(y),len(y),y.shape[1],int(shift),int(stride),int(offset),int(S),5.,ptr(H))
 if n<0:raise ValueError(n)
 H=.5*(H+H[:,(-np.arange(S))%S])
 return H,n
@lru_cache(maxsize=1)
def _scores():
 _hist();dll=ctypes.CDLL(str(HERE/'steady_hist.so'));f=dll.steady_scores_cpp
 pd=ctypes.POINTER(ctypes.c_double);pi=ctypes.POINTER(ctypes.c_longlong)
 f.argtypes=[pd,ctypes.c_longlong,ctypes.c_int,ctypes.c_int,pi,ctypes.c_int,ctypes.c_int,ctypes.c_double,pd,pd];f.restype=ctypes.c_int
 return f
def linear_scores(y,shift,J,origins):
 y=np.ascontiguousarray(y,dtype=float);J=np.ascontiguousarray(J,dtype=float);ix=np.ascontiguousarray(origins,dtype=np.int64);out=np.zeros(len(ix))
 pd=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double));pi=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
 err=_scores()(pd(y),len(y),y.shape[1],int(shift),pi(ix),len(ix),J.shape[0],5.,pd(J),pd(out))
 if err:raise ValueError(err)
 return out
def probabilities(H,m,E,tau,K):return prob_from_kernels(H,*kernel_matrices(float(m['Gamma']),float(m['Q']),float(E),float(tau),int(K),H.shape[0]))

@lru_cache(maxsize=16)
def j_weights(K):
 # Average sign(end_label-start_label) over uniform starting cell.
 # Labels are m=i_y*K+i_x, with x the fastest varying index.
 x,y=np.meshgrid(np.arange(K),np.arange(K),indexing='ij')
 return np.where(y!=0,1-2*y/K,np.where(x!=0,1-2*x/K,0.))
def current_mean(p):return float(np.sum(p*j_weights(p.shape[0])))
def tur_rate(mean,second,tau):
 variance=float(second-mean*mean)
 if variance<=0:raise FloatingPointError(f'Invalid variance {variance}')
 return float(np.log1p(2*mean*mean/variance)/tau)

def nested_cv(peers,tau):
 values=np.empty((len(CAPS),4))
 for j in range(4):
  train=peers[np.arange(4)!=j].mean(axis=0);raw=fit(train);maxf=float(np.max(np.abs(raw)));p=peers[j];pr=reverse(p)
  for k,cap in enumerate(CAPS):
   f=np.clip(raw,-cap,cap);values[k,j]=(np.sum(p*f)-np.sum(pr*np.expm1(f)))/tau
   if cap>=maxf:values[k:,j]=values[k,j];break
 return values
def work_statistics(meta,stats,y):
 work=stats.internal_work_ibp.to_numpy();blocks=np.array_split(work,min(40,max(2,len(work)//10)));bm=np.array([v.mean() for v in blocks]);se=bm.std(ddof=1)/np.sqrt(len(bm))
 base=meta['E']**2*meta['Q']**2/meta['Gamma']
 return dict(N=meta['N'],E=meta['E'],true_rate=base+float(work.mean()),collective_true_rate=base,true_rate_sampling_se=float(se),internal_work=float(work.mean()),production_periods=meta['production_periods'],correlation_frames=len(y),dt=meta['dt'],acceptance=meta['acceptance'],true_rate_upper=meta['E']**2*meta['B'])
