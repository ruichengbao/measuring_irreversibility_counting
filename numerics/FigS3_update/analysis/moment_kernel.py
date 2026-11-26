#!/usr/bin/env python3
"""Conditional QMC moments of the physical two-time box-counting current.

The inner integration averages J and J**2 over a common uniform initial
translation and its Gaussian collective displacement. It never substitutes
the variance of an averaged current for the physical current variance.
"""
from pathlib import Path
import ctypes
import os
import subprocess
import time
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

HERE=Path(__file__).resolve().parent
_function=None

def _load():
    global _function
    if _function is not None:return _function
    src=HERE/'moment_kernel.cpp';lib=HERE/'libmoment_kernel.so'
    if not lib.exists() or lib.stat().st_mtime<src.stat().st_mtime:
        tmp=HERE/f'.moment_kernel_{os.getpid()}.so'
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(tmp)],check=True)
        os.replace(tmp,lib)
    dll=ctypes.CDLL(str(lib))
    f=dll.box_current_moments_cpp
    pd=ctypes.POINTER(ctypes.c_double);pi=ctypes.POINTER(ctypes.c_longlong)
    f.argtypes=[pd,ctypes.c_longlong,ctypes.c_int,pi,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                ctypes.c_double,ctypes.c_double,ctypes.c_double,pd,ctypes.c_longlong,ctypes.c_int,pd]
    f.restype=ctypes.c_int
    _function=f
    return f

def current_weights(K):
    """Exact mean-current weight for displacement probabilities P[dx,dy]."""
    a=np.zeros((K,K),dtype=float)
    for dx in range(K):
        for dy in range(K):
            if dy:a[dx,dy]=1-2*dy/K
            elif dx:a[dx,dy]=1-2*dx/K
    return a

def moments(y,lagsteps,K,Gamma,Q,E,tau,origins=128,power=9,seed=1731,L=5.,reflect_y=True):
    """Return QMC mean, physical second moment, and per-origin integrals.

    Parameters
    ----------
    y : float array [frames, particles, 2]
        Relative unwrapped coordinates.
    lagsteps : int
        Separation of the two recorded frames.
    origins : int or array
        Number of evenly distributed origins, or explicit origin indices.
    power : int
        2**power Sobol integration points per selected origin. One scrambled
        point set is reused at all origins; use an independent seed to assess
        quadrature error, which is distinct from trajectory sampling error.
    reflect_y : bool
        Average each configuration and its transverse reflection, an exact
        symmetry of the steady model. For x-fast box labels J=Jx+Jy, where
        Jy orders different rows and Jx orders boxes within a row. Reflection
        changes Jy to -Jy. Consequently its two contributions are averaged
        exactly as mean Jx and second moment Jx**2+Jy**2.
    """
    tic=time.perf_counter()
    y=np.ascontiguousarray(y,dtype=np.float64)
    if y.ndim!=3 or y.shape[-1]!=2:raise ValueError('Expected y[frames,N,2].')
    lagsteps=int(lagsteps);available=len(y)-lagsteps
    if available<=0:raise ValueError('Lag exceeds record length.')
    if np.isscalar(origins):
        ix=np.unique(np.rint(np.linspace(0,available-1,min(int(origins),available))).astype(np.int64))
    else:ix=np.asarray(origins,dtype=np.int64)
    ix=np.ascontiguousarray(ix,dtype=np.int64)
    if not len(ix) or np.min(ix)<0 or np.max(ix)>=available:raise ValueError('Invalid origins.')
    points=qmc.Sobol(4,scramble=True,seed=int(seed)).random_base2(int(power))
    points[:,2:]=ndtri(np.clip(points[:,2:],np.finfo(float).eps,1-np.finfo(float).eps))
    points=np.ascontiguousarray(points,dtype=np.float64)
    out=np.zeros((len(ix),2),dtype=np.float64)
    pd=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pi=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    err=_load()(pd(y),len(y),y.shape[1],pi(ix),len(ix),lagsteps,int(K),float(L),
                float(Q*E*tau/Gamma),float(np.sqrt(2*tau/Gamma)),pd(points),len(points),int(reflect_y),pd(out))
    if err:raise RuntimeError(f'Current-moment kernel status {err}.')
    mean=float(out[:,0].mean());second=float(out[:,1].mean())
    return dict(mean=mean,second_moment=second,variance=second-mean**2,
                origin_indices=ix,per_origin_mean=out[:,0],per_origin_second=out[:,1],
                sobol_power=int(power),sobol_points=len(points),sobol_seed=int(seed),
                origins=len(ix),seconds=time.perf_counter()-tic,
                transverse_reflection=bool(reflect_y),
                box_index='iy*K+ix',integration='uniform_translation_gaussian_increment_scrambled_sobol')
