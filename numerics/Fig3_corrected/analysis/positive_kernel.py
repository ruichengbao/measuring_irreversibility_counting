#!/usr/bin/env python3
"""Positive real-space density correlations with analytic Gaussian center motion.

The relative pair displacement distribution is approximated by cloud-in-cell
(CIC) interpolation on a periodic S by S grid. Every histogram weight and every
Gaussian-integrated bin kernel is nonnegative, preserving rare-event tails.
Validate S against 2S to bound this spatial quadrature error.
"""
from pathlib import Path
import ctypes,subprocess,os
from functools import lru_cache
import numpy as np
from scipy.special import erfcx

L=5.
HERE=Path(__file__).resolve().parent
LIB=HERE/'pair_hist.so'

def _library():
    source=HERE/'pair_hist.cpp'
    if not LIB.exists() or LIB.stat().st_mtime<source.stat().st_mtime:
        temporary=LIB.with_name(f'.pair_hist_{os.getpid()}.so')
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(temporary)],check=True)
        os.replace(temporary,LIB)
    dll=ctypes.CDLL(str(LIB))
    f=dll.pair_histograms_cpp
    f.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_longlong,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_double)]
    f.restype=ctypes.c_int
    return f

_histogram_cpp=None

def pair_histograms(y,lag,S=512):
    """Return H[phase,x,y], normalized separately; phase0=min, phase1=max.

    Frames are spaced by T/2, with the first at a drive minimum. Lag is in
    complete periods, hence the frame shift is 2*lag. Coordinates may be
    unwrapped. Pair weights include all N² pairs, including the same particle.
    """
    global _histogram_cpp
    if _histogram_cpp is None:_histogram_cpp=_library()
    y=np.ascontiguousarray(y,dtype=np.float64)
    if y.ndim!=3 or y.shape[2]!=2:raise ValueError('Expected frames[T,N,2]')
    shift=round(2*lag)
    if abs(shift-2*lag)>1e-12:raise ValueError('lag must align with half-period frames')
    H=np.zeros((2,S,S),dtype=np.float64)
    err=_histogram_cpp(y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(y),y.shape[1],shift,S,L,H.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if err:raise ValueError(f'Histogram failed: status {err}')
    return H


def _R(a,sigma):
    """sigma*phi(abs(a)/sigma)-abs(a)*Phi(-abs(a)/sigma)."""
    t=np.abs(a)/sigma
    # erfcx avoids an underflowing division of two Gaussian tail factors.
    return sigma/np.sqrt(2*np.pi)*np.exp(-.5*t*t)*(1-t*np.sqrt(np.pi/2)*erfcx(t/np.sqrt(2)))


def gaussian_bin_kernel(z,h,sigma):
    """Probability of a width-h target bin averaged over a width-h source bin.

    k(z) = 1/h int[-h,h] (h-|u|) g_wrapped(z+u) du.
    The positive-part contribution to the second difference of J is evaluated
    as a triangle, avoiding cancellation between large linear terms.
    """
    z=(np.asarray(z)+L/2)%L-L/2
    if sigma >= .7:
        # The broad wrapped Gaussian has no rare tails on this torus. Its
        # Fourier series converges rapidly with an absolute truncation error
        # far below double-precision resolution; positivity is checked.
        cutoff=int(np.ceil(np.sqrt(2*np.log(1e20))/(2*np.pi*sigma/L)))
        result=np.ones_like(z,dtype=float)
        for n in range(1,cutoff+1):
            amplitude=np.sinc(n*h/L)**2*np.exp(-.5*(2*np.pi*n*sigma/L)**2)
            result+=2*amplitude*np.cos((2*np.pi*n/L)*z)
        result*=h/L
        if np.any(result<=0):raise FloatingPointError('Nonpositive broad Gaussian Fourier kernel')
        return result
    nimages=max(1,int(np.ceil((L/2+h+10*sigma)/L)))
    result=np.zeros_like(z,dtype=float)
    for image in range(-nimages,nimages+1):
        a=z+image*L
        part=(np.maximum(h-np.abs(a),0)+_R(a+h,sigma)-2*_R(a,sigma)+_R(a-h,sigma))/h
        result+=part
    if np.any(result<0):
        raise FloatingPointError(f'Negative analytic Gaussian bin kernel: {result.min()}')
    return result


@lru_cache(maxsize=256)
def kernel_matrices(Gamma,Q,E,lag,K,S=512):
    """Cached Hx,Hy arrays, shape[K,S], with positive drift along x."""
    h=L/K;sigma=np.sqrt(2*lag/Gamma)
    g=np.arange(S)*L/S
    z=np.arange(K)[:,None]*h-g[None,:]
    Hx=gaussian_bin_kernel(z-Q/Gamma*E*lag,h,sigma)
    Hy=gaussian_bin_kernel(z,h,sigma)
    return Hx,Hy


def prob_from_kernels(H,Hx,Hy):
    """Contract a normalized relative displacement histogram with the kernels."""
    H=np.asarray(H)
    if H.ndim!=2:raise ValueError('Pass one phase histogram or the phase mean')
    ix=np.flatnonzero(H.sum(axis=1)>0);iy=np.flatnonzero(H.sum(axis=0)>0)
    p=Hx[:,ix]@H[np.ix_(ix,iy)]@Hy[:,iy].T
    if np.any(p<0):raise FloatingPointError('Negative probability')
    p/=p.sum()
    return p


def correlation_probability(H,Gamma,Q,E,lag,K):
    """P[d]=K²*C[0,d], normalized; no artificial probability floor."""
    H=np.asarray(H)
    if H.ndim==3:H=H.mean(axis=0)
    return prob_from_kernels(H,*kernel_matrices(float(Gamma),float(Q),float(E),float(lag),int(K),H.shape[0]))


_score_cpp=None

def pair_scores(y,lag,score_grid):
    """Linear score for each origin, averaging all N² particle pairs.

    score_grid[x,y] is evaluated by exactly the same periodic CIC interpolation
    as pair_histograms. For a fixed variational function f on displacement
    bins, form W=f-expm1(reverse(f)) and score_grid=Hx.T@W@Hy. The result is
    the pre-division KL score. Divide by lag to obtain a rate, and select even
    or odd origin indices to retain one sampling phase.
    """
    global _score_cpp
    if _score_cpp is None:
        _library()  # Compiles through an atomic rename, safe for live readers.
        dll=ctypes.CDLL(str(LIB))
        _score_cpp=dll.pair_scores_cpp
        _score_cpp.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_longlong,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
        _score_cpp.restype=ctypes.c_int
    y=np.ascontiguousarray(y,dtype=np.float64)
    J=np.ascontiguousarray(score_grid,dtype=np.float64)
    if y.ndim!=3 or y.shape[2]!=2:raise ValueError('Expected frames[T,N,2]')
    if J.ndim!=2 or J.shape[0]!=J.shape[1]:raise ValueError('Expected square score grid')
    shift=round(2*lag)
    if abs(shift-2*lag)>1e-12:raise ValueError('lag must align with half-period frames')
    if len(y)<=shift:raise ValueError('Insufficient frames for lag')
    out=np.empty(len(y)-shift,dtype=np.float64)
    ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    err=_score_cpp(ptr(y),len(y),y.shape[1],shift,J.shape[0],L,ptr(J),ptr(out))
    if err:raise ValueError(f'Pair scores failed: status {err}')
    return out
