#!/usr/bin/env python3
"""Exact endpoint-density benchmark on a two-dimensional Brownian torus.

Homogeneous noninteracting limit of the many-body model: L=5, D=mu=q=1,
k_B T=1. E(t)=E[1+sin(2*pi*t/T)], T=1. Observation starts at t0=T/4
and tau is an integer multiple of T. The drive is time symmetric on the
observation interval, and its integrated displacement is E*tau.

No simulation, fitted coefficient, or original-data modification is used.
The Gaussian propagator is integrated analytically over each square box.

For K boxes per side, let a=L/K, k_h=2*pi*h/L, and d=0,...,K-1.
The one-dimensional conditional cell displacement probability is

 p_d(E)=1/K [1+2 sum_{h>=1} exp(-D*k_h^2*tau)
                 sinc(h/K)^2 cos(2*pi*h*d/K-k_h*D*E*tau)].

Here numpy.sinc(z)=sin(pi*z)/(pi*z). The 2D one-particle endpoint
joint probability for a specified ordered pair of cells is
 P_d = p_dx(E)*p_dy(0)/M, M=K^2.
The normalized-density correlation of N independent particles is
 C_d=(1-1/N)/M^2+P_d/N.
Each displacement d occurs for M ordered cell pairs, so
 rate = M/(2*tau) sum_d (C_d-C_-d) log(C_d/C_-d).

Writing B_d=partial_E C_d|0 and C0_d=C_d(0), the exact weak-field
coefficient is A=2*M/tau sum_d B_d^2/C0_d. Thus rate=A*E^2+O(E^4).
The actual entropy-production rate averaged over the drive period is
 (3/2)*N*D*E^2. Numerical uncertainty is Fourier truncation only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def one_dimensional(K, E, tau=1., L=5., D=1., harmonics=32):
    """Integrated conditional cell probabilities and their E derivative."""
    h = np.arange(1, harmonics+1, dtype=float)
    kh = 2*np.pi*h/L
    d = np.arange(K, dtype=float)
    amp = np.exp(-D*kh*kh*tau)*np.sinc(h/K)**2
    phase = 2*np.pi*d[:, None]*h[None, :]/K-kh[None, :]*D*E*tau
    p = (1.+2.*np.cos(phase)@amp)/K
    dp = 2.*np.sin(phase)@(amp*kh*D*tau)/K
    return p, dp


def correlation(N, K, E, tau=1., harmonics=32):
    M = K*K
    px, dpx = one_dimensional(K, E, tau=tau, harmonics=harmonics)
    py, _ = one_dimensional(K, 0., tau=tau, harmonics=harmonics)
    C = (1.-1./N)/(M*M)+np.outer(px, py)/(N*M)
    B = np.outer(dpx, py)/(N*M)
    return C, B


def endpoint_rate(N, K, E, tau=1., harmonics=32):
    M = K*K
    C, _ = correlation(N, K, E, tau=tau, harmonics=harmonics)
    rev = (-np.arange(K)) % K
    Cr = C[np.ix_(rev, rev)]
    difference = C-Cr
    # log1p avoids loss of accuracy in the E -> 0 limit.
    return M/(2*tau)*np.sum(difference*np.log1p(difference/Cr))


def weak_field_coefficient(N, K, tau=1., harmonics=32):
    M = K*K
    C0, B = correlation(N, K, 0., tau=tau, harmonics=harmonics)
    return 2*M/tau*np.sum(B*B/C0)


def validation():
    convergence = []
    normalization_errors = []
    for N in [1, 5, 20]:
        for K in [1, 2, 4, 5, 8, 10, 16, 20, 32, 40]:
            for E in [0., .001, .1, 1.]:
                C, _ = correlation(N, K, E)
                normalization_errors.append(abs(K*K*C.sum()-1.))
                r16 = endpoint_rate(N, K, E, harmonics=16)
                r32 = endpoint_rate(N, K, E, harmonics=32)
                convergence.append(abs(r16-r32))
                assert C.min()>0
                assert r32 >= -1e-14
                assert r32 <= 1.5*N*E*E+1e-12
            for chain in [[1, 2, 4, 8, 16, 32], [1, 5, 10, 20, 40]]:
                for E in [.001, .1, 1.]:
                    vals=[endpoint_rate(N, k, E) for k in chain]
                    assert np.min(np.diff(vals)) > -1e-13
    ratios = []
    for N in [1, 5, 20]:
        for K in [5, 10, 20, 40]:
            A = weak_field_coefficient(N, K)
            r = endpoint_rate(N, K, 1e-4)/(A*1e-8)
            ratios.append(r)
            assert abs(r-1.) < 1e-6
    # Independent central derivative comparison of analytic coefficient.
    derivative_error=[]
    for N in [1,5,20]:
        C0,B=correlation(N,10,0.)
        Cp,_=correlation(N,10,1e-5)
        Cm,_=correlation(N,10,-1e-5)
        derivative_error.append(np.max(abs(B-(Cp-Cm)/(2e-5))))
    return {
        'max_probability_normalization_error':float(max(normalization_errors)),
        'max_absolute_rate_change_16_to_32_Fourier_modes':float(max(convergence)),
        'weak_field_exact_rate_over_A_E2_at_E1e_minus4_range':list(map(float,[min(ratios),max(ratios)])),
        'max_analytic_derivative_vs_central_difference_error':float(max(derivative_error)),
        'nested_resolution_monotonicity_checked':True,
        'entropy_production_bound_checked':True,
    }


def main(outdir):
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for N in [1,5,20]:
        for K in [1,2,4,5,8,10,16,20,32,40]:
            for E in [.1,.5,1.]:
                A=weak_field_coefficient(N,K)
                rows.append(dict(N=N,K=K,M=K*K,E=E,tau=1.,
                                 estimator_rate=endpoint_rate(N,K,E),
                                 weak_field_coefficient=A,
                                 quadratic_rate=A*E*E,
                                 true_rate=1.5*N*E*E))
    pd.DataFrame(rows).to_csv(outdir/'exact_resolution.csv',index=False)
    rows=[]
    for N in [1,5,20]:
        for K in [5,10,20]:
            A=weak_field_coefficient(N,K)
            for E in np.unique(np.r_[np.geomspace(.001,1.,81),.1,.2,.4,.5,.8,1.]):
                rate=endpoint_rate(N,K,E)
                rows.append(dict(N=N,K=K,M=K*K,E=E,tau=1.,
                                 estimator_rate=rate,
                                 weak_field_coefficient=A,
                                 quadratic_rate=A*E*E,
                                 quadratic_ratio=rate/(A*E*E),
                                 true_rate=1.5*N*E*E))
    pd.DataFrame(rows).to_csv(outdir/'exact_weak_field.csv',index=False)
    rows=[]
    for N in [1,2,5,10,20,50,100,200,500,1000]:
        for K in [5,10,20]:
            A=weak_field_coefficient(N,K)
            rows.append(dict(N=N,K=K,M=K*K,tau=1.,
                             weak_field_coefficient=A,
                             coefficient_times_N2=A*N*N,
                             true_weak_field_coefficient=1.5*N))
    pd.DataFrame(rows).to_csv(outdir/'exact_N_scaling.csv',index=False)
    metadata={
        'model':'Homogeneous noninteracting Brownian particles on a 2D torus',
        'L':5.,'D':1.,'mu':1.,'q':1.,'kBT':1.,
        'field':'E(t)=E[1+sin(2*pi*t/T)]','T':1.,'t0':.25,'tau':1.,
        'method':'Analytically integrated Fourier propagator; 32 positive Fourier modes',
        'old_data_modified':False,
        'coefficient_method':'Analytic derivative at E=0, no fitting',
        'nested_grid_chains':[[1,2,4,8,16,32],[1,5,10,20,40]],
        'validation':validation(),
    }
    (outdir/'exact_benchmark_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata['validation'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',default=str(Path(__file__).resolve().parent.parent/'data'))
    args=parser.parse_args()
    main(args.outdir)
