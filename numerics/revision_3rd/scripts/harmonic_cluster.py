#!/usr/bin/env python3
"""Exact density-counting benchmark for a driven harmonic Brownian cluster.

State: R mod L and y in R^N with sum(y)=0; x_a=(R+y_a) mod L is
observed. Pair springs act on unwrapped y_a-y_b. The field is
E(t)=E[1+sin(2*pi*t/T)] and observations start at t0=T/4 with tau=T.
All results are deterministic evaluations of exact Gaussian propagators,
not fitted simulations and not replacements for the submitted MD data.

The script writes numerical tables and verification_results.json beside it.
Dependencies: numpy, scipy, pandas.  Run: python harmonic_cluster.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import roots_legendre

L = 5.0
MU = Q = BETA = D = TAU = PERIOD = 1.0
OUT = Path(__file__).resolve().parent.parent / 'data'
OUT.mkdir(exist_ok=True)


def variances(n, kappa):
    """Variances of same-particle and distinct-particle displacements."""
    if kappa == 0:
        return 2 * D * TAU, np.inf
    if np.isinf(kappa):
        return 2 * D * TAU / n, 2 * D * TAU / n
    z = MU * kappa * n * TAU
    one_minus = -np.expm1(-z)
    base = 2 * D * TAU / n
    prefactor = 2 * D / (MU * kappa * n)
    same = base + prefactor * (1 - 1 / n) * one_minus
    distinct = base + prefactor * (1 - 1 / n + np.exp(-z) / n)
    return same, distinct


def box_gaussian_fourier(variance, field, boxes, modes=512):
    """C[delta] and dC/dE from the wrapped Gaussian Fourier series."""
    if np.isinf(variance):
        return np.full(boxes, 1 / boxes**2), np.zeros(boxes)
    r = np.arange(1, modes + 1, dtype=float)
    wave = 2 * np.pi * r / L
    amp = np.sinc(r / boxes)**2 * np.exp(-variance * wave**2 / 2)
    theta = (2 * np.pi / boxes) * np.outer(np.arange(boxes), r)
    theta -= wave[None, :] * MU * Q * field * TAU
    corr = (1 + 2 * np.cos(theta) @ amp) / boxes**2
    deriv = 2 * np.sin(theta) @ (amp * wave * MU * Q * TAU) / boxes**2
    return corr, deriv


def box_gaussian_quadrature(variance, field, boxes, order=64):
    """Positive Gaussian integral for tails subject to Fourier cancellation.

    C_d = L^-1 int_-h^h (h-|u|) sum_w phi_v(d*h+u-drift+w*L) du.
    Each Gaussian image is included through at least 12 standard deviations.
    """
    if np.isinf(variance):
        return np.full(boxes, 1 / boxes**2), np.zeros(boxes)
    h = L / boxes
    nodes, weights = roots_legendre(order)
    u = np.concatenate(((nodes - 1) * h / 2, (nodes + 1) * h / 2))
    weight = np.tile(weights * h / 2, 2) * (h - abs(u)) / L
    drift = MU * Q * field * TAU
    nimages = int(np.ceil((abs(drift) + 12 * np.sqrt(variance)) / L)) + 2
    images = np.arange(-nimages, nimages + 1)
    z = (np.arange(boxes)[:, None, None] * h + u[None, :, None]
         - drift + L * images[None, None, :])
    density = np.exp(-z*z/(2*variance)) / np.sqrt(2*np.pi*variance)
    corr = np.einsum('duw,u->d', density, weight)
    deriv = np.einsum('duw,u->d', density*z/variance, weight)*MU*Q*TAU
    return corr, deriv


def box_gaussian(variance, field, boxes):
    corr, deriv = box_gaussian_fourier(variance, field, boxes)
    # Positive integration avoids subtracting Fourier sums near probability zero.
    if np.min(corr) < 1e-12:
        corr, deriv = box_gaussian_quadrature(variance, field, boxes)
    return corr, deriv


def correlations(n, kappa, field, boxes=20):
    same, distinct = variances(n, kappa)
    cs, ds = box_gaussian(same, field, boxes)
    if n == 1:
        return cs, ds
    cd, dd = box_gaussian(distinct, field, boxes)
    return cs/n + (1-1/n)*cd, ds/n + (1-1/n)*dd


def weak_coefficient(n, kappa, boxes=20):
    corr, deriv = correlations(n, kappa, 0, boxes)
    rev = (-np.arange(boxes)) % boxes
    corr = (corr + corr[rev])/2
    deriv = (deriv - deriv[rev])/2
    # M equal rows; sum over unordered box pairs = M/2 sum over displacements.
    return PERIOD/TAU * 2*boxes*np.sum(deriv**2/corr)


def entropy(n, kappa, field, boxes=20):
    corr, _ = correlations(n, kappa, field, boxes)
    rev = (-np.arange(boxes)) % boxes
    delta = corr-corr[rev]
    estimate = PERIOD/TAU * boxes/2*np.sum(delta*np.log(corr/corr[rev]))
    true = 1.5*BETA*MU*Q**2*n*field**2*PERIOD
    return estimate, true


def main():
    rows=[]
    for n in (1, 5, 20, 50):
        for kappa in np.logspace(-3, 1, 81):
            a = weak_coefficient(n, kappa)
            rows.append(dict(N=n,kappa=kappa,M=20,tau=TAU,
                             weak_est_coefficient=a,
                             weak_true_coefficient=1.5*n,
                             weak_quality=a/(1.5*n),
                             weak_est_coefficient_per_N=a/n,
                             relaxation_rate=MU*kappa*n))
    pd.DataFrame(rows).to_csv(OUT/'harmonic_interaction_scaling.csv',index=False)

    rows=[]
    for n in (5,20):
        for kappa in (.01,.1,1.):
            a=weak_coefficient(n,kappa)
            for field in np.geomspace(1e-3,1,81):
                est,true=entropy(n,kappa,field)
                rows.append(dict(N=n,kappa=kappa,E=field,M=20,tau=TAU,
                                 sigma_est=est,sigma_true=true,
                                 weak_est_prediction=a*field**2,
                                 ratio_to_weak=est/(a*field**2),
                                 quality=est/true))
    pd.DataFrame(rows).to_csv(OUT/'harmonic_weak_field_curves.csv',index=False)

    rows=[]
    for n in (1,5,20,50):
        for kappa in (0,.01,.1,1,np.inf):
            for boxes in (5,10,20,40):
                a=weak_coefficient(n,kappa,boxes)
                est,true=entropy(n,kappa,.1,boxes)
                rows.append(dict(N=n,kappa=kappa,M=boxes,E=.1,
                                 weak_est_coefficient=a,
                                 sigma_est=est,sigma_true=true))
    resolution=pd.DataFrame(rows)
    resolution.to_csv(OUT/'harmonic_nested_resolution.csv',index=False)

    checks={}
    normerr=0.; symmetryerr=0.; fouriererr=0.; derivativeerr=0.
    for n in (1,5,20,50):
        for kappa in (0,.001,.01,.1,1,10,np.inf):
            c,d=correlations(n,kappa,0)
            normerr=max(normerr,abs(20*c.sum()-1))
            rev=(-np.arange(20))%20
            symmetryerr=max(symmetryerr,np.max(abs(c-c[rev])))
            cp,_=correlations(n,kappa,1e-5)
            cm,_=correlations(n,kappa,-1e-5)
            derivativeerr=max(derivativeerr,np.max(abs((cp-cm)/2e-5-d)))
            for var in variances(n,kappa):
                c1,d1=box_gaussian_fourier(var,.1,20,256)
                c2,d2=box_gaussian_fourier(var,.1,20,512)
                fouriererr=max(fouriererr,np.max(abs(c1-c2)),np.max(abs(d1-d2)))
    checks.update(max_normalization_error=normerr,max_equilibrium_symmetry_error=symmetryerr,
                  max_256_vs_512_fourier_error=fouriererr,
                  max_analytic_vs_centered_derivative_error=derivativeerr)
    independent_errors=[]; rigid_errors=[]
    for n in (1,5,20,50):
        c0,_=correlations(n,0,.1)
        cw,_=correlations(n,1e-9,.1)
        cr,_=correlations(n,np.inf,.1)
        cs,_=correlations(n,1e9,.1)
        independent_errors.append(float(np.max(abs(c0-cw))))
        rigid_errors.append(float(np.max(abs(cr-cs))))
    checks['independent_limit_max_abs_error']=max(independent_errors)
    checks['rigid_limit_max_abs_error']=max(rigid_errors)
    nested=[]
    for _,g in resolution.groupby(['N','kappa'],dropna=False):
        nested.extend(np.diff(g.sort_values('M').sigma_est))
    checks['minimum_nested_resolution_increment']=float(min(nested))
    weak=pd.read_csv(OUT/'harmonic_weak_field_curves.csv')
    checks['max_estimator_to_true_ratio']=float(weak.quality.max())
    checks['max_relative_weak_error_at_E_0p001']=float(abs(weak[weak.E==.001].ratio_to_weak-1).max())
    checks['minimum_entropy_estimate']=float(weak.sigma_est.min())
    assert normerr<1e-10 and symmetryerr<1e-10 and fouriererr<1e-12
    assert derivativeerr<1e-7
    assert min(nested)>-1e-10
    assert weak.quality.max()<=2/3+1e-9
    (OUT/'harmonic_verification_results.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
