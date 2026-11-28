#!/usr/bin/env python3
"""Check coarse-grid contraction against direct Gaussian/bin integration."""
from pathlib import Path
import numpy as np
import pandas as pd
from shortlag_core import load_run,edge_histograms,kernels,prob_from_kernels,coarsen
from positive_kernel import kernel_matrices

PACKAGE=Path(__file__).resolve().parent.parent

if __name__=='__main__':
    rows=[]
    for N,E in [(5,1.),(100,.1)]:
        run=load_run(PACKAGE/'engine/raw'/f'N{N:03d}_E{E:.1f}_r0')
        meta=run['meta'];S=320 if N<=20 else 640;M=meta['M']
        for end,steps in [(0,1),(4,4),(7,7)]:
            tau=steps/M;t1=end/M;t0=t1-tau
            effectiveE=E*(tau+(np.sin(2*np.pi*t1)-np.sin(2*np.pi*t0))/(2*np.pi))/tau
            hf,hr=edge_histograms(run,end,steps,S)
            for name,hist in [('forward',hf),('reverse',hr)]:
                fine=prob_from_kernels(hist,*kernels(meta['Gamma'],meta['Q'],E,M,end,steps,160,S))
                for K in [5,10,20,40,80]:
                    direct=prob_from_kernels(hist,*kernel_matrices(meta['Gamma'],meta['Q'],effectiveE,tau,K,S))
                    coarse=coarsen(fine,K)
                    error=np.abs(direct-coarse)
                    rows.append(dict(N=N,E=E,end_phase=end,lag_steps=steps,protocol=name,K=K,
                        maximum_absolute_error=error.max(),L1_error=error.sum()))
    frame=pd.DataFrame(rows)
    if frame.maximum_absolute_error.max()>2e-13:raise AssertionError('Direct coarse-grid contraction differs')
    destination=PACKAGE/'analysis/validation/coarsening_direct_contraction.csv'
    destination.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(destination,index=False)
    print(frame[['maximum_absolute_error','L1_error']].max().to_string())
