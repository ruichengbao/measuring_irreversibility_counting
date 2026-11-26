#!/usr/bin/env python3
"""Exact N=1 phase-grid convergence, separate from the uniform-M=8 main results."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from shortlag_core import gaussian_bin_kernel,optimal_schedule,L

HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parent

def main(raw,output):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for file in sorted(raw.glob('N001*_meta.json')):
        meta=json.loads(file.read_text());E=meta['E'];Gamma=meta['Gamma'];charge=meta['Q']
        true=1.5*E*E*charge*charge/Gamma
        for K in [10,20,160]:
            for M in [8,16,32]:
                edges=np.zeros((M,M));d=np.arange(K)*L/K;h=L/K
                for end in range(M):
                    t1=end/M
                    for k in range(1,M+1):
                        tau=k/M;t0=t1-tau
                        drift=charge/Gamma*E*(tau+(np.sin(2*np.pi*t1)-np.sin(2*np.pi*t0))/(2*np.pi))
                        p=gaussian_bin_kernel(d-drift,h,np.sqrt(2*tau/Gamma));p/=p.sum()
                        q=p[(-np.arange(K))%K]
                        edges[end,k-1]=np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300))))
                best,candidates=optimal_schedule(edges,edges,strict_short=False)
                common=max((r for r in candidates if r['strategy']=='common_lag'),key=lambda r:r['test'])
                rows.append(dict(N=1,E=E,rep=meta['rep'],K=K,M=M,true_rate=true,
                                 phase_rate=best['test'],quality=best['test']/true,
                                 strategy=best['strategy'],partition=str(best['edges']),
                                 common_rate=common['test'],common_lag=common['lag_steps']/M,
                                 smallest_lag_quality=float(edges[:,0].sum()/true)))
    d=pd.DataFrame(rows);d.to_csv(output/'single_particle_phase_convergence.csv',index=False)
    means=d.groupby(['E','K','M']).agg(phase_rate=('phase_rate','mean'),true_rate=('true_rate','mean'),quality_mean=('quality','mean'),quality_min=('quality','min'),quality_max=('quality','max')).reset_index()
    means['quality_ratio_of_means']=means.phase_rate/means.true_rate
    means.to_csv(output/'single_particle_phase_convergence_summary.csv',index=False)
    print(means.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,default=PACKAGE/'engine/raw')
    p.add_argument('--output',type=Path,default=HERE/'validation');a=p.parse_args();main(a.raw,a.output)
