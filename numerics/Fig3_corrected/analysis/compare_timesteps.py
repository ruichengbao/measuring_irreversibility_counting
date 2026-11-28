#!/usr/bin/env python3
"""Compare direct internal-work and structural statistics at two timesteps."""
from pathlib import Path
import json,argparse,numpy as np
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--main-dir',type=Path,default=HERE.parent/'raw')
parser.add_argument('--validation-dir',type=Path,default=HERE.parent/'validation/raw')
parser.add_argument('--output',type=Path,default=HERE.parent/'validation/work_timestep_comparison.json')
options=parser.parse_args()
def values(prefix):
    m=json.loads(Path(str(prefix)+'_meta.json').read_text())
    a=np.genfromtxt(str(prefix)+'_periods.csv',delimiter=',',names=True)
    work=a['internal_work']; n=len(work)//10
    block=work[:10*n].reshape(n,10).mean(axis=1)
    se=float(block.std(ddof=1)/np.sqrt(n))
    base=1.5*m['E']**2*m['Q']**2/m['Gamma']
    x=np.fromfile(str(prefix)+'_snapshots.bin',dtype='<f8').reshape(-1,1+2*m['N'])[:,1:].reshape(-1,m['N'],2)
    rg=(x*x).sum(axis=2).mean(axis=1)
    return dict(N=m['N'],E=m['E'],dt=m['dt'],true_ep=float(base+work.mean()),
                true_ep_block_sem=se,internal_ep=float(work.mean()),
                radius_squared_mean=float(rg.mean()),radius_squared_sem=float(rg.std(ddof=1)/np.sqrt(len(rg))),
                acceptance=m['acceptance'])
rows=[]
for N in [50,100]:
    for E in [.1,1.]:
        name=f'N{N:03d}_E{E:.1f}_r0'
        try:
            coarse=values(options.main_dir/name);fine=values(options.validation_dir/name)
        except (FileNotFoundError,json.JSONDecodeError):
            continue
        delta=coarse['true_ep']-fine['true_ep'];se=np.hypot(coarse['true_ep_block_sem'],fine['true_ep_block_sem'])
        rows.append(dict(N=N,E=E,coarse=coarse,fine=fine,
                         true_ep_difference=float(delta),relative_true_ep_difference=float(delta/fine['true_ep']),
                         difference_combined_sem=float(se)))
options.output.write_text(json.dumps(rows,indent=2))
for r in rows:print(r['N'],r['E'],'relative EP difference',r['relative_true_ep_difference'],'difference / combined SEM',r['true_ep_difference']/r['difference_combined_sem'])
