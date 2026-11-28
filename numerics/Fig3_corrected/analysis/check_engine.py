#!/usr/bin/env python3
"""Check saved trajectory invariants and work quadrature independently."""
from pathlib import Path
import argparse,json
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
parser.add_argument('--expected-runs',type=int,default=None)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
records=[]
for path in sorted(args.directory.glob('*_meta.json')):
    m=json.loads(path.read_text())
    N=m['N'];prefix=str(path).removesuffix('_meta.json')
    snapshots=np.fromfile(prefix+'_snapshots.bin',dtype='<f8').reshape(-1,1+2*N)
    xyz=snapshots[:,1:].reshape(-1,N,2)
    w=np.array([1/p['D'] for p in m['particles']]);w/=w.sum()
    centering=float(np.max(np.abs(np.einsum('fnd,n->fd',xyz,w))))
    stats=np.genfromtxt(prefix+'_periods.csv',delimiter=',',names=True)
    work_error=float(np.max(np.abs(stats['internal_work']-stats['internal_work_ibp'])))
    minimum_energy=float(np.min(stats['potential']))
    maximum_energy=float(np.max(stats['potential']))
    assert len(snapshots)==2*m['production_periods'],path
    assert len(stats)==m['production_periods'],path
    assert np.all(np.isfinite(snapshots)),path
    assert np.all(np.isfinite(stats['internal_work'])),path
    assert np.allclose(snapshots[:,0],np.arange(1,len(snapshots)+1)/2,atol=1e-10),path
    assert centering<1e-10,path
    assert 0<m['acceptance']<=1,path
    assert all(abs(p['D']*p['sigma']-.01)<1e-12 for p in m['particles']),path
    assert abs(sum(p['q'] for p in m['particles'])-m['Q'])<1e-10,path
    assert abs(sum(1/p['D'] for p in m['particles'])-m['Gamma'])<1e-10,path
    if m['E']==0:
        assert np.max(np.abs(stats['internal_work']))==0,path
    records.append(dict(run=path.stem,N=N,E=m['E'],dt=m['dt'],
                        periods=m['production_periods'],acceptance=m['acceptance'],
                        maximum_center_residual=centering,
                        maximum_work_quadrature_difference=work_error,
                        minimum_potential=minimum_energy,maximum_potential=maximum_energy))
if args.expected_runs is not None:
    assert len(records)==args.expected_runs,(len(records),args.expected_runs)
result=dict(runs_checked=len(records),all_invariants_passed=True,records=records)
if args.output: args.output.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
