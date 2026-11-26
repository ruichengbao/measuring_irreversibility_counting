#!/usr/bin/env python3
"""Validate invariants and independently compare the stored particle assignments."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd

def inspect(prefix,reference=None):
    prefix=Path(prefix);m=json.loads(Path(str(prefix)+'_meta.json').read_text())
    data=np.fromfile(str(prefix)+'_snapshots.bin',dtype='<f8').reshape(-1,1+2*m['N']);t=data[:,0];y=data[:,1:].reshape(len(t),m['N'],2)
    a=pd.read_csv(str(prefix)+'_periods.csv');weights=1/np.array([z['D'] for z in m['particles']])/m['Gamma']
    identity=np.max(np.abs(a.internal_work-m['E']*(a.p_end-a.p_start)))
    continuity=np.max(np.abs(a.p_end.to_numpy()[:-1]-a.p_start.to_numpy()[1:])) if len(a)>1 else 0.
    err=np.max(np.abs(np.einsum('tnd,n->td',y,weights)))
    assert np.isfinite(data).all() and np.isfinite(a.to_numpy()).all()
    assert m['field_protocol']=='constant' and len(t)==m['record_frames'] and len(a)==m['production_periods']
    assert abs(t[0]-m['record_time_start'])<1e-8 and abs(t[-1]-m['record_time_end'])<1e-8
    assert np.max(np.abs(np.diff(t)-m['record_interval']))<1e-8
    assert identity<1e-8 and continuity<1e-8 and err<1e-10
    assert np.max(np.abs(a.internal_work_ibp-m['E']*(a.p_end-a.p_start)))<1e-8
    assert m['maximum_relative_cache_error']<1e-8 and 0<m['acceptance']<=1
    expected_gamma=sum(1/z['D'] for z in m['particles']);expected_Q=sum(z['q'] for z in m['particles'])
    assert abs(m['Gamma']-expected_gamma)<1e-8 and abs(m['Q']-expected_Q)<1e-8
    for z in m['particles']:assert abs(z['D']-.01/z['sigma'])<1e-12
    same=True
    if reference is not None:
        r=Path(reference)/(prefix.name+'_meta.json')
        assert r.exists(),f'Missing reference metadata: {r}'
        same=json.loads(r.read_text())['particles']==m['particles'];assert same
    cm=m['E']**2*m['Q']**2/m['Gamma'];internal=a.internal_work_ibp.mean()
    blocks=np.array_split(a.internal_work_ibp.to_numpy(),min(50,max(2,len(a)//10)))
    se=np.std([b.mean() for b in blocks],ddof=1)/np.sqrt(len(blocks))
    return dict(name=prefix.name,N=m['N'],E=m['E'],dt=m['dt'],production_duration=len(a),frames=len(t),collective_rate=cm,internal_rate=internal,true_rate=cm+internal,internal_rate_block_se=se,work_identity_error=float(identity),period_continuity_error=float(continuity),center_error=float(err),force_cache_error=m['maximum_relative_cache_error'],acceptance=m['acceptance'],same_particles_as_reference=same)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,default=Path(__file__).resolve().parent/'raw');p.add_argument('--reference',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    rows=[inspect(Path(str(f)[:-10]),a.reference) for f in sorted(a.runs.glob('*_meta.json'))]
    out=dict(total_runs=len(rows),all_invariants_passed=True,rows=rows)
    if a.output:a.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}));print('max center error',max([r['center_error'] for r in rows],default=0))
