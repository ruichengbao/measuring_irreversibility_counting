"""Validate short-lag file dimensions, phase references and model assignments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate(prefix, expected=None):
    prefix=Path(prefix)
    m=json.loads(Path(str(prefix)+'_meta.json').read_text())
    N,M=int(m['N']),int(m['M'])
    f=np.fromfile(str(prefix)+'_forward.bin',dtype='<f8').reshape(-1,1+2*N)
    r=np.fromfile(str(prefix)+'_reverse.bin',dtype='<f8').reshape(m['n_branches'],M,N,2)
    o=pd.read_csv(str(prefix)+'_origins.csv')
    assert f.shape[0]==m['n_frames']==m['production_periods']*M+1
    assert len(o)==m['n_branches']
    assert np.isfinite(f).all() and np.isfinite(r).all()
    assert np.max(np.abs(f[:,0]-np.arange(len(f))/M))<1e-10
    assert np.all(o.branch_id.values==np.arange(len(o)))
    ids=o.forward_end_index.to_numpy(dtype=int)
    assert np.min(ids)>=M and np.max(ids)<len(f)
    assert np.all(ids%M==o.phase_index.values)
    assert np.max(np.abs(f[ids,0]-o.end_time.values))<1e-10
    assert o.groupby('phase_index').size().nunique()==1
    assert o.noise_seed.nunique()==len(o)
    weights=np.array([1/p['D'] for p in m['particles']])/m['Gamma']
    y=f[:,1:].reshape(-1,N,2)
    cf=np.max(np.abs(np.einsum('fai,a->fi',y,weights)))
    cr=np.max(np.abs(np.einsum('blai,a->bli',r,weights)))
    assert max(cf,cr)<1e-10
    assert abs(sum(p['q'] for p in m['particles'])-m['Q'])<1e-10
    assert abs(sum(1/p['D'] for p in m['particles'])-m['Gamma'])<1e-10
    if expected:
        assert m['N']==expected['N'] and abs(m['E']-expected['E'])<1e-12 and m['rep']==expected['rep']
        assert m['disorder_seed']==expected['disorder_seed']
        assert len(m['particles'])==len(expected['particles'])
        for a,b in zip(m['particles'],expected['particles']):
            for k in ('sigma','D','q'):assert abs(a[k]-b[k])<1e-12
    assert m['forward_cache_error']<1e-7 and m['reverse_cache_error']<1e-7
    if N==1:assert np.max(np.abs(y))==0 and np.max(np.abs(r))==0
    result={'passed':True,'N':N,'E':m['E'],'rep':m['rep'],'n_frames':len(f),'n_branches':len(o),'maximum_forward_center_error':float(cf),'maximum_reverse_center_error':float(cr),'branch_seeds_unique':True,'original_assignments_preserved':expected is not None,'forward_cache_error':m['forward_cache_error'],'reverse_cache_error':m['reverse_cache_error']}
    Path(str(prefix)+'_checks.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('prefix');a=p.parse_args();print(json.dumps(validate(a.prefix),indent=2))
