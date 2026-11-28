#!/usr/bin/env python3
"""Fourfold time-step checks and one equilibrium control."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,time
from run_steady import execute,jobs,HERE,ensure_compiled

if __name__=='__main__':
    ensure_compiled()
    out=HERE/'validation';out.mkdir(exist_ok=True)
    refs=jobs();chosen=[]
    for E in [.1,1.]:
        j=next(j.copy() for j in refs if j['N']==100 and j['E']==E and j['rep']==0)
        j['dt']=.00005;chosen.append(j)
    j=next(j.copy() for j in refs if j['N']==20 and j['E']==.1 and j['rep']==0)
    j.update(name='N020_E0.0_r0',E=0.,production_periods=160,noise_seed=3900011);chosen.append(j)
    (HERE/'validation_manifest.json').write_text(json.dumps(chosen,indent=2))
    begin=time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs={pool.submit(execute,j,out,False):j for j in chosen}
        for f in as_completed(fs):print(f'elapsed={time.time()-begin:.1f}s {f.result()}',flush=True)
