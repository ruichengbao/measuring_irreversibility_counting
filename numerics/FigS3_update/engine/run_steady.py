#!/usr/bin/env python3
"""Run the constant-field counterpart of the archived many-body model."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,subprocess,time
HERE=Path(__file__).resolve().parent

def jobs():
    durations={2:{.1:10000,.2:4000,.5:2000,.8:1000,1.:1000},5:{.1:5000,.2:2000,.5:1000,.8:1000,1.:1000},10:{.1:2000,.2:1000,.5:500,.8:500,1.:500},20:{.1:1000,.2:400}}
    result=[]
    for N in [2,5,10,20,50,60,80,100]:
        for E in [.1,.2,.5,.8,1.]:
            for rep in range(5):
                result.append(dict(name=f'N{N:03d}_E{E:.1f}_r{rep}',N=N,E=E,rep=rep,dt=.0002,burn_periods=10,
                    production_periods=durations.get(N,{}).get(E,160),record_duration=160,record_interval=.05,
                    disorder_seed=170100+1009*N+47*rep,noise_seed=281100000+100003*N+1009*round(10*E)+79*rep))
    priority=['N100_E1.0_r0']+[f'N020_E0.1_r{i}' for i in range(5)]+[f'N100_E1.0_r{i}' for i in range(1,5)]+[f'N100_E0.1_r{i}' for i in range(5)]
    result.sort(key=lambda j:priority.index(j['name']) if j['name'] in priority else 1000+j['N'])
    return result

def ensure_compiled():
    source=HERE/'simulate_steady.cpp';binary=HERE/'simulate_steady'
    dependencies = [source, HERE.parent.parent/'particle_species.hpp']
    if not binary.exists() or binary.stat().st_mtime < max(p.stat().st_mtime for p in dependencies):
        subprocess.run(['g++','-O3','-march=native','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True)

def execute(j,output,force=False):
    prefix=output/j['name'];meta=prefix.with_name(prefix.name+'_meta.json')
    if not force and meta.exists():
        m=json.loads(meta.read_text())
        snapshot=Path(str(prefix)+'_snapshots.bin');statistics=Path(str(prefix)+'_periods.csv')
        matching=['N','E','dt','burn_periods','production_periods','disorder_seed','noise_seed','record_duration','record_interval']
        complete=(snapshot.exists() and statistics.exists() and statistics.stat().st_size>0
                  and snapshot.stat().st_size==m.get('record_frames',0)*(1+2*j['N'])*8)
        if complete and all(m.get(k)==j[k] for k in matching) and m.get('field_protocol')=='constant':
            return j['name']+' already complete'
    keys=['N','E','dt','burn_periods','production_periods','disorder_seed','noise_seed']
    args=[str(HERE/'simulate_steady')]+[str(j[k]) for k in keys]+[str(prefix),str(j['record_duration']),str(j['record_interval'])]
    r=subprocess.run(args,capture_output=True,text=True)
    if r.returncode:raise RuntimeError(j['name']+' '+r.stderr)
    return r.stdout.strip()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--force',action='store_true');p.add_argument('--output-dir',type=Path,default=HERE/'raw');a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    ensure_compiled()
    alljobs=jobs();(HERE/'jobs_manifest_steady.json').write_text(json.dumps(alljobs,indent=2))
    begin=time.time();completed=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(execute,j,a.output_dir,a.force):j for j in alljobs}
        for i,f in enumerate(as_completed(fs),1):
            line=f.result();completed.append(fs[f]['name']);print(f'{i}/{len(alljobs)} elapsed={time.time()-begin:.1f}s {line}',flush=True)
            (HERE/'production_status.json').write_text(json.dumps(dict(completed=len(completed),total=len(alljobs),elapsed_seconds=time.time()-begin,names=completed),indent=2))
