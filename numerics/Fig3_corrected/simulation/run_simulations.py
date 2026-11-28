#!/usr/bin/env python3
"""Reproduce the specified many-body ensemble using independent C++ runs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, subprocess, time, argparse

HERE=Path(__file__).resolve().parent
RUNS=HERE.parent/'raw'
jobs=[]
parser=argparse.ArgumentParser()
parser.add_argument('--pilot',action='store_true',help='Use 160 production periods at every point.')
parser.add_argument('--workers',type=int,default=8)
parser.add_argument('--force',action='store_true',help='Recompute complete saved runs.')
parser.add_argument('--output-dir',type=Path,default=RUNS,help='Directory for raw simulation outputs.')
options=parser.parse_args()
RUNS=options.output_dir.resolve()
RUNS.mkdir(parents=True,exist_ok=True)
durations={2:{.1:10000,.2:4000,.4:2000,.5:2000,.8:1000,1.:1000},
           5:{.1:5000,.2:2000,.4:1000,.5:1000,.8:1000,1.:1000},
           10:{.1:2000,.2:1000,.4:500,.5:500,.8:500,1.:500},
           20:{.1:1000,.2:400}}
for N in [2,5,10,20,50,60,80,100]:
    for E in [.1,.2,.4,.5,.8,1.]:
        for rep in range(5):
            name=f'N{N:03d}_E{E:.1f}_r{rep}'
            jobs.append(dict(name=name,N=N,E=E,rep=rep,dt=2e-4,
                             burn_periods=10,production_periods=(160 if options.pilot else durations.get(N,{}).get(E,160)),
                             disorder_seed=170100+1009*N+47*rep,
                             noise_seed=81100000+100003*N+1009*round(10*E)+79*rep))
final_manifest=HERE/'jobs_manifest_driven_final.json'
if final_manifest.exists() and not options.pilot:
    jobs=json.loads(final_manifest.read_text())
else:
    for j in jobs:j['implementation']='uncached'
(HERE/'jobs_manifest_driven.json').write_text(json.dumps(jobs,indent=2))

def run(j):
    prefix=RUNS/j['name']
    if not options.force and (RUNS/(j['name']+'_meta.json')).exists():
        try:
            m=json.loads((RUNS/(j['name']+'_meta.json')).read_text())
            if m['production_periods']>=j['production_periods']:
                return j['name']+' already complete'
        except (ValueError,KeyError): pass
    binary='simulate_driven_cached' if j.get('implementation')=='pair_cache' else 'simulate_driven_uncached'
    args=[str(HERE/binary)]+[str(j[k]) for k in ['N','E','dt','burn_periods','production_periods','disorder_seed','noise_seed']]+[str(prefix)]
    result=subprocess.run(args,text=True,capture_output=True)
    if result.returncode: raise RuntimeError(j['name']+' '+result.stderr)
    return result.stdout.strip()

if __name__=='__main__':
    for source,binary in [('simulate_driven.cpp','simulate_driven_uncached'),('simulate_driven_cached.cpp','simulate_driven_cached')]:
        executable = HERE / binary
        dependencies = [HERE / source, HERE.parent.parent / 'particle_species.hpp']
        if not executable.exists() or executable.stat().st_mtime < max(p.stat().st_mtime for p in dependencies):
            subprocess.run(['g++','-O3','-march=native','-ffp-contract=off','-std=c++17',str(HERE/source),'-o',str(HERE/binary)],check=True)
    start=time.time()
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures={pool.submit(run,j):j for j in jobs}
        for number,f in enumerate(as_completed(futures),1):
            print(f'{number}/{len(jobs)} elapsed={time.time()-start:.1f}s {f.result()}',flush=True)
