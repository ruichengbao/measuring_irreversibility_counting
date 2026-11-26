#!/usr/bin/env python3
"""Finer-timestep checks using the same particle realizations as production."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, subprocess, time,argparse
HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'validation/raw'
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=OUT)
parser.add_argument('--workers',type=int,default=2)
options=parser.parse_args()
OUT=options.output_dir.resolve()
OUT.mkdir(parents=True,exist_ok=True)
jobs=[]
for N in [100,50]:
    for E in [.1,1.]:
        jobs.append(dict(N=N,E=E,dt=5e-5,burn_periods=10,production_periods=80,
                         disorder_seed=170100+1009*N,
                         noise_seed=91100000+100003*N+1009*round(10*E),
                         name=f'N{N:03d}_E{E:.1f}_r0'))
for N in [20,100]:
    jobs.append(dict(N=N,E=0.,dt=2e-4,burn_periods=10,production_periods=160,
                     disorder_seed=170100+1009*N,
                     noise_seed=99100000+100003*N,
                     name=f'N{N:03d}_E0.0_r0'))
(OUT/'jobs_manifest.json').write_text(json.dumps(jobs,indent=2))
def run(j):
    prefix=OUT/j['name']
    args=[str(HERE/'simulate_driven_cached')]+[str(j[k]) for k in ['N','E','dt','burn_periods','production_periods','disorder_seed','noise_seed']]+[str(prefix)]
    return subprocess.run(args,text=True,capture_output=True,check=True).stdout.strip()
if __name__=='__main__':
    binary = HERE/'simulate_driven_cached'
    sources = [HERE/'simulate_driven_cached.cpp', HERE.parent.parent/'particle_species.hpp']
    if not binary.exists() or binary.stat().st_mtime < max(p.stat().st_mtime for p in sources):
        subprocess.run(['g++','-O3','-march=native','-ffp-contract=off','-std=c++17',str(HERE/'simulate_driven_cached.cpp'),'-o',str(HERE/'simulate_driven_cached')],check=True)
    start=time.time()
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            print(f'elapsed={time.time()-start:.1f}s {f.result()}',flush=True)
