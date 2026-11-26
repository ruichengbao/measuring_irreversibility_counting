#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from analyze_group import analyze_group

HERE=Path(__file__).resolve().parent

def worker(task):
    N,E,base,out=task
    result=analyze_group(N,E,base,out)
    return dict(N=N,E=E,mean_Q=sum(r['quality_factor_Q'] for r in result)/5,
        mean_true=sum(r['true_rate'] for r in result)/5,
        mean_bound=sum(r['estimator_rate'] for r in result)/5,
        seconds=result[-1]['analysis_group_seconds'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--force',action='store_true')
    ap.add_argument('--runs',default=str(HERE.parent/'raw'));ap.add_argument('--output',default=str(HERE.parent/'data/analysis_runs'));args=ap.parse_args()
    jobs=[(N,E) for N in [2,5,10,20,50,60,80,100] for E in [.1,.2,.4,.5,.8,1.]]
    submitted=set();futures={};start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            for N,E in jobs:
                if (N,E) in submitted:continue
                files=[Path(args.runs)/f'N{N:03d}_E{E:.1f}_r{r}_meta.json' for r in range(5)]
                try:metas=[json.loads(f.read_text()) for f in files]
                except (ValueError,OSError):continue
                existing=[Path(args.output)/f'N{N:03d}_E{E:.1f}_r{r}_result.json' for r in range(5)]
                if not args.force and all(f.exists() for f in existing):
                    old=[json.loads(f.read_text()) for f in existing]
                    if all(v.get('production_periods')==m['production_periods'] and 'estimator_K160' in v for v,m in zip(old,metas)):
                        submitted.add((N,E));continue
                submitted.add((N,E));futures[pool.submit(worker,(N,E,args.runs,args.output))]=(N,E)
            for future in [f for f in futures if f.done()]:
                job=futures.pop(future)
                print(f'elapsed={time.time()-start:.1f}s '+json.dumps(future.result()),flush=True)
            if len(submitted)==len(jobs) and not futures:break
            time.sleep(2)
    print('All48many-bodygroupsanalyzed',flush=True)
