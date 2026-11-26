#!/usr/bin/env python3
"""Resume the complete 54-point analysis, optionally waiting for simulations.

This launcher checks complete five-replica groups, runs up to --workers groups,
and rebuilds the small plotting tables after each completed group. Existing
complete outputs are left intact unless --force is supplied.
"""
from pathlib import Path
import argparse,concurrent.futures,json,subprocess,sys,time
from aggregate_full_results import aggregate

HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parent
NS=[1,2,5,10,20,50,60,80,100]
ES=[.1,.2,.4,.5,.8,1.]

def run_group(N,E,args):
    log=args.output/f'N{N:03d}_E{E:.1f}.log'
    command=[sys.executable,str(HERE/'analyze_full_group.py'),str(N),str(E),
             '--raw',str(args.raw),'--old',str(args.old),'--output',str(args.output)]
    with log.open('w') as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    return result.returncode,str(log)

def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    all_jobs=[(N,E) for N in NS for E in ES]
    pending=[]
    for N,E in all_jobs:
        files=[args.output/f'N{N:03d}_E{E:.1f}_r{r}_result.json' for r in range(5)]
        complete=all(f.exists() for f in files)
        if complete:
            complete=all(json.loads(f.read_text()).get('selection_rule')=='training_one_standard_error_fallback' for f in files)
        if args.force or not complete:pending.append((N,E))
    completed=[];failed=[];running={}
    start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending or running:
            for job in list(pending):
                if len(running)>=args.workers:break
                N,E=job
                ready=all((args.raw/f'N{N:03d}_E{E:.1f}_r{r}_meta.json').exists() for r in range(5))
                if not ready:continue
                pending.remove(job);running[pool.submit(run_group,N,E,args)]=job
            done=[f for f in running if f.done()]
            for future in done:
                job=running.pop(future);code,log=future.result()
                (completed if code==0 else failed).append(dict(N=job[0],E=job[1],returncode=code,log=log))
                print(f'Finished N={job[0]}, E={job[1]}: returncode={code}',flush=True)
                if code==0:aggregate(args.output,args.summary)
            status=dict(completed=completed,failed=failed,pending=pending,running=list(running.values()),seconds=time.time()-start)
            (args.output/'analysis_status.json').write_text(json.dumps(status,indent=2))
            if pending and not running and not args.wait:
                print('Some simulation groups are not yet complete. Use --wait to keep watching.',flush=True);break
            if running or pending:time.sleep(2)
    aggregate(args.output,args.summary)
    if failed:raise SystemExit(f'{len(failed)} groups failed; inspect listed logs.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=PACKAGE/'engine/raw')
    p.add_argument('--old',type=Path,default=PACKAGE/'baseline/analysis_runs')
    p.add_argument('--output',type=Path,default=HERE/'results')
    p.add_argument('--summary',type=Path,default=HERE/'summary')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--wait',action='store_true');p.add_argument('--force',action='store_true')
    main(p.parse_args())
