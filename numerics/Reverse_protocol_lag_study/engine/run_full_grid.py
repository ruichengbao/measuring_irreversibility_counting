#!/usr/bin/env python3
"""Compile and run the full forward/reverse grid. Model seeds are in the manifest."""
from pathlib import Path
import argparse,json,subprocess,time,concurrent.futures,threading
from validate_run import validate
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'jobs_manifest.json')
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--raw',type=Path,default=ROOT/'raw')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.raw.mkdir(parents=True,exist_ok=True)
    jobs=json.loads(a.manifest.read_text())['jobs']
    exe=ROOT/'simulate_shortlag'
    dependencies = [ROOT/'simulate_shortlag.cpp', ROOT.parent.parent/'particle_species.hpp']
    if not exe.exists() or exe.stat().st_mtime < max(p.stat().st_mtime for p in dependencies):
        subprocess.run(['g++','-O3','-std=c++17','-march=native',str(ROOT/'simulate_shortlag.cpp'),'-o',str(exe)],check=True)
    start=time.time();lock=threading.Lock();completed=[];failed=[]
    def status():
        obj={'total':len(jobs),'completed':len(completed),'failed':len(failed),'elapsed_seconds':time.time()-start,'completed_jobs':completed,'failures':failed}
        tmp=ROOT/'status.tmp.json';tmp.write_text(json.dumps(obj,indent=2));tmp.replace(ROOT/'status.json')
    def run(j):
        prefix=a.raw/j['name'];meta=Path(str(prefix)+'_meta.json')
        if a.resume and meta.exists():
            validate(prefix,j);return {'name':j['name'],'resumed':True,'seconds':json.loads(meta.read_text())['seconds']}
        cmd=[str(exe),str(j['N']),str(j['E']),str(j['dt']),str(j['burn_periods']),str(j['production_periods']),str(j['origin_stride']),str(j['M']),str(j['disorder_seed']),str(j['noise_seed']),str(prefix),str(j['rep'])]
        with open(str(prefix)+'.log','w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        validate(prefix,j);return {'name':j['name'],'resumed':False,'seconds':json.loads(meta.read_text())['seconds']}
    status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(run,j):j for j in jobs}
        for f in concurrent.futures.as_completed(fs):
            j=fs[f]
            try:
                result=f.result();completed.append(result);print(f"[{len(completed)}/{len(jobs)}] {j['name']} {result['seconds']:.2f}s",flush=True)
            except Exception as e:failed.append({'name':j['name'],'error':repr(e)});print('FAILED',j['name'],repr(e),flush=True)
            status()
    if failed:raise SystemExit(f'{len(failed)} jobs failed')
    print('Complete',len(completed),'elapsed',time.time()-start,flush=True)
if __name__=='__main__':main()
