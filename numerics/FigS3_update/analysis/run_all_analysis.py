#!/usr/bin/env python3
"""Analyze completed steady groups, optionally waiting for simulation output."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,csv,json,time
from concurrent.futures import ProcessPoolExecutor
from analyze_steady import analyze_group
HERE=Path(__file__).resolve().parent
def ready(runs,N,E):
 for rep in range(5):
  p=runs/f'N{N:03d}_E{E:.1f}_r{rep}_meta.json'
  try:m=json.loads(p.read_text())
  except (OSError,ValueError):return False
  if m.get('record_frames',0)<=1:return False
 return True
def complete(output,N,E,stride,qmc_power,qmc_origins):
 """Reuse a group only when every output and its numerical settings agree."""
 for rep in range(5):
  base=output/f'N{N:03d}_E{E:.1f}_r{rep}'
  files=[Path(str(base)+suffix) for suffix in ['_result.json','_scan.csv','_selected.npz']]
  if not all(p.is_file() and p.stat().st_size>0 for p in files):return False
  try:
   result=json.loads(files[0].read_text())
   if any(result.get(key)!=value for key,value in [('histogram_origin_stride',stride),('qmc_power',qmc_power),('qmc_origins',qmc_origins)]):return False
   with files[1].open(newline='') as stream:rows=list(csv.DictReader(stream))
   if not rows:return False
   for row in rows:
    expected_stride=min(stride,max(1,round(float(row['tau'])/.05)))
    if int(row['histogram_origin_stride'])!=expected_stride or int(row['qmc_power'])!=qmc_power or int(row['qmc_origins'])!=qmc_origins:return False
  except (OSError,ValueError,KeyError):return False
 return True
def main(args):
 tasks=[(N,E) for N in [2,5,10,20,50,60,80,100] for E in [.1,.2,.5,.8,1.]]
 pending=[]
 for N,E in tasks:
  if not args.force and complete(args.output,N,E,args.stride,args.qmc_power,args.qmc_origins):continue
  pending.append((N,E))
 running={};completed=0;start=time.time()
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  while pending or running:
   for key in pending[:]:
    if len(running)>=args.workers:break
    if not args.force and complete(args.output,*key,args.stride,args.qmc_power,args.qmc_origins):pending.remove(key);continue
    if ready(args.runs,*key):
     N,E=key;f=pool.submit(analyze_group,N,E,args.runs,args.output,0,args.stride,args.qmc_power,args.qmc_origins);running[f]=key;pending.remove(key)
   for f,key in list(running.items()):
    if f.done():
     f.result();completed+=1;del running[f];print(f'COMPLETE {key} groups={completed} seconds={time.time()-start:.1f}',flush=True)
   if pending and not running and not args.wait:print('Groups not ready:',pending,flush=True);break
   if pending or running:time.sleep(1)
 print('Analysis runner finished.',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,default=HERE.parent/'engine/raw');p.add_argument('--output',type=Path,default=HERE/'results');p.add_argument('--workers',type=int,default=2);p.add_argument('--qmc-power',type=int,default=11);p.add_argument('--qmc-origins',type=int,default=256);p.add_argument('--stride',type=int,default=10);p.add_argument('--wait',action='store_true');p.add_argument('--force',action='store_true');a=p.parse_args();main(a)
