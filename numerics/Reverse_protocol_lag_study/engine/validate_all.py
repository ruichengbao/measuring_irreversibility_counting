#!/usr/bin/env python3
"""Verify complete forward/reverse coverage and global thermal-seed uniqueness."""
from pathlib import Path
import argparse,json
import pandas as pd
from validate_run import validate
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=ROOT/'jobs_manifest.json');p.add_argument('--raw',type=Path,default=ROOT/'raw');p.add_argument('--output',type=Path,default=ROOT/'all_checks.json');a=p.parse_args()
 jobs=json.loads(a.manifest.read_text())['jobs'];keys={(j['N'],j['E'],j['rep']) for j in jobs}
 expected={(n,e,r) for n in [1,2,5,10,20,50,60,80,100] for e in [.1,.2,.4,.5,.8,1.] for r in range(5)}
 assert keys==expected and len(jobs)==270
 checks=[];seeds=[];metas=[];raw_bytes=0
 for j in jobs:
  prefix=a.raw/j['name'];checks.append(validate(prefix,j));m=json.loads(Path(str(prefix)+'_meta.json').read_text());metas.append(m)
  o=pd.read_csv(str(prefix)+'_origins.csv');seeds.extend([int(m['noise_seed'])]+list(map(int,o.noise_seed)))
  for suffix in ['_forward.bin','_reverse.bin','_origins.csv','_periods.csv','_meta.json']:raw_bytes+=Path(str(prefix)+suffix).stat().st_size
 assert len(seeds)==len(set(seeds)), 'Thermal seed collision'
 out={'passed':True,'runs':len(checks),'parameter_points':len({(j['N'],j['E']) for j in jobs}),'original_disorder_realizations_per_point':5,'reverse_branches':sum(m['n_branches'] for m in metas),'thermal_seeds':len(seeds),'all_thermal_seeds_unique':True,'all_particle_assignments_match':all(c['original_assignments_preserved'] for c in checks),'maximum_center_error':max(max(c['maximum_forward_center_error'],c['maximum_reverse_center_error']) for c in checks),'maximum_cache_error':max(max(c['forward_cache_error'],c['reverse_cache_error']) for c in checks),'raw_bytes':raw_bytes,'maximum_relative_radius':max(m['maximum_relative_radius'] for m in metas),'minimum_forward_acceptance':min(m['forward_acceptance'] for m in metas),'minimum_reverse_acceptance':min(m['reverse_acceptance_mean'] for m in metas),'sum_run_wall_seconds':sum(m['seconds'] for m in metas)}
 a.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
