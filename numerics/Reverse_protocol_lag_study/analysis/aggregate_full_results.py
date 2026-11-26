#!/usr/bin/env python3
"""Summarize independently evaluated replicas without clipping or test selection."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

def aggregate(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[json.loads(p.read_text()) for p in sorted(source.glob('*_result.json'))]
    long=[]
    for r in records:
        for grid,g in r['grids'].items():
            long.append(dict(N=r['N'],E=r['E'],rep=r['rep'],K=int(grid),true_rate=r['true_rate'],
                             true_rate_sampling_se=r['true_rate_sampling_se'],
                             old_rate=g['baseline_rate'],short_rate=g['phase_resolved_rate'],
                             expanded_rate=g['expanded_rate'],old_cv=g['baseline_cv'],short_cv=g['phase_resolved_cv'],
                             short_cv_se=g.get('phase_resolved_cv_se',np.nan),
                             expanded_unpenalized_rate=g.get('expanded_unpenalized_rate',g['expanded_rate']),
                             selection_rule=r.get('selection_rule','unpenalized_training_maximum'),
                             expanded_selected=g['expanded_selected'],phase_strategy=g['phase_strategy'],
                             selected_has_subperiod=g['selected_has_subperiod'],
                             phase_candidate_has_subperiod=g['selected_has_subperiod'],
                             expanded_has_subperiod=g['expanded_selected']=='phase_resolved' and g['selected_has_subperiod'],
                             selected_lags=';'.join(str(e['lag']) for e in g['edges']),
                             origins_per_phase=r['origins_per_phase'],M=r['M'],S=r['S']))
    d=pd.DataFrame(long)
    if d.empty:return
    d.to_csv(output/'replicates.csv',index=False)
    for K,a in d.groupby('K'):
        rows=[]
        for (N,E),g in a.groupby(['N','E']):
            if len(g)!=5:continue
            true=g.true_rate.mean();row=dict(N=N,E=E,K=K,realizations=5,
                                           true_rate=true,true_rate_sd=g.true_rate.std(ddof=1),
                                           true_rate_thermal_sem=np.linalg.norm(g.true_rate_sampling_se)/5)
            for label in ('old','short','expanded'):
                v=g[label+'_rate'];row[label+'_rate']=v.mean();row[label+'_rate_sd']=v.std(ddof=1)
                row[label+'_quality']=v.mean()/true;row[label+'_quality_sd']=(v/g.true_rate).std(ddof=1)
            row['expanded_over_old']=row['expanded_rate']/row['old_rate']
            row['short_over_old']=row['short_rate']/row['old_rate']
            row['short_fraction_selected']=(g.expanded_selected=='phase_resolved').mean()
            row['expanded_unpenalized_rate']=g.expanded_unpenalized_rate.mean()
            row['subperiod_fraction']=g.selected_has_subperiod.mean()
            row['phase_candidate_subperiod_fraction']=g.phase_candidate_has_subperiod.mean()
            row['expanded_subperiod_fraction']=g.expanded_has_subperiod.mean()
            row['paired_gain_mean']=(g.expanded_rate-g.old_rate).mean()
            row['paired_gain_sd']=(g.expanded_rate-g.old_rate).std(ddof=1)
            row['origins_per_phase_min']=g.origins_per_phase.min()
            rows.append(row)
        pd.DataFrame(rows).to_csv(output/f'fig3_K{K}_summary.csv',index=False)
    print(f'Aggregated {len(records)} replica records.')

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source',type=Path,default=Path(__file__).resolve().parent/'results')
    a.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'summary')
    args=a.parse_args();aggregate(args.source,args.output)
