#!/usr/bin/env python3
"""Audit complete data coverage, preserved baseline and independent schedule selection."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

PACKAGE=Path(__file__).resolve().parent.parent
NS=[1,2,5,10,20,50,60,80,100]
ES=[.1,.2,.4,.5,.8,1.]
KS=[5,10,20,40,80,160]

def audit(results,old,allow_partial=False):
    results=Path(results);old=Path(old)
    records=[json.loads(p.read_text()) for p in sorted(results.glob('*_result.json'))]
    actual={(r['N'],r['E'],r['rep']) for r in records}
    expected={(N,E,r) for N in NS for E in ES for r in range(5)}
    if len(actual)!=len(records):raise AssertionError('Duplicate target records')
    if actual-expected:raise AssertionError('Unexpected parameter point')
    if not allow_partial and actual!=expected:raise AssertionError(f'Missing {len(expected-actual)} target records')
    report=dict(records=len(records),expected_records=len(expected),missing_records=len(expected-actual),
                maximum_probability_normalization_error=0.,minimum_probability=1.,
                maximum_phase_quality=-np.inf,phase_selected=0,integer_selected=0,
                guarded_rate_below_saved_integer=0)
    for r in records:
        if r['M']!=8:raise AssertionError('Main data do not use a uniform M=8 grid')
        if r['selection_rule']!='training_one_standard_error_fallback':raise AssertionError('Mixed selection rules')
        if r['rep'] in r['training_replicas'] or len(r['training_replicas'])!=4:raise AssertionError('Target included in training')
        if set(map(int,r['grids']))!=set(KS):raise AssertionError('Observation-grid coverage')
        baseline=json.loads((old/(r['name']+'_result.json')).read_text())
        if r['true_rate']!=baseline['true_rate']:raise AssertionError('True EP baseline changed')
        for K in KS:
            g=r['grids'][str(K)]
            if g['baseline_rate']!=baseline[f'estimator_K{K}']:raise AssertionError('Integer estimate changed')
            folds=np.array(g['phase_resolved_cv_folds'])
            if not np.isclose(folds.mean(),g['phase_resolved_cv'],atol=1e-10,rtol=1e-10):raise AssertionError('Fold mean')
            if not np.isclose(folds.std(ddof=1)/2,g['phase_resolved_cv_se'],atol=1e-10,rtol=1e-10):raise AssertionError('Fold standard error')
            selected=g['phase_resolved_cv']-r['fallback_se_multiplier']*g['phase_resolved_cv_se']>g['baseline_cv']
            if (g['expanded_selected']=='phase_resolved')!=selected:raise AssertionError('Selection does not follow training-only rule')
            expected_rate=g['phase_resolved_rate'] if selected else g['baseline_rate']
            if g['expanded_rate']!=expected_rate:raise AssertionError('Selected score changed or clipped')
            unpenalized=g['phase_resolved_cv']>g['baseline_cv']
            if g['expanded_unpenalized_rate']!=(g['phase_resolved_rate'] if unpenalized else g['baseline_rate']):raise AssertionError('Unpenalized score changed')
            edges=g['edges'];M=r['M']
            if any(not 1<=e['lag_steps']<=M for e in edges):raise AssertionError('Lag outside phase grid')
            if g['phase_strategy']=='partition':
                if sum(e['lag_steps'] for e in edges)!=M or g['normalizer']!=1:raise AssertionError('Partition does not cover one period')
                phase=g['start_phase']
                for e in edges:
                    phase=(phase+e['lag_steps'])%M
                    if phase!=e['end_phase']:raise AssertionError('Partition is not contiguous')
            elif g['phase_strategy']=='common_lag':
                if sorted(e['end_phase'] for e in edges)!=list(range(M)):raise AssertionError('Missing sampling phases')
                if any(e['lag_steps']!=g['common_lag_steps'] for e in edges):raise AssertionError('Common lag mismatch')
                if g['normalizer']!=g['common_lag_steps']:raise AssertionError('Common-lag normalization')
            else:raise AssertionError('Unknown schedule')
            report['phase_selected' if selected else 'integer_selected']+=1
            report['guarded_rate_below_saved_integer']+=int(g['expanded_rate']<g['baseline_rate'])
            report['maximum_phase_quality']=max(report['maximum_phase_quality'],g['phase_resolved_rate']/r['true_rate'])
    for p in results.glob('*_probability_checks.csv'):
        f=pd.read_csv(p)
        report['minimum_probability']=min(report['minimum_probability'],float(f.min_probability.min()))
        report['maximum_probability_normalization_error']=max(report['maximum_probability_normalization_error'],float(f.max_normalization_error.max()))
    if report['minimum_probability']<0 or report['maximum_probability_normalization_error']>1e-12:
        raise AssertionError('Invalid contracted probability')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=PACKAGE/'analysis/results')
    p.add_argument('--old',type=Path,default=PACKAGE/'baseline/analysis_runs')
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();report=audit(a.results,a.old,a.allow_partial)
    destination=PACKAGE/'analysis/validation/output_audit.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
