#!/usr/bin/env python3
"""Fixed-function, fixed-lag timestep and quadrature checks for steady estimates."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from steady_core import *
from moment_kernel import moments

def block_se(values,nblocks=16):
 b=np.array([v.mean() for v in np.array_split(np.asarray(values),min(nblocks,len(values)))])
 return float(b.std(ddof=1)/np.sqrt(len(b)))

def write_report(rows,output):
 """Record the primary and auxiliary checks separately, without causal inference."""
 lines=['Fixed-function validation of the steady-state comparison','',
  'The displayed steady-state panel uses K=5. K=20 is an auxiliary calculation.',
  'Every timestep comparison keeps the learned density function and both selected lags fixed.',
  'The timestep changes from 0.0002 to 0.00005. Sampling errors use 16 time blocks.',
  'The refined current second moment uses 512 origins and 8192 scrambled Sobol points.',
  'A second independent scramble checks quadrature, and the auxiliary pair grid is doubled.',
  'The finite-sample weak FTUR substitutes estimated moments into a nonlinear formula.','']
 for K,label in [(5,'Primary K=5'),(20,'Auxiliary K=20')]:
  lines.append(label)
  for row in rows:
   if row['K']!=K:continue
   lines.append('E={E:g}: density timestep change {dd:+.4f}% ({dz:+.3f} combined sampling SE); FTUR timestep change {td:+.4f}% ({tz:+.3f} combined sampling SE).'.format(E=row['E'],dd=100*row['density_relative_step_change'],dz=row['density_step_change_in_sampling_se'],td=100*row['tur_relative_step_change'],tz=row['tur_step_change_in_sampling_se']))
   lines.append('Grid doubling: density {dg:+.4f}%, FTUR {tg:+.4f}%. Independent refined QMC scramble: FTUR {qq:+.4f}%.'.format(dg=100*row['quadrature_density_relative_change'],tg=100*row['quadrature_tur_relative_change'],qq=100*(row['production_other_scramble_tur_rate']/row['production_tur_rate']-1)))
  lines.append('')
 lines.extend(['The 24.8% FTUR difference at auxiliary K=20, E=0.1 is 2.94 combined sampling SE. This comparison alone does not identify its cause. It does not enter the displayed steady-state panel.',
  'All measured values and sampling errors are retained in steady_timestep_comparison.csv and .json.',
  'Reproduce with: python3 analysis/validate_steady.py --fine engine/validation',''])
 (output/'validation_report.txt').write_text('\n'.join(lines))

def evaluate(prefix,result,selected,K,power=13,origins=512,seed=99175):
 m,t,y,stats=load_run(prefix);S=int(result['quadrature_grid']);dtobs=float(np.median(np.diff(t)))
 tau=result[f'lag_K{K}'];shift=round(tau/dtobs);stride=min(result['histogram_origin_stride'],max(1,round(tau/.05)));ix=np.arange(0,len(y)-shift,stride,dtype=np.int64)
 f=selected[f'K{K}_f'];w=f-np.expm1(reverse(f));hx,hy=kernel_matrices(float(m['Gamma']),float(m['Q']),float(m['E']),float(tau),K,S)
 J=hx.T@w@hy;J=.5*(J+J[:,(-np.arange(S))%S]);values=linear_scores(y,shift,J,ix)/tau
 ttau=result[f'tur_lag_K{K}'];tshift=round(ttau/dtobs);qm=moments(y,tshift,K,m['Gamma'],m['Q'],m['E'],ttau,origins=origins,power=power,seed=seed)
 hx,hy=kernel_matrices(float(m['Gamma']),float(m['Q']),float(m['E']),float(ttau),K,S);W=j_weights(K);J=hx.T@W@hy;J=.5*(J+J[:,(-np.arange(S))%S])
 jm=linear_scores(y,tshift,J,qm['origin_indices']);j2=qm['per_origin_second'];mu=float(jm.mean());s=float(j2.mean());tur=tur_rate(mu,s,ttau)
 # Delta-method score across time blocks, retaining covariance of the two moments.
 var=s-mu*mu;den=s+mu*mu;dmu=4*mu*s/(ttau*var*den);ds=-2*mu*mu/(ttau*var*den)
 tv=dmu*(jm-mu)+ds*(j2-s)
 true=work_statistics(m,stats,y)
 return dict(prefix=str(prefix),K=K,dt=m['dt'],density_lag=tau,tur_lag=ttau,density_rate=float(values.mean()),density_sampling_se=block_se(values),tur_rate=tur,tur_sampling_se=block_se(tv),current_mean=mu,current_second_moment=s,true_rate=true['true_rate'],true_sampling_se=true['true_rate_sampling_se'],qmc_power=power,qmc_origins=qm['origins'],qmc_seed=seed)

def main(a):
 a.output.mkdir(parents=True,exist_ok=True);rows=[]
 for E in [.1,1.]:
  name=f'N100_E{E:.1f}_r0';result=json.loads((a.results/(name+'_result.json')).read_text());selected=np.load(a.results/(name+'_selected.npz'))
  for K in [5,20]:
   coarse=evaluate(a.runs/name,result,selected,K);fine=evaluate(a.fine/name,result,selected,K,seed=99176)
   second=evaluate(a.runs/name,result,selected,K,seed=99177)
   doubled=evaluate(a.runs/name,{**result,'quadrature_grid':2*result['quadrature_grid']},selected,K)
   row=dict(N=100,E=E,K=K,**{f'production_{k}':v for k,v in coarse.items() if k not in ['K','prefix']},**{f'fine_{k}':v for k,v in fine.items() if k not in ['K','prefix']},production_other_scramble_tur_rate=second['tur_rate'],quadrature_density_relative_change=doubled['density_rate']/coarse['density_rate']-1,quadrature_tur_relative_change=doubled['tur_rate']/coarse['tur_rate']-1,density_saved_score_error=coarse['density_rate']-result[f'estimator_K{K}'])
   for what in ['density','tur']:
    delta=fine[f'{what}_rate']-coarse[f'{what}_rate'];se=np.hypot(fine[f'{what}_sampling_se'],coarse[f'{what}_sampling_se'])
    row[f'{what}_relative_step_change']=delta/coarse[f'{what}_rate'];row[f'{what}_step_change_in_sampling_se']=delta/se if se else None
   row['figure_role']='primary' if K==5 else 'auxiliary'
   rows.append(row);print('Validated',E,K,flush=True)
 pd.DataFrame(rows).to_csv(a.output/'steady_timestep_comparison.csv',index=False)
 (a.output/'steady_timestep_comparison.json').write_text(json.dumps(rows,indent=2))
 write_report(rows,a.output)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,default=HERE.parent/'engine/raw');p.add_argument('--fine',type=Path,required=True);p.add_argument('--results',type=Path,default=HERE/'results');p.add_argument('--output',type=Path,default=HERE/'validation');a=p.parse_args();main(a)
