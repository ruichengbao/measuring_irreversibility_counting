#!/usr/bin/env python3
"""Plot Fig. 3 variants and the comparison of expanded versus integer lags.

The input is a long-format CSV with one row per grid K, particle number N and
field E. Required columns are K,N,E,true_rate,true_rate_sd,baseline_rate,
expanded_rate,expanded_rate_sd. Optional short_rate and short_rate_sd columns
are retained in the exported tables. All rates are per unit physical time.

The expanded strategy is selected using training data. This script never
maximizes, clips, smooths, or reselects estimates. Error bars are the supplied
standard deviations across realizations, not standard errors.

Example:
 python plot_reverse_lag_figures.py --summary ../analysis/summary.csv
"""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

FIELDS=[.1,.2,.4,.5,.8,1.]
NUMBERS=[1,2,5,10,20,50,60,80,100]
GRIDS=[10,20,160]
COLORS=['#4c78a8','#72b7b2','#54a24b','#e89c38','#b279a2','#9d755d']
MARKERS=['o','s','^','D','v','P']
STYLES=['-','--','-.',':','--','-.']
plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':12.,'mathtext.fontset':'stix',
 'axes.labelsize':14.,'axes.linewidth':1.1,'xtick.labelsize':12.,
 'ytick.labelsize':12.,'legend.fontsize':10.5,'lines.linewidth':2.,
 'lines.markersize':4.,'xtick.direction':'in','ytick.direction':'in',
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.transparent':False,
})


def read_summary(path):
 path=Path(path)
 if path.is_dir():
  frames=[]
  for f in sorted(path.glob('fig3_K*_summary.csv')):
   a=pd.read_csv(f)
   if 'K' not in a:a['K']=int(f.stem.split('_')[1][1:])
   frames.append(a)
  if not frames:raise ValueError('No per-grid summaries found')
  d=pd.concat(frames,ignore_index=True)
 else:
  d=pd.read_csv(path)
 if 'baseline_rate' not in d and 'old_rate' in d:
  d=d.rename(columns={'old_rate':'baseline_rate','old_rate_sd':'baseline_rate_sd'})
 required=['K','N','E','true_rate','true_rate_sd','baseline_rate',
           'expanded_rate','expanded_rate_sd','short_rate','short_rate_sd']
 missing=[c for c in required if c not in d]
 if missing:raise ValueError('Missing plotting columns: '+', '.join(missing))
 if d.duplicated(['K','N','E']).any():raise ValueError('Duplicate (K,N,E) rows')
 if 'realizations' in d and not (d.realizations==5).all():
  raise ValueError('Each point must contain all five realizations')
 for k in [5,10,20,40,80,160]:
  for e in FIELDS:
   s=d[(d.K==k)&np.isclose(d.E,e)]
   if sorted(s.N.astype(int).tolist())!=NUMBERS:
    raise ValueError(f'Incomplete particle-number grid: K={k}, E={e}')
 if not np.isfinite(d[required].to_numpy(dtype=float)).all():
  raise ValueError('Non-finite plotting data')
 if (d.true_rate<=0).any() or (d.baseline_rate<=0).any():
  raise ValueError('True and baseline rates must be positive')
 if (d.expanded_rate_sd<0).any() or (d.true_rate_sd<0).any():
  raise ValueError('Standard deviations cannot be negative')
 d['quality_factor_Q']=d.expanded_rate/d.true_rate
 d['expanded_over_integer']=d.expanded_rate/d.baseline_rate
 if 'short_rate' in d:
  d['short_over_integer']=d.short_rate/d.baseline_rate
 return d.sort_values(['K','E','N'])


def base(ax):
 ax.grid(True,ls='--',lw=.55,color='#dddddd')
 ax.set_axisbelow(True)
 ax.tick_params(length=4,width=.9)


def save(fig,out):
 out.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight')
 fig.savefig(out.with_suffix('.png'),dpi=300,bbox_inches='tight')
 plt.close(fig)


def draw_main(d,k,out):
 data=d[d.K==k]
 fig,axs=plt.subplots(2,1,figsize=(5.2,7.4))
 fig.subplots_adjust(left=.18,right=.98,bottom=.08,top=.965,hspace=.42)
 a,b=axs
 for ax in axs:
  base(ax);ax.set_xlabel(r'$N$');ax.set_xlim(-2,105)
  ax.set_xticks([0,20,40,60,80,100])
 s=data[np.isclose(data.E,.1)].sort_values('N')
 a.errorbar(s.N,s.true_rate,yerr=s.true_rate_sd,color=COLORS[0],marker='o',
            ls='-',capsize=0,label=r'$\sigma_{[0,\mathcal{T}]}/\mathcal{T}$')
 a.errorbar(s.N,s.expanded_rate,yerr=s.expanded_rate_sd,color=COLORS[1],
            marker='s',ls='--',capsize=0,
            label=r'$\sigma^{\rm est,*}_{[0,\mathcal{T}]}/\mathcal{T}$')
 a.set_ylabel('Entropy production')
 amin=min(0.,float((s.expanded_rate-s.expanded_rate_sd).min()))
 amax=max(float((s.true_rate+s.true_rate_sd).max()),
          float((s.expanded_rate+s.expanded_rate_sd).max()))
 a.set_ylim(amin*1.1,amax*1.1)
 a.yaxis.set_major_locator(MaxNLocator(nbins=6))
 a.legend(loc='upper left',frameon=True,borderpad=.45,handlelength=2.4)
 # This inset reports the same expanded strategy at each observation grid.
 si=d[(d.N==5)&np.isclose(d.E,.1)&d.K.isin([5,10,20,40,80,160])].sort_values('K')
 if len(si)==6:
  ia=a.inset_axes([.70,.115,.27,.17])
  yy=si.expanded_rate/(si.E**2);err=si.expanded_rate_sd/(si.E**2)
  ia.errorbar(si.K,yy,yerr=err,color=COLORS[4],marker='s',lw=1.4,
              markersize=3.,elinewidth=.8,capsize=1.5)
  ia.set_xscale('log',base=2)
  ia.text(.95,.10,r'$N=5,\ E=0.1$',ha='right',va='bottom',fontsize=8.,
          transform=ia.transAxes)
  ia.set_xlabel('Boxes per side',fontsize=8.5,labelpad=1.)
  ia.set_ylabel(r'$\sigma^{\rm est,*}/(\mathcal{T}E^2)$',fontsize=8.5,labelpad=.5)
  ia.set_xticks([5,10,20,40,80,160]);ia.set_xticklabels(['5','10','20','40','80','160'])
  ia.minorticks_off();ia.set_xlim(4,200)
  lo=float((yy-err).min());hi=float((yy+err).max());span=max(hi-lo,.1)
  ia.set_ylim(min(0.,lo-.15*span) if lo<0 else max(0.,lo-.7*span),hi+.15*span)
  ia.yaxis.set_major_locator(MaxNLocator(nbins=3))
  ia.tick_params(labelsize=7.8,length=2.5)
  ia.grid(True,ls='--',lw=.4,color='#dddddd')
  for spine in ia.spines.values():spine.set_linewidth(.8)
 for ii,e in enumerate(FIELDS):
  s=data[np.isclose(data.E,e)].sort_values('N')
  b.plot(s.N,s.quality_factor_Q,color=COLORS[ii],marker=MARKERS[ii],
          ls=STYLES[ii],label=rf'$E={e:.1f}$')
 qlo=min(0.,float(data.quality_factor_Q.min())-.03)
 qhi=max(1.05,float(data.quality_factor_Q.max())+.03)
 b.set_ylim(qlo,qhi)
 if qlo==0 and qhi==1.05:b.set_yticks(np.arange(0,1.01,.2))
 b.set_ylabel(r'$\mathcal{Q}=\sigma^{\rm est,*}_{[0,\mathcal{T}]}/\sigma_{[0,\mathcal{T}]}$')
 b.legend(loc='upper right',ncol=2,columnspacing=1.,handlelength=2.,frameon=True)
 for label,ax in zip(['(a)','(b)'],axs):
  ax.text(-.17,1.065,label,transform=ax.transAxes,fontsize=20,ha='left',va='bottom')
 save(fig,out)


def draw_gain(d,out):
 if 'short_over_integer' not in d:
  raise ValueError('short_rate is required for the strategy comparison figure')
 fig,axs=plt.subplots(3,2,figsize=(9.2,8.6),sharex=True,sharey='row')
 fig.subplots_adjust(left=.10,right=.98,bottom=.075,top=.88,hspace=.20,wspace=.12)
 handles=labels=None
 columns=[('short_over_integer','Phase-resolved windows'),
          ('expanded_over_integer','Including integer-lag strategy')]
 # A common scale and a tick at unity make both losses and gains explicit.
 shown=d[d.K.isin(GRIDS)]
 low=min(float(shown.short_over_integer.min()),float(shown.expanded_over_integer.min()))
 high=max(float(shown.short_over_integer.max()),float(shown.expanded_over_integer.max()))
 ylim=(min(0.,np.floor(low)),max(2.,np.ceil(1.08*high)))
 for row,k in enumerate(GRIDS):
  q=d[d.K==k]
  for col,(column,title) in enumerate(columns):
   ax=axs[row,col];base(ax)
   ax.axhline(1.,color='#777777',lw=1.,ls='--',zorder=0)
   for ii,e in enumerate(FIELDS):
    s=q[np.isclose(q.E,e)].sort_values('N')
    ax.plot(s.N,s[column],color=COLORS[ii],marker=MARKERS[ii],
            ls=STYLES[ii],label=rf'$E={e:.1f}$')
   ax.set_ylim(*ylim)
   ax.set_xlim(-2,105);ax.set_xticks([0,20,40,60,80,100])
   ax.set_yticks(np.arange(ylim[0],ylim[1]+.5,1.))
   label=chr(ord('a')+2*row+col)
   ax.text(.965,.90,rf'({label})  ${k}\times{k}$',transform=ax.transAxes,
           ha='right',va='top',fontsize=13.,
           bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2.))
   if row==0:ax.set_title(title,fontsize=12.5,pad=10.)
   if row==2:ax.set_xlabel(r'$N$')
   handles,labels=ax.get_legend_handles_labels()
 fig.text(.02,.48,'Bound / integer-lag bound',rotation=90,ha='center',va='center',fontsize=14.)
 fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.995),
            ncol=6,columnspacing=1.1,handlelength=2.,frameon=True,fontsize=10.5)
 save(fig,out)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--summary',type=Path,required=True,help='Combined CSV or directory containing fig3_K*_summary.csv')
 p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
 a=p.parse_args();d=read_summary(a.summary)
 a.output_dir.mkdir(parents=True,exist_ok=True)
 for k in GRIDS:draw_main(d,k,a.output_dir/f'Fig3_reverse_lag_K{k}')
 draw_gain(d,a.output_dir/'FigS_reverse_lag_gain')
 d.to_csv(a.output_dir/'reverse_lag_plotting_table.csv',index=False)
 print(json.dumps({'grids':GRIDS,'points_per_grid':54,
  'ratio_range':[float(d.expanded_over_integer.min()),float(d.expanded_over_integer.max())]},indent=2))

if __name__=='__main__':main()
