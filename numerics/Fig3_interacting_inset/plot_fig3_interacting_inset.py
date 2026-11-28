#!/usr/bin/env python3
"""Plot the integer-period counting bound with the interacting-system inset.

The main curves use the grid-specific CSV tables. The inset uses the five
N=5, E=0.1 realizations at each resolution. Error bars are sample standard
deviations across these realizations. Use --tables and --inset to plot a rerun.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
COLORS=['#4c78a8','#72b7b2','#54a24b','#e89c38','#b279a2','#9d755d']
MARKERS=['o','s','^','D','v','P']
STYLES=['-','--','-.',':','--','-.']
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':12.,
    'mathtext.fontset':'stix', 'axes.labelsize':14.,
    'axes.linewidth':1.1, 'xtick.labelsize':12., 'ytick.labelsize':12.,
    'legend.fontsize':10.5, 'lines.linewidth':2., 'lines.markersize':4.,
    'xtick.direction':'in', 'ytick.direction':'in',
    'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.transparent':False,
})

def field(table,*names,default=None):
    for name in names:
        if name in table: return table[name].to_numpy()
    if default is not None:return np.full(len(table),default,dtype=float)
    raise ValueError('Expected one of columns '+str(names))

def base(ax):
    ax.grid(True,ls='--',lw=.55,color='#dddddd')
    ax.set_axisbelow(True)
    ax.tick_params(length=4,width=.9)

def draw(table,inset,output):
    data=pd.read_csv(table)
    fig,axs=plt.subplots(2,1,figsize=(5.2,7.4))
    fig.subplots_adjust(left=.18,right=.98,bottom=.08,top=.965,hspace=.42)
    a,b=axs
    for ax in axs:
        base(ax)
        ax.set_xlabel(r'$N$')
        ax.set_xlim(-2,105)
        ax.set_xticks([0,20,40,60,80,100])
    s=data[np.isclose(data.E,.1)].sort_values('N')
    true=field(s,'true_rate','sigma_real_mean','true_rate_mean')
    true_sd=field(s,'true_rate_sd','sigma_real_std','true_rate_se',default=0)
    est=field(s,'estimator_rate','sigma_est_mean','estimator_rate_mean')
    est_sd=field(s,'estimator_rate_sd','sigma_est_std','estimator_rate_se',default=0)
    a.errorbar(s.N,true,yerr=true_sd,color=COLORS[0],marker='o',ls='-',
               capsize=0,label=r'$\sigma_{[0,\mathcal{T}]}/\mathcal{T}$')
    a.errorbar(s.N,est,yerr=est_sd,color=COLORS[1],marker='s',ls='--',
               capsize=0,label=r'$\sigma^{\rm est,*}_{[0,\mathcal{T}]}/\mathcal{T}$')
    a.set_ylabel('Entropy production')
    a.set_ylim(0,1.10*max(np.max(true+true_sd),np.max(est+est_sd)))
    a.yaxis.set_major_locator(MaxNLocator(nbins=6))
    a.legend(loc='upper left',frameon=True,borderpad=.45,handlelength=2.4)
    if inset is not None:
        resolution=pd.read_csv(inset)
        ia=a.inset_axes([.70,.115,.27,.17])
        si=resolution.sort_values('K')
        ia.errorbar(si.K, si.estimator_rate/(si.E**2),
                    yerr=si.estimator_rate_sd/(si.E**2),
                    color=COLORS[4], marker='s', lw=1.4, markersize=3.,
                    elinewidth=.8, capsize=1.5)
        ia.set_xscale('log',base=2)
        ia.text(.95,.10,r'$N=5,\ E=0.1$',ha='right',va='bottom',
                fontsize=8.,transform=ia.transAxes)
        ia.set_xlabel('Boxes per side',fontsize=8.5,labelpad=1.)
        ia.set_ylabel(r'$\sigma^{\rm est,*}/(\tau E^2)$',
                      fontsize=8.5,labelpad=.5)
        ia.set_xticks([5,10,20,40,80,160])
        ia.set_xticklabels(['5','10','20','40','80','160'])
        ia.minorticks_off()
        ia.set_xlim(4,200)
        ia.set_ylim(1.2,2.95)
        ia.set_yticks([1.5,2.,2.5])
        ia.tick_params(labelsize=7.8,length=2.5)
        ia.grid(True,ls='--',lw=.4,color='#dddddd')
        for spine in ia.spines.values():spine.set_linewidth(.8)
    for ii,E in enumerate([.1,.2,.4,.5,.8,1.]):
        s=data[np.isclose(data.E,E)].sort_values('N')
        q=field(s,'quality_factor_Q','quality_ratio','Q')
        b.plot(s.N,q,color=COLORS[ii],marker=MARKERS[ii],ls=STYLES[ii],label=rf'$E={E:.1f}$')
    b.set_ylim(0,1.05);b.set_yticks(np.arange(0,1.01,.2))
    b.set_ylabel(r'$\mathcal{Q}=\sigma^{\rm est,*}_{[0,\mathcal{T}]}/\sigma_{[0,\mathcal{T}]}$')
    b.legend(loc='upper right',ncol=2,columnspacing=1.,handlelength=2.,frameon=True)
    for label,ax in zip(['(a)','(b)'],axs):
        ax.text(-.17,1.065,label,transform=ax.transAxes,fontsize=20,ha='left',va='bottom')
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'),dpi=300,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid',choices=['10','20','all'],default='all')
    p.add_argument('--output-dir',type=Path,default=ROOT)
    p.add_argument('--tables',type=Path,default=ROOT.parent/'Fig3_grid_update'/'plotting_tables')
    p.add_argument('--inset',type=Path,default=ROOT/'data'/'inset_interacting.csv')
    args=p.parse_args()
    grids=[10,20] if args.grid=='all' else [int(args.grid)]
    for K in grids:
        draw(args.tables/f'fig3_K{K}_summary.csv',args.inset,
             args.output_dir/f'Fig3_K{K}_interacting_inset')
