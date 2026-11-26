#!/usr/bin/env python3
"""Exact particle-number sweep for the supplementary harmonic benchmark."""
from pathlib import Path
import numpy as np
import pandas as pd
from harmonic_cluster import weak_coefficient

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'data'/'harmonic_N_scaling.csv'
rows=[]
for n in np.unique(np.r_[np.arange(1,11),np.round(np.geomspace(10,100,31)).astype(int)]):
    for kappa in [.01,.1,1.]:
        rows.append(dict(N=int(n),kappa=kappa,M=20,tau=1.,
                         weak_est_coefficient=weak_coefficient(int(n),kappa),
                         weak_true_coefficient=1.5*n))
pd.DataFrame(rows).to_csv(OUT,index=False)
print(OUT)
