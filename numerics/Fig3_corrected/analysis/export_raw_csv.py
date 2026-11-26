#!/usr/bin/env python3
"""Export a saved relative-coordinate trajectory to a readable long CSV."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prefix',help='Run path without _meta.json/_snapshots.bin suffix')
    p.add_argument('--output',type=Path);a=p.parse_args()
    prefix=Path(a.prefix);m=json.loads(Path(str(prefix)+'_meta.json').read_text())
    data=np.fromfile(str(prefix)+'_snapshots.bin',dtype='<f8').reshape(-1,1+2*m['N'])
    y=data[:,1:].reshape(-1,m['N'],2)
    df=pd.DataFrame(dict(time=np.repeat(data[:,0],m['N']),particle=np.tile(np.arange(m['N']),len(data)),
        x_relative=y[:,:,0].ravel(),y_relative=y[:,:,1].ravel()))
    out=a.output or Path(str(prefix)+'_snapshots.csv');df.to_csv(out,index=False);print(out)
