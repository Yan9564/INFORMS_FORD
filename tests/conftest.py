from __future__ import annotations
import numpy as np, pandas as pd, pytest
from pathlib import Path

@pytest.fixture()
def ford_dirs(tmp_path: Path):
    acc=tmp_path/'accept'; rej=tmp_path/'reject'; acc.mkdir(); rej.mkdir()
    cols=[f'feature_{i}' for i in range(48)]
    for i in range(3):
        df=pd.DataFrame(np.arange(20*48).reshape(20,48)+i, columns=cols); df.to_csv(acc/f'Sweep_trans_{i}_accept.csv', index=False)
    for i in range(2):
        df=pd.DataFrame(np.arange(20*48).reshape(20,48)+100+i, columns=cols); df['feature_55']=[0]*10+[1]*10; df.to_csv(rej/f'Sweep_trans_{i}_reject.csv', index=False)
    return acc, rej

def cfg(acc, rej, **kw):
    d={'name':'ford','accepted_dir':str(acc),'rejected_dir':str(rej),'feature_columns':{'prefix':'feature_','start':0,'end':47},'label_column':'feature_55','normal_label':0,'anomaly_label':1,'validation_fraction':0.34,'random_seed':42,'missing_values':{'strategy':'median'},'clipping':{'enabled':True,'lower_quantile':0.001,'upper_quantile':0.999},'scaling':{'method':'standard'},'windowing':{'lookback':3,'stride':1,'representation':'full_window_flattened'},'downsampling':{'every_n_rows':1}}
    d.update(kw); return d
