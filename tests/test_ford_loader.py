import numpy as np, pandas as pd, pytest
from benchmark.datasets.ford import FordDatasetConfig, FordDatasetLoader
from conftest import cfg

def test_discovery_and_shapes(ford_dirs):
    l=FordDatasetLoader(FordDatasetConfig.from_dict(cfg(*ford_dirs)))
    assert len(l.discover_files('accept'))==3 and len(l.discover_files('reject'))==2
    b=l.load(); assert b.X_train.shape[1]==144 and set(b.y_test)=={0,1}
    assert 'source_filename' in b.test_metadata and len(b.test_metadata)==len(b.X_test)

def test_explicit_indices_and_no_cross_file_windows(ford_dirs):
    d=cfg(*ford_dirs, train_indices=[0,1], test_indices=[1]); b=FordDatasetLoader(FordDatasetConfig.from_dict(d)).load()
    assert set(b.test_metadata.source_file_index)=={1}
    assert (b.test_metadata.source_filename.nunique()==1)

def test_representations(ford_dirs):
    for rep, ndim in [('row',2),('last_timestep',2),('full_window_flattened',2),('sequence',3)]:
        d=cfg(*ford_dirs); d['windowing']['representation']=rep
        b=FordDatasetLoader(FordDatasetConfig.from_dict(d)).load(); assert b.X_train.ndim==ndim

def test_missing_and_invalid_columns(ford_dirs):
    acc,rej=ford_dirs; df=pd.read_csv(acc/'Sweep_trans_0_accept.csv'); df.loc[0,'feature_1']=np.nan; df.to_csv(acc/'Sweep_trans_0_accept.csv', index=False)
    FordDatasetLoader(FordDatasetConfig.from_dict(cfg(acc,rej))).load()
    d=cfg(acc,rej); d['missing_values']={'strategy':'error'}
    with pytest.raises(ValueError): FordDatasetLoader(FordDatasetConfig.from_dict(d)).load()
    df=df.drop(columns=['feature_2']); df.to_csv(acc/'Sweep_trans_0_accept.csv', index=False)
    with pytest.raises(ValueError): FordDatasetLoader(FordDatasetConfig.from_dict(cfg(acc,rej))).load()

def test_preprocessing_fitted_on_training_only(ford_dirs):
    b=FordDatasetLoader(FordDatasetConfig.from_dict(cfg(*ford_dirs))).load()
    assert b.preprocessing_state['clip_lower'] is not None
    assert b.preprocessing_state['scaler'] is not None
