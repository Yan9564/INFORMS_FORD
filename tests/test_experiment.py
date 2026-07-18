import yaml, pytest
from benchmark.runners.experiment import run_experiment
from conftest import cfg

def test_end_to_end_and_no_overwrite(ford_dirs, tmp_path):
    c={'random_seed':7,'dataset':cfg(*ford_dirs),'model':{'name':'isolation_forest','parameters':{'n_estimators':10,'random_state':7}},'threshold':{'method':'percentile','percentile':90},'run_id':'synthetic'}
    p=tmp_path/'exp.yaml'; p.write_text(yaml.safe_dump(c))
    r=run_experiment(p,tmp_path/'raw'); assert r['status']=='success' and r['schema_version']=='1.0'
    with pytest.raises(FileExistsError): run_experiment(p,tmp_path/'raw')
