import json
from benchmark.reporting.build_tables import build_tables

def test_reporting(tmp_path):
    p=tmp_path/'raw/ford/isolation_forest'; p.mkdir(parents=True)
    (p/'r.json').write_text(json.dumps({'dataset':'ford','model':'isolation_forest','input_representation':'row','random_seed':1,'threshold_method':'fixed','precision':1,'recall':1,'f1':1,'auroc':1,'auprc':1,'total_runtime_seconds':.1,'status':'success','run_id':'r'}))
    out=build_tables(tmp_path/'raw', tmp_path/'agg')
    assert out['csv'].exists() and out['markdown'].exists()
