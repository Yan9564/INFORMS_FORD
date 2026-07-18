import numpy as np, pytest
from benchmark.models.isolation_forest import IsolationForestDetector

def test_iforest_score_direction_and_shape(tmp_path):
    X=np.r_[np.random.default_rng(1).normal(0,1,(50,2)), [[8,8]]]
    m=IsolationForestDetector(random_state=1, n_estimators=20).fit(X[:50])
    s=m.score_samples(X); assert s[-1] > np.median(s[:50])
    with pytest.raises(ValueError): m.score_samples(X.reshape(51,1,2))
    p=tmp_path/'m.joblib'; m.save(p); assert len(IsolationForestDetector.load(p).score_samples(X[:2]))==2
