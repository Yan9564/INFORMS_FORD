import pytest
from benchmark.metrics.thresholding import select_threshold

def test_threshold_methods():
    s=[0,1,2,3]; y=[0,0,1,1]
    assert select_threshold(s, method='fixed', fixed_threshold=1.5).threshold==1.5
    assert select_threshold(s, method='percentile', percentile=50).threshold==1.5
    assert select_threshold(s, method='contamination', contamination=.25).threshold==2.25
    assert select_threshold(s, y, method='best_f1').threshold==2
    with pytest.raises(ValueError): select_threshold(s, method='bad')
