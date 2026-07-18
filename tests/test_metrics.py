from benchmark.metrics.classification import evaluate_binary

def test_metrics_counts():
    m=evaluate_binary([0,0,1,1],[.1,.9,.8,.2],.5)
    assert m['true_positives']==1 and m['false_positives']==1 and m['f1']==0.5
