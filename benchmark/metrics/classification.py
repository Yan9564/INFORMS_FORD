"""Point-wise/sample-wise classification metrics; no point adjustment is applied."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

def evaluate_binary(y_true, scores, threshold: float) -> dict[str, Any]:
    y = np.asarray(y_true, int); s = np.asarray(scores, float); pred = (s >= threshold).astype(int)
    if len(y)==0 or len(y)!=len(s): raise ValueError("Labels and scores must be non-empty and equal length")
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); tn=int(((pred==0)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    p,r,f,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    warnings=[]
    try: auroc=float(roc_auc_score(y,s))
    except ValueError as e: auroc=None; warnings.append(f"AUROC unavailable: {e}")
    try: auprc=float(average_precision_score(y,s))
    except ValueError as e: auprc=None; warnings.append(f"AUPRC unavailable: {e}")
    spec = tn/(tn+fp) if (tn+fp) else 0.0
    return {"true_positives":tp,"false_positives":fp,"true_negatives":tn,"false_negatives":fn,"precision":float(p),"recall":float(r),"f1":float(f),"specificity":float(spec),"accuracy":float(accuracy_score(y,pred)),"auroc":auroc,"auprc":auprc,"warnings":warnings}
