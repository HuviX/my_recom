import numpy as np
import pandas as pd
from numpy import log2


def avg_precision(actual, predicted, k=10):
    score = 0.0
    true_predictions = 0.0
    for i, predict in enumerate(predicted[:k]):
        if predict in actual:
            true_predictions += 1.0
            score += true_predictions / (i + 1.0)
    return score / min(len(actual), k)


def map_at_k(df: pd.DataFrame, k=10):
    true, predictions = df.values.T
    ap_at_k = [
        avg_precision(act, pred, k) for act, pred in zip(true, predictions)
    ]
    return np.mean(ap_at_k)


def dcg(prediction, actual, k=10):
    k = min(k, len(prediction))
    score = 0
    for i, x in enumerate(actual[:k]):
        if x in prediction[: i + 1]:
            score += 1 / log2(i + 2)
    return score


def ndcg(actual, predicted, k=10):
    num = dcg(predicted, actual, k)
    den = np.sum(1 / log2(i + 2) for i in range(k))
    return num / den

def get_ndcg(preds):
    true = sorted(preds, reverse=True)
    return ndcg(true, preds)

def get_mrr(preds):
    preds = np.array(preds)
    idx = np.where(preds == 1)[0]
    size = idx.size
    return 1 / (idx[0] + 1) if size > 0 else 0.
