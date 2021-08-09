import datetime
import os
from collections import defaultdict

import addict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import log2
from numpy.lib.npyio import load
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.logger_dict = defaultdict(lambda: 0)

    def write(self, name, value):
        self.writer.add_scalar(name, value, self.logger_dict[name])
        self.logger_dict[name] += 1

    def log(self, value_dict):
        for k, v in value_dict.items():
            self.write(k, v)


class DCG:
    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int DCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        self.k = k
        self.discount = self._make_discount(256)
        if gain_type in ['exp2', 'identity']:
            self.gain_type = gain_type
        else:
            raise ValueError('gain type not equal to exp2 or identity')

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        return np.sum(np.divide(gain, discount))

    def _get_gain(self, targets):
        t = targets[: self.k]
        if self.gain_type == 'exp2':
            return np.power(2.0, t) - 1.0
        else:
            return t

    def _get_discount(self, k):
        if k > len(self.discount):
            self.discount = self._make_discount(2 * len(self.discount))
        return self.discount[:k]

    @staticmethod
    def _make_discount(n):
        x = np.arange(1, n + 1, 1)
        discount = np.log2(x + 1)
        return discount


class NDCG(DCG):
    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int NDCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        super().__init__(k, gain_type)

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        dcg = super().evaluate(targets)
        ideal = np.sort(targets)[::-1]
        idcg = super().evaluate(ideal)
        return dcg / idcg

    def maxDCG(self, targets):
        """
        :param targets: ranked list with relevance
        :return:
        """
        ideal = np.sort(targets)[::-1]
        return super().evaluate(ideal)


def estimate_ndcg(loader: DataLoader, model: nn.Module, device, k=10):
    all_ndcg = []
    for query, rel in loader:
        query = query.float().squeeze().to(device)
        scores = model(query)
        rel = rel.squeeze().numpy()
        scores = scores.squeeze().cpu().numpy()
        all_ndcg.append(ndcg(rel, scores, k))
    return np.mean(all_ndcg)


def avg_precision(actual: np.ndarray, predicted: np.ndarray, k=10):
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


def dcg(element_list: np.ndarray, k: int):
    score = 0.0
    for order, rank in enumerate(element_list[:k]):
        score += float(rank) / log2((order + 2))
    return score


def ndcg(actual: np.ndarray, pred: np.ndarray, k: int = 10):
    return dcg(pred, k) / dcg(actual, k)


def mean_ndcg(df: pd.DataFrame, k=10):
    true, predictions = df.values.T
    ndcg_list = [ndcg(act, pred, k) for act, pred in zip(true, predictions)]
    return np.mean(ndcg_list)


def eval_ndcg_at_k(model, loader, k: int, device):
    ndcg_metric = NDCG(k)
    qids, rels, scores = [], [], []
    model.eval()
    with torch.no_grad():
        for qid, (x, rel) in enumerate(loader):
            if x is None or x.shape[0] == 0:
                continue
            y_tensor = model.forward(torch.Tensor(x).to(device))
            scores.append(y_tensor.cpu().numpy().squeeze())
            qids.append([qid] * x.shape[1])
            rels.append(rel)

    qids = np.hstack(qids)
    rels = np.hstack(rels)
    scores = np.hstack(scores)
    result_df = pd.DataFrame({'qid': qids, 'rel': rels[0], 'score': scores})
    session_ndcgs = []
    for qid in result_df.qid.unique():
        result_qid = result_df[result_df.qid == qid].sort_values(
            'score', ascending=False
        )
        rel_rank = result_qid.rel.values
        if ndcg_metric.maxDCG(rel_rank) == 0:
            continue
        ndcg_k = ndcg_metric.evaluate(rel_rank)
        if not np.isnan(ndcg_k):
            session_ndcgs.append(ndcg_k)
    return np.mean(session_ndcgs)


def eval_cross_entropy_loss(model, loader, device, sigma=1.0):
    """
    formula in https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

    C = 0.5 * (1 - S_ij) * sigma * (si - sj) + log(1 + exp(-sigma * (si - sj)))
    when S_ij = 1:  C = log(1 + exp(-sigma(si - sj)))
    when S_ij = -1: C = log(1 + exp(-sigma(sj - si)))
    sigma can change the shape of the curve
    """
    model.eval()
    with torch.no_grad():
        total_cost = 0
        # total_pairs = len(loader)
        pairs_in_compute = 0
        for X, Y in loader:
            Y = Y.reshape(-1, 1)
            rel_diff = Y - Y.T
            pos_pairs = (rel_diff > 0).numpy().astype(np.float32)
            num_pos_pairs = np.sum(pos_pairs, (0, 1))
            # skip negative sessions, no relevant info:
            if num_pos_pairs == 0:
                continue
            neg_pairs = (rel_diff < 0).numpy().astype(np.float32)
            num_pairs = (
                2 * num_pos_pairs
            )  # num pos pairs and neg pairs are always the same
            pos_pairs = torch.tensor(pos_pairs).to(device)
            neg_pairs = torch.tensor(neg_pairs).to(device)
            Sij = pos_pairs - neg_pairs
            # only calculate the different pairs
            diff_pairs = pos_pairs + neg_pairs
            pairs_in_compute += num_pairs

            X_tensor = X.squeeze().to(device)
            y_pred = model(X_tensor)
            y_pred_diff = y_pred - y_pred.t()

            # logsigmoid(x) = log(1 / (1 + exp(-x))) equivalent to log(1 + exp(-x))
            C = 0.5 * (1 - Sij) * sigma * y_pred_diff - F.logsigmoid(
                -sigma * y_pred_diff
            )
            C = C * diff_pairs
            cost = torch.sum(C, (0, 1))
            total_cost += cost
        avg_cost = total_cost / pairs_in_compute
    return avg_cost


def eval_map(model, loader, device):
    aps = []
    with torch.no_grad():
        for X, Y in loader:
            pred = model(X.to(device))
            act = Y.numpy()[0].tolist()
            pred = pred[0].cpu().numpy().ravel().tolist()
            ap = average_precision_score(act, pred)
            if not np.isnan(ap):
                aps.append(ap)
    return np.mean(aps)


def to_float32(data: pd.DataFrame) -> pd.DataFrame:
    for col in list(data.columns[1:]):
        try:
            data[col] = data[col].astype(np.float32)
        except:
            pass
    return data
