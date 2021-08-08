from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .utils import eval_cross_entropy_loss, eval_map, eval_ndcg_at_k


def init_weights(m):
    # print("weights init")
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


class LambdaRank(nn.Module):
    def __init__(
        self, net_structures,  device, sigma=1.0,
    ):
        """Fully Connected Layers with Sigmoid activation at the last layer

        :param net_structures: list of int for LambdaRank FC width
        """
        super().__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            setattr(
                self,
                'fc' + str(i + 1),
                nn.Linear(net_structures[i], net_structures[i + 1]),
            )
            setattr(self, 'act' + str(i + 1), nn.ReLU())
        setattr(
            self,
            'fc' + str(len(net_structures)),
            nn.Linear(net_structures[-1], 1),
        )
        self.activation = nn.Sigmoid()
        self.sigma = sigma
        self.device = device

    def forward(self, input1):
        # from 1 to N - 1 layer, use ReLU as activation function
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            act = getattr(self, 'act' + str(i))
            input1 = act(fc(input1))

        fc = getattr(self, 'fc' + str(self.fc_layers))
        return self.activation(fc(input1)) * self.sigma

    def get_lambda(self, Y, y_pred, rank_order, N):
        pos_pairs_score_diff = 1.0 + torch.exp(
            self.sigma * (y_pred - y_pred.t())
        )
        rel_diff = Y - Y.t()
        pos_pairs = (rel_diff > 0).type(torch.float32)
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        rank_order_tensor = (
            torch.tensor(rank_order)
            .to(self.device)
            .reshape(-1, 1)
        )
        decay_diff = 1.0 / torch.log2(
            rank_order_tensor + 1.0
        ) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)

        delta_ndcg = torch.abs(N * rel_diff * decay_diff)
        lambda_update = (
            self.sigma
            * (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff)
            * delta_ndcg
        )
        lambda_update = torch.sum(lambda_update, 1, keepdim=True)
        return lambda_update


    def train_step(
        self,
        model: nn.Module,
        loader: DataLoader,
        writer: SummaryWriter,
        batch_size: int,
        optimizer: torch.optim,
        ideal_dcg: float,
        epoch: int,
        scheduler,
        k: int = 10,
    ):
        grad_batch, y_pred_batch = [], []
        model.train()
        pbar = tqdm(total=len(loader))
        for i, (X, Y) in enumerate(loader):
            X, Y = X.squeeze().to(self.device), Y.squeeze().numpy()
            N = 1.0 / ideal_dcg.maxDCG(Y)
            y_pred = model(X)
            y_pred_batch.append(y_pred)
            # compute the rank order of each document
            rank_df = pd.DataFrame({"Y": Y, "doc": np.arange(Y.shape[0])})
            rank_df = rank_df.sort_values("Y").reset_index(drop=True)
            rank_order = rank_df.sort_values("doc").index.values + 1
            with torch.no_grad():
                Y = torch.tensor(Y).view(-1, 1).to(self.device)
                lambda_update = model.get_lambda(Y, y_pred, rank_order, N)
                assert lambda_update.shape == y_pred.shape
                grad_batch.append(lambda_update)
            pbar.update(1)
            if i % batch_size == 0:
                for grad, y_pred in zip(grad_batch, y_pred_batch):
                    y_pred.backward(grad / batch_size, retain_graph=True)
                optimizer.step()
                model.zero_grad()
                grad_batch, y_pred_batch = [], []

        to_write = {
            "NDCG Train": eval_ndcg_at_k(model, loader, k, self.device),
            "Loss Train": eval_cross_entropy_loss(
                model, loader, self.device
            ),
            "MAP Train": eval_map(model, loader,  self.device),
        }
        writer.log(to_write)
        pbar.close()


def validation_step(
    model: nn.Module,
    loader: DataLoader,
    writer: SummaryWriter,
    device,
    epoch: int,
    k: int = 10,
):
    to_write = {
        'NDCG Val': eval_ndcg_at_k(model, loader, k, device),
        'Loss Val': eval_cross_entropy_loss(model, loader, device),
        'MAP Val': eval_map(model, loader, device),
    }
    writer.log(to_write)
