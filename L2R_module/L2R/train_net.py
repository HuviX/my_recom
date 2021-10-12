from typing import List, Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import QueryDS
from .LambdaNet import LambdaRank, init_weights, validation_step
from .utils import NDCG, Logger, to_float32


def main(
    input_data,
    features_list: List[str],
    query_id_col: str,
    target_col: str,
    model_params: Dict[str, Any],
    model_path: str,
    **kwargs,
):
    data = pd.read_pickle(input_data)
    data = data.rename({"index": query_id_col}, axis=1)
    train = data[data["type"] == "train"]
    test = data[data["type"] == "test"]
    sc = StandardScaler()
    train[features_list] = sc.fit_transform(train[features_list].values)
    print(test.shape)
    for col in ["prob_view", "prob_cart"]:
        test[col] = test[col].apply(
            lambda x: np.nan if x in [np.inf, -np.inf] else x
        )
    test = test.dropna(subset=["prob_view", "prob_cart"])
    print(test.shape)
    test[features_list] = sc.transform(test[features_list])
    test.head()

    test = to_float32(test)
    train = to_float32(train)
    print("Getting datasets and dataloaders")
    train_dataset = QueryDS(train, features_list, query_id_col, target_col)
    test_dataset = QueryDS(
        test, features_list, query_id_col, target_col, type="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
    )
    input_size = len(features_list)
    model_params = model_params
    print("Initializing Neural Net to train ...")
    print(f"NNet params:\n  input_size: {input_size}\n" 
        "hidden_layers: {*model_params['layers']}")
    lambdarank_structure = [input_size, *model_params["layers"]]
    ndcg_gain_in_train = "exp2"
    sigma = 1.0
    device = torch.device("cpu")
    my_net = LambdaRank(
        net_structures=lambdarank_structure,
        sigma=sigma,
        device=device,
    )
    my_net.to(device)
    my_net.apply(init_weights)
    print(my_net)

    ideal_dcg = NDCG(10, ndcg_gain_in_train)
    batch_size = model_params["batch_size"]
    lr = model_params["learning_rate"]
    optimizer = torch.optim.Adam(my_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.75
    )

    writer = Logger(model_params["logdir"])
    save_dir = model_params["save_dir"]

    for epoch in range(10):
        my_net.train_step(
            my_net,
            train_loader,
            writer,
            batch_size,
            optimizer,
            ideal_dcg,
            epoch,
            scheduler, # TODO: refactor this
        )
        validation_step(my_net, test_loader, writer, device, epoch)

        torch.save(my_net.state_dict(), f"{save_dir}/model_{epoch}.pth")
    print("Saving model...")
    torch.save(my_net.state_dict(), model_path)
