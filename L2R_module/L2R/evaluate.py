from typing import Dict, Any

from catboost import Pool, CatBoost
import numpy as np
import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from models_pack.metrics import map_at_k
from .LambdaNet import LambdaRank
from .data import QueryDS
from .utils import to_float32


def load_model_from_checkpoint(
    input_size,
    layers,
    device,
    sigma,
    weights,
    **kwargs,
):
    lambdarank_structure = [input_size, layers]
    model = LambdaRank(
        net_structures=lambdarank_structure,
        sigma=sigma,
        device=device,
    )
    model.to(device)
    model.load_state_dict(torch.load(weights))
    return model


def eval_catboost(
    data: pd.DataFrame,
    labels_df: pd.DataFrame,
    model_path: str,
    top: int = 50,
):
    model = CatBoost().load_model(model_path)
    features = model.feature_names_
    data = data[data.type == "test"]
    target = data["target"]
    prediction_item_id = data["prediction"]
    queries_test = data["index"]
    pool = Pool(
        data=data[features],
        label=target,
        group_id=queries_test,
    )
    prediction = model.predict(pool)
    test_prediction = pd.DataFrame(
        data={
            "index": queries_test,
            "pred": prediction,
            "target": target,
            "item_id": prediction_item_id,
        }
    )
    
    preds = (
        test_prediction
        .sort_values(["index", "pred"], ascending=[True, False])
        .groupby("index")["item_id"]
        .agg(list)
        .apply(lambda x: x[:top])
        .to_frame()
    )
    labels_df = labels_df[labels_df["type"] == "test"]
    new_df = (
        preds
        .merge(labels_df.reset_index()[["index", "order"]], on="index")
        .drop(["index"], axis=1)[["order", "item_id"]]
    )
    return {
        f"map_at_{top}": map_at_k(new_df, top),
    }
    # print(f"Catboost map@{top}: {map_at_k(new_df, top)}")


def eval_lambda(
    data: pd.DataFrame,
    labels_df: pd.DataFrame,
    model_params: Dict[str, Any],
    device: torch.device,
    top: int = 50,
):
    input_size = len(model_params["features"])
    lambdarank_structure = [input_size, *model_params["layers"]]
    sigma = 1.0
    my_net = LambdaRank(
        net_structures=lambdarank_structure, sigma=sigma, device=device,
    )
    my_net.to(device)
    my_net.load_state_dict(torch.load(model_params["model_path"]))
    test = data[data["type"] == "test"]
    test = test.rename({"index": "q_id"}, axis=1)
    for col in ["prob_view", "prob_cart"]:
        test[col] = test[col].apply(lambda x: np.nan
                                            if x in [np.inf, -np.inf]
                                            else x)


    sc = StandardScaler()
    test[model_params["features"]] = (
        sc.fit_transform(test[model_params["features"]].values)
    )
    print(test.shape)

    test = test.dropna(subset=["prob_view", "prob_cart"])
    print(test.shape)

    model_params["data"] = test
    test = to_float32(test)
    test_dataset = QueryDS(
        **model_params, type="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
    )

    preds = []
    with torch.no_grad():
        for _, (data, rel) in enumerate(test_loader):
            data = data.cuda()
            pred = my_net(data)
            preds.append(pred.squeeze(0).cpu().numpy()[:, 0])
    
    test_prediction = pd.DataFrame(
        data={
            "index": test["q_id"].values,
            "pred":  [x for pred in preds for x in pred],
            "item_id": test["prediction"].values.astype(int)
        }
    )

    preds = (
        test_prediction
        .sort_values(["index", "pred"], ascending=[True, False])
        .groupby("index")["item_id"]
        .agg(list)
        .apply(lambda x: x[:top])
    )
    preds = preds.to_frame()
    labels_df = labels_df[labels_df["type"] == "test"]
    new_df = (
        preds
        .merge(labels_df.reset_index()[["index", "order"]], on="index")
        .drop(["index"], axis=1)[["order", "item_id"]]
    )
    return {
        f"map_at_{top}": map_at_k(new_df, top),
    }


def log_metrics(
    metrics: Dict[str, Dict[str, float]]
):
    for model in metrics.keys():
        with mlflow.start_run():
            for metric, value in metrics[model].items():
                mlflow.log_metric(metric, value)
                mlflow.log_param("model", model)
                mlflow.log_metric("iteration", 1)
                print(model, metric, value)


def main(
    input_data: Dict[str, str],
    models: Dict[str, Any],
    mlflow_experiment_name: str,
):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    mlflow.set_experiment(mlflow_experiment_name)
    data = pd.read_pickle(input_data["data"])
    labels_df = pd.read_pickle(input_data["labels_df"])
    metrics_catboost = (
        eval_catboost(data, labels_df, models["catboost"]["model_path"])
    )
    metrics_labmdarank = (
        eval_lambda(data, labels_df, models["lambdarank"], device)
    )
    metrics = {
        "catboost": metrics_catboost,
        "lambda_rank": metrics_labmdarank,
    }
    log_metrics(metrics)
