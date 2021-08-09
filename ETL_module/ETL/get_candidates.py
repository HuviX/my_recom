import gc
import multiprocessing
import random
import sys
import time
import types
from typing import Tuple, List, Callable, Dict

import numpy as np
import pandas as pd

from models_pack.models import CombinedProd2Vec, CosineModel, Top


def get_dataset(df: pd.DataFrame, size: int) -> pd.DataFrame:
    df = df.sample(frac=1)
    return df[:size]


def load_models(
    cart_w2vec: str,
    view_w2vec: str,
    top50: str,
    cosine_folder: str = "sparse_data/",
) -> Tuple[List[str], CombinedProd2Vec, CosineModel]:
    top50 = Top(top50)
    p2vec = CombinedProd2Vec([view_w2vec, cart_w2vec])
    cos_model = CosineModel(cosine_folder)
    return top50, p2vec, cos_model


def process_batch(k: int) -> pd.DataFrame:
    def paralled_prediction(row, scoring_method: Callable, n: int):
        session = {"views": row[0], "to_cart": row[1]}
        p2vec_pred = p2vec.get_prediction_for_session(
            session, n, scoring_method
        )
        cos_pred = cosine_model._predict(session, n, scoring_method)
        return pd.Series([p2vec_pred, cos_pred])

    from executor_pool_storage import (
        data,
        n,
        scoring_method,
        p2vec,
        cosine_model,
    )

    df = data[k]

    df[["w2vec_pred", "cos_pred"]] = df[["view", "to_cart"]].apply(
        lambda x: paralled_prediction(x, scoring_method, n), axis=1
    )
    return df


def get_predictions(
    df: pd.DataFrame,
    n: int,
    scoring_method: Callable,
    p2vec: CombinedProd2Vec,
    cosine_model: CosineModel,
    task_cpus=8,
) -> pd.DataFrame:
    df = np.array_split(df, task_cpus)
    LOCALS = {
        "data": df,
        "n": n,
        "scoring_method": scoring_method,
        "p2vec": p2vec,
        "cosine_model": cosine_model,
    }
    sys.modules["executor_pool_storage"] = types.ModuleType(
        "executor_pool_storage"
    )
    sys.modules["executor_pool_storage"].__dict__.update(LOCALS)

    with multiprocessing.Pool(processes=task_cpus) as pool:
        res = pool.map(process_batch, range(1))
    return pd.concat(res)


def main(
    train_path: str,
    test_path: str,
    size_of_train: int,
    size_of_test: int,
    model_paths: Dict[str, str],
    top_n: int,
    output_data: str,
    **kwargs,
) -> None:
    print("Reading Tables ...")
    df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    scoring_method = np.mean
    print("Getting data and loading models ...")
    df = get_dataset(df, size_of_train)
    test_df = get_dataset(test_df, size_of_test)
    top50, p2vec, cos_model = load_models(**model_paths)
    print("Getting predictions ...")
    pred_df = get_predictions(df, top_n, scoring_method, p2vec, cos_model)
    test_pred_df = get_predictions(
        test_df, top_n, scoring_method, p2vec, cos_model
    )

    # Getting top 50 prediction
    pred_df["top_pred"] = pred_df.apply(
        lambda x: top50.get_prediction(), axis=1
    )
    pred_df = pred_df.reset_index(drop=True)
    pred_df["type"] = "train"
    test_pred_df = test_pred_df.reset_index(drop=True)
    test_pred_df["type"] = "test"
    test_pred_df["top_pred"] = test_pred_df.apply(
        lambda x: top50.get_prediction(), axis=1
    )
    print("Saving Tables ...")
    candidates_df = pd.concat([pred_df, test_pred_df], axis=0)
    candidates_df.to_pickle(output_data)
