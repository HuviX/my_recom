import gc
import multiprocessing
import random
import sys
import time
import types
from typing import Tuple, List, Callable, Dict

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from scipy import sparse as sp


def avg_score(x):
    return np.mean([i[1] for i in x])


def sum_score(x):
    return sum(i[1] for i in x)


def transform(df, num_of_carts, num_of_views, categories_1, categories_2):
    exploded = df.explode("ovr_pred")
    exploded = exploded.reset_index()
    # exploded = exploded.drop(['index'], axis=1)
    # exploded.drop(['to_cart','view'], axis=1, inplace=True)
    pred_score = exploded["ovr_pred"].apply(get_prediction_and_score)

    exploded["prediction"] = [x[0] for x in pred_score]
    exploded["score"] = [x[1] for x in pred_score]
    exploded["target"] = exploded[["order", "prediction"]].apply(
        lambda x: 1 if x[1] in x[0] else 0, axis=1
    )
    exploded["mean_score_for_prediction"] = exploded.groupby(["prediction"])[
        "score"
    ].transform("mean")

    exploded["num_of_cart"] = exploded["prediction"].map(num_of_carts)
    exploded["num_of_views"] = exploded["prediction"].map(num_of_views)
    # exploded['num_of_ords'] = exploded['prediction'].map(num_of_orders)
    exploded["num_of_cart"].fillna(0, inplace=True)
    exploded["num_of_views"].fillna(0, inplace=True)
    # exploded['num_of_ords'].fillna(0, inplace=True)

    exploded["category_1"] = exploded["prediction"].map(categories_1)
    exploded["category_2"] = exploded["prediction"].map(categories_2)
    exploded["category_1"].fillna("None", inplace=True)
    exploded["category_2"].fillna("None", inplace=True)

    exploded["mean_ovr"].fillna(0, inplace=True)
    exploded["avg_w2vec"].fillna(0, inplace=True)
    exploded["avg_cosine"].fillna(0, inplace=True)

    # Были ли предикты для этой сесии от косинусной модели или от в2век
    exploded["is_popular"] = exploded["len_of_pred"] > 0

    exploded.drop(
        ["ovr_pred", "cos_pred", "w2vec_pred", "top_pred", "order"],
        axis=1,
        inplace=True,
    )
    # exploded.fillna(0, inplace=True)
    return exploded


def get_prediction_and_score(x):
    if not isinstance(x, np.int64):
        prediction = x[0]
        score = x[1]
    else:
        prediction = x
        score = 0.0
    return prediction, score


def get_prob_features(
    external_data: Dict[str, str], df: pd.DataFrame
) -> pd.Series:
    # "sparse_data/similarities_view.npz"
    sim_view = sp.load_npz(external_data["sparse_similarities_view"])
    # "sparse_data/similarities_cart_add.npz"
    sim_cart = sp.load_npz(external_data["sparse_similarities_to_cart"])

    # Proba features martices and vecs
    # "sparse_data/PAB_view.npz"
    PAB_view = sp.load_npz(external_data["PAB_view"])
    # "sparse_data/PAB_cart.npz"
    PAB_cart = sp.load_npz(external_data["PAB_cart"])
    # "sparse_data/PB_view.npy"
    with open(external_data["PB_view"], "rb") as f:
        PB_view = np.load(f).T

    with open(external_data["PB_cart"], "rb") as f:
        PB_cart = np.load(f).T

    def prob_features(row):
        view = row[0]
        cart = row[1]
        prediction = row[2]
        ovr_view = 0.0
        ovr_cart = 0.0
        max_view = 0.0
        max_cart = 0.0
        prob_view = [0.0]
        prob_cart = [0.0]

        for item in view:
            prob_view += PAB_view[prediction, item] / PB_view[item]
            sim_view_ = sim_view[prediction, item]
            ovr_view += sim_view_
            max_view = max(sim_view_, max_view)

        for item in cart:
            prob_cart += PAB_cart[prediction, item] / PB_cart[item]
            sim_cart_ = sim_cart[prediction, item]
            ovr_cart += sim_cart_
            max_cart = max(sim_cart_, max_cart)
        return pd.Series(
            [
                ovr_cart,
                ovr_view,
                prob_cart[0],
                prob_view[0],
                max_cart,
                max_view,
            ]
        )

    df[
        [
            "pred_cart_sim",
            "pred_view_sim",
            "prob_cart",
            "prob_view",
            "max_cart_sim",
            "max_view_sim",
        ]
    ] = df[["view", "to_cart", "prediction"]].apply(prob_features, axis=1)

    df[["prob_cart", "prob_view"]] = df[["prob_cart", "prob_view"]].fillna(0.0)
    return df


def category_mean_encoding(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df[df.type == "train"].copy()
    test_df = df[df.type == "test"].copy()
    global_mean = train_df["target"].mean()

    cat1_target_mean = train_df.groupby("category_1")["target"].mean()
    cumsum = (
        train_df.groupby("category_1")["target"].cumsum() - train_df["target"]
    )
    cumcnt = train_df.groupby("category_1").cumcount()
    train_df["cat1_encoded_feature"] = cumsum / cumcnt
    train_df["cat1_encoded_feature"].fillna(global_mean, inplace=True)

    test_df["cat1_encoded_feature"] = test_df["category_1"].map(
        cat1_target_mean
    )
    test_df["cat1_encoded_feature"].fillna(global_mean, inplace=True)

    cat2_target_mean = train_df.groupby("category_2")["target"].mean()
    cumsum = (
        train_df.groupby("category_2")["target"].cumsum() - train_df["target"]
    )
    cumcnt = train_df.groupby("category_2").cumcount()
    train_df["cat2_encoded_feature"] = cumsum / cumcnt
    train_df["cat2_encoded_feature"].fillna(global_mean, inplace=True)

    test_df["cat2_encoded_feature"] = test_df["category_2"].map(
        cat2_target_mean
    )
    test_df["cat2_encoded_feature"].fillna(global_mean, inplace=True)
    return train_df, test_df


def main(
    candidates_df: str,
    external_data: Dict[str, str],
    output_data: Dict[str, str],
    **kwargs,
):
    candidates_df = pd.read_pickle(candidates_df)

    candidates_df["avg_w2vec"] = candidates_df["w2vec_pred"].apply(avg_score)
    candidates_df["avg_cosine"] = candidates_df["cos_pred"].apply(avg_score)

    candidates_df["sum_w2vec"] = candidates_df["w2vec_pred"].apply(sum_score)
    candidates_df["sum_cosine"] = candidates_df["cos_pred"].apply(sum_score)

    candidates_df["len_of_w2vec"] = candidates_df["w2vec_pred"].apply(
        lambda x: len(x)
    )
    candidates_df["len_of_cos"] = candidates_df["cos_pred"].apply(
        lambda x: len(x)
    )
    candidates_df["len_of_pred"] = (
        candidates_df["len_of_cos"] + candidates_df["len_of_w2vec"]
    )

    candidates_df["ovr_pred"] = (
        candidates_df["w2vec_pred"] + candidates_df["cos_pred"]
    )

    candidates_df["mean_ovr"] = candidates_df["ovr_pred"].apply(avg_score)
    candidates_df["sum_ovr"] = candidates_df["ovr_pred"].apply(sum_score)

    # top_pred = df.iloc[0, 5]
    # def intersection_with_top(x):
    #     return np.intersect1d(top_pred, x).shape[0]

    # candidates_df['intersection_w2vec'] = candidates_df['w2vec_pred'].apply(intersection_with_top)
    # candidates_df['intersection_cosine'] = candidates_df['cos_pred'].apply(intersection_with_top)
    # candidates_df['ovr_pred'] = candidates_df['w2vec_pred'] + candidates_df['cos_pred'] + candidates_df['top_pred']

    # 'data/num_of_to_cart.pkl'
    with open(external_data["cart_num"], "rb") as handle:
        num_of_carts = pickle.load(handle)
    # 'data/num_of_view.pkl'
    with open(external_data["view_num"], "rb") as handle:
        num_of_views = pickle.load(handle)

    num_of_views = dict((k, v) for k, v in num_of_views.items())
    num_of_carts = dict((k, v) for k, v in num_of_carts.items())

    categories = pd.read_pickle(external_data["categories"])
    categories.head()
    categories["itemid"] = categories["itemid"].astype(int)

    categories_1 = dict(
        categories[["itemid", "category_name_1_level"]].values.tolist()
    )
    categories_2 = dict(
        categories[["itemid", "category_name_2_level"]].values.tolist()
    )

    del categories
    gc.collect()

    size = candidates_df.shape[0]
    df = transform(
        candidates_df, num_of_carts, num_of_views, categories_1, categories_2
    )
    df.to_pickle(output_data["exploded_df"])
    del candidates_df, num_of_views, num_of_carts, categories_1, categories_2
    gc.collect()

    df = get_prob_features(external_data, df)
    train_df, test_df = category_mean_encoding(df)
    df = pd.concat([train_df, test_df])
    df.to_pickle(output_data["features_df"])
