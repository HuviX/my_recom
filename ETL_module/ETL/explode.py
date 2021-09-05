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


def main(
    candidates_df: str,
    external_data: Dict[str, str],
    output_data: Dict[str, str],
    **kwargs,
):
    print("READING TABLE...")
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
    print("GETTING EXTERNAL DATA...")
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

    print("TRANSFORM TABLES...")
    df = transform(
        candidates_df, num_of_carts, num_of_views, categories_1, categories_2
    )
    print("SAVING TABLE...")
    df.to_pickle(output_data["exploded_df"])
