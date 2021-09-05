from typing import Dict

import pandas as pd
import scipy.sparse as sp
import numpy as np


def get_prob_features(
    external_data: Dict[str, str], df: pd.DataFrame
) -> pd.Series:
    # "sparse_data/similarities_view.npz"
    sim_view = sp.load_npz(external_data["sparse_similarities_view"])
    # "sparse_data/similarities_cart_add.npz"
    print("HERE")
    sim_cart = sp.load_npz(external_data["sparse_similarities_to_cart"])
    print("HERE")
    # Proba features martices and vecs
    # "sparse_data/PAB_view.npz"
    PAB_view = sp.load_npz(external_data["PAB_view"])
    # "sparse_data/PAB_cart.npz"
    print("HERE")
    PAB_cart = sp.load_npz(external_data["PAB_cart"])
    # "sparse_data/PB_view.npy"
    print("HERE")
    with open(external_data["PB_view"], "rb") as f:
        PB_view = np.load(f).T
    print("HERE")
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
    train_df = df[df.type == "train"] #.count()
    test_df = df[df.type == "test"] #.count()
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
    exploded_df: str,
    external_data: Dict[str, str],
    output_data: Dict[str, str],
    **kwargs,
):
    df = pd.read_pickle(exploded_df)
    print("GETTING FEATURES...")
    df = get_prob_features(external_data, df)
    print("GETTING MEAN ENCODING...")
    train_df, test_df = category_mean_encoding(df)
    print("SAVING TABLES...")
    train_df.to_pickle(output_data["train_data"])
    test_df.to_pickle(output_data["test_data"])
