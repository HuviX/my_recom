import argparse
import pickle
from typing import Dict

import matplotlib.pyplot as plt
from gensim.models import Word2Vec

from w2vec.utils import callback


def load_data(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def init_model(window_len: int, min_count: int) -> Word2Vec:
    model = Word2Vec(
        window=window_len,
        sg=1,
        hs=0,
        negative=10,
        vector_size=128,
        workers=32,
        alpha=0.01,
        min_alpha=0.0001,
        min_count=min_count,
        seed=14,
        compute_loss=True,
    )
    return model


def train(n_epochs: int, min_count: int, cfg: Dict, model_dir: str):
    paths = [cfg.data.cart, cfg.data.views]
    model_names = ["cart", "view"]
    for model_name, path in zip(model_names, paths):
        data = load_data(path)
        max_len = max(len(x) for x in data)
        print(max_len, min_count)
        if model_name == "cart":
            model = init_model(max_len, min_count // 2)
        else:
            model = init_model(max_len, min_count)
        # train model
        model.build_vocab(data, progress_per=200)
        model.train(
            data,
            total_examples=model.corpus_count,
            epochs=n_epochs,
            report_delay=1,
            compute_loss=True,
            callbacks=[callback()],
        )

        model.save(f"{model_dir}/{model_name}.model")
    print("Done")


def main(
    n_epochs: int,
    min_count: int,
    model_dir: str,
    data: Dict[str, str],
    **kwargs,
):
    paths = [data["cart"], data["views"]]
    model_names = ["cart", "view"]
    for model_name, path in zip(model_names, paths):
        data = load_data(path)
        max_len = max(len(x) for x in data)
        print(max_len, min_count)
        if model_name == "cart":
            model = init_model(max_len, min_count // 2)
        else:
            model = init_model(max_len, min_count)
        # train model
        model.build_vocab(data, progress_per=200)
        model.train(
            data,
            total_examples=model.corpus_count,
            epochs=n_epochs,
            report_delay=1,
            compute_loss=True,
            callbacks=[callback()],
        )

        model.save(f"{model_dir}/{model_name}.model")
    print("Done")
    
