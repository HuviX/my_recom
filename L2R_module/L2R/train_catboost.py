from typing import Dict, Any, Optional
import os

from catboost import Pool, CatBoost
import pandas as pd


def fit_model(
    loss: str,
    train_dir: str,
    train_pool: Pool,
    test_pool: Pool,
    default_params: Dict[str, Any],
    add_params: Optional[Dict[str, Any]] = None,
) -> CatBoost:
    params = default_params
    params["loss_function"] = loss
    params["train_dir"] = train_dir
    if add_params is not None:
        params.update(add_params)
    model = CatBoost(params)
    model.fit(train_pool, eval_set=test_pool, plot=False)
    return model


def main(
    input_data: str,
    catboost_params: Dict[str, Any],
    model_fit_params: Dict[str, Any],
    save_dir: str,
    **kwargs,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = kwargs["device"]
    df = pd.read_pickle(input_data)
    train = df[
        (df.type == 'train')
    ].drop(['type', 'view', 'to_cart'], axis=1).reset_index(drop=True)
    test = df[
        (df.type == 'test')
    ].drop(['type', 'view', 'to_cart'], axis=1).reset_index(drop=True)
    cols_to_drop = [
        "index", "target", "prediction",
        "category_1", "category_2"
    ]
    X_train = train.drop(cols_to_drop, axis=1)
    X_test = test.drop(cols_to_drop, axis=1)

    y_train = train["target"]
    y_test = test["target"]

    queries_train = train["index"]
    queries_test = test["index"]

    train_pool = Pool(
        data=X_train,
        label=y_train,
        group_id=queries_train,
        # cat_features=['category_1', 'category_2']
    )

    test_pool = Pool(
        data=X_test,
        label=y_test,
        group_id=queries_test,
        # cat_features=['category_1', 'category_2']
    )
    model = fit_model(
        **model_fit_params,
        train_pool=train_pool,
        test_pool=test_pool,
        default_params=catboost_params,
    )
    model.save_model(save_dir)
