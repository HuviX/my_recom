from typing import List

import addict
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class QueryDS(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        query_id_col: str,
        target_col: str,
        type="train",
        k: int = 30,  # by default from each sessions get 30 recommendations
    ):
        self.target_col = target_col
        self.features = features
        self.query_id_col = query_id_col
        self.data = data
        self.data = drop_noninformative_sessions(
            data, self.target_col, self.query_id_col, type, k
        )
        self.queries = self.data[self.query_id_col].unique()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_id = self.queries[idx]
        query_df = self.data[self.data[self.query_id_col] == query_id]
        query = query_df[self.features].values
        relevance = query_df[self.target_col].values
        return query, relevance


def drop_noninformative_sessions(
    data: pd.DataFrame, target_col: str, query_id_col: str, type: str, k: int
) -> pd.DataFrame:
    """
    Sessions that have zero relevant objects (i.e. target is 0) give
    no information during train time
    """
    print(f"{data[query_id_col].unique().size} unique sessions before filter.")
    print(f"train shape: {data.shape}")
    if type == "train":
        data = (
            data.groupby(query_id_col, group_keys=False)
            .apply(
                lambda x: x.sort_values(by="target", ascending=False).head(k)
            )
            .groupby(query_id_col, group_keys=False)
            .filter(lambda x: x[target_col].sum() > 1)
        )
    print(f"{data[query_id_col].unique().size} unique sessions after filter.")
    print(f"train shape: {data.shape}")
    return data
