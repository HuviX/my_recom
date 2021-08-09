from typing import Dict

import pandas as pd
import pickle


def main(
    data: Dict[str, str],
):
    categories = pd.read_csv(data["session_items_categories"])
    with open(data["mapping"], "rb") as handle:
        mapping = pickle.load(handle)
    categories['itemid'] = categories['itemid'].astype(str).map(mapping)
    categories.drop(
        [
            'category_id_1_level',
            'category_id_2_level',
            'catalogid'
        ], axis=1, inplace=True)
    categories.to_pickle(data["output"])
