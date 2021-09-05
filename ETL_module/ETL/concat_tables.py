from typing import Dict

import pandas as pd

def main(input_data: Dict[str, pd.DataFrame], output_data: str):
    pd.concat(
        [
            pd.read_pickle(input_data["train_data"]),
            pd.read_pickle(input_data["test_data"])
        ]
    ).to_pickle(output_data)
