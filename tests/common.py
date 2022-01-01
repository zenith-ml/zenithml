import numpy as np
import pandas as pd


def create_mini_test_df(path=None):
    df: pd.DataFrame = pd.DataFrame(
        data={
            "f_bools": [True, False, True, False, True],
            "f_ints": [1, 1, 2, 2, 1],
            "f_buk": [100, 100, 200, 200, 300],
            "f_cat": ["A", "B", "C", "A", "B"],
            "f_float": [0.1, 0.1, 0.2, 0.2, 0.1],
            "f_vec1": [[0.1, 0.1], [0.4761, 0.421], [0.57973867, 0.81480247], [0.11, 0.21], [0.1, 0.1]],
            "f_vec2": [[0.1, 0.1], [0.5671, 0.231], [0.57973867, 0.0], [0.32, 0.4], [0.1, 0.1]],
            "f2_bools": [True, False, True, False, True],
            "f2_ints": [1, 1, 2, 2, 1],
            "f2_buk": [100, 100, 200, 200, 300],
            "f2_cat": [["A"], ["B", "C"], [], ["A"], ["B"]],
            "f2_float": [0.1, 0.1, 0.2, 0.2, 0.1],
            "y": list(np.random.rand(5)),
            "ids": [f"id_{i}" for i in list(np.arange(5))],
        }
    )
    if path:
        df.to_parquet(path)
    return df
