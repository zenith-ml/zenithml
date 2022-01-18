import abc
from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd


class PandasAnalyzer(abc.ABC):
    def __init__(self, input_col: Union[str, List[str]], feature: str, default_value: Optional[Any], **kwargs):
        self.feature = feature
        self.input_col = input_col
        self.default_value = default_value
        self.kwargs = kwargs

    @abc.abstractmethod
    def analyze(self, df: pd.DataFrame) -> Optional[Dict]:
        pass

    @staticmethod
    def fit(input_configs: List, df: pd.DataFrame):
        analyze_data: Dict[str, Any] = {}
        for layer in input_configs:
            data = layer.pandas_analyzer().analyze(df) if layer.pandas_analyzer() is not None else None
            if data:
                analyze_data.update(**data)
        for layer in input_configs:
            layer.load(analyze_data)
        return analyze_data


class NumericalPandasAnalyzer(PandasAnalyzer):
    def analyze(self, df: pd.DataFrame) -> Optional[Dict]:
        percentile = self.kwargs.get("percentile")
        bins = self.kwargs.get("bins")

        analyze_data = {
            f"{self.feature}_max": df[self.input_col].max(),
            f"{self.feature}_min": df[self.input_col].min(),
            f"{self.feature}_avg": df[self.input_col].mean(),
            f"{self.feature}_stddev": df[self.input_col].std(),
        }

        if percentile:
            analyze_data.update(
                {
                    f"{self.feature}_percentile": np.percentile(
                        df[df[self.input_col] > 0][self.input_col].values, percentile
                    )
                }
            )
        if bins:
            analyze_data.update({f"{self.feature}_bins": list(np.histogram(df[self.input_col].values, bins=bins)[1])})
        return analyze_data


class CategoricalPandasAnalyzer(PandasAnalyzer):
    def analyze(self, df: pd.DataFrame) -> Optional[Dict]:
        top_k = self.kwargs.get("top_k")
        if top_k is not None:
            return {f"{self.feature}_cat": list(df[self.input_col].value_counts()[:top_k].index)}
        else:
            return {f"{self.feature}_cat": list(df[self.input_col].value_counts().index)}


class CategoricalListPandasAnalyzer(PandasAnalyzer):
    def analyze(self, df: pd.DataFrame) -> Optional[Dict]:
        u, count = np.unique(
            np.asarray([item for _, row in df[self.input_col].iteritems() for item in row]), return_counts=True
        )
        count_sort_ind = np.argsort(-count)
        top_k = self.kwargs.get("top_k")
        if top_k is not None:
            return {f"{self.feature}_cat": list(u[count_sort_ind][:top_k])}
        else:
            return {f"{self.feature}_cat": list(u[count_sort_ind])}
