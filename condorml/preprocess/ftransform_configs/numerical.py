from typing import Optional

import numpy as np

from condorml.preprocess.analyze import (
    PandasAnalyzer,
    StandardScalerBQAnalyzer,
    BQAnalyzer,
    LogScalerBQAnalyzer,
    NVTAnalyzer,
)
from condorml.preprocess.analyze.nvt_analyzer import NormalizeMinMaxNVTAnalyzer
from condorml.preprocess.analyze.nvt_analyzer import NormalizeNVTAnalyzer
from condorml.preprocess.analyze.pandas_analyzer import NumericalPandasAnalyzer
from condorml.preprocess.base_transformer import (
    NumericalBaseNVTTransformer,
    NVTColType,
    BaseNVTTransformer,
)
from condorml.preprocess.ftransform_configs.ftransform_config import FTransformConfig


class Numerical(FTransformConfig):
    def __init__(
        self,
        input_col: str,
        default_value: float = 0.0,
        dimension: int = 1,
        dtype=None,
    ):
        super().__init__(input_col=input_col, default_value=default_value, dtype=dtype)
        self.dimension = dimension

    def base_transformer(self) -> BaseNVTTransformer:
        return NumericalBaseNVTTransformer(
            input_col=self.input_col,
            col_type=NVTColType.CONT,
            default_value=self.default_value,
            is_list=self.dimension > 1,
        )

    def pandas_analyzer(self, **kwargs) -> Optional[PandasAnalyzer]:
        return None

    def bq_analyzer(self) -> Optional[BQAnalyzer]:
        return None

    def nvt_analyzer(self, **kwargs) -> Optional[NVTAnalyzer]:
        return None

    def tf_preprocess_layer(self):
        from condorml.tf.layers.preprocess.numerical import NumericalLayer

        return NumericalLayer(name=self.name, dimension=self.dimension)

    def torch_preprocess_layer(self):
        from condorml.torch.layers.preprocess.numerical import NumericalLayer

        return NumericalLayer(dimension=self.dimension)


class StandardNormalizer(Numerical):
    def pandas_analyzer(self, **kwargs) -> Optional[PandasAnalyzer]:
        return NumericalPandasAnalyzer(
            input_col=self.input_col,
            default_value=self.default_value,
            feature=self.name,
        )

    def bq_analyzer(self) -> BQAnalyzer:
        return StandardScalerBQAnalyzer(input_col=self.input_col, feature=self.name)

    def nvt_analyzer(self, **kwargs) -> Optional[NVTAnalyzer]:
        return NormalizeNVTAnalyzer(input_col=self.input_col, feature=self.name, default_value=self.default_value)

    def load(self, analyze_data):
        self._analyze_data = {
            "mean_val": analyze_data[f"{self.name}_avg"],
            "variance_val": np.power(analyze_data[f"{self.name}_stddev"], 2),
        }

    def tf_preprocess_layer(self):
        from keras.layers import Normalization

        return Normalization(
            name=self.name,
            mean=self.analyze_data.get("mean_val"),
            variance=self.analyze_data.get("variance_val"),
        )

    def torch_preprocess_layer(self):
        from condorml.torch.layers.preprocess.normalizers import NormalizationLayer

        return NormalizationLayer(
            mean=self.analyze_data.get("mean_val"),
            variance=self.analyze_data.get("variance_val"),
        )


class MinMaxNormalizer(Numerical):
    def pandas_analyzer(self, **kwargs) -> Optional[PandasAnalyzer]:
        return NumericalPandasAnalyzer(
            input_col=self.input_col,
            default_value=self.default_value,
            feature=self.name,
        )

    def bq_analyzer(self) -> BQAnalyzer:
        return StandardScalerBQAnalyzer(input_col=self.input_col, feature=self.name)

    def nvt_analyzer(self, **kwargs) -> Optional[NVTAnalyzer]:
        return NormalizeMinMaxNVTAnalyzer(
            input_col=self.input_col, feature=self.name, default_value=self.default_value
        )

    def load(self, analyze_data):
        self._analyze_data = {
            "min_val": analyze_data[f"{self.name}_min"],
            "max_val": analyze_data[f"{self.name}_max"],
        }

    def tf_preprocess_layer(self):
        import tensorflow as tf
        from condorml.tf.layers.preprocess.normalizers import MinMaxNormalizeLayer

        return MinMaxNormalizeLayer(
            name=self.name,
            min_val=tf.constant(self.analyze_data.get("min_val"), dtype=tf.float32),
            max_val=tf.constant(self.analyze_data.get("max_val"), dtype=tf.float32),
        )

    def torch_preprocess_layer(self):
        from condorml.torch.layers.preprocess.normalizers import MinMaxNormalizeLayer

        return MinMaxNormalizeLayer(min_val=self.analyze_data.get("min_val"), max_val=self.analyze_data.get("max_val"))


class LogNormalizer(Numerical):
    def __init__(
        self,
        input_col: str,
        default_value: float = 0.0,
        percentile: float = 0.05,
        dtype=None,
    ):
        super().__init__(input_col=input_col, default_value=default_value, dtype=dtype)
        self.percentile = percentile

    def load(self, analyze_data):
        self._analyze_data = {
            "percentile": analyze_data[f"{self.name}_percentile"],
        }

    def pandas_analyzer(self, **kwargs) -> PandasAnalyzer:
        return NumericalPandasAnalyzer(
            input_col=self.input_col,
            default_value=self.default_value,
            feature=self.name,
            percentile=self.percentile,
        )

    def nvt_analyzer(self, **kwargs) -> Optional[NVTAnalyzer]:
        raise NotImplementedError

    def bq_analyzer(self) -> BQAnalyzer:
        return LogScalerBQAnalyzer(input_col=self.input_col, feature=self.name, percentile=self.percentile)

    def tf_preprocess_layer(self):
        import tensorflow as tf
        from condorml.tf.layers.preprocess.normalizers import LogNormalizeLayer

        log_threshold = tf.constant(self.analyze_data.get("percentile"), dtype=tf.float32)
        return LogNormalizeLayer(name=self.name, log_threshold=log_threshold)

    def torch_preprocess_layer(self):
        import torch
        from condorml.torch.layers.preprocess.normalizers import LogNormalizeLayer

        log_threshold = torch.tensor(self.analyze_data.get("percentile"), dtype=torch.float32)
        return LogNormalizeLayer(log_threshold=log_threshold)
