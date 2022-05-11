from typing import List, Optional, Union

import pandas as pd

from zenithml.preprocess.analyze import (
    PandasAnalyzer,
    BQAnalyzer,
    NVTAnalyzer,
    CategoricalBQAnalyzer,
    WeightedCategoricalBQAnalyzer,
    BucketizedBQAnalyzer,
    CategoricalPandasAnalyzer,
    CategoricalListPandasAnalyzer,
    NumericalPandasAnalyzer,
)
from zenithml.preprocess.analyze.nvt_analyzer import CategorifyNVTAnalyzer
from zenithml.preprocess.base_transformer import (
    NVTColType,
    BaseNVTTransformer,
    CategoricalBaseNVTTransformer,
    BucketizeBaseNVTTransformer, NumericalBaseNVTTransformer,
)
from zenithml.preprocess.ftransform_configs.ftransform_config import FTransformConfig


class Categorical(FTransformConfig):
    def __init__(
        self,
        input_col: Union[str, List[str]],
        dimension: int = 1,
        num_buckets: Optional[int] = None,
        top_k: int = 100000,
        vocab: Optional[List[str]] = None,
        dtype=None,
        weights=None,
        export_as_parquet: bool = False,
        seq_length: int = 1,
    ):
        super().__init__(input_col=input_col, default_value=None, dtype=dtype)
        self.dimension = dimension
        self.num_buckets = num_buckets
        self.top_k = top_k
        self.weights = weights
        self.export_as_parquet = export_as_parquet
        self.seq_length = seq_length
        if self.weights is not None:
            assert dimension > 1, "weights can only be set for Embeddings (i.e., dimension > 1)"

        if vocab is not None:
            self._analyze_data = {"vocab": vocab}

    def pandas_analyzer(self, **kwargs) -> PandasAnalyzer:
        return CategoricalPandasAnalyzer(
            input_col=self.input_col, default_value=self.default_value, feature=self.name, top_k=self.top_k
        )

    def base_transformer(self) -> Union[List[BaseNVTTransformer], BaseNVTTransformer]:
        return CategoricalBaseNVTTransformer(
            input_col=self.input_col,
            col_type=NVTColType.CAT,
            default_value=self.default_value,
            vocab=self.analyze_data.get("vocab"),
        )

    def bq_analyzer(self) -> Optional[BQAnalyzer]:
        return CategoricalBQAnalyzer(
            input_col=self.input_col,
            feature=self.name,
            top_k=self.top_k,
            export_as_parquet=self.export_as_parquet,
        )

    def nvt_analyzer(self, dask_working_dir=None, **kwargs) -> Optional[NVTAnalyzer]:
        return CategorifyNVTAnalyzer(input_col=self.input_col, feature=self.name, default_value=self.default_value)

    def load(self, analyze_data):
        if self._analyze_data.get("vocab") is None:
            if isinstance(analyze_data[f"{self.name}_cat"], str):
                df = pd.read_parquet(analyze_data[f"{self.name}_cat"])
                analyze_data = {k: v for k, v in df.to_dict(orient="list").items()}
            self._analyze_data = {"vocab": list(analyze_data[f"{self.name}_cat"])}
            self.num_buckets = len(self._analyze_data.get("vocab")) + 1

    def get_num_buckets(self):
        return len(self.analyze_data.get("vocab")) + 1

    def tf_preprocess_layer(self):
        from zenithml.tf.layers.preprocess.embedding import EmbeddingLayer
        from zenithml.tf.layers.preprocess.nhotencoder import NHotEncodingLayer

        num_buckets = self.get_num_buckets()
        if self.dimension == 1:
            return NHotEncodingLayer(name=self.name, num_buckets=num_buckets)
        else:
            return EmbeddingLayer(
                name=self.name,
                num_buckets=num_buckets,
                dimensions=self.dimension,
                weights=self.weights,
                seq_length=self.seq_length,
            )

    def torch_preprocess_layer(self):
        from zenithml.torch.layers.preprocess.embedding import EmbeddingLayer
        from zenithml.torch.layers.preprocess.nhotencoder import NHotEncodingLayer

        num_buckets = self.get_num_buckets()
        if self.dimension == 1:
            return NHotEncodingLayer(num_buckets=num_buckets)
        else:
            return EmbeddingLayer(
                num_buckets=num_buckets,
                dimensions=self.dimension,
                weights=self.weights,
                seq_length=self.seq_length,
            )


class CategoricalList(Categorical):
    def pandas_analyzer(self, **kwargs) -> PandasAnalyzer:
        return CategoricalListPandasAnalyzer(
            input_col=self.input_col,
            default_value=self.default_value,
            feature=self.name,
            top_k=self.top_k,
        )

    def bq_analyzer(self) -> Optional[BQAnalyzer]:
        return WeightedCategoricalBQAnalyzer(
            input_col=self.input_col,
            feature=self.name,
            top_k=self.top_k,
            export_as_parquet=self.export_as_parquet,
        )

    def base_transformer(self) -> Union[List[BaseNVTTransformer], BaseNVTTransformer]:
        return CategoricalBaseNVTTransformer(
            input_col=self.input_col,
            col_type=NVTColType.SPARSE_AS_DENSE if self.seq_length > 1 else NVTColType.CAT,
            default_value=self.default_value,
            vocab=self.analyze_data.get("vocab"),
            seq_length=self.seq_length,
        )


class WeightedCategoricalList(CategoricalList):
    def __init__(
        self,
        input_col: str,
        weight_col: str,
        num_buckets: Optional[int] = None,
        top_k: int = 100000,
        vocab: Optional[List[str]] = None,
        dtype=None,
        weights=None,
        export_as_parquet: bool = False,
    ):

        super().__init__(
            [input_col, weight_col],
            dimension=1,
            num_buckets=num_buckets,
            top_k=top_k,
            vocab=vocab,
            dtype=dtype,
            weights=weights,
            export_as_parquet=export_as_parquet,
            seq_length=1,
        )
        self.cat_col = input_col
        self.weight_col = weight_col

    @property
    def name(self):
        return f"{self.input_col}_weight_{self.weight_col}"

    def nvt_analyzer(self, dask_working_dir=None, **kwargs) -> Optional[NVTAnalyzer]:
        return CategorifyNVTAnalyzer(input_col=self.cat_col, feature=self.name, default_value=self.default_value)

    def pandas_analyzer(self, **kwargs) -> PandasAnalyzer:
        return CategoricalListPandasAnalyzer(
            input_col=self.cat_col,
            default_value=self.default_value,
            feature=self.name,
            top_k=self.top_k,
        )

    def bq_analyzer(self) -> Optional[BQAnalyzer]:
        return WeightedCategoricalBQAnalyzer(
            input_col=self.cat_col,
            feature=self.name,
            top_k=self.top_k,
            export_as_parquet=self.export_as_parquet,
        )

    def base_transformer(self) -> List[BaseNVTTransformer]:
        return [
            CategoricalBaseNVTTransformer(
                input_col=self.cat_col,
                col_type=NVTColType.CAT,
                default_value=self.default_value,
                vocab=self.analyze_data.get("vocab"),
            ),
            NumericalBaseNVTTransformer(
                input_col=self.weight_col,
                col_type=NVTColType.CONT,
                default_value=0.0,
                is_list=True,
            ),
        ]

    def tf_preprocess_layer(self):
        from zenithml.tf.layers.preprocess.weighted_nhotencoder import WeightedNHotEncodingLayer

        num_buckets = self.get_num_buckets()
        if self.dimension == 1:
            return WeightedNHotEncodingLayer(input_col=self.name, weight_col=self.weight_col, num_buckets=num_buckets)
        else:
            raise Exception("dimension must be None or 1 for WeightedCategoricalList")

    def torch_preprocess_layer(self):
        from zenithml.torch.layers.preprocess.weighted_nhotencoder import WeightedNHotEncodingLayer

        num_buckets = self.get_num_buckets()
        if self.dimension == 1:
            return WeightedNHotEncodingLayer(num_buckets=num_buckets)
        else:
            raise Exception("dimension must be None or 1 for WeightedCategoricalList")


class Bucketized(Categorical):
    def __init__(
        self,
        input_col: str,
        default_value: Optional[float] = 0.0,
        bins: Optional[int] = None,
        bin_boundaries: Optional[List[float]] = None,
        dtype=None,
    ):
        super().__init__(input_col=input_col, dtype=dtype)
        self._default_value = default_value
        self.bins = bins
        self.bin_boundaries = bin_boundaries
        self.dimension = 1

    def base_transformer(self) -> BaseNVTTransformer:
        return BucketizeBaseNVTTransformer(
            input_col=self.input_col,
            col_type=NVTColType.CAT,
            default_value=self.default_value,
            bin_boundaries=self.bin_boundaries or self.analyze_data.get("bin_boundaries"),
        )

    def bq_analyzer(self) -> BQAnalyzer:
        assert self.bins, "bins must be set when using bq_analyzer."
        return BucketizedBQAnalyzer(input_col=self.input_col, feature=self.name, bins=self.bins)

    def nvt_analyzer(self, dask_working_dir=None, **kwargs) -> Optional[NVTAnalyzer]:
        assert self.bin_boundaries, "bin_boundaries must be set when using nvt_analyzer."
        return None

    def pandas_analyzer(self, **kwargs) -> PandasAnalyzer:
        return NumericalPandasAnalyzer(
            input_col=self.input_col,
            default_value=self.default_value,
            feature=self.name,
            bins=self.bins if self.bin_boundaries is not None else None,
        )

    def load(self, analyze_data):
        self._analyze_data = {
            "bin_boundaries": analyze_data[f"{self.name}_bins"],
        }

    def get_num_buckets(self):
        return len(self.analyze_data.get("bin_boundaries")) + 1
