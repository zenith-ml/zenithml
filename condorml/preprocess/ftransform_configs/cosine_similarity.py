from typing import List, Optional

from condorml.preprocess.analyze import (
    PandasAnalyzer,
    BQAnalyzer,
    NVTAnalyzer,
)
from condorml.preprocess.base_transformer import (
    NumericalBaseNVTTransformer,
    NVTColType,
    BaseNVTTransformer,
)
from condorml.preprocess.ftransform_configs.ftransform_config import FTransformConfig


class CosineSimilarity(FTransformConfig):
    def __init__(
        self,
        input_col1: str,
        input_col2: str,
        default_value: float = 0.0,
        dimension: int = 1,
        dtype=None,
    ):
        super().__init__(input_col=[input_col1, input_col2], default_value=default_value, dtype=dtype)
        self.dimension = dimension

    @property
    def name(self):
        return f"{self.input_col[0]}_cosine_{self.input_col[1]}"

    def base_transformer(self) -> List[BaseNVTTransformer]:
        return [
            NumericalBaseNVTTransformer(
                input_col=col,
                col_type=NVTColType.CONT,
                default_value=self.default_value,
                is_list=self.dimension > 1,
            )
            for col in self.input_col
        ]

    def pandas_analyzer(self, **kwargs) -> Optional[PandasAnalyzer]:
        return None

    def nvt_analyzer(self, **kwargs) -> Optional[NVTAnalyzer]:
        return None

    def bq_analyzer(self) -> Optional[BQAnalyzer]:
        return None

    def tf_preprocess_layer(self):
        from condorml.tf.layers.preprocess.cosine_similarity import CosineSimilarityLayer

        return CosineSimilarityLayer(
            var1_name=self.input_col[0], var2_name=self.input_col[1], dimension=self.dimension
        )

    def torch_preprocess_layer(self):
        from condorml.torch.layers.preprocess.cosine_similarity import CosineSimilarityLayer

        return CosineSimilarityLayer(dimension=self.dimension)
