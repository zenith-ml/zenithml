import abc
from pathlib import Path
from typing import Optional, Any, List, Union

import numpy as np
import nvtabular as nvt
import pandas as pd
from nvtabular.ops import FillMissing, Bucketize, LambdaOp

from zenithml.nvt.ops.vocab_categorify import VocabCategorify
from zenithml.nvt.workflow import CustomNVTWorkflow
from zenithml.preprocess.constants import NVTColType


def _cast_col(col, gdf):
    if col.dtype==bool:
        return col.astype(np.float32, copy=False)
    else:
        return col


def _merge_nvt_ops(nvt_ops):
    ops = nvt_ops[0]

    if len(nvt_ops) == 1:
        return ops

    for op in nvt_ops[1:]:
        ops += op
    return ops


class BaseNVTTransformer(abc.ABC):
    def __init__(
        self,
        input_col: Union[str, List[str]],
        col_type: NVTColType,
        default_value: Optional[Any],
        **kwargs,
    ):
        self.input_col = input_col
        self.col_type = col_type
        self.default_value = default_value
        self.kwargs = kwargs

    @abc.abstractmethod
    def ops(self, **kwargs) -> nvt.ops.Operator:
        raise NotImplementedError

    @staticmethod
    def fit(nvt_ds, input_group_dict, var_dict, client, dask_working_dir):

        ops = []
        for group_name, vals in input_group_dict.items():
            for _feature in vals:
                for analyzer in BaseNVTTransformer.get_analyzers(_feature):
                    ops.append(analyzer.ops(dask_working_dir=dask_working_dir))

        outcome_var = var_dict.get("outcome", [])
        outcome_var = outcome_var if isinstance(outcome_var, list) else [outcome_var]
        nvt_ops = _merge_nvt_ops(ops) + outcome_var

        nvt_workflow = CustomNVTWorkflow(nvt_ops, client=client)
        nvt_workflow.fit(nvt_ds)
        return nvt_workflow

    @staticmethod
    def get_analyzers(feature):
        analyzers = feature.base_transformer()
        if not isinstance(analyzers, list):
            analyzers = [analyzers]
        else:
            analyzers = analyzers
        return analyzers


class NumericalBaseNVTTransformer(BaseNVTTransformer):
    def ops(self, **kwargs):
        if self.kwargs.get("is_list", False):
            op = [self.input_col] >> FillMissing(fill_val=self.default_value)
        else:
            op = [self.input_col] >> FillMissing(fill_val=self.default_value) >> LambdaOp(_cast_col)
        return op


class BucketizeBaseNVTTransformer(BaseNVTTransformer):
    def ops(self, **kwargs) -> nvt.ops.Operator:
        bin_boundaries = self.kwargs.get("bin_boundaries")
        op = (
            [self.input_col]
            >> FillMissing(fill_val=self.default_value)
            >> Bucketize(boundaries={self.input_col: bin_boundaries})
        )

        return op


class CategoricalBaseNVTTransformer(BaseNVTTransformer):
    def ops(self, dask_working_dir: Optional[Union[str, Path]] = None, **kwargs) -> nvt.ops.Operator:
        assert dask_working_dir, "dask_working_dir must be set for CategoricalNVTPreprocessAnalyzer"
        assert "vocab" in self.kwargs, "vocab must be set for CategoricalNVTPreprocessAnalyzer"

        dask_working_dir = Path(dask_working_dir) if isinstance(dask_working_dir, str) else dask_working_dir
        vocab = self.kwargs.get("vocab")
        cat_vocabs = {}
        if isinstance(vocab, list):
            cat_vocabs[self.input_col] = pd.Series(vocab)
        else:
            cat_vocabs[self.input_col] = vocab
        op = [self.input_col] >> VocabCategorify(
            vocabs=cat_vocabs, out_path=str(dask_working_dir / "preprocessor" / "categorify")
        )
        return op
