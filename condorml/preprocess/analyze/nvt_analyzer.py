import abc
from pathlib import Path
from typing import Optional, Union, List

import nvtabular as nvt
import pandas as pd
from nvtabular.ops import FillMissing, Categorify, Normalize, NormalizeMinMax

from condorml.preprocess.base_transformer import _merge_nvt_ops


class NVTAnalyzer(abc.ABC):
    def __init__(
        self, input_col: Union[str, List[str]], feature: str, default_value: Optional[Union[str, float, int]] = None
    ):
        self.input_col = input_col
        self.default_value = default_value
        self.feature = feature

    @abc.abstractmethod
    def ops(self, **kwargs):
        pass

    @staticmethod
    def fit(nvt_ds, input_group_dict, client, dask_working_dir):

        ops = []
        analyze_data = {}
        for group_name, vals in input_group_dict.items():
            for _feature in vals:
                ops.append(_feature.nvt_analyzer().ops(dask_working_dir=dask_working_dir))

        nvt_ops = _merge_nvt_ops(ops)
        nvt_workflow = nvt.Workflow(nvt_ops, client=client)
        nvt_workflow.fit(nvt_ds)

        # TODO: This can be inefficient when number of features are large
        for parent_node in nvt_workflow.output_node.parents_with_dependencies:
            for group_name, vals in input_group_dict.items():
                for _feature in vals:
                    analyze_data.update(_feature.nvt_analyzer().get_data(parent_node.op))

        for group_name, vals in input_group_dict.items():
            for _feature in vals:
                _feature.load(analyze_data)
        return analyze_data


class NormalizeNVTAnalyzer(NVTAnalyzer):
    def ops(self, **kwargs):
        assert self.default_value is not None, "default_value must be set"
        return [self.input_col] >> FillMissing(fill_val=self.default_value) >> Normalize()

    def get_data(self, op: Normalize):
        data = {}
        if isinstance(op, Normalize):
            for var, val in op.means.items():
                data[f"{self.feature}_avg"] = val
            for var, val in op.stds.items():
                data[f"{self.feature}_stddev"] = val
        return data


class NormalizeMinMaxNVTAnalyzer(NVTAnalyzer):
    def ops(self, **kwargs):
        assert self.default_value is not None, "default_value must be set"
        return [self.input_col] >> FillMissing(fill_val=self.default_value) >> NormalizeMinMax()

    def get_data(self, op: NormalizeMinMax):
        data = {}
        if isinstance(op, NormalizeMinMax):
            for var, val in op.mins.items():
                data[f"{self.feature}_min"] = val
            for var, val in op.maxs.items():
                data[f"{self.feature}_max"] = val
        return data


class CategorifyNVTAnalyzer(NVTAnalyzer):
    def ops(self, dask_working_dir: Optional[Union[str, Path]] = None, **kwargs):
        assert dask_working_dir, "dask_working_dir must be set for CategorifyNVTAnalyzer"
        dask_working_dir = Path(dask_working_dir) if isinstance(dask_working_dir, str) else dask_working_dir
        return [self.input_col] >> Categorify(out_path=str(dask_working_dir / "Categorify"))

    def get_data(self, op: Categorify):
        data = {}
        if isinstance(op, Categorify):
            for var, val in op.categories.items():
                data[f"{self.feature}_cat"] = list(pd.read_parquet(val)[var].values)

        return data
