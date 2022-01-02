import pytest
import sqlparse
import nvtabular as nvt
from nvtabular.workflow import WorkflowNode

from condorml.preprocess.analyze.nvt_analyzer import (
    NormalizeNVTAnalyzer,
    NormalizeMinMaxNVTAnalyzer,
    CategorifyNVTAnalyzer,
    NVTAnalyzer,
)


def create_ftransform_config(analyzer):
    class DummyFTransformConfig:
        def nvt_analyzer(self):
            return analyzer

        def load(self, op):
            pass

    return DummyFTransformConfig()


def test_nvt_analyzer_fit(test_df, datasets, tmp_path):
    input_group_dict = {
        "features": [
            create_ftransform_config(NormalizeNVTAnalyzer("f_ints", "features_f_ints", default_value=0.0)),
            create_ftransform_config(NormalizeMinMaxNVTAnalyzer("f_float", "features_f_float", default_value=0.0)),
            create_ftransform_config(CategorifyNVTAnalyzer("f_cat", "features_f_cat")),
        ]
    }
    analyze_data = NVTAnalyzer.fit(nvt.Dataset(test_df), input_group_dict, client=None, dask_working_dir=tmp_path)
    expected_set = {
        f"features_{f}" for f in {"f_ints_avg", "f_ints_stddev", "f_float_min", "f_float_max", "f_cat_cat"}
    }
    assert expected_set - set(analyze_data.keys()) == set()


@pytest.mark.parametrize(
    "analyzer",
    [
        NormalizeNVTAnalyzer,
        NormalizeMinMaxNVTAnalyzer,
        CategorifyNVTAnalyzer,
    ],
)
def test_nvt_analyzer_check_op(analyzer):
    analyzer = analyzer(input_col="input_c", default_value="something", feature="input_c")
    assert isinstance(analyzer.ops(dask_working_dir="dummy"), WorkflowNode)
