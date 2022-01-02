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

        def get_data(self, op):
            return analyzer.get_data(op)

    return DummyFTransformConfig()


def test_nvt_analyzer_fit(test_df, datasets, tmp_path):
    input_group_dict = {
        "features": [
            create_ftransform_config(NormalizeNVTAnalyzer("f_ints", default_value=0.0)),
            create_ftransform_config(NormalizeMinMaxNVTAnalyzer("f_float", default_value=0.0)),
            create_ftransform_config(CategorifyNVTAnalyzer("f_cat")),
        ]
    }
    analyze_data = NVTAnalyzer.fit(nvt.Dataset(test_df), input_group_dict, client=None, dask_working_dir=tmp_path)
    assert "features" in analyze_data
    assert {"f_ints_avg", "f_ints_stddev", "f_float_min", "f_float_max", "f_cat_cat"} - set(
        analyze_data["features"].keys()
    ) == set()


@pytest.mark.parametrize(
    "analyzer",
    [
        NormalizeNVTAnalyzer,
        NormalizeMinMaxNVTAnalyzer,
        CategorifyNVTAnalyzer,
    ],
)
def test_nvt_analyzer_check_op(analyzer):
    analyzer = analyzer(input_col="input_c", default_value="something")
    assert isinstance(analyzer.ops(dask_working_dir="dummy"), WorkflowNode)
