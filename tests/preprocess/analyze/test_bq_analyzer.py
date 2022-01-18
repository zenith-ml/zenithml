import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import sqlparse

from zenithml.preprocess.analyze import (
    BucketizedBQAnalyzer,
    StandardScalerBQAnalyzer,
    LogScalerBQAnalyzer,
    CategoricalBQAnalyzer,
    WeightedCategoricalBQAnalyzer,
)
from zenithml.preprocess.analyze.bq_analyzer import _generate_query_str, _generate_parquet_export


def bq_analyzer_args(analyer_class):
    kwargs = {"input_col": "input_col", "feature": "feature"}
    kwargs.update(
        {
            BucketizedBQAnalyzer: {"bins": 10},
            StandardScalerBQAnalyzer: {},
            LogScalerBQAnalyzer: {"percentile": 0.1},
            CategoricalBQAnalyzer: {"top_k": 10},
            WeightedCategoricalBQAnalyzer: {"top_k": 10},
        }[analyer_class]
    )
    return kwargs


@pytest.mark.parametrize(
    "analyzer",
    [
        BucketizedBQAnalyzer,
        StandardScalerBQAnalyzer,
        LogScalerBQAnalyzer,
        CategoricalBQAnalyzer,
        WeightedCategoricalBQAnalyzer,
    ],
)
@patch("zenithml.preprocess.analyze.bq_analyzer.BQRunner")
def test_bq_analyzer_query(mock_bq_runner, analyzer):
    analyzer = analyzer(**bq_analyzer_args(analyzer))
    qtxt = _generate_query_str([analyzer], "bq_table", where_clause=None)
    sqlparse.parse(qtxt)


@pytest.mark.parametrize(
    "analyzer",
    [
        BucketizedBQAnalyzer,
        StandardScalerBQAnalyzer,
        LogScalerBQAnalyzer,
        CategoricalBQAnalyzer,
        WeightedCategoricalBQAnalyzer,
    ],
)
@patch("zenithml.preprocess.analyze.bq_analyzer.BQRunner")
def test_bq_analyzer_parquet_exports(mock_bq_runner, analyzer, tmp_path):
    mock_bq_runner().query.to_parquet.return_value = pd.DataFrame([1, 2, 3])

    analyzer_obj = analyzer(export_as_parquet=True, **bq_analyzer_args(analyzer))
    data = _generate_parquet_export(
        [analyzer_obj],
        "bq_table",
        where_clause=None,
        output_path=tmp_path,
        renew_cache=True,
    )
    if analyzer in [CategoricalBQAnalyzer, WeightedCategoricalBQAnalyzer]:

        assert data == {"feature_cat": Path(tmp_path) / "feature_cat.vocab"}
    else:
        assert data == {}
