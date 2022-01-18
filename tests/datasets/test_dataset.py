from pathlib import Path
from unittest.mock import patch

import pytest

from zenithml.data import BQDataset
from zenithml.data import ParquetDataset
from zenithml.preprocess import Numerical, StandardNormalizer
from zenithml.preprocess import Preprocessor


def test_parquet_dataset(test_df, datasets, tmp_path):
    test_df_path = str(datasets["dummy_df"])
    transformed_data_loc = str(Path(test_df_path) / "transformed_dataset")
    ds = ParquetDataset(
        data_loc=test_df_path,
        working_dir=tmp_path,
        transformed_data_loc=transformed_data_loc,
    )

    assert ds.base_nvt_dataset.num_rows == len(test_df)
    assert isinstance(ds.preprocessor, Preprocessor)
    assert str(ds.transformed_data_loc) == transformed_data_loc


def test_parquet_dataset_variables(datasets, tmp_path):
    test_df_path = str(datasets["dummy_df"])

    ds = ParquetDataset(test_df_path, tmp_path)
    ds.add_outcome_variable("y")
    ds.add_variable_group("features", [Numerical("dummy_col")])

    assert ds.preprocessor.outcome_variable == "y"
    assert ds.preprocessor.variable_group_keys == ["features"]


@pytest.mark.parametrize("transform", [Numerical, StandardNormalizer])
def test_parquet_dataset_analyze(transform, test_df, datasets, tmp_path):
    test_df_path = str(datasets["dummy_df"])
    preprocessor_loc = str(Path(tmp_path) / "preprocessor")

    ds = ParquetDataset(test_df_path, tmp_path)
    ds.add_outcome_variable("y")
    ds.add_variable_group("features", [transform("f_ints")])
    ds.analyze(pandas_df=test_df, preprocessor_loc=preprocessor_loc)

    load_preprocessor = Preprocessor()
    load_preprocessor.load(preprocessor_loc)
    assert load_preprocessor.outcome_variable == "y"
    assert load_preprocessor.variable_group_keys == ["features"]
    if transform == Numerical:
        assert load_preprocessor.analysis_data == {"features": {}}
    elif transform == StandardNormalizer:
        for k in ["avg", "max", "min", "stddev"]:
            assert f"features_f_ints_{k}" in load_preprocessor.analysis_data["features"]


def test_parquet_dataset_to_tf(test_df, datasets, tmp_path):
    test_df_path = str(datasets["dummy_df"])
    preprocessor_loc = str(Path(tmp_path) / "preprocessor")
    transformed_data_loc = str(Path(tmp_path) / "transformed_data")

    ds = ParquetDataset(test_df_path, tmp_path, transformed_data_loc=transformed_data_loc)
    ds.add_outcome_variable("y")
    ds.add_variable_group("features", [Numerical("f_ints")])
    ds.analyze_transform(preprocessor_loc=preprocessor_loc, pandas_df=test_df, out_files_per_proc=1)
    batch = next(ds.to_tf(batch_size=2))
    assert list(batch[0]) == ["f_ints"]
    assert len(batch[0]["f_ints"]) == 2
    assert len(batch[1]) == 2


def test_parquet_dataset_to_torch(test_df, datasets, tmp_path):
    test_df_path = str(datasets["dummy_df"])
    preprocessor_loc = str(Path(tmp_path) / "preprocessor")
    transformed_data_loc = str(Path(tmp_path) / "transformed_data")

    ds = ParquetDataset(test_df_path, tmp_path, transformed_data_loc=transformed_data_loc)
    ds.add_outcome_variable("y")
    ds.add_variable_group("features", [Numerical("f_ints")])
    ds.analyze_transform(preprocessor_loc=preprocessor_loc, pandas_df=test_df, out_files_per_proc=1)
    batch = next(iter(ds.to_torch(batch_size=2)))
    assert list(batch[0]) == ["f_ints"]
    assert len(batch[0]["f_ints"]) == 2
    assert len(batch[1]) == 2


@patch("zenithml.data.core.BQRunner")
def test_bq_dataset(mock_bq_runner, datasets, tmp_path):
    test_df_path = str(datasets["dummy_df"])

    mock_bq_runner().to_parquet.return_value = test_df_path

    ds = BQDataset(bq_table="project.dataset.table", gcs_datasets_dir="gs://dummy", working_dir=tmp_path)

    assert ds.base_nvt_dataset.num_rows == 5
