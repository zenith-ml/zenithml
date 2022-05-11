import os

import nvtabular as nvt
import pytest

from zenithml import preprocess as pp
from zenithml import tf as zmtf
from zenithml import torch as zmtorch
from zenithml.preprocess.constants import Backend


def test_preprocessor_end2end(test_df, datasets, tmp_path):
    dummy_df_path = datasets["dummy_df"]
    working_dir = tmp_path / "working_dir"
    working_dir.mkdir()
    preprocessor = pp.Preprocessor().add_variable_group(
        "features",
        [
            pp.Numerical(input_col="f_bools"),
            pp.StandardNormalizer(input_col="f_float"),
            pp.MinMaxNormalizer(input_col="f_ints"),
            pp.Categorical(input_col="f_cat"),
            pp.CategoricalList(input_col="f2_cat"),
            pp.Numerical(input_col="f_vec1", dimension=2),
        ],
    )
    preprocessor.add_outcome_variable("y")
    nvt_ds = nvt.Dataset(os.path.join(str(dummy_df_path), "*.parquet"))
    preprocessor.analyze(
        pandas_df=test_df,
        nvt_ds=nvt_ds,
        dask_working_dir=str(working_dir),
    )
    ds = zmtf.NVTKerasDataset.from_preprocessor(
        paths_or_dataset=preprocessor.transform(data=nvt_ds),
        batch_size=5,
        preprocessor=preprocessor,
        drop_last=True,
        shuffle=False,
    )
    pp_layers = preprocessor.get_preprocess_layers(backend=Backend.TF)
    batch_x, batch_y = next(ds)
    dense_batch = zmtf.concat_layers(batch_x, dict(pp_layers["features"]))
    assert dense_batch.shape == (5, 13)


@pytest.mark.parametrize("backend", [Backend.TF, Backend.Torch])
def test_preprocessor_end2end_nvtanalyze(test_df, datasets, tmp_path, backend):
    dummy_df_path = datasets["dummy_df"]
    working_dir = tmp_path / "working_dir"
    working_dir.mkdir()
    preprocessor = pp.Preprocessor().add_variable_group(
        "features",
        [
            pp.Numerical(input_col="f_bools"),
            pp.StandardNormalizer(input_col="f_float"),
            pp.MinMaxNormalizer(input_col="f_ints"),
            pp.Categorical(input_col="f_cat"),
            pp.WeightedCategoricalList(input_col="f2_cat", weight_col="f2_cat_weight"),
            pp.Numerical(input_col="f_vec1", dimension=2),
        ],
    )
    preprocessor.add_outcome_variable("y")
    nvt_ds = nvt.Dataset(os.path.join(str(dummy_df_path), "*.parquet"))
    preprocessor.analyze(
        nvt_ds=nvt_ds,
        dask_working_dir=str(working_dir),
    )

    DatasetClass = zmtf.NVTKerasDataset if backend == Backend.TF else zmtorch.TorchDataset
    ds = DatasetClass.from_preprocessor(
        paths_or_dataset=preprocessor.transform(data=nvt_ds),
        batch_size=5,
        preprocessor=preprocessor,
        drop_last=True,
        shuffle=False,
    )
    batch_x, batch_y = next(iter(ds))
    if backend == Backend.TF:
        pp_layers = zmtf.get_preprocess_layers_and_dims(preprocessor.get_preprocess_layers(backend=backend))
        dense_batch = zmtf.concat_layers(batch_x, dict(pp_layers["features"]))
    else:
        pp_layers, _ = zmtorch.get_preprocess_layers_and_dims(preprocessor.get_preprocess_layers(backend=backend))
        dense_batch = zmtorch.concat_layers(batch_x, dict(pp_layers["features"]))

    assert dense_batch.shape == (5, 15)
