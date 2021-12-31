# flake8: noqa
import pytest
from keras.layers import Normalization

from condorml import preprocess as pp
from condorml.preprocess.analyze import (
    CategoricalPandasAnalyzer,
    NumericalPandasAnalyzer,
    CategoricalListPandasAnalyzer,
)
from condorml.preprocess.analyze.nvt_analyzer import NumericalNVTAnalyzer, CategoricalNVTAnalyzer
from condorml.tf import layers as pp_layers


def test_numerical_input():
    input_col = pp.Numerical(input_col="f_ints")
    input_col.set_prefix("test")
    input_col.load({"test_f_ints_max": 1.0, "test_f_ints_min": 0.0})

    assert input_col.name == f"test_f_ints"
    assert isinstance(input_col.analyze_data, dict)
    assert input_col.analyze_data == {}
    assert isinstance(input_col.preprocess_layer(backend=pp.Backend.TF), pp_layers.NumericalLayer)
    assert isinstance(input_col.nvt_analyzer(), NumericalNVTAnalyzer)
    assert input_col.pandas_analyzer() is None


def test_std_normalizer_input():
    input_col = pp.StandardNormalizer(input_col="f_ints")
    input_col.set_prefix("test")
    input_col.load({"test_f_ints_avg": 1.0, "test_f_ints_stddev": 2.0})

    assert input_col.name == f"test_f_ints"
    assert isinstance(input_col.analyze_data, dict)
    assert input_col.analyze_data["mean_val"] == 1.0
    assert input_col.analyze_data["variance_val"] == 4.0
    assert isinstance(input_col.preprocess_layer(backend=pp.Backend.TF), Normalization)
    assert isinstance(input_col.nvt_analyzer(), NumericalNVTAnalyzer)
    assert isinstance(input_col.pandas_analyzer(), NumericalPandasAnalyzer)


def test_minmax_normalizer_input():
    input_col = pp.MinMaxNormalizer(input_col="f_ints")
    input_col.set_prefix("test")
    input_col.load({"test_f_ints_min": 1.0, "test_f_ints_max": 2.0})

    assert input_col.name == f"test_f_ints"
    assert isinstance(input_col.analyze_data, dict)
    assert input_col.analyze_data["min_val"] == 1.0
    assert input_col.analyze_data["max_val"] == 2.0
    assert isinstance(input_col.preprocess_layer(backend=pp.Backend.TF), pp_layers.MinMaxNormalizeLayer)
    assert isinstance(input_col.nvt_analyzer(), NumericalNVTAnalyzer)
    assert isinstance(input_col.pandas_analyzer(), NumericalPandasAnalyzer)


def test_log_normalizer_input():
    input_col = pp.LogNormalizer(input_col="f_ints")
    input_col.set_prefix("test")
    input_col.load({"test_f_ints_percentile": 2.0})

    assert input_col.name == f"test_f_ints"
    assert isinstance(input_col.analyze_data, dict)
    assert input_col.analyze_data["percentile"] == 2.0
    assert isinstance(input_col.preprocess_layer(backend=pp.Backend.TF), pp_layers.LogNormalizeLayer)
    assert isinstance(input_col.nvt_analyzer(), NumericalNVTAnalyzer)
    assert isinstance(input_col.pandas_analyzer(), NumericalPandasAnalyzer)


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("vocab", [None, ["a", "b", "c"]])
@pytest.mark.parametrize("weights", [None, True])
def test_categorical_input(dimension, vocab, weights):
    weights = None if not weights else [10, 20]
    if weights and dimension == 1:
        with pytest.raises(AssertionError):
            pp.Categorical(input_col="f_cat", dimension=dimension, vocab=vocab, weights=weights)
    else:
        input_col = pp.Categorical(input_col="f_cat", dimension=dimension, vocab=vocab, weights=weights)
        input_col.set_prefix("test")
        input_col.load({"test_f_cat_cat": ["a", "b", "c"]})

        assert input_col.name == f"test_f_cat"
        assert isinstance(input_col.analyze_data, dict)
        assert input_col.analyze_data["vocab"] == ["a", "b", "c"]

        if dimension == 1:
            assert isinstance(input_col.preprocess_layer(backend=pp.Backend.TF), pp_layers.NHotEncodingLayer)
        else:
            assert isinstance(input_col.preprocess_layer(backend=pp.Backend.TF), pp_layers.EmbeddingLayer)

        assert isinstance(input_col.nvt_analyzer(), CategoricalNVTAnalyzer)
        assert isinstance(input_col.pandas_analyzer(), CategoricalPandasAnalyzer)


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("vocab", [None, ["a", "b", "c"]])
@pytest.mark.parametrize("weights", [None, True])
def test_categoricallist_input(dimension, vocab, weights):
    weights = None if not weights else [10, 20]
    if weights and dimension == 1:
        with pytest.raises(AssertionError):
            pp.Categorical(input_col="f_cat", dimension=dimension, vocab=vocab, weights=weights)
    else:
        input_col = pp.CategoricalList(input_col="f_cat", dimension=dimension, vocab=vocab, weights=weights)
        assert isinstance(input_col.pandas_analyzer(), CategoricalListPandasAnalyzer)


def test_cosine_similarity_input():
    input_col = pp.CosineSimilarity(input_col1="vec1", input_col2="vec2", dimension=2)

    assert input_col.name == f"vec1_cosine_vec2"
    assert isinstance(input_col.analyze_data, dict)
    assert input_col.pandas_analyzer() is None
    assert len(input_col.nvt_analyzer()) == 2
    for nvt_analyzer in input_col.nvt_analyzer():
        assert isinstance(nvt_analyzer, NumericalNVTAnalyzer)
    assert isinstance(input_col.preprocess_layer(backend=pp.Backend.TF), pp_layers.CosineSimilarityLayer)
