import numpy as np
import pytest

from condorml import preprocess as pp
from condorml.preprocess.analyze import (
    NumericalPandasAnalyzer,
    CategoricalPandasAnalyzer,
    CategoricalListPandasAnalyzer,
    PandasAnalyzer,
)


@pytest.mark.parametrize("percentile", [None, 0.05])
@pytest.mark.parametrize("bins", [None, 2])
def test_numerical_analyzer( test_df, bins, percentile ):
    input_col = "f_buk" if bins is not None else "f_ints"
    feature = f"feature_{input_col}"
    analyzer = NumericalPandasAnalyzer(
        input_col=input_col,
        default_value=0.00,
        feature=feature,
        bins=bins,
        percentile=percentile,
    )
    analyzer_data = analyzer.analyze(test_df)
    assert test_df[input_col].max(), pytest.approx(analyzer_data.get(f"{feature}_max"), 0.001)
    assert test_df[input_col].min(), pytest.approx(analyzer_data.get(f"{feature}_min"), 0.001)
    assert test_df[input_col].mean(), pytest.approx(analyzer_data.get(f"{feature}_avg"), 0.001)
    assert test_df[input_col].std(), pytest.approx(analyzer_data.get(f"{feature}_std"), 0.001)
    assert analyzer.feature == feature
    if bins is not None:
        assert sorted(analyzer_data.get(f"{feature}_bins")) == sorted([100.0, 200.0, 300.0])
    if percentile is not None:
        assert analyzer_data.get(f"{feature}_percentile") == np.percentile(test_df[input_col], percentile)


@pytest.mark.parametrize("top_k", [None, 2])
def test_categorical_analyzer( test_df, top_k ):
    input_col = "f_cat"
    feature = f"feature_{input_col}"
    analyzer = CategoricalPandasAnalyzer(input_col=input_col, feature=feature, default_value="", top_k=top_k)
    analyzer_data = analyzer.analyze(test_df)
    if top_k:
        assert set(analyzer_data.get(f"{feature}_cat")) == {"A", "B"}
    else:
        assert set(analyzer_data.get(f"{feature}_cat")) == {"A", "B", "C"}


@pytest.mark.parametrize("top_k", [None, 2])
def test_catagorical_list_analyzer( test_df, top_k ):
    input_col = "f2_cat"  # this is a list column
    feature = f"feature_{input_col}"
    analyzer = CategoricalListPandasAnalyzer(input_col=input_col, feature=feature, default_value="", top_k=top_k)
    analyzer_data = analyzer.analyze(test_df)
    if top_k:
        assert set(analyzer_data.get(f"{feature}_cat")) == {"A", "B"}
    else:
        assert set(analyzer_data.get(f"{feature}_cat")) == {"A", "B", "C"}


def test_pandas_analyzer_fit( test_df ):
    feature_configs = [
        pp.Numerical(input_col="f_bool"),
        pp.StandardNormalizer(input_col="f_float"),
        pp.MinMaxNormalizer(input_col="f_ints"),
        pp.Categorical(input_col="f_cat"),
        pp.CategoricalList(input_col="f2_cat"),
        pp.CosineSimilarity(input_col1="f_vec1", input_col2="f_vec2", dimension=2),
    ]
    _ = [f.set_prefix("test") for f in feature_configs]
    analyze_data = PandasAnalyzer.fit(feature_configs, test_df)

    assert len(analyze_data.keys()) == 10
    for feature in feature_configs:
        if not isinstance(feature, pp.Numerical) and not isinstance(feature, pp.CosineSimilarity):
            assert feature.analyze_data != {}
