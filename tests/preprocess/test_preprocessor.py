import shutil

import pandas as pd
import tempfile

import condorml
from condorml import preprocess as pp


def test_variable_groups( test_df ):
    preprocessor = pp.Preprocessor()
    preprocessor.add_variable_group(
        "features",
        [
            pp.StandardNormalizer(input_col="f_ints"),
            pp.Numerical(input_col="f_float"),
            # pp.BucketizedLayer(bins=2, input_col="f_buk"),
            pp.Categorical(top_k=100, input_col="f_cat"),
        ],
    )

    preprocessor.analyze(nvt_ds=None, pandas_df=pd.DataFrame(test_df))
    assert {"features_f_cat", "features_f_float", "features_f_ints"} == set(
        [i.name for i in preprocessor.get_variable_group("features")]
    )


def test_add_outcome_variable():
    preprocessor = pp.Preprocessor()
    preprocessor.add_outcome_variable("t")
    assert preprocessor.outcome_variable == "t"


def test_save_load( test_df ):
    input_layers = [
        pp.StandardNormalizer(input_col="f_ints"),
        pp.Numerical(input_col="f_float"),
        # pp.BucketizedLayer(bins=2, input_col="f_buk"),
        pp.Categorical(top_k=100, input_col="f_cat"),
    ]
    preprocessor = pp.Preprocessor()
    preprocessor.add_variable_group(
        "features",
        input_layers,
    )
    preprocessor.add_variable("treatment", "t")

    preprocessor.analyze(nvt_ds=None, pandas_df=pd.DataFrame(test_df))
    analyze_data = preprocessor.analysis_data
    path = tempfile.mkdtemp(prefix=condorml.__name__)
    path = str(path)

    preprocessor.save(path)
    loaded_preprocessor = pp.Preprocessor()
    loaded_preprocessor.load(path)
    assert loaded_preprocessor.analysis_data == analyze_data
    # self.assertEqual(len(loaded_vars.get("features")), len(layers))
    # self.assertEqual(loaded_vars.get("treatment"), "t")
    shutil.rmtree(path)
