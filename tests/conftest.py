import pytest

from .common import create_mini_test_df


@pytest.fixture(scope="session", autouse=True)
def datasets(tmpdir_factory):
    datadir = {"dummy_df": tmpdir_factory.mktemp("dummy_df")}
    create_mini_test_df().to_parquet(str(datadir["dummy_df"].join("dataset-0.parquet")))
    return datadir


@pytest.fixture(scope="function")
def test_df():
    return create_mini_test_df()
