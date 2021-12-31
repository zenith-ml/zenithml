import logging
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from condorml.gcp import BQRunner


def test_set_projectid():
    bq = BQRunner(project="dummy")
    assert bq.bq_client.project == "dummy"


def test_query_dry_run(caplog):
    bq = BQRunner("project")
    query = """SELECT * FROM `project.dataset.table`"""
    with caplog.at_level(logging.INFO):
        bq.query(query=query, dry_run=True, renew_cache=True)

    assert """SELECT *\nFROM `project.dataset.table`\n""" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        bq.query(query=query, dry_run=True, renew_cache=False)

    assert """SELECT *\nFROM `project.dataset.table`\n""" in caplog.text


@pytest.mark.parametrize("renew_cache", [True, False])
@patch("condorml.gcp.bq.bigquery.QueryJobConfig")
@patch("condorml.gcp.bq.bigquery.Client")
def test_query_no_cache(mock_bq_client, mock_job_config, renew_cache):
    mock_query_job = MagicMock()
    mock_query_job.results.to_dataframe.return_value = pd.DataFrame([1, 2, 3])
    mock_bq_client().query.return_value = mock_query_job

    bq = BQRunner("project")
    mock_job_config().use_legacy_sql.return_value = False
    mock_job_config().allow_large_results.return_value = True
    mock_job_config().dry_run.return_value = False
    query = """SELECT * FROM `project.dataset.table`"""

    formatted_query = "SELECT *\nFROM `project.dataset.table`"
    if renew_cache:
        bq.clear_cache()
    bq.query(query=query, renew_cache=renew_cache)

    mock_bq_client().query.assert_called_once_with(formatted_query, job_config=mock_job_config())


def test_clear_cache():
    try:
        bq = BQRunner("project")
        bq.clear_cache()
    except Exception as e:
        pytest.fail(f"Clear Cache failed with exception {e}")


@patch("condorml.gcp.bq.GCSFileSystem")
@patch("condorml.gcp.bq.bigquery.Client")
def test_to_parquet(mock_bq_client, mock_gcsfs):
    mock_gcsfs().exists.return_value = False
    bq = BQRunner()
    outpath = bq.to_parquet("project.dataset.table", "gs://dummy")
    mock_bq_client().extract_table.assert_called_once()
    assert outpath == "gs://dummy/project/dataset/table/part-*.parquet"


@patch("condorml.gcp.bq.GCSFileSystem")
@patch("condorml.gcp.bq.bigquery.Client")
def test_to_parquet_exists(mock_bq_client, mock_gcsfs, caplog):
    mock_gcsfs().exists.return_value = True
    bq = BQRunner()

    with caplog.at_level(logging.INFO):
        outpath = bq.to_parquet("project.dataset.table", "gs://dummy")
        mock_bq_client().extract_table.assert_not_called()
    assert outpath == "gs://dummy/project/dataset/table/part-*.parquet"
    assert "Destination GCS Path already exist, skipping big query export!" in caplog.text
