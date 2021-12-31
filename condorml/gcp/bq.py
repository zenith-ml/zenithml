import os
import tempfile
from typing import Optional

import pandas as pd
import sqlparse
from absl import logging
from gcsfs.core import GCSFileSystem
from google.cloud import bigquery
from joblib import Memory
from condorml.utils import rich_logging


def run_query(
    query: str,
    bq_client: bigquery.Client,
    dialect: str = "standard",
    allow_large_results: bool = True,
    dry_run: bool = False,
):
    job_config = bigquery.QueryJobConfig()
    job_config.use_legacy_sql = dialect == "legacy"
    job_config.allow_large_results = allow_large_results
    job_config.dry_run = dry_run
    formatted_query = sqlparse.format(query, reindent=True, keyword_case="upper")
    if dry_run:
        rich_logging().info(formatted_query)
        result = formatted_query
    else:
        query_job = bq_client.query(formatted_query, job_config=job_config)
        result = query_job.result().to_dataframe()
    return result


class BQRunner:
    def __init__(self, project: Optional[str] = None, location: str = "EU", temp_dir: Optional[str] = None):
        """
        A light-weight wrapper around BigQuery Client to perform operations such as
            query (w/ local caching), export/import parquet files.
        Args:
            project (:obj:`str`, optional): Google Project ID. If not passed then google.cloud.biquery will
                fallback to the default inferred from the environment.
            location (str): GCP Location. default=EU
            temp_dir (:obj:`str`, optional): Path to a local directory to store the cache. If None,
                tempfile.gettempdir() is used to figure out the best tmp folder for the calling user.
        """
        self.bq_client = bigquery.Client(project=project, location=location)
        self._location = location
        self._cache_memory = Memory(temp_dir or tempfile.gettempdir(), verbose=20)
        self._run_query_fn = self._cache_memory.cache(func=run_query, ignore=["bq_client", "dialect", "dry_run"])

    def query(
        self,
        query: str,
        dialect: str = "standard",
        allow_large_results: bool = True,
        dry_run: bool = False,
        renew_cache: bool = False,
    ) -> pd.DataFrame:
        """
        Runs BigQuery using the google.cloud.biquery API. Additionally, check if the results are available in the
            local cache to speedup repeat queries.
        Args:
            query (str): SQL query string to run.
            dialect (str): standard or legacy BQ dialect.
            allow_large_results (bool): if large results must be permitted
            dry_run (bool): if set to true, the formatted query is printed but not executed.
            renew_cache (bool): flag indicating if the local cache needs to be ignored.

        Returns (pd.Dataframe):
            The output from running the query as a pandas dataframe.
        """

        if renew_cache or dry_run:
            result = self._run_query_fn.call(
                query,
                bq_client=self.bq_client,
                dialect=dialect,
                allow_large_results=allow_large_results,
                dry_run=dry_run,
            )
        else:
            result = self._run_query_fn(
                query=query,
                bq_client=self.bq_client,
                dialect=dialect,
                allow_large_results=allow_large_results,
                dry_run=dry_run,
            )
        return result

    def clear_cache(self):
        """Clears the local cache directory."""

        self._cache_memory.clear(warn=False)

    def to_parquet(self, source_bq: str, destination_gcs: str):
        """
        Exports BigQuery table to Google Cloud Storage.
        Args:
            source_bq (str): BQ table in `projectid.datasetid.tableid` format.
            destination_gcs (str): Google Cloud Storage(GCS) destination path to export the table.
        Returns (str): the output GCS path where the BQ table was exported.

        """
        assert len(source_bq.split(".")) == 3, "invalid source_bq format"
        project, dataset_id, table_id = source_bq.split(".")
        destination_gcs = os.path.join(destination_gcs, project, dataset_id, table_id)
        out_path = os.path.join(destination_gcs, "part-*.parquet")

        fs = GCSFileSystem()
        if fs.exists(destination_gcs):
            print("here")
            rich_logging().info("Destination GCS Path already exist, skipping big query export!")
            return out_path

        project, dataset_id, table_id = source_bq.split(".")
        dataset_ref = bigquery.DatasetReference(project, dataset_id)
        table_ref = dataset_ref.table(table_id)
        job_config = bigquery.job.ExtractJobConfig()
        job_config.destination_format = bigquery.DestinationFormat.PARQUET
        self.bq_client.extract_table(
            table_ref,
            out_path,
            location=self._location,
            job_config=job_config,
        ).result()

        return out_path
