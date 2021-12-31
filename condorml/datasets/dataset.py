from pathlib import Path
from typing import List, Union

import nvtabular as nvt

from condorml.gcp import BQRunner
from condorml.preprocess import Preprocessor
from condorml.preprocess.ftransform_configs.ftransform_config import FTransformConfig


def _get_hvd_params(use_hvd, init_hvd_fn):
    if use_hvd:
        hvd, seed_fn = init_hvd_fn()
        global_size, global_rank = hvd.size(), hvd.rank()
    else:
        global_size, global_rank, seed_fn = None, None, None
    return global_rank, global_size, seed_fn


class ParquetDataset:
    def __init__(self, path: Union[str, Path], working_dir: Union[str, Path]):
        working_dir = Path(working_dir) if isinstance(working_dir, str) else working_dir
        self._path: Path = Path(path) if isinstance(path, str) else path

        self._base_nvt_dataset = nvt.Dataset(str(self._path / "*.parquet"))
        self._working_dir: Path = working_dir
        self._preprocessor: Preprocessor = Preprocessor()

    @property
    def base_nvt_dataset(self):
        return self._base_nvt_dataset

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def preprocessor_dir(self) -> Path:
        return self._working_dir / "preprocessor"

    @property
    def transformed_data_dir(self) -> Path:
        return self._working_dir / "transformed_dataset"

    @property
    def dask_working_dir(self) -> Path:
        return self._working_dir / "dask_working_dir"

    def add_variable_group(self, key: str, value: List[FTransformConfig]):
        """

        Args:
            key (str): Name of the variable group.
            value (List[FTransformConfig]):
        """
        self._preprocessor.add_variable_group(key, value)

    def add_outcome_variable(self, value: Union[str, List[str]]):
        """
        A convenience function to add a outcome variable.
        `add_outcome("y")` is equivalent to  `add_variable("outcome", "y")
        Args:
            value (str): A string indicating the column name of the outcome variable in the dataset.
        """
        self._preprocessor.add_outcome_variable(value)

    def analyze(self, pandas_df, client=None):
        self.preprocessor.analyze(
            nvt_ds=self.base_nvt_dataset,
            pandas_df=pandas_df,
            client=client,
            dask_working_dir=self.dask_working_dir,
        )
        self.preprocessor.save(self._working_dir / "preprocessor")

    def transform(self, out_files_per_proc=20, additional_cols=None, **kwargs):
        if not str(self.transformed_data_dir).startswith("gs"):
            self.transformed_data_dir.mkdir(exist_ok=True, parents=True)
        self.preprocessor.transform(
            data=self._base_nvt_dataset,
            output_data_path=self.transformed_data_dir,
            out_files_per_proc=out_files_per_proc,
            additional_cols=additional_cols,
            **kwargs
        )

    def analyze_transform(self, pandas_df, client=None, out_files_per_proc=20, additional_cols=None, **kwargs):
        self.analyze(pandas_df=pandas_df, client=client)
        self.transform(out_files_per_proc=out_files_per_proc, additional_cols=additional_cols, **kwargs)

    def to_torch(
        self,
        batch_size: int,
        use_hvd: bool = False,
        buffer_size=0.06,
        parts_per_chunk=1,
        shuffle: bool = True,
    ):
        from condorml.torch import TorchDataset, init_hvd

        global_rank, global_size, seed_fn = _get_hvd_params(use_hvd, init_hvd)
        self.preprocessor.load(self.preprocessor_dir)
        return TorchDataset.from_preprocessor(
            paths_or_dataset=str(self.transformed_data_dir),
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            buffer_size=buffer_size,
            parts_per_chunk=parts_per_chunk,
            global_size=global_size,
            global_rank=global_rank,
            seed_fn=seed_fn,
        )

    def to_tf(
        self,
        batch_size: int,
        use_hvd: bool = False,
        buffer_size=0.06,
        parts_per_chunk=1,
        shuffle: bool = True,
    ):
        from condorml.tf import NVTKerasDataset, init_hvd

        global_rank, global_size, seed_fn = _get_hvd_params(use_hvd, init_hvd)
        self.preprocessor.load(self.preprocessor_dir)
        return NVTKerasDataset.from_preprocessor(
            paths_or_dataset=str(self.transformed_data_dir),
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            buffer_size=buffer_size,
            parts_per_chunk=parts_per_chunk,
            global_size=global_size,
            global_rank=global_rank,
            seed_fn=seed_fn,
        )


class BQDataset(ParquetDataset):
    def __init__(self, bq_table: str, gcs_datasets_dir: str, working_dir: Union[str, Path]):
        path = BQRunner().to_parquet(source_bq=bq_table, destination_gcs=gcs_datasets_dir)
        super().__init__(path=path, working_dir=working_dir)
        self.bq_table = bq_table

    def analyze(self, renew_cache=False, client=None):
        self.preprocessor.analyze(
            bq_table=self.bq_table,
            nvt_ds=self.base_nvt_dataset,
            bq_analyzer_out_path=self._working_dir / "vocabs",
            renew_cache=renew_cache,
            client=client,
            dask_working_dir=self._working_dir,
        )
        self.preprocessor.save(self._working_dir / "preprocessor")
