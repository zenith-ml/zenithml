from pathlib import Path
from typing import List, Union, Optional

import numpy as np

from zenithml.gcp import BQRunner
from zenithml.nvt.utils import validate_data_path
from zenithml.preprocess import Preprocessor
from zenithml.preprocess.ftransform_configs.ftransform_config import FTransformConfig
from zenithml.utils import rich_logging


def _get_hvd_params(use_hvd, init_hvd_fn):
    if use_hvd:
        hvd, seed_fn = init_hvd_fn()
        global_size, global_rank = hvd.size(), hvd.rank()
    else:
        global_size, global_rank, seed_fn = None, None, None
    return global_rank, global_size, seed_fn


class ParquetDataset:
    def __init__(
        self,
        path: Union[str, Path, List[str], List[Path]],
        working_dir: Union[str, Path],
        preprocessor: Optional[Preprocessor] = None,
    ):
        working_dir = Path(working_dir) if isinstance(working_dir, str) else working_dir
        self._base_nvt_dataset = None
        self._dataset_path = path
        self._working_dir: Path = working_dir
        self.preprocessor: Preprocessor = preprocessor or Preprocessor()

    @property
    def base_nvt_dataset(self):
        if self._base_nvt_dataset is None:
            rich_logging().debug(f"Reading dataset from {self._dataset_path}")
            print(self._dataset_path)
            self._base_nvt_dataset = validate_data_path(self._dataset_path)
        return self._base_nvt_dataset

    @property
    def dataset_path(self) -> Union[str, Path, List[str], List[Path]]:
        return self._dataset_path

    @property
    def preprocessor_dir(self) -> Path:
        return self._working_dir / "preprocessor"

    @property
    def working_dir(self) -> Path:
        return self._working_dir

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
        self.preprocessor.add_variable_group(key, value)

    def add_outcome_variable(self, value: Union[str, List[str]]):
        """
        A convenience function to add a outcome variable.
        `add_outcome("y")` is equivalent to  `add_variable("outcome", "y")
        Args:
            value (str): A string indicating the column name of the outcome variable in the dataset.
        """
        self.preprocessor.add_outcome_variable(value)

    def load_preprocessor(self):
        if self.preprocessor_dir.exists():
            self.preprocessor.load(self.preprocessor_dir)
        else:
            raise Exception(
                f"Preprocessor does not exists at {self.preprocessor_dir}, "
                f"Make sure to call analyze_transform() before calling load(), to_tf(), to_torch()"
            )

    def analyze(self, pandas_df=None, client=None):
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
            **kwargs,
        )

    def analyze_transform(self, pandas_df=None, client=None, out_files_per_proc=20, additional_cols=None, **kwargs):
        self.analyze(pandas_df=pandas_df, client=client)
        self.transform(out_files_per_proc=out_files_per_proc, additional_cols=additional_cols, **kwargs)

    def to_torch(
        self,
        batch_size: int,
        transform_data: bool = False,
        buffer_size=0.06,
        parts_per_chunk=1,
        shuffle: bool = True,
        seed_fn=None,
        drop_last=False,
        global_size=None,
        global_rank=None,
        side_cols=None,
        map_fn=None,
    ):
        from zenithml.torch import TorchDataset

        # global_rank, global_size, seed_fn = _get_hvd_params(use_hvd, init_hvd)
        if self.preprocessor is None:
            self.load_preprocessor()
        if transform_data:
            paths_or_dataset = self.preprocessor.transform(
                data=validate_data_path(str(self.dataset_path)),
                additional_cols=side_cols,
            )
        else:
            paths_or_dataset = str(self.transformed_data_dir)
        return TorchDataset.from_preprocessor(
            paths_or_dataset=paths_or_dataset,
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            buffer_size=buffer_size,
            parts_per_chunk=parts_per_chunk,
            global_size=global_size,
            global_rank=global_rank,
            seed_fn=seed_fn,
            side_cols=side_cols,
            drop_last=drop_last,
            map_fn=map_fn,
        )

    def to_tf(
        self,
        batch_size: int,
        use_hvd: bool = False,
        buffer_size=0.06,
        parts_per_chunk=1,
        shuffle: bool = True,
    ):
        from zenithml.tf import NVTKerasDataset, init_hvd

        global_rank, global_size, seed_fn = _get_hvd_params(use_hvd, init_hvd)
        self.load_preprocessor()
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

    def analyze(self, renew_cache=False, client=None, **kwargs):
        self.preprocessor.analyze(
            bq_table=self.bq_table,
            nvt_ds=self.base_nvt_dataset,
            bq_analyzer_out_path=self._working_dir / "vocabs",
            renew_cache=renew_cache,
            client=client,
            dask_working_dir=self._working_dir,
        )
        self.preprocessor.save(self._working_dir / "preprocessor")


def load_dataset(
    name: str,
    data_dir: Optional[str] = None,
    working_dir: Optional[str] = None,
    features: Optional[Preprocessor] = None,
    **kwargs,
) -> ParquetDataset:
    from zenithml.data import public as public_datasets

    dataset_cls = public_datasets.__dict__[name.capitalize()]
    return dataset_cls(data_dir=data_dir, working_dir=working_dir, **kwargs)
