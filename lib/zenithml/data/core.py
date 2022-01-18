import typing
from pathlib import Path
from typing import List, Union, Optional

from zenithml.gcp import BQRunner
from zenithml.nvt.utils import validate_data_path
from zenithml.preprocess import Preprocessor
from zenithml.preprocess.ftransform_configs.ftransform_config import FTransformConfig
from zenithml.utils import rich_logging, fs


class ParquetDataset:
    def __init__(
        self,
        data_loc: Union[str, Path, List[str], List[Path]],
        working_dir: Union[str, Path],
        preprocessor: Optional[Preprocessor] = None,
        transformed_data_loc: Optional[Union[str, Path, List[str], List[Path]]] = None,
    ):
        self._base_nvt_dataset = None
        self._transformed_data_loc = transformed_data_loc
        self._dataset_path = data_loc
        self._working_dir: Path = fs.local_path(working_dir)
        self.preprocessor: Preprocessor = preprocessor or Preprocessor()

    @property
    def base_nvt_dataset(self):
        if self._base_nvt_dataset is None:
            rich_logging().debug(f"Reading dataset from {self._dataset_path}")
            self._base_nvt_dataset = validate_data_path(self._dataset_path)
        return self._base_nvt_dataset

    @property
    def dataset_loc(self) -> Union[str, Path, List[str], List[Path]]:
        return self._dataset_path

    @property
    def transformed_data_loc(self) -> Optional[Union[str, Path, List[str], List[Path]]]:
        return self._transformed_data_loc

    @property
    def dask_working_loc(self) -> Path:
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

    def load_preprocessor(self, path: Union[str, Path]):
        if fs.exists(path):
            self.preprocessor.load(path)
        else:
            raise Exception(
                f"Preprocessor does not exists at {path}, "
                f"Make sure to call analyze_transform() before calling load(), to_tf(), to_torch()"
            )

    def analyze(self, preprocessor_loc: str, pandas_df=None, bq_table=None, bq_analyzer_out_path=None, client=None):
        self.preprocessor.analyze(
            bq_table=bq_table,
            bq_analyzer_out_path=bq_analyzer_out_path,
            nvt_ds=self.base_nvt_dataset,
            pandas_df=pandas_df,
            client=client,
            dask_working_dir=self.dask_working_loc,
        )
        self.preprocessor.save(preprocessor_loc)

    def transform(self, out_files_per_proc=20, additional_cols=None, **kwargs):
        assert self.transformed_data_loc, "transformed_data_loc must be set before calling transform()"
        fs.mkdir(self.transformed_data_loc)
        self.preprocessor.transform(
            data=self._base_nvt_dataset,
            output_data_path=self.transformed_data_loc,
            out_files_per_proc=out_files_per_proc,
            additional_cols=additional_cols,
            **kwargs,
        )

    def analyze_transform(
        self,
        preprocessor_loc: str,
        pandas_df=None,
        client=None,
        out_files_per_proc=20,
        additional_cols=None,
        **kwargs,
    ):
        self.analyze(preprocessor_loc=preprocessor_loc, pandas_df=pandas_df, client=client)
        self.transform(out_files_per_proc=out_files_per_proc, additional_cols=additional_cols, **kwargs)

    def to_torch(
        self,
        batch_size: int,
        preprocessor_loc: Optional[str] = None,
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

        paths_or_dataset = self._get_transformed_dataset_path(preprocessor_loc, side_cols, transform_data)
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
        preprocessor_loc: Optional[str] = None,
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
        from zenithml.tf import NVTKerasDataset

        paths_or_dataset = self._get_transformed_dataset_path(preprocessor_loc, side_cols, transform_data)
        ds = NVTKerasDataset.from_preprocessor(
            paths_or_dataset=paths_or_dataset,
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            buffer_size=buffer_size,
            parts_per_chunk=parts_per_chunk,
            global_size=global_size,
            global_rank=global_rank,
            seed_fn=seed_fn,
            drop_last=drop_last,
        )
        if map_fn:
            return ds.map(map_fn)
        else:
            return ds

    def _get_transformed_dataset_path(self, preprocessor_loc, side_cols, transform_data):
        if self.preprocessor is None:
            assert preprocessor_loc, "preprocessor_loc must be set when preprocessor is None"
            self.load_preprocessor(preprocessor_loc)
        if transform_data:
            paths_or_dataset = self.preprocessor.transform(
                data=validate_data_path(str(self.dataset_loc)),
                additional_cols=side_cols,
            )
        else:
            paths_or_dataset = str(self.transformed_data_loc)
        return paths_or_dataset


class BQDataset(ParquetDataset):
    def __init__(
        self,
        bq_table: str,
        gcs_datasets_dir: str,
        working_dir: Union[str, Path],
        transformed_data_loc: Optional[Union[str, Path, List[str], List[Path]]] = None,
    ):
        path = BQRunner().to_parquet(source_bq=bq_table, destination_gcs=gcs_datasets_dir)
        super().__init__(data_loc=path, working_dir=working_dir, transformed_data_loc=transformed_data_loc)
        self.bq_table = bq_table

    @typing.no_type_check
    def analyze(
        self,
        preprocessor_loc: str,
        renew_cache=False,
        client=None,
        bq_analyzer_out_path=None,
        **kwargs,
    ):
        bq_analyzer_out_path = (
            fs.join(preprocessor_loc, "vocabs") if bq_analyzer_out_path is None else bq_analyzer_out_path
        )
        super().analyze(
            preprocessor_loc=preprocessor_loc,
            bq_table=self.bq_table,
            bq_analyzer_out_path=bq_analyzer_out_path,
            client=client,
        )
