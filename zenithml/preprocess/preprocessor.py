import logging
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple

import nvtabular as nvt
import pandas as pd
from cloudpickle import pickle

from zenithml.nvt.workflow import CustomNVTWorkflow
from zenithml.preprocess.analyze import BQAnalyzer, NVTAnalyzer
from zenithml.preprocess.analyze import PandasAnalyzer
from zenithml.preprocess.base_transformer import BaseNVTTransformer
from zenithml.preprocess.constants import Backend
from zenithml.preprocess.constants import NVTColType
from zenithml.preprocess.constants import _OUTCOME_VAR
from zenithml.preprocess.ftransform_configs.ftransform_config import FTransformConfig
from zenithml.utils import fs


class Preprocessor:
    def __init__(self):
        """
        Preprocessor holds the configs related to the variables in the dataset.
        It is a convenience class that groups variables and specifies the
        transformations to perform for each feature.

        Note: We use variable and features interchangeably in the context of Preprocessor class.
        """
        self._var_group_dict: Dict[str, List[FTransformConfig]] = {}
        self._var_dict: Dict = {}
        self.analysis_data: Dict = {}
        self.nvt_workflow: Optional[CustomNVTWorkflow] = None
        self.nvt_cols = None

    @property
    def outcome_variable(self):
        """Returns the outcome variable values."""
        return self._var_dict.get(_OUTCOME_VAR)

    @property
    def variable_group_keys(self) -> List[str]:
        """Returns a list of variable group keys."""
        return list(self._var_group_dict.keys())

    def add_variable_group(self, key: str, value: List[FTransformConfig]):
        """

        Args:
            key (str): Name of the variable group.
            value (List[FTransformConfig]):
        """
        assert key not in self._var_dict, "duplicate key"
        for v in value:
            v.set_prefix(key)
        self._var_group_dict[key] = value
        return self

    def add_variable(self, key: str, value: Union[str, List[str]]):
        """
        Adds a variable such as outcome, treatment to the preprocessor. This allows us to work with abstract
        variable names when developing or evaluating models.
        Args:
            key (str): Name of the variable (such as outcome, treatment, etc.)
            value (Union[str, List[str]]): A string indicating the column name of the variable in the dataset.
        """
        assert key not in self._var_group_dict, "duplicate key"
        self._var_dict[key] = value

    def add_outcome_variable(self, value: Union[str, List[str]]):
        """
        A convenience function to add a outcome variable.
        `add_outcome("y")` is equivalent to  `add_variable("outcome", "y")
        Args:
            value (str): A string indicating the column name of the outcome variable in the dataset.
        """
        assert _OUTCOME_VAR not in self._var_group_dict, "duplicate key"
        self._var_dict[_OUTCOME_VAR] = value

    def get_variable_group(self, name):
        """
        Returns a list of FTransformConfig obj for the given variable group.
        Args:
            name (str): Key of the variable group or variable.
        """
        if name in self._var_group_dict:
            return self._var_group_dict.get(name)

    def get_nvt_cols(self):
        cont_cols = []
        cat_cols = []

        sparse_max = {}
        for group_name, vals in self._var_group_dict.items():
            for _feature in vals:
                for analyzer in BaseNVTTransformer.get_analyzers(_feature):
                    if analyzer.col_type == NVTColType.CONT:
                        cont_cols.append(analyzer.input_col)
                    elif analyzer.col_type == NVTColType.CAT:
                        cat_cols.append(analyzer.input_col)
                    elif analyzer.col_type == NVTColType.SPARSE_AS_DENSE:
                        sparse_max[analyzer.input_col] = analyzer.kwargs.get("seq_length")

        return cont_cols, cat_cols, list(sparse_max.keys()), sparse_max

    def get_preprocess_layers(self, backend: Backend = Backend.Torch):
        """
        Args:
            backend (Backend): Specifies the base framework (Allowed: Backend.TF or Backend.Torch)
        Returns: Framework specific preprocessing layers

        """
        pp_layers: Dict[str, List[Tuple[str, FTransformConfig]]] = {}
        for group in self.variable_group_keys:
            pp_layers[group] = []
            for input_config in self.get_variable_group(group):
                pp_layers[group].append((input_config.input_col, input_config.preprocess_layer(backend)))
        return pp_layers

    def analyze(
        self,
        nvt_ds: nvt.Dataset,
        bq_table: Optional[str] = None,
        pandas_df: Optional[pd.DataFrame] = None,
        where_clause: Optional[str] = None,
        renew_cache: bool = False,
        bq_analyzer_out_path: Optional[Path] = None,
        client=None,
        dask_working_dir: Optional[Union[str, Path]] = None,
    ):

        if bq_table is None and pandas_df is None:
            self.analysis_data = NVTAnalyzer.fit(
                nvt_ds=nvt_ds,
                input_group_dict=self._var_group_dict,
                client=client,
                dask_working_dir=dask_working_dir,
            )
        else:
            for group_name, group_input_values in self._var_group_dict.items():
                if pandas_df is not None:
                    logging.debug("data is pandas DataFrame. Using AnalyzerType.PandasAnalyzer")
                    self.analysis_data[group_name] = PandasAnalyzer.fit(group_input_values, df=pandas_df)

                elif bq_table is not None:
                    logging.debug("data is str. Using AnalyzerType.BQAnalyzer")
                    self.analysis_data[group_name], _ = BQAnalyzer.fit(
                        group_input_values,
                        bq_table=bq_table,
                        where_clause=where_clause,
                        renew_cache=renew_cache,
                        output_path=bq_analyzer_out_path,
                    )

        # This check is mostly to make testing easier but in practice nvt_ds must always be set.
        if nvt_ds:
            self.nvt_workflow = BaseNVTTransformer.fit(
                nvt_ds=nvt_ds,
                input_group_dict=self._var_group_dict,
                var_dict=self._var_dict,
                client=client,
                dask_working_dir=dask_working_dir,
            )

    def transform(self, data=None, output_data_path=None, out_files_per_proc=20, additional_cols=None, **kwargs):
        """

        Args:
            data ():
            output_data_path ():
            out_files_per_proc ():
            additional_cols ():

        Returns:

        """
        data = nvt.Dataset(data) if isinstance(data, str) else data
        if output_data_path:
            self.nvt_workflow.transform(data, additional_cols=additional_cols).to_parquet(
                output_data_path, out_files_per_proc=out_files_per_proc, **kwargs
            )
        else:
            return self.nvt_workflow.transform(data, additional_cols=additional_cols)

    def save(self, path: Union[str, Path]):
        """
        Save the preprocessor assets to file.
        Args:
            path (str): Path to a local directory to store the preprocess assets.
        """
        fs.mkdir(path)
        data = self.analysis_data, self.get_nvt_cols(), self._var_dict, self._var_group_dict

        if self.nvt_workflow:
            nvt_workflow_path = fs.join(path, "nvt_workflow")
            self.nvt_workflow.save(str(nvt_workflow_path))
            persist_obj = data, str(nvt_workflow_path)
        else:
            persist_obj = data, ""

        _file_prt = fs.open_fileptr(fs.join(path, "preprocessor.pkl"), mode="wb")
        pickle.dump(persist_obj, _file_prt)
        fs.close_fileptr(_file_prt)

    def load(self, path: Union[str, Path]):
        """
        Load the preprocessor assets from file.
        Args:
            path (str): Path to a local directory containing preprocess assets.
        """
        base_path = fs.join(path, "preprocessor.pkl")
        assert fs.exists(base_path), f"{base_path} does not exist"

        _file_prt = fs.open_fileptr(base_path, mode="rb")
        _data, nvt_workflow_path = pickle.load(_file_prt)
        fs.close_fileptr(_file_prt)

        self.analysis_data, nvt_cols, self._var_dict, self._var_group_dict = _data
        for group_name, vals in self._var_group_dict.items():
            for _feature in vals:
                _feature.load(self.analysis_data[group_name])
        if nvt_workflow_path:
            self.nvt_workflow = nvt.Workflow.load(nvt_workflow_path)
