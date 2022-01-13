import importlib
import json
from copy import deepcopy
from os import path
from typing import Optional, Dict, Any

import tensorflow as tf
from rich import pretty


class ConfigManager:
    """A light-weight container for all configurations to run an experiment"""

    _var_dict = {}

    def __init__(
        self,
        experiment_loc: str,
        tmp_working_dir: str,
        experiment_group: str = "default",
        experiment_name: str = "default",
        train_config: Optional[dict] = None,
        infer_config: Optional[dict] = None,
        eval_config: Optional[dict] = None,
        **kwargs,
    ):
        for k, v in locals().items():
            if k != "self" and k != "kwargs":
                self._var_dict[k] = v

        for var_name, var_value in kwargs.items():
            if var_value is not None:
                self._var_dict[var_name] = var_value

    def init_locations(self):
        # Mkdirs
        # tf.io.gfile.makedirs(working_dir)
        # tf.io.gfile.makedirs(self.dask_working_dir)
        # tf.io.gfile.makedirs(self.local_data_dir)

        pass

    @property
    def train_config(self) -> Dict[str, Any]:
        return self._var_dict.get("train_config")

    @property
    def eval_config(self) -> Dict[str, Any]:
        return self._var_dict.get("eval_config")

    @property
    def tmp_working_dir(self):
        return self._var_dict["working_dir"]

    @property
    def experiment_group(self):
        return self._var_dict["experiment_group"]

    @property
    def experiment_name(self):
        return self._var_dict["experiment_name"]

    @property
    def dask_working_dir(self):
        return path.join(self.working_dir, "dask_working_dir")

    @property
    def model_dir(self):
        return path.join(self.working_dir, "model_dir")

    @property
    def gcs_datasets_dir(self):
        return path.join(self.gcs_bucket, "parquet_datasets", self.experiment_group)

    def __getattribute__(self, item):
        _var = object.__getattribute__(self, "_var_dict").get(item)
        return _var or object.__getattribute__(self, item)

    def add_config(self, name, value):
        self._var_dict[name] = value
        self.__dict__[name] = self._var_dict.get(name)

    def show(self):
        config_vars = deepcopy(self._var_dict)
        pretty.pprint(config_vars)

    def as_json(self):
        config_vars = self._var_dict
        return json.dumps(config_vars)

    def as_dict(self):
        return self._var_dict
