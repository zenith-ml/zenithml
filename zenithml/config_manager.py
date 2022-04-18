import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, Any, Union

from rich import pretty

from zenithml.utils import fs


class ConfigManager:
    """A light-weight container for configs.
    The class tries to add some convention around directory locations.
    """

    _var_dict: Dict[str, Any] = {}

    def __init__(
        self,
        experiment_loc: str,
        working_dir: str,
        experiment_group: str = "default",
        experiment_name: str = "default",
        data_loc: Optional[str] = None,
        train_config: Optional[dict] = None,
        infer_config: Optional[dict] = None,
        eval_config: Optional[dict] = None,
        overwrite: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        for k, v in locals().items():
            if k != "self" and k != "kwargs":
                self._var_dict[k] = v

        for var_name, var_value in kwargs.items():
            if var_value is not None:
                self._var_dict[var_name] = var_value

        self.init_locations(overwrite, verbose)

    def init_locations(self, overwrite: bool, verbose: bool = False):
        dirs = [self.working_dir, self.dask_working_dir, self.experiment_loc]
        if overwrite:
            # TODO: Implement rm
            raise NotImplementedError
        for d in dirs:
            fs.mkdir(d, exist_ok=True, parents=True)

    @property
    def train_config(self) -> Optional[Dict[str, Any]]:
        return self._var_dict.get("train_config")

    @property
    def eval_config(self) -> Optional[Dict[str, Any]]:
        return self._var_dict.get("eval_config")

    @property
    def working_dir(self) -> Path:
        _loc = self._var_dict["working_dir"]
        assert fs.is_local(_loc), f"working_dir must be a local path but found {_loc}"
        return Path(_loc)

    @property
    def dask_working_dir(self) -> Union[str, Path]:
        return fs.join(self.working_dir, "dask_working_dir")

    @property
    def experiment_group(self) -> str:
        return self._var_dict["experiment_group"]

    @property
    def experiment_name(self) -> str:
        return self._var_dict["experiment_name"]

    # Experiment Specific locations
    @property
    def preprocessor_loc(self) -> Union[str, Path]:
        return fs.join(self.experiment_loc, [self.experiment_group, self.experiment_name, "preprocessor"])

    @property
    def transformed_data_loc(self) -> Union[str, Path]:
        return fs.join(self.experiment_loc, [self.experiment_group, self.experiment_name, "transformed_data"])

    @property
    def model_loc(self) -> Union[str, Path]:
        return fs.join(self.experiment_loc, [self.experiment_group, self.experiment_name, "model"])

    @property
    def predictions_loc(self) -> Union[str, Path]:
        return fs.join(self.experiment_loc, [self.experiment_group, self.experiment_name, "predictions"])

    @property
    def eval_loc(self) -> Union[str, Path]:
        return fs.join(self.experiment_loc, [self.experiment_group, self.experiment_name, "eval"])

    @property
    def data_loc(self) -> Union[str, Path]:
        return self._var_dict["data_loc"]

    def __getattribute__(self, item):
        _var = object.__getattribute__(self, "_var_dict").get(item)
        if _var is not None:
            return _var
        return object.__getattribute__(self, item)

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
