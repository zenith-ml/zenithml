import abc
from pathlib import Path
from typing import Union, Optional


class ZenithBaseDataset(abc.ABC):
    def __init__(
        self,
        name: str,
        base_data_loc: Union[Path, str],
        working_dir: Union[Path, str],
        transformed_data_loc: Optional[Union[Path, str]] = None,
    ):
        self._name = name
        self._base_data_loc = base_data_loc
        self._transformed_data_loc = transformed_data_loc
        self.working_dir = working_dir

    @property
    def name(self) -> str:
        return self._name

    @property
    def data_loc(self) -> Union[Path, str]:
        return self._base_data_loc

    @property
    def transformed_data_loc(self) -> Optional[Union[Path, str]]:
        return self._transformed_data_loc

    @abc.abstractmethod
    def info(self) -> str:
        """Return a short description about the dataset."""
        pass

    @abc.abstractmethod
    def download(self, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def train(self):
        pass

    @property
    @abc.abstractmethod
    def validation(self):
        pass

    @property
    @abc.abstractmethod
    def test(self):
        pass


class LocalZenithBaseDataset(ZenithBaseDataset, abc.ABC):
    @property
    def data_loc(self) -> Path:
        return Path(self._base_data_loc) / self.name
