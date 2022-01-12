import abc
from pathlib import Path


class BaseDatasetMixin(abc.ABC):
    def __init__(self, name: str, base_data_dir: str, working_dir: str):
        self._name = name
        self._base_data_dir = base_data_dir
        self.working_dir = working_dir

    @property
    def name(self) -> str:
        return self._name

    @property
    def data_dir(self) -> Path:
        return Path(self._base_data_dir) / self.name

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
