import abc
from typing import Union, List, Dict, Any, Optional

from zenithml.preprocess.analyze import BQAnalyzer
from zenithml.preprocess.analyze import NVTAnalyzer
from zenithml.preprocess.analyze import PandasAnalyzer
from zenithml.preprocess.base_transformer import BaseNVTTransformer
from zenithml.preprocess.constants import Backend


class FTransformConfig(abc.ABC):
    def __init__(
        self,
        input_col: Union[List[str], str],
        default_value: Optional[Any],
        dtype,
    ):
        self._name = None
        self._prefix: Optional[str] = None
        self._analyze_data: Dict[str, Any] = {}
        self._input_col = input_col
        self._dtype = dtype
        self._default_value = default_value

    @property
    def name(self) -> str:
        return f"{self._prefix}_{self.input_col}"

    @property
    def default_value(self) -> Optional[Any]:
        return self._default_value

    @property
    def dtype(self):
        return self._dtype

    @property
    def input_col(self) -> Union[str, List[str]]:
        return self._input_col

    @property
    def analyze_data(self) -> Dict[str, Any]:
        return self._analyze_data

    @analyze_data.setter
    def analyze_data(self, data: Dict[str, Any]):
        self._analyze_data = data

    def set_prefix(self, prefix: str):
        self._prefix = prefix

    def load(self, analyze_data):
        pass

    @abc.abstractmethod
    def base_transformer(self) -> Union[BaseNVTTransformer, List[BaseNVTTransformer]]:
        raise NotImplementedError

    @abc.abstractmethod
    def pandas_analyzer(self, **kwargs) -> Optional[PandasAnalyzer]:
        raise NotImplementedError

    @abc.abstractmethod
    def bq_analyzer(self) -> Optional[BQAnalyzer]:
        raise NotImplementedError

    @abc.abstractmethod
    def nvt_analyzer(self) -> Optional[NVTAnalyzer]:
        raise NotImplementedError

    @abc.abstractmethod
    def torch_preprocess_layer(self):
        raise NotImplementedError

    @abc.abstractmethod
    def tf_preprocess_layer(self):
        raise NotImplementedError

    def preprocess_layer(self, backend: Backend):
        if backend == Backend.Torch:
            return self.torch_preprocess_layer()
        elif backend == Backend.TF:
            return self.tf_preprocess_layer()
        else:
            raise Exception(f"Invalid Backend {backend}")
