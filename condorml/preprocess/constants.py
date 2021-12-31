from enum import Enum

_OUTCOME_VAR = "outcome"


class AnalyzerType(Enum):
    PandasAnalyzer = 0
    BQAnalyzer = 1
    NVTPreprocessAnalyzer = 2
    NVTAnalyzer = 3


class NVTColType(Enum):
    CONT = 0
    CAT = 1
    SPARSE_AS_DENSE = 2
    SPARSE = 3

    def __eq__(self, other):
        return self.value == other.value


class Backend(Enum):
    Torch = 0
    TF = 1

    def __eq__(self, other):
        return self.value == other.value
