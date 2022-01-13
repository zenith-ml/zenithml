from typing import Optional, Callable

from zenithml.data import ParquetDataset
from zenithml.preprocess import Preprocessor
from zenithml.utils import rich_logging


class DatasetRegistry:
    """The factory class for managing datasets"""

    registry = {}
    """ Internal registry for available datasets """

    @classmethod
    def register(cls, name: str) -> Callable:
        logger = rich_logging()

        def inner_wrapper(wrapped_class: DatasetRegistry) -> Callable:
            if name in cls.registry:
                logger.warning("Dataset %s already exists. Will replace it", name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_dataset(cls, name: str, **kwargs) -> Optional[ParquetDataset]:
        logger = rich_logging()

        if name not in cls.registry:
            logger.warning("Dataset %s does not exist in the registry", name)
            return None

        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor


def load_dataset(
    name: str,
    data_dir: Optional[str] = None,
    working_dir: Optional[str] = None,
    features: Optional[Preprocessor] = None,
    **kwargs,
) -> ParquetDataset:

    return DatasetRegistry.create_dataset(name=name, data_dir=data_dir, working_dir=working_dir, **kwargs)
