from typing import Optional, Callable, Union, List, Dict, Any

from zenithml.data.base import ZenithBaseDataset
from zenithml.preprocess import Preprocessor
from zenithml.utils import rich_logging


class DatasetRegistry:
    """The factory class for managing datasets"""

    registry: Dict[str, Any] = {}
    """ Internal registry for available datasets """

    @classmethod
    def register(cls, name: str) -> Callable:
        logger = rich_logging()

        def inner_wrapper(wrapped_class: DatasetRegistry) -> DatasetRegistry:
            if name in cls.registry:
                logger.warning("Dataset %s already exists. Will replace it", name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_dataset(cls, name: str, **kwargs) -> ZenithBaseDataset:
        logger = rich_logging()

        if name not in cls.registry:
            logger.warning("Dataset %s does not exist in the registry", name)
            raise Exception("Dataset %s does not exist in the registry", name)

        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor


def load_dataset(
    name: str,
    data_loc: Optional[str] = None,
    transformed_data_loc: Optional[str] = None,
    working_dir: Optional[str] = None,
    features: Optional[Preprocessor] = None,
    outcome_var: Optional[Union[List[str], str]] = None,
    **kwargs,
) -> ZenithBaseDataset:
    import logging

    logging.getLogger().setLevel(logging.INFO)

    return DatasetRegistry.create_dataset(
        name=name,
        data_loc=data_loc,
        transformed_data_loc=transformed_data_loc,
        working_dir=working_dir,
        features=features,
        outcome_var=outcome_var,
        **kwargs,
    )
