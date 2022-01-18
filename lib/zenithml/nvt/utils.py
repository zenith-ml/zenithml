from pathlib import Path
from typing import Union, List

import nvtabular as nvt


def validate_data_path(paths_or_dataset: Union[str, Path, List[str], nvt.Dataset]):
    paths_or_dataset = str(paths_or_dataset) if isinstance(paths_or_dataset, Path) else paths_or_dataset

    if isinstance(paths_or_dataset, str):
        paths_or_dataset += "/*.parquet" if not paths_or_dataset.endswith(".parquet") else ""
        dataset = nvt.Dataset(paths_or_dataset)
    elif isinstance(paths_or_dataset, list):
        print(paths_or_dataset)
        dataset = nvt.Dataset(paths_or_dataset)
    elif isinstance(paths_or_dataset, nvt.Dataset):
        dataset = paths_or_dataset
    else:
        raise Exception(
            f"paths_or_dataset must be of type str, List[str], nvt.Dataset " f"but found {type(paths_or_dataset)}"
        )
    return dataset
