from pathlib import Path
from typing import Union, List, IO

import fsspec


def _is_gcs_path(path: Union[str, Path]) -> bool:
    if isinstance(path, str):
        return path.startswith("gs://") or path.startswith("gcs://")
    else:
        return False


def is_local(path: Union[str, Path]):
    if isinstance(path, Path):
        return True
    else:
        return not (path.startswith("gs://") or path.startswith("gcs://") or path.startswith("s3://"))


def local_path(path: Union[str, Path]) -> Path:
    assert path is not None and is_local(path), f"{path} must be a local path"
    return path if isinstance(path, Path) else Path(path)


def ls(path: Union[str, Path]) -> List[Union[str, Path]]:
    if _is_gcs_path(path):
        _, _, paths = fsspec.get_fs_token_paths(path)
        return [f"gs://{f}" for f in paths]
    else:
        path = local_path(path)
        return [x for x in path.glob("**/*") if path.is_file()]


def exists(path: Union[str, Path]) -> bool:
    if _is_gcs_path(path):
        return fsspec.filesystem("gs").exists(path)
    else:
        path = local_path(path)
        return path.exists()


def open_fileptr(path: Union[str, Path], **kwargs):
    if _is_gcs_path(path):
        return fsspec.filesystem("gcs").open(path, **kwargs)
    else:
        path = local_path(path)
        return path.open(**kwargs)


def close_fileptr(path: Union[IO, fsspec.core.OpenFile]):
    if hasattr(path, "close"):
        path.close()


def mkdir(path: Union[str, Path], parents=True, exist_ok=True, **kwargs):
    if _is_gcs_path(path):
        fsspec.filesystem("gcs").mkdir(path)
    else:
        path = local_path(path)
        path.mkdir(parents=parents, exist_ok=exist_ok, **kwargs)


def join(path: Union[str, Path], join_str: Union[str, List[str]]) -> Union[str, Path]:
    join_str = join_str if isinstance(join_str, list) else [join_str]
    if _is_gcs_path(path):
        base_path: str = path.rstrip("/")  # type: ignore
        for j in join_str:
            base_path += "/" + j
        return base_path
    else:
        joined_path: Path = local_path(path)
        for j in join_str:
            joined_path = joined_path / j
        return joined_path
