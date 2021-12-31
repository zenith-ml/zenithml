import logging
from contextlib import contextmanager
from time import time
from typing import Iterator

from rich.console import Console
from rich.logging import RichHandler

import condorml

console = Console()


def rich_logging():
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(markup=True, rich_tracebacks=True),
        ],
    )
    log = logging.getLogger(condorml.__name__.upper())
    return log


@contextmanager
def timer_logging(description: str) -> Iterator[None]:
    console.print(f"[BEGIN] {description}", style="magenta")
    start = time()
    yield
    elapsed_time = round(time() - start, 3)

    console.print(f"[END] {description} in {elapsed_time}s", style="green")
