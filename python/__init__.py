from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

try:
    from .pyharp import *
    _data_dir = Path(__file__).with_name("data")
    if _data_dir.exists():
        add_resource_directory(str(_data_dir), prepend=False)
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.pyharp":
        raise

from .disort import *
from .compile import *

try:
    __version__ = version("pyharp")
except PackageNotFoundError:
    __version__ = "0.0.0"
