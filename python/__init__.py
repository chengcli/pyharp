from importlib.metadata import PackageNotFoundError, version

import torch

try:
    from .pyharp import *
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.pyharp":
        raise

from .disort import *
from .compile import *

try:
    __version__ = version("pyharp")
except PackageNotFoundError:
    __version__ = "0.0.0"
