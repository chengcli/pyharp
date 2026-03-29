from importlib.metadata import PackageNotFoundError, version

import torch

from .pyharp import *
from .disort import *
from .rfmlib import *
from .compile import *

try:
    __version__ = version("pyharp")
except PackageNotFoundError:
    __version__ = "0.0.0"
