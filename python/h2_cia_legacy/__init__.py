import os
import pyharp
from .h2_cia_legacy import *

folder = os.path.dirname(__file__)
pyharp.add_resource_directory(folder, prepend=False)
