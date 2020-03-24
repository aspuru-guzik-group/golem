from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import pyximport
import numpy as np
pyximport.install(
        setup_args={'include_dirs': np.get_include()},
        reload_support=True)

from .golem import Golem
from . import dists
