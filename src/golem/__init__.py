from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import pyximport
import numpy as np
pyximport.install(
        setup_args={'include_dirs': np.get_include()},
        reload_support=True)

from .golem import Golem
from .extensions import BaseDist, Normal, TruncatedNormal, FoldedNormal
from .extensions import Uniform, TruncatedUniform, BoundedUniform, Gamma
from .extensions import Poisson, DiscreteLaplace, Categorical
from .extensions import FrozenNormal, FrozenUniform, FrozenGamma
from .extensions import FrozenPoisson, FrozenDiscreteLaplace, FrozenCategorical
