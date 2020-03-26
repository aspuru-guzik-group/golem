from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import pyximport
import numpy as np
pyximport.install(
        setup_args={'include_dirs': np.get_include()},
        reload_support=True)

from .golem import Golem
from .convolution import BaseDist, Normal, TruncatedNormal, FoldedNormal
from .convolution import Uniform, TruncatedUniform, BoundedUniform, Gamma
from .convolution import Poisson, DiscreteLaplace, Categorical
from .convolution import FrozenNormal, FrozenUniform, FrozenGamma
from .convolution import FrozenPoisson, FrozenDiscreteLaplace, FrozenCategorical
