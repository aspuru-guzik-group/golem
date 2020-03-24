#cython: language_level=3

import  cython
cimport cython

import  numpy as np
cimport numpy as np

from libc.math cimport sqrt, erf, exp, floor, ceil, abs, INFINITY

from scipy.special import gammainc

import logging


# =========================
# Probability Distributions
# =========================
cdef class Delta:

    def __init__(self):
        """Delta function. This is used internally by Golem for the dimensions with no uncertainty.
        """
        pass

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        # use > instead of >= because of how the nodes are split in sklearn: the tiles include the lower
        # boundary
        if x > loc:
            return 1.
        else:
            return 0.


cdef class Normal:

    cdef readonly double std
    cdef readonly double frozen_loc

    def __init__(self, std, frozen_loc=None):
        """Gaussian distribution.

        Parameters
        ----------
        std : float
            The scale (one standard deviation) of the Gaussian distribution.
        frozen_loc : float, optional
            Whether to fix the location of the distribution. If this is defined, the location of the distribution
            (representing the uncertainty in the inputs) will not depend on the input locations. Default is None.
        """
        self.std = std

        # if frozen_loc is not defined, we assign inf
        if frozen_loc is not None:
            self.frozen_loc = frozen_loc
        else:
            self.frozen_loc = INFINITY

    cpdef double pdf(self, x, loc=0):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Normal distribution.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """

        if self.frozen_loc != INFINITY:
            loc = self.frozen_loc

        pass

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Normal distribution.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        # freeze loc if needed
        if self.frozen_loc != INFINITY:
            loc = self.frozen_loc

        # calc cdf
        return _normal_cdf(x, loc, self.std)


cdef class TruncatedNormal:

    cdef readonly double std
    cdef readonly double low_bound
    cdef readonly double high_bound
    cdef readonly double frozen_loc

    def __init__(self, std, low_bound=-INFINITY, high_bound=INFINITY, frozen_loc=None):
        """Truncated Normal distribution.

        Parameters
        ----------
        std : float
            The scale (one standard deviation) of the Gaussian distribution.
        low_bound : float, optional
            Lower bound for the distribution. Default is -inf.
        high_bound : float, optional
            Upper bound for the distribution. Default is inf.
        frozen_loc : float, optional
            Whether to fix the location of the distribution. If this is defined, the location of the distribution
            (representing the uncertainty in the inputs) will not depend on the input locations. Default is None.
        """
        self.std = std
        self.low_bound = low_bound
        self.high_bound = high_bound

        # if frozen_loc is not defined, we assign inf
        if frozen_loc is not None:
            self.frozen_loc = frozen_loc
        else:
            self.frozen_loc = INFINITY

        # perform checks
        _warn_if_no_bounds(type(self).__name__, self.low_bound, self.high_bound)

    cpdef double pdf(self, x, loc=0):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Normal distribution.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """

        if self.frozen_loc != INFINITY:
            loc = self.frozen_loc
        pass

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Normal distribution.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        # define variables and freeze loc if needed
        cdef double cdf_x
        cdef double cdf_upper_bound
        cdef double cdf_lower_bound

        if self.frozen_loc != INFINITY:
            loc = self.frozen_loc

        # calc cdf
        if x < self.low_bound:
            return 0.
        elif x > self.high_bound:
            return 1.
        else:
            cdf_x = _normal_cdf(x, loc, self.std)
            cdf_upper_bound = _normal_cdf(self.high_bound, loc, self.std)
            cdf_lower_bound = _normal_cdf(self.low_bound, loc, self.std)
            return  (cdf_x - cdf_lower_bound) / (cdf_upper_bound - cdf_lower_bound)


cdef class FoldedNormal:

    cdef readonly double std
    cdef readonly double low_bound
    cdef readonly double high_bound
    cdef readonly double frozen_loc

    def __init__(self, std, low_bound=-INFINITY, high_bound=INFINITY, frozen_loc=None):
        """Folded Normal distribution.

        Parameters
        ----------
        std : float
            The scale (one standard deviation) of the Gaussian distribution.
        low_bound : float, optional
            Lower bound for the distribution. Default is -inf.
        high_bound : float, optional
            Upper bound for the distribution. Default is inf.
        frozen_loc : float, optional
            Whether to fix the location of the distribution. If this is defined, the location of the distribution
            (representing the uncertainty in the inputs) will not depend on the input locations. Default is None.
        """
        self.std = std
        self.low_bound = low_bound
        self.high_bound = high_bound

        # if frozen_loc is not defined, we assign inf
        if frozen_loc is not None:
            self.frozen_loc = frozen_loc
        else:
            self.frozen_loc = INFINITY

        # perform checks
        _warn_if_no_bounds(type(self).__name__, self.low_bound, self.high_bound)

    cpdef double pdf(self, x, loc=0):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Normal distribution.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """

        if self.frozen_loc != INFINITY:
            loc = self.frozen_loc
        pass

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Normal distribution.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        # define variables and freeze loc if needed
        cdef double cdf
        cdef double cdf_left
        cdef double cdf_right
        cdef double x_low
        cdef double x_high
        cdef double i

        if self.frozen_loc != INFINITY:
            loc = self.frozen_loc

        # calc cdf
        if x < self.low_bound:
            return 0.
        elif x > self.high_bound:
            return 1.
        else:
            # -----------------------------------------
            # if no bounds ==> same as normal gauss_cdf
            # -----------------------------------------
            # this is just to catch the case where the user does not enter bounds
            if self.high_bound == INFINITY and self.low_bound == -INFINITY:
                return _normal_cdf(x, loc, self.std)
            # -------------------
            # if lower bound only
            # -------------------
            elif self.high_bound == INFINITY:
                # if x is infinity, return 1 (otherwise x_low=NaN)
                if x == INFINITY:
                    return 1.
                else:
                    x_low  = x - 2 * (x - self.low_bound)
                    cdf = _normal_cdf(x, loc, self.std) - _normal_cdf(x_low, loc, self.std)
                    return cdf

            # -------------------
            # if upper bound only
            # -------------------
            elif self.low_bound == -INFINITY:
                # if x is -infinity, return 0 (otherwise x_high=NaN)
                if x == -INFINITY:
                    return 0.
                else:
                    x_high = x + 2 * (self.high_bound - x)
                    cdf = 1. - (_normal_cdf(x_high, loc, self.std) - _normal_cdf(x, loc, self.std))
                    return cdf

            # -------------------------
            # if lower and upper bounds
            # -------------------------
            else:
                cdf = 0.
                i = 0.
                while True:
                    # "fold" on the left
                    x_high = x - i*(self.high_bound - self.low_bound)
                    x_low  = x - i*(self.high_bound - self.low_bound) - (2 * (x - self.low_bound))
                    cdf_left = _normal_cdf(x_high, loc, self.std) - _normal_cdf(x_low, loc, self.std)

                    # if i == 0, +/- i*domain_range is the same and we double count the same area
                    if i == 0.:
                        cdf += cdf_left
                        i += 2.
                        continue

                    # "fold" on the right
                    x_high = x + i*(self.high_bound - self.low_bound)
                    x_low  = x + i*(self.high_bound - self.low_bound) - (2 * (x - self.low_bound))
                    cdf_right = _normal_cdf(x_high, loc, self.std) - _normal_cdf(x_low, loc, self.std)

                    # add delta cdf
                    delta_cdf = cdf_right + cdf_left
                    cdf += delta_cdf

                    # break if delta less than some tolerance
                    if delta_cdf < 10e-6:
                        break

                    # fold at lower bound every 2 folds
                    i += 2.

                return cdf


cdef class Uniform:

    cdef readonly double urange
    cdef readonly (double, double) frozen_interval

    def __init__(self, urange, frozen_interval=None):
        """Uniform distribution.

        Parameters
        ----------
        range : float
            The range of the Uniform distribution.
        frozen_interval : [low, high], optional
            Whether to fix the interval of the distribution. If this is defined, the location of the distribution
            (representing the uncertainty around the inputs) will not depend on the input locations. Default is None.
        """
        self.urange = urange

        # if frozen_loc is not defined, we assign inf
        if frozen_interval is not None:
            self.frozen_interval = frozen_interval
        else:
            self.frozen_interval = (INFINITY, INFINITY)

    cpdef double pdf(self, x, loc=0):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Uniform distribution.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """

        if self.frozen_interval[0] != INFINITY:
            loc = self.frozen_interval

        pass

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Uniform distribution.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        # define variables and freeze loc if needed
        cdef double a
        cdef double b

        if self.frozen_interval[0] != INFINITY:
            a = self.frozen_interval[0]
            b = self.frozen_interval[1]
        else:
            a = loc - 0.5 * self.urange
            b = loc + 0.5 * self.urange

        # calc cdf
        if x < a:
            return 0.
        elif x > b:
            return 1.
        else:
            return (x - a) / (b - a)


cdef class TruncatedUniform:

    cdef readonly double urange
    cdef readonly double low_bound
    cdef readonly double high_bound

    def __init__(self, urange, low_bound=-INFINITY, high_bound=INFINITY):
        """Truncated uniform distribution.

        Parameters
        ----------
        range : float
            The range of the Uniform distribution.
        low_bound : float, optional
            Lower bound for the distribution. Default is -inf.
        high_bound : float, optional
            Upper bound for the distribution. Default is inf.
        """
        self.urange = urange
        self.low_bound = low_bound
        self.high_bound = high_bound

        # perform checks
        _warn_if_no_bounds(type(self).__name__, self.low_bound, self.high_bound)

    cpdef double pdf(self, x, loc=0):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Uniform distribution.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """

        pass

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Uniform distribution.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        # define variables and freeze loc if needed
        cdef double a
        cdef double b

        a = loc - 0.5 * self.urange
        b = loc + 0.5 * self.urange

        # truncate if close to bounds
        if 0.5 * self.urange > (loc - self.low_bound):
            a = self.low_bound
        elif 0.5 * self.urange > (self.high_bound - loc):
            b = self.high_bound

        # calc cdf
        if x < a:
            return 0.
        elif x > b:
            return 1.
        else:
            return (x - a) / (b - a)


cdef class BoundedUniform:

    cdef readonly double urange
    cdef readonly double low_bound
    cdef readonly double high_bound

    def __init__(self, urange, low_bound=-INFINITY, high_bound=INFINITY):
        """Bounded uniform distribution.

        Parameters
        ----------
        range : float
            The range of the Uniform distribution.
        low_bound : float, optional
            Lower bound for the distribution. Default is -inf.
        high_bound : float, optional
            Upper bound for the distribution. Default is inf.
        """
        self.urange = urange
        self.low_bound = low_bound
        self.high_bound = high_bound

        # perform checks
        _warn_if_no_bounds(type(self).__name__, self.low_bound, self.high_bound)

    cpdef double pdf(self, x, loc=0):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Uniform distribution.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """

        pass

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Uniform distribution.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        # define variables and freeze loc if needed
        cdef double a
        cdef double b

        a = loc - 0.5 * self.urange
        b = loc + 0.5 * self.urange

        # fix based on lower bound
        if 0.5 * self.urange > (loc - self.low_bound):
            a = self.low_bound
            b = self.low_bound + self.urange
        # fix based on upper bound
        elif 0.5 * self.urange > (self.high_bound - loc):
            b = self.high_bound
            a = self.high_bound - self.urange
        # "standard" uniform
        else:
            a = loc - 0.5 * self.urange
            b = loc + 0.5 * self.urange

        # calc cdf
        if x < a:
            return 0.
        elif x > b:
            return 1.
        else:
            return (x - a) / (b - a)


cdef class Gamma:

    cdef readonly double std
    cdef readonly double low_bound
    cdef readonly double high_bound
    cdef readonly double frozen_loc
    cdef double no_bounds

    def __init__(self, std, low_bound=-INFINITY, high_bound=INFINITY, frozen_loc=None):
        """Gamma distribution parametrized by its standard deviation and mode. These are used to fit the k and
        theta parameters.

        Parameters
        ----------
        std : float
            The scale (one standard deviation) of the distribution.
        low_bound : float, optional
            Lower bound for the distribution. Default is zero.
        high_bound : float, optional
            Upper bound for the distribution. Default is inf.
        frozen_loc : float, optional
            Whether to fix the location (mode) of the distribution. If this is defined, the location of the distribution
            (representing the uncertainty in the inputs) will not depend on the input locations. Default is None.
        """
        self.std = std
        self.low_bound = low_bound
        self.high_bound = high_bound

        # if frozen_loc is not defined, we assign inf
        if frozen_loc is not None:
            self.frozen_loc = frozen_loc
        else:
            self.frozen_loc = INFINITY

        # perform checks
        no_bounds = _warn_if_no_bounds(type(self).__name__, self.low_bound, self.high_bound)
        if no_bounds == 1.:
            self.low_bound = 0.
        _check_single_bound(type(self).__name__, self.low_bound, self.high_bound)

    cpdef double pdf(self, x, loc=0):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mode) of the Gamma distribution.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """

        if self.frozen_loc != INFINITY:
            loc = self.frozen_loc
        pass

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mode) of the Gamma distribution.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        # define variables and freeze loc if needed
        cdef double k
        cdef double theta

        if self.frozen_loc != INFINITY:
            loc = self.frozen_loc

        # calc cdf
        if x < self.low_bound:
            return 0.
        if x > self.high_bound:
            return 1.

        # if we have lower bound
        if self.high_bound == INFINITY:
            x = x - self.low_bound
            loc = loc - self.low_bound

            var = self.std**2.
            theta = sqrt(var + (loc**2.)/4.) - loc/2.
            k = loc/theta + 1.

            return gammainc(k, x/theta)

        # if we have an upper bound
        elif self.low_bound == -INFINITY:
            x = self.high_bound - x
            loc = self.high_bound - loc

            var = self.std**2.
            theta = sqrt(var + (loc**2.)/4.) - loc/2.
            k = loc/theta + 1.

            return 1. - gammainc(k, x/theta)


cdef class DiscreteLaplace:

    cdef readonly double scale
    cdef readonly double frozen_loc

    def __init__(self, scale, frozen_loc=None):
        """Discrete Laplace distribution.

        Parameters
        ----------
        scale : float
            The scale of the discrete Laplace distribution, which controls its variance.
        frozen_loc : float, optional
            Whether to fix the location of the distribution. If this is defined, the location of the distribution
            (representing the uncertainty in the inputs) will not depend on the input locations. Default is None.
        """
        self.scale = scale

        # if frozen_loc is not defined, we assign inf
        if frozen_loc is not None:
            self.frozen_loc = frozen_loc
        else:
            self.frozen_loc = INFINITY

    cpdef double pdf(self, x, loc=0):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Normal distribution.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """
        cdef double p

        if self.frozen_loc != INFINITY:
            loc = self.frozen_loc

        p = np.exp(-1 / self.scale)
        return (1 - p) / (1 + p) * (p ** abs(x-loc))

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Normal distribution.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        # define variables and freeze loc if needed
        cdef double p
        if self.frozen_loc != INFINITY:
            loc = self.frozen_loc

        # calc cdf
        p = exp(-1. / self.scale)
        if x < loc:
            return p ** (-floor(x - loc)) / (1. + p)
        else:
            return 1. - (p ** (floor(x - loc) + 1.) / (1. + p))


cdef class Categorical:

    cdef readonly list categories
    cdef readonly double unc
    cdef readonly int num_categories
    cdef readonly double frozen_loc
    cdef readonly double [:] frozen_prob

    def __init__(self, categories, unc, frozen_prob=None):
        """Simple categorical distribution.

        Parameters
        ----------
        categories : list
            List of categories
        unc : float
            The uncertainty in the categorical choice, i.e. probability that the queried category is not the category
            being evaluated.
        frozen_prob : list, optional
            Whether to fix the probabilities of the categorical distribution. If this is defined, the distribution
            will not depend on the input location. Default is None.
        """
        self.categories = categories
        self.num_categories = len(categories)
        self.unc = unc

        # if frozen_loc is not defined, we assign inf
        if frozen_prob is not None:
            self.frozen_prob = frozen_prob
        else:
            self.frozen_prob = np.array([INFINITY] * self.num_categories)

    cpdef double pdf(self, x, loc=0):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Normal distribution.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """

        pass

    @cython.cdivision(True)
    cpdef double cdf(self, double x, int loc):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mean) of the Normal distribution.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        # define variables and freeze loc if needed
        cdef int upper_cat
        cdef double cdf
        cdef int cat_idx
        #if self.frozen_loc != INFINITY:
        #    loc = self.frozen_loc

        # put bounds on x
        if x == -INFINITY:
            x = -0.5  # encoding starts from 0
        if x == INFINITY:
            x = self.num_categories - 0.5  # last category encoded as num_categories-1

        # the category in with highest integer encoding
        upper_cat = <int>ceil(x)

        # calc cdf
        cdf = 0.
        for cat_idx in range(upper_cat):
            if cat_idx == loc:
                cdf +=  1. - self.unc
            else:
                cdf += self.unc / (self.num_categories - 1.)
        return cdf



@cython.cdivision(True)
cdef double _normal_cdf(double x, double loc, double scale):
    """Helper function to calculate Normal CDF.
    """
    cdef double arg
    arg = (x - loc) / (1.4142135623730951 * scale)
    if arg > 3.:
        return 1.
    elif arg < -3.:
        return 0.
    else:
        return (1. + erf( arg )) * 0.5


def _check_single_bound(dist, l_bound, h_bound):
    if not np.isinf(l_bound) and not np.isinf(h_bound):
        raise ValueError(f'{dist} allows to define either a lower or an upper bound, not both')


def _warn_if_no_bounds(dist, l_bound, h_bound):
    if np.isinf(l_bound) and np.isinf(h_bound):
        logging.warning(f'No bounds provided to the bounded distribution {dist}. Verify your input.')
        return 1.
    return 0.

