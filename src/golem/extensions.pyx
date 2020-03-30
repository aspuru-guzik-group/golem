# cython: language_level=3

import  cython 
cimport cython

import  numpy as np 
cimport numpy as np 

from libc.math cimport sqrt, erf, exp, floor, abs, INFINITY
from scipy.special import gammainc, pdtr, xlogy, gammaln

import logging
import time

# ==============
# Main Functions
# ==============

@cython.boundscheck(False)
cpdef get_bboxes(double [:, :] X, int [:, :] node_indexes, double [:] value, long [:] leave_id, long [:] feature,
                      double [:] threshold):

    start = time.time()

    cdef int num_dim, num_tile, num_sample, tile_id, node_id, num_node, n

    cdef int num_tiles = np.unique(leave_id).shape[0]  # number of terminal leaves
    cdef int num_samples = np.shape(X)[0]  # number of samples
    cdef int num_dims = np.shape(X)[1]  # number of features
    cdef int tree_depth = np.shape(node_indexes)[1]  # max number of nodes to reach a leaf

    # to store the y_pred value of each tile/leaf
    cdef double [:] preds = np.empty(num_tiles)
    # to store the lower/upper bounds of each tile/leaf
    cdef double [:, :, :] bounds = np.empty(shape=(num_tiles, num_dims, 2))

    # initialise bounds with -inf and +inf, since later we will store bounds only if tighter than previous ones
    for num_tile in range(num_tiles):
        for num_dim in range(num_dims):
            bounds[num_tile, num_dim, 0] = -INFINITY
            bounds[num_tile, num_dim, 1] = +INFINITY

    # -----------------------------------------------
    # Iterate through the paths leading to all leaves
    # -----------------------------------------------
    tile_id = -1  # this is to keep an index for the tiles
    for num_sample in range(num_samples):
        # if we have duplicate paths (due to multiple samples ending up in in the same leaf)
        # skip them, otherwise we will be counting some tiles multiple times
        if _path_already_visited(num_sample, node_indexes, tree_depth) == 1:
            continue
        tile_id = tile_id + 1

        # -------------------------------------------------------
        # extract decisions made at each node leading to the leaf
        # -------------------------------------------------------
        for num_node in range(tree_depth):
            node_id = node_indexes[num_sample, num_node]

            # we assigned -1 as dummy nodes to pad the arrays to the same length, so if < 0 skip dummy node
            # also, we know that after a -1 node we only have other -1 nodes ==> break
            if node_id < 0:
                break

            # if it is a terminal node, no decision is made: store the y_pred value of this node in preds
            if leave_id[num_sample] == node_id:
                preds[tile_id] = value[node_id]
                continue

            # check if the feature being evaluated is above/below node decision threshold
            # note that feature[node_id] = the feature/dimension used for splitting the node
            if X[num_sample, feature[node_id]] <= threshold[node_id]:  # upper threshold
                # if upper threshold is lower than the previously stored one
                if threshold[node_id] < bounds[tile_id, feature[node_id], 1]:
                    bounds[tile_id, feature[node_id], 1] = threshold[node_id]
            else:  # lower threshold
                # if lower threshold is higher than the previously stored one
                if threshold[node_id] > bounds[tile_id, feature[node_id], 0]:
                    bounds[tile_id, feature[node_id], 0] = threshold[node_id]

    # check that the number of tiles found in node_indexes is the expected one
    assert tile_id == num_tiles-1
    end = time.time()
    logging.info('Tree parsed in %.2f %s' % parse_time(start, end))
    return np.asarray(bounds), np.asarray(preds)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef convolute(double [:, :] X, BaseDist [:] dists, double [:] preds, double [:, :, :] bounds):
    start = time.time()

    cdef int num_dim, num_tile, num_sample

    cdef int num_tiles = np.shape(preds)[0]

    cdef BaseDist dist
    cdef double low, high
    cdef double low_cat, high_cat
    cdef double scale, num_cats, num_cats_in_tile

    cdef double xi
    cdef double joint_prob

    cdef int num_samples = np.shape(X)[0]  # number of samples
    cdef int num_dims = np.shape(X)[1]  # number of features
    cdef double [:] newy = np.empty(num_samples)
    cdef double [:] newy_std = np.empty(num_samples)

    cdef double yi_reweighted
    cdef double yi_reweighted_squared

    cdef double yi_cache

    # ------------------------
    # Iterate over all samples
    # ------------------------
    for num_sample in range(num_samples):

        yi_reweighted         = 0.
        yi_reweighted_squared = 0.

        # ----------------------
        # iterate over all tiles
        # ----------------------
        for num_tile in range(num_tiles):

            joint_prob = 1.  # joint probability of the tile

            # ---------------------------
            # iterate over all dimensions
            # ---------------------------
            # Note you have to do this, you cannot iterate over uncertain dimensions only.
            # This because for dims with no uncertainty join_prob needs to be multiplied by 0 or 1 depending
            # whether the sample is in the tile or not. And the only way to do this is to check the tile bounds
            # in the certain dimension.
            for num_dim in range(num_dims):

                dist = <BaseDist>dists[num_dim]  # distribution object
                xi = X[num_sample, num_dim]  # location of input

                # get boundaries of the tile
                low  = bounds[num_tile, num_dim, 0]
                high = bounds[num_tile, num_dim, 1]

                # multiply joint probability by probability in num_dim
                joint_prob *= dist.cdf(high, xi) - dist.cdf(low, xi)

            # do the sum already within the loop
            yi_cache               = joint_prob * preds[num_tile]
            yi_reweighted         += yi_cache
            yi_reweighted_squared += yi_cache * preds[num_tile]

        # store robust y value for the kth sample
        newy[num_sample] = yi_reweighted
        newy_std[num_sample] = sqrt(yi_reweighted_squared - yi_reweighted**2)

    end = time.time()
    logging.info('Convolution performed in %.2f %s' % parse_time(start, end))
    return np.asarray(newy), np.asarray(newy_std)


# ==================================================================
# Probability Distributions that depend on the input/sample location
# ==================================================================
cdef class BaseDist(object):
    cpdef double pdf(self, x, loc):
        pass
    cpdef double cdf(self, double x, double loc):
        pass


cdef class Delta(BaseDist):

    def __init__(self):
        """Delta function. This is used internally by Golem for the dimensions with no uncertainty.
        """
        pass

    cpdef double pdf(self, x, loc):
        pass

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        # use > instead of >= because of how the nodes are split in sklearn: the tiles include the lower
        # boundary
        if x > loc:
            return 1.
        else:
            return 0.


cdef class Normal(BaseDist):

    cdef readonly double std

    def __init__(self, std):
        """Normal distribution.

        Parameters
        ----------
        std : float
            The scale (one standard deviation) of the Normal distribution.
        """
        self.std = std

    cpdef double pdf(self, x, loc):
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
        return _normal_pdf(x, loc, self.std)

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
        return _normal_cdf(x, loc, self.std)


cdef class TruncatedNormal(BaseDist):

    cdef readonly double std
    cdef readonly double low_bound
    cdef readonly double high_bound

    def __init__(self, std, low_bound=-INFINITY, high_bound=INFINITY):
        """Truncated Normal distribution.

        Parameters
        ----------
        std : float
            The scale (one standard deviation) of the Normal distribution.
        low_bound : float, optional
            Lower bound for the distribution. Default is -inf.
        high_bound : float, optional
            Upper bound for the distribution. Default is inf.
        """
        self.std = std
        self.low_bound = low_bound
        self.high_bound = high_bound

        # perform checks
        _warn_if_no_bounds(type(self).__name__, self.low_bound, self.high_bound)

    cpdef double pdf(self, x, loc):
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
        cdef double Z
        if x < self.low_bound:
            return 0.
        elif x > self.high_bound:
            return 0.
        else:
            Z = _normal_cdf(self.high_bound, loc, self.std) - _normal_cdf(self.low_bound, loc, self.std)
            return _normal_pdf(x, loc, self.std) / Z

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


cdef class FoldedNormal(BaseDist):

    cdef readonly double std
    cdef readonly double low_bound
    cdef readonly double high_bound

    def __init__(self, std, low_bound=-INFINITY, high_bound=INFINITY):
        """Folded Normal distribution.

        Parameters
        ----------
        std : float
            The scale (one standard deviation) of the Normal distribution.
        low_bound : float, optional
            Lower bound for the distribution. Default is -inf.
        high_bound : float, optional
            Upper bound for the distribution. Default is inf.
        """
        self.std = std
        self.low_bound = low_bound
        self.high_bound = high_bound

        # perform checks
        _warn_if_no_bounds(type(self).__name__, self.low_bound, self.high_bound)

    cpdef double pdf(self, x, loc):
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
        # define variables and freeze loc if needed
        cdef double pdf
        cdef double pdf_left
        cdef double pdf_right
        cdef double x_low
        cdef double x_high
        cdef double i

        # calc pdf
        if x < self.low_bound:
            return 0.
        elif x > self.high_bound:
            return 0.
        else:
            # -----------------------------------
            # if no bounds ==> same as normal pdf
            # -----------------------------------
            # this is just to catch the case where the user does not enter bounds
            if self.high_bound == INFINITY and self.low_bound == -INFINITY:
                return _normal_pdf(x, loc, self.std)
            # -------------------
            # if lower bound only
            # -------------------
            elif self.high_bound == INFINITY:
                x_low  = x - 2 * (x - self.low_bound)
                return _normal_pdf(x, loc, self.std) + _normal_pdf(x_low, loc, self.std)

            # -------------------
            # if upper bound only
            # -------------------
            elif self.low_bound == -INFINITY:
                x_high = x + 2 * (self.high_bound - x)
                return _normal_pdf(x, loc, self.std) + _normal_pdf(x_high, loc, self.std)

            # -------------------------
            # if lower and upper bounds
            # -------------------------
            else:
                x_low  = x - 2 * (x - self.low_bound)
                x_high = x + 2 * (self.high_bound - x)
                pdf = _normal_pdf(x, loc, self.std) + _normal_pdf(x_low, loc, self.std) + _normal_pdf(x_high, loc, self.std)
                i = 2.
                while True:
                    # pdf points on the left
                    x_high = x - i*(self.high_bound - self.low_bound)
                    x_low  = x - i*(self.high_bound - self.low_bound) - (2 * (x - self.low_bound))
                    pdf_left = _normal_pdf(x_low, loc, self.std) + _normal_pdf(x_high, loc, self.std)

                    # pdf points on the right
                    x_high = x + i*(self.high_bound - self.low_bound) + (2 * (self.high_bound - x))
                    x_low  = x + i*(self.high_bound - self.low_bound)
                    pdf_right = _normal_pdf(x_low, loc, self.std) + _normal_pdf(x_high, loc, self.std)

                    delta_pdf = pdf_left + pdf_right
                    pdf += delta_pdf
                    # break if delta less than some tolerance
                    if delta_pdf < 10e-6:
                        break
                    i += 2.

                return pdf

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


cdef class Uniform(BaseDist):

    cdef readonly double urange

    def __init__(self, urange):
        """Uniform distribution.

        Parameters
        ----------
        urange : float
            The range of the Uniform distribution.
        """
        self.urange = urange

    cpdef double pdf(self, x, loc):
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
        cdef double a
        cdef double b

        a = loc - 0.5 * self.urange
        b = loc + 0.5 * self.urange

        if x < a:
            return 0.
        elif x > b:
            return 0.
        else:
            return 1. / (b - a)

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

        # calc cdf
        if x < a:
            return 0.
        elif x > b:
            return 1.
        else:
            return (x - a) / (b - a)


cdef class TruncatedUniform(BaseDist):

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

    cpdef double pdf(self, x, loc):
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
        cdef double a
        cdef double b

        a = loc - 0.5 * self.urange
        b = loc + 0.5 * self.urange

        # truncate if close to bounds
        if 0.5 * self.urange > (loc - self.low_bound):
            a = self.low_bound
        elif 0.5 * self.urange > (self.high_bound - loc):
            b = self.high_bound

        # a has to be < b
        if a == b:
            logging.error('uniform distribution with zero range encountered')

        if x < a:
            return 0.
        elif x > b:
            return 0.
        else:
            return 1. / (b - a)

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
            # NOTE: if b=a, then we have a zero division
            return (x - a) / (b - a)


cdef class BoundedUniform(BaseDist):

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

    cpdef double pdf(self, x, loc):
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
            return 0.
        else:
            return 1 / (b - a)

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


cdef class Gamma(BaseDist):

    cdef readonly double std
    cdef readonly double low_bound
    cdef readonly double high_bound
    cdef double no_bounds

    def __init__(self, std, low_bound=-INFINITY, high_bound=INFINITY):
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
        """
        self.std = std
        self.low_bound = low_bound
        self.high_bound = high_bound

        # perform checks
        no_bounds = _warn_if_no_bounds(type(self).__name__, self.low_bound, self.high_bound)
        if no_bounds == 1.:
            self.low_bound = 0.
        _check_single_bound(type(self).__name__, self.low_bound, self.high_bound)

    cpdef double pdf(self, x, loc):
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
        cdef double logpdf
        cdef double var

        if x < self.low_bound:
            return 0.
        if x > self.high_bound:
            return 0.

        # if we have lower bound
        if self.high_bound == INFINITY:
            x = x - self.low_bound
            loc = loc - self.low_bound
            var = self.std**2.
            theta = sqrt(var + (loc**2.)/4.) - loc/2.
            k = loc/theta + 1.

            logpdf = xlogy(k - 1., x) - x/theta - gammaln(k) - xlogy(k, theta)
            return exp(logpdf)

        # if we have an upper bound
        elif self.low_bound == -INFINITY:
            x = self.high_bound - x
            loc = self.high_bound - loc
            var = self.std**2.
            theta = sqrt(var + (loc**2.)/4.) - loc/2.
            k = loc/theta + 1.

            logpdf = xlogy(k - 1., x) - x/theta - gammaln(k) - xlogy(k, theta)
            return exp(logpdf)

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


cdef class Poisson(BaseDist):

    cdef double shift
    cdef int low_bound

    def __init__(self, shift=0.5, low_bound=0):
        """Poisson distribution. The lambda parameter will be fitted based on the location of the input sample.

        Parameters
        ----------
        shift : float (0,1], optional
            Shift parameter between 0 and 1. The rate parameter :math:`\lambda` will be determined by the location of the input sample plus this
            value. If :math:`x_k` is the input location and :math:`\delta` is the value of this argument,
            then :math:`\lambda = x_k + \delta`. Having ``shift != 0`` ensures the distribution has a unique mode.
        low_bound : float, optional
            Lower bound for the distribution. Default is zero.
        """
        self.shift = shift
        self.low_bound = low_bound

    cpdef double pdf(self, x, loc):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mode) of the Poisson distribution.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """
        cdef double l
        cdef int arg

        l = loc + self.shift - self.low_bound
        if l <= 0:
            logging.error('lambda <= 0 encountered in Poisson distribution')

        if x < self.low_bound:
            return 0.
        else:
            arg = <int>floor(x - self.low_bound)
            return (l**(x-self.low_bound) * np.exp(-l)) / np.math.factorial(arg)

    @cython.cdivision(True)
    cpdef double cdf(self, double x, double loc):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : float
            The location (mode) of the Poisson distribution.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        cdef double l
        l = loc + self.shift - self.low_bound
        if x < self.low_bound:
            return 0.
        else:
            return pdtr(x - self.low_bound, l)


cdef class DiscreteLaplace(BaseDist):

    cdef readonly double scale

    def __init__(self, scale):
        """Discrete Laplace distribution.

        Parameters
        ----------
        scale : float
            The scale of the discrete Laplace distribution, which controls its variance.
        """
        self.scale = scale

    cpdef double pdf(self, x, loc):
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

        # calc cdf
        p = exp(-1. / self.scale)
        if x < loc:
            return p ** (-floor(x - loc)) / (1. + p)
        else:
            return 1. - (p ** (floor(x - loc) + 1.) / (1. + p))


cdef class Categorical(BaseDist):
    """
    Simple categorical distribution where an uncertainty describing the probability of error, i.e. selecting the
    wrong category, is spread among all other available categories.
    """

    cdef readonly list categories
    cdef readonly double unc
    cdef readonly int num_categories

    def __init__(self, categories, unc):
        """Categories will be encoded alphabetically as an ordered variable.
        In practice, because true categorical variables are not yet supported in sklearn, we implement this  distribution
        as a discrete one with the first category being encoded as 0, the second as 1, et cetera, in alphabetical
        order.

        Parameters
        ----------
        categories : list
            List of categories
        unc : float
            The uncertainty in the categorical choice, i.e. probability that the queried category is not the category
            being evaluated.
        """
        self.categories = categories
        self.num_categories = len(categories)
        self.unc = unc

    cpdef double pdf(self, x, loc):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
        loc : int
            Integer encoding corresponding to a category.
            
        Returns
        -------
        pdf : float
            Probability evaluated at ``x``.
        """
        if int(x) == int(loc):
            return 1. - self.unc
        else:
            return self.unc / (self.num_categories - 1.)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double cdf(self, double x, double loc):
        """Cumulative density function.

        Parameters
        ----------
        x : int
            The point where to evaluate the cdf.
            
        loc : int
            Integer encoding corresponding to a category.
            
        Returns
        -------
        cdf : float
            Cumulative probability evaluated at ``x``.
        """

        # define variables and freeze loc if needed
        cdef int upper_cat
        cdef double cdf
        cdef int cat_idx

        # put bounds on x
        if x == -INFINITY:
            x = -0.5  # encoding starts from 0
        if x == INFINITY:
            x = self.num_categories - 0.5  # last category encoded as num_categories-1

        # the category in with highest integer encoding
        upper_cat = <int>floor(x) + 1

        # calc cdf
        cdf = 0.
        for cat_idx in range(upper_cat):
            if cat_idx == <int>loc:
                cdf +=  1. - self.unc
            else:
                cdf += self.unc / (self.num_categories - 1.)
        return cdf


# ==================================================================================
# "Frozen" probability distributions that do not depend on the input/sample location
# ==================================================================================
cdef class FrozenNormal:

    cdef readonly double std
    cdef readonly double mean

    def __init__(self, mean, std):
        """Normal distribution.

        Parameters
        ----------
        mean : float
            Mean of the distribution.
        std : float
            Standard deviation of the distribution.
        """
        self.std = std
        self.mean = mean

    cpdef double pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """
        return _normal_pdf(x, self.mean, self.std)

    @cython.cdivision(True)
    cpdef double cdf(self, double x):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the cdf.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """
        return _normal_cdf(x, self.mean, self.std)


cdef class FrozenUniform:

    cdef readonly double a
    cdef readonly double b

    def __init__(self, a, b):
        """Uniform distribution.

        Parameters
        ----------
        a : float
            Lower bound of the distribution.
        b : float
            Upper bound of the distribution.
        """
        if a > b:
            raise ValueError('argument `a` needs to be <= `b`')
        self.a = a
        self.b = b

    cpdef double pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """

        if x < self.a:
            return 0.
        elif x > self.b:
            return 0.
        else:
            return 1. / (self.b - self.a)

    @cython.cdivision(True)
    cpdef double cdf(self, double x):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        if x < self.a:
            return 0.
        elif x > self.b:
            return 1.
        else:
            return (x - self.a) / (self.b - self.a)


cdef class FrozenGamma:

    cdef readonly double k
    cdef readonly double theta
    cdef readonly double low_bound
    cdef readonly double high_bound
    cdef readonly double no_bounds

    def __init__(self, k, theta, low_bound=-INFINITY, high_bound=INFINITY):
        """Gamma distribution.

        Parameters
        ----------
        k : float
            Shape parameter for the Gamma distribution.
        theta : float
            Scale parameter for the Gamma distribution.
        low_bound : float, optional
            Lower bound for the distribution. Default is zero.
        high_bound : float, optional
            Upper bound for the distribution. Default is inf.
        """
        self.k = k
        self.theta = theta
        self.low_bound = low_bound
        self.high_bound = high_bound

        # perform checks
        no_bounds = _warn_if_no_bounds(type(self).__name__, self.low_bound, self.high_bound)
        if no_bounds == 1.:
            self.low_bound = 0.
        _check_single_bound(type(self).__name__, self.low_bound, self.high_bound)

    cpdef double pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """
        cdef double logpdf
        if x < self.low_bound:
            return 0.
        if x > self.high_bound:
            return 0.

        # if we have lower bound
        if self.high_bound == INFINITY:
            logpdf = (xlogy(self.k - 1., x-self.low_bound) - (x-self.low_bound)/self.theta -
                      gammaln(self.k) - xlogy(self.k, self.theta))
            return exp(logpdf)
        # if we have an upper bound
        elif self.low_bound == -INFINITY:
            logpdf = (xlogy(self.k - 1., self.high_bound-x) - (self.high_bound-x)/self.theta -
                      gammaln(self.k) - xlogy(self.k, self.theta))
            return exp(logpdf)

    @cython.cdivision(True)
    cpdef double cdf(self, double x):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """

        # calc cdf
        if x < self.low_bound:
            return 0.
        if x > self.high_bound:
            return 1.

        # if we have lower bound
        if self.high_bound == INFINITY:
            return gammainc(self.k, (x - self.low_bound)/self.theta)
        # if we have an upper bound
        elif self.low_bound == -INFINITY:
            return 1. - gammainc(self.k, (self.high_bound - x) / self.theta)


cdef class FrozenPoisson:

    cdef double l
    cdef int low_bound

    def __init__(self, l, low_bound=0):
        """Poisson distribution.

        Parameters
        ----------
        l : float
            The rate parameter.
        low_bound : float, optional
            Lower bound for the distribution. Default is zero.
        """
        self.l = l
        self.low_bound = low_bound

    cpdef double pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """
        cdef int arg

        if x < self.low_bound:
            return 0.
        else:
            arg = <int>floor(x - self.low_bound)
            return (self.l**(x - self.low_bound) * np.exp(-self.l)) / np.math.factorial(arg)

    @cython.cdivision(True)
    cpdef double cdf(self, double x):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """
        if x < self.low_bound:
            return 0.
        else:
            return pdtr(x - self.low_bound, self.l)


cdef class FrozenDiscreteLaplace:

    cdef readonly double mean
    cdef readonly double scale

    def __init__(self, mean, scale):
        """Discrete Laplace distribution.

        Parameters
        ----------
        mean : float
            Mean of the distribution.
        scale : float
            The scale of the discrete Laplace distribution, which controls its variance.
        """
        self.mean = mean
        self.scale = scale

    cpdef double pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
            
        Returns
        -------
        pdf : float
            Probability density evaluated at ``x``.
        """
        cdef double p
        p = np.exp(-1 / self.scale)
        return (1 - p) / (1 + p) * (p ** abs(x-self.mean))

    @cython.cdivision(True)
    cpdef double cdf(self, double x):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the pdf.
            
        Returns
        -------
        cdf : float
            Cumulative density evaluated at ``x``.
        """
        cdef double p
        p = exp(-1. / self.scale)
        if x < self.mean:
            return p ** (-floor(x - self.mean)) / (1. + p)
        else:
            return 1. - (p ** (floor(x - self.mean) + 1.) / (1. + p))


cdef class FrozenCategorical:

    cdef readonly np.ndarray categories
    cdef readonly int num_categories
    cdef readonly np.ndarray probabilities

    def __init__(self, categories, probabilities):
        """Simple categorical distribution. Categories will be encoded alphabetically as an ordered variable.
        In practice, because true categorical variables are not yet supported in sklearn, we implement this  distribution
        as a discrete one with the first category being encoded as 0, the second as 1, et cetera, in alphabetical
        order.

        Parameters
        ----------
        categories : array
            List of categories.
        probabilities : array
            List of probabilities corresponding to each category.
        """
        _check_is_on_simplex(probabilities)

        # sort categories alphabetically and probabilities accordingly
        self.categories, self.probabilities = (np.array(l) for l in zip(*sorted(zip(categories, probabilities))))
        self.num_categories = len(categories)

    cpdef double pdf(self, x):
        """Probabilities.

        Parameters
        ----------
        x : int
            Integer encoding corresponding to a category.
            
        Returns
        -------
        pdf : float
            Probability of category ``x``.
        """
        return self.probabilities[int(x)]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double cdf(self, double x):
        """Cumulative density function.

        Parameters
        ----------
        x : float
            The point where to evaluate the cdf.
            
        Returns
        -------
        cdf : float
            Cumulative probability evaluated at ``x``.
        """
        cdef double [:] probabilities = self.probabilities
        cdef int upper_cat
        cdef double cdf
        cdef int i

        # put bounds on x
        if x == -INFINITY:
            x = -0.5  # encoding starts from 0
        if x == INFINITY:
            x = self.num_categories - 0.5  # last category encoded as num_categories-1

        # the category in with highest integer encoding
        upper_cat = <int>floor(x) + 1

        # calc cdf
        cdf = 0.
        for i in range(upper_cat):
            cdf += probabilities[i]
        return cdf


# ================
# Helper functions
# ================
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


@cython.cdivision(True)
cdef double _normal_pdf(double x, double loc, double scale):
    return exp(-0.5 * ((x-loc) / scale)**2.) / (scale * 2.5066282746310002)


def _check_single_bound(dist, l_bound, h_bound):
    if not np.isinf(l_bound) and not np.isinf(h_bound):
        raise ValueError(f'{dist} allows to define either a lower or an upper bound, not both')


def _check_is_on_simplex(probs):
    probs = np.array(probs)
    if np.any(probs > 1.) or np.any(probs < 0.):
        raise ValueError('probabilities need to be between zero and one')
    if not np.isclose(np.sum(probs), 1.):
        raise ValueError('the sum of all probabilities needs to be equal to one')


def _warn_if_no_bounds(dist, l_bound, h_bound):
    if np.isinf(l_bound) and np.isinf(h_bound):
        logging.warning(f'No bounds provided to the bounded distribution {dist}. Verify your input.')
        return 1.
    return 0.


@cython.boundscheck(False)
cdef int _path_already_visited(num_sample, node_indexes, tree_depth):
    """
    Checks whether an array equal to ``node_indexes[num_sample]`` is present in ``node_indexes[ : num_sample]``.
    
    Parameters
    ----------
    num_sample : int
        Index of sample/tree path being considered.
    node_indexes : array
        2D array where each row is a path through the tree.
    tree_depth : int
        Maximum depth of the tree. 

    Returns
    -------
    int
        Returns 0 if False, 1 of True.
    """
    cdef int n

    # for each sample up to num_sample
    for n in range(num_sample):
        # check is node_indexes[num_sample] is the same as node_indexes[n]
        if _all_the_same(node_indexes, tree_depth, num_sample, n) == 0:
            # if it is not, go to next sample
            continue
        else:
            # if it is, node_indexes[num_sample] has been visited before ==> return True
            return 1
    # we could not find an equal array ==> return False
    return 0


@cython.boundscheck(False)
cdef int _all_the_same(node_indexes, ncols, sample1, sample2):
    """
    Checks whether two arrays contain the same integer elements. Array 1 is ``node_indexes[num_sample]`` and array 2
    is ``node_indexes[n]``.
    
    Parameters
    ----------
    node_indexes : array
        2D array of which 2 rows are compared.
    ncols : int
        Number of elements in each row of ``node_indexes``. 
    sample1 : int
        Index of first sample to consider, i.e. a row in ``node_indexes``.
    sample2 : int
        Index of first sample to consider, i.e. a row in ``node_indexes``.

    Returns
    -------
    int
        Returns 0 if False, 1 of True.
    """

    cdef int ncols_mem =  ncols
    cdef int sample1_mem =  sample1
    cdef int sample2_mem =  sample2
    cdef int [:, :] node_indexes_mem  = node_indexes
    cdef int i

    # for each element in each of the two arrays, check if they are different
    for i in range(ncols_mem):
        if node_indexes_mem[sample1_mem, i] != node_indexes_mem[sample2_mem, i]:
            # if at least one element is different ==> return False
            return 0
    # if we could not find any difference between the 2 arrays, they are equal ==> return True
    return 1


cdef tuple parse_time(start, end):
    cdef double elapsed = end-start  # elapsed time in seconds
    if elapsed < 1.0:
        return elapsed * 1000., 'ms'
    else:
        return elapsed, 's'
