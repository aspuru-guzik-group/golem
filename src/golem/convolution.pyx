#cython: language_level=3

import  cython 
cimport cython

import  numpy as np 
cimport numpy as np 

from libc.math cimport sqrt, erf, abs
from numpy.math cimport INFINITY

from scipy.special import gammainc

import time
import sys

# ====================================
# Cumulative Probability Distributions
# ====================================
@cython.cdivision(True)
cdef double gauss_cdf(double x, double loc, double scale):
    """
    Gaussian distribution.
    
    Parameters
    ----------
    x : float
        the point where the cdf is evaluated.
    loc : float
        the location of the distribution.
    scale: float
        the scale (one standard deviation) of the Gaussian distribution.

    Returns
    -------
    cdf : float
        the cumulative distribution function evaluated at `x`.
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
cdef double truncated_gauss_cdf(double x, double loc, double scale, double low_bound, double high_bound):
    """
    Truncated Gaussian distribution.
    """

    cdef double cdf_x
    cdef double cdf_upper_bound
    cdef double cdf_lower_bound

    if x < low_bound:
        return 0.
    elif x > high_bound:
        return 1.
    else:
        cdf_x = gauss_cdf(x, loc, scale)
        cdf_upper_bound = gauss_cdf(high_bound, loc, scale)
        cdf_lower_bound = gauss_cdf(low_bound, loc, scale)
        return  (cdf_x - cdf_lower_bound) / (cdf_upper_bound - cdf_lower_bound)

@cython.cdivision(True)
cdef double folded_gauss_cdf(double x, double loc, double scale, double low_bound, double high_bound):
    """
    Folded Gaussian distribution. Note: this is a slow cdf to evaluate.
    """

    cdef double cdf
    cdef double cdf_left
    cdef double cdf_right
    cdef double x_low
    cdef double x_high
    cdef double i

    if x < low_bound:
        return 0.
    elif x > high_bound:
        return 1.
    else:
        # -----------------------------------------
        # if no bounds ==> same as normal gauss_cdf
        # -----------------------------------------
        # this is just to catch the case where the user does not enter bounds
        if high_bound == INFINITY and low_bound == -INFINITY:
            return gauss_cdf(x, loc, scale)
        # -------------------
        # if lower bound only
        # -------------------
        elif high_bound == INFINITY:
            # if x is infinity, return 1 (otherwise x_low=NaN)
            if x == INFINITY:
                return 1.
            else:
                x_low  = x - 2 * (x - low_bound)
                cdf = gauss_cdf(x, loc, scale) - gauss_cdf(x_low, loc, scale)
                return cdf

        # -------------------
        # if upper bound only
        # -------------------
        elif low_bound == -INFINITY:
            # if x is -infinity, return 0 (otherwise x_high=NaN)
            if x == -INFINITY:
                return 0.
            else:
                x_high = x + 2 * (high_bound - x)
                cdf = 1. - (gauss_cdf(x_high, loc, scale) - gauss_cdf(x, loc, scale))
                return cdf

        # -------------------------
        # if lower and upper bounds
        # -------------------------
        else:
            cdf = 0.
            i = 0.
            while True:
                # "fold" on the left
                x_high = x - i*(high_bound - low_bound)
                x_low  = x - i*(high_bound - low_bound) - (2 * (x - low_bound))
                cdf_left = gauss_cdf(x_high, loc, scale) - gauss_cdf(x_low, loc, scale)

                # if i == 0, +/- i*domain_range is the same and we double count the same area
                if i == 0.:
                    cdf += cdf_left
                    i += 2.
                    continue

                # "fold" on the right
                x_high = x + i*(high_bound - low_bound)
                x_low  = x + i*(high_bound - low_bound) - (2 * (x - low_bound))
                cdf_right = gauss_cdf(x_high, loc, scale) - gauss_cdf(x_low, loc, scale)

                # add delta cdf
                delta_cdf = cdf_right + cdf_left
                cdf += delta_cdf

                # break if delta less than some tolerance
                if delta_cdf < 10e-6:
                    break

                # fold at lower bound every 2 folds
                i += 2.

            return cdf

@cython.cdivision(True)
cdef double uniform_cdf(double x, double loc, double scale):
    """
    Uniform distribution.
    
    Parameters
    ----------
    x : float
        the point where the cdf is evaluated.
    loc : float
        the location of the distribution.
    scale : float
        the range of the uniform.

    Returns
    -------
    cdf : float
        the cumulative distribution function evaluated at `x`.
    """
    cdef double a = loc - 0.5 * scale
    cdef double b = loc + 0.5 * scale
    if x < a:
        return 0.
    elif x > b:
        return 1.
    else:
        return (x - a) / (b - a)


@cython.cdivision(True)
cdef double truncated_uniform_cdf(double x, double loc, double scale, double low_bound, double high_bound):
    """
    Truncated uniform distribution. 
    """

    cdef double a = loc - 0.5 * scale
    cdef double b = loc + 0.5 * scale

    # truncate if close to bounds
    if 0.5 * scale > (loc - low_bound):
        a = low_bound
    elif 0.5 * scale > (high_bound - loc):
        b = high_bound

    if x < a:
        return 0.
    elif x > b:
        return 1.
    else:
        return (x - a) / (b - a)


@cython.cdivision(True)
cdef double bounded_uniform_cdf(double x, double loc, double scale, double low_bound, double high_bound):
    """
    Bounded uniform distribution. 
    """

    cdef double a = loc - 0.5 * scale
    cdef double b = loc + 0.5 * scale

    # fix based on lower bound
    if 0.5 * scale > (loc - low_bound):
        a = low_bound
        b = low_bound + scale
    # fix based on upper bound
    elif 0.5 * scale > (high_bound - loc):
        b = high_bound
        a = high_bound - scale
    # "standard" uniform
    else:
        a = loc - 0.5 * scale
        b = loc + 0.5 * scale

    if x < a:
        return 0.
    elif x > b:
        return 1.
    else:
        return (x - a) / (b - a)


@cython.cdivision(True)
cdef double gamma_cdf(double x, double loc, double scale, double low_bound, double high_bound):
    """
    Gamma distribution. Mode (loc) and standard deviation (scale) will be used to fit the k and theta parameters.
    
    x : float
        the point where the cdf is evaluated.
    loc : float
        the mode of the distribution.
    scale: float
        the standard deviation of the distribution. 
    low_bound: float
        lower bound of the distribution. 
    high_bound: float
        upper bound of the distribution. 
    """

    cdef double k
    cdef double theta

    if x < low_bound:
         return 0.
    if x > high_bound:
        return 1.

    # TODO: this transformation could be done only once for all data in the python Golem class
    # i.e. we have lower bound
    if high_bound == INFINITY:
        x = x - low_bound
        loc = loc - low_bound

        var = scale**2.
        theta = np.sqrt(var + (loc**2.)/4.) - loc/2.
        k = loc/theta + 1.

        return gammainc(k, x/theta)

    # i.e. we have an upper bound
    elif low_bound == -INFINITY:
        high_bound = -high_bound
        x = -x - high_bound
        loc = -loc - high_bound

        var = scale**2.
        theta = np.sqrt(var + (loc**2.)/4.) - loc/2.
        k = loc/theta + 1.

        return 1. - gammainc(k, x/theta)


# ==========
# Main Class
# ==========
cdef class cGolem:

    cdef np.ndarray np_X
    cdef np.ndarray np_node_indexes
    cdef np.ndarray np_value
    cdef np.ndarray np_leave_id
    cdef np.ndarray np_feature
    cdef np.ndarray np_threshold

    cdef np.ndarray np_dists
    cdef np.ndarray np_preds
    cdef np.ndarray np_bounds
    cdef np.ndarray np_y_robust
    cdef np.ndarray np_y_robust_std

    cdef int num_samples, num_dims, num_tiles, verbose

    cdef double start, end

    def __init__(self, X, dists, node_indexes, value, leave_id, feature, threshold, verbose):
        self.np_X            = X
        self.np_dists        = dists
        self.np_node_indexes = node_indexes
        self.np_value        = value
        self.np_leave_id     = leave_id
        self.np_feature      = feature
        self.np_threshold    = threshold
        self.verbose         = verbose

        self.num_samples = X.shape[0]
        self.num_dims    = X.shape[1]
        # num of leaves/tiles = num of unique leave nodes associated with all input samples
        self.num_tiles   = np.unique(leave_id).shape[0]

        self.np_bounds = np.empty(shape=(self.num_tiles, self.num_dims, 2))
        self.np_preds =  np.empty(self.num_tiles)

        end = time.time()

    @cython.boundscheck(False)
    cdef void _get_bboxes(self):
        if self.verbose == 1:
            start = time.time()
            print('Parsing the tree...', end='')

        # -----------------------
        # Initialise memory views
        # -----------------------
        cdef double [:, :]  X             = self.np_X
        cdef int [:, :]     node_indexes  = self.np_node_indexes
        cdef double [:]     value         = self.np_value
        cdef long [:]       leave_id      = self.np_leave_id
        cdef long [:]       feature       = self.np_feature
        cdef double [:]     threshold     = self.np_threshold

        cdef int num_dim, num_tile, num_sample, tile_id, node_id, num_node, n
        cdef int tree_depth = np.shape(self.np_node_indexes)[1]  # max number of nodes to reach a leaf

        # to store the y_pred value of each tile/leaf
        cdef double [:] preds = np.empty(self.num_tiles)
        # to store the lower/upper bounds of each tile/leaf
        cdef double [:, :, :] bounds = np.empty(shape=(self.num_tiles, self.num_dims, 2))

        # initialise bounds with -inf and +inf, since later we will store bounds only if tighter than previous ones
        for num_tile in range(self.num_tiles):
            for num_dim in range(self.num_dims):
                bounds[num_tile, num_dim, 0] = -INFINITY
                bounds[num_tile, num_dim, 1] = +INFINITY

        # -----------------------------------------------
        # Iterate through the paths leading to all leaves
        # -----------------------------------------------
        tile_id = -1  # this is to keep an index for the tiles
        for num_sample in range(self.num_samples):
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
        assert tile_id == self.num_tiles-1
        self.np_bounds = np.asarray(bounds)
        self.np_preds = np.asarray(preds)

        if self.verbose == 1:
            print('done', end=' ')
            end = time.time()
            print('[%.2f %s]' % parse_time(start, end))


    @cython.boundscheck(False)
    cdef void _convolute(self):
        if self.verbose == 1:
            start = time.time()
            print('Convoluting...', end='')

        # -----------------------
        # Initialise memory views
        # -----------------------
        cdef double [:, :]    X      = self.np_X
        cdef double [:, :]    dists  = self.np_dists
        cdef double [:]       preds  = self.np_preds
        cdef double [:, :, :] bounds = self.np_bounds

        cdef int num_dim, num_tile, num_sample

        cdef double dist_type, dist_scale, dist_lb, dist_ub, dist_loc
        cdef double low, high
        cdef double low_cat, high_cat
        cdef double scale, num_cats, num_cats_in_tile

        cdef double xi
        cdef double joint_prob

        cdef double [:] newy = np.empty(self.num_samples)
        cdef double [:] newy_std = np.empty(self.num_samples)

        cdef double yi_reweighted
        cdef double yi_reweighted_squared

        cdef double cache

        # ------------------------
        # Iterate over all samples
        # ------------------------
        for num_sample in range(self.num_samples):

            yi_reweighted         = 0.
            yi_reweighted_squared = 0.

            # ----------------------
            # iterate over all tiles
            # ----------------------
            for num_tile in range(self.num_tiles):

                joint_prob = 1.  # joint probability of the tile

                # ---------------------------
                # iterate over all dimensions
                # ---------------------------
                # Note you have to do this, you cannot iterate over uncertain dimensions only.
                # This because for dims with no uncertainty join_prob needs to be multiplied by 0 or 1 depending
                # whether the sample is in the tile or not. And the only way to do this is to check the tile bounds
                # in the certain dimension.
                for num_dim in range(self.num_dims):

                    dist_type  = dists[num_dim, 0]  # distribution type
                    dist_scale = dists[num_dim, 1]  # scale parameter
                    dist_lb    = dists[num_dim, 2]  # lower bound (not always used)
                    dist_ub    = dists[num_dim, 3]  # upper bound (not always used)
                    dist_loc   = dists[num_dim, 4]  # set a fixed location for dist (not always used)

                    # if loc == inf, then we are not fixing it
                    if dist_loc == INFINITY:
                        xi = X[num_sample, num_dim]
                    # otherwise, xi and thus the location is fixed
                    else:
                        xi = dist_loc

                    # delta function (used for dims with no uncertainty)
                    # -------------------------------------------------
                    if dist_type == -1.:
                        # boundaries of the tile in this dimension
                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        if low <= xi < high:
                            joint_prob *= 1.
                        else:
                            joint_prob *= 0.

                    # gaussian
                    # --------
                    elif dist_type == 0.:
                        # boundaries of the tile in this dimension
                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        joint_prob *= gauss_cdf(high, xi, dist_scale) - gauss_cdf(low, xi, dist_scale)

                    # truncated gaussian
                    # ------------------
                    elif dist_type == 0.1:
                        # boundaries of the tile in this dimension
                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        joint_prob *= (truncated_gauss_cdf(high, xi, dist_scale, dist_lb, dist_ub) -
                                       truncated_gauss_cdf(low, xi, dist_scale, dist_lb, dist_ub))

                    # folded gaussian
                    # ----------------
                    elif dist_type == 0.2:
                        # boundaries of the tile in this dimension
                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        joint_prob *= (folded_gauss_cdf(high, xi, dist_scale, dist_lb, dist_ub) -
                                       folded_gauss_cdf(low, xi, dist_scale, dist_lb, dist_ub))

                    # uniform
                    # -------
                    elif dist_type == 1.:
                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        joint_prob *= uniform_cdf(high, xi, dist_scale) - uniform_cdf(low, xi, dist_scale)

                    # truncated uniform
                    # -----------------
                    elif dist_type == 1.1:
                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        joint_prob *= (truncated_uniform_cdf(high, xi, dist_scale, dist_lb, dist_ub) -
                                       truncated_uniform_cdf(low, xi, dist_scale, dist_lb, dist_ub))

                    # bounded uniform
                    # ---------------
                    elif dist_type == 1.2:
                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        joint_prob *= (bounded_uniform_cdf(high, xi, dist_scale, dist_lb, dist_ub) -
                                       bounded_uniform_cdf(low, xi, dist_scale, dist_lb, dist_ub))
                    # gamma
                    # -----
                    elif dist_type == 2.:
                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        joint_prob *= (gamma_cdf(high, xi, dist_scale, dist_lb, dist_ub) -
                                       gamma_cdf(low, xi, dist_scale, dist_lb, dist_ub))

                    # categorical
                    # -----------
                    elif dist_type == -2.:
                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        # get info about categories needed to compute probabilities
                        num_cats = np.floor(dist_scale)  # number of categories
                        scale = dist_scale - num_cats  # uncertain fraction
                        # figure out how many categories we have in this tile
                        if low == -INFINITY:
                            low_cat = -0.5
                        else:
                            low_cat = low
                        if high == INFINITY:
                            high_cat = num_cats - 0.5
                        else:
                            high_cat = high
                        num_cats_in_tile = high_cat - low_cat

                        if low <= xi < high:
                            # probability of current category + probability of other categories in this tile
                            joint_prob *= (1.0 - scale) + (num_cats_in_tile - 1.) * (scale / (num_cats - 1.))
                        else:
                            # probability of all categories in this tile
                            # distribute uncertain fraction across all other cats
                            joint_prob *= (scale / (num_cats - 1.)) * num_cats_in_tile

                    else:
                        sys.exit(f'[ ERROR ]: unrecognized index "{dist_type}" key for distribution selection')

                # do the sum already within the loop
                cache                  = joint_prob * preds[num_tile]
                yi_reweighted         += cache
                yi_reweighted_squared += cache * preds[num_tile]

            # store robust y value for the kth sample
            newy[num_sample] = yi_reweighted
            newy_std[num_sample] = sqrt(yi_reweighted_squared - yi_reweighted**2)

        self.np_y_robust = np.asarray(newy)
        self.np_y_robust_std = np.asarray(newy_std)

        if self.verbose == 1:
            print('done', end=' ')
            end = time.time()
            print('[%.2f %s]' % parse_time(start, end))


# ================
# Helper Functions
# ================
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


# ===========================
# Functions exposed to Python
# ===========================
cpdef convolute(X, dists, node_indexes, value, leave_id, feature, threshold, verbose):
    golem = cGolem(X, dists, node_indexes, value, leave_id, feature, threshold, verbose)
    golem._get_bboxes()
    golem._convolute()
    return golem.np_y_robust, golem.np_y_robust_std, golem.np_bounds, golem.np_preds