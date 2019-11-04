#!/usr/bin/env python 

from __future__ import print_function
import  cython 
cimport cython

import  numpy as np 
cimport numpy as np 

from libc.math cimport exp, sqrt, erf
from libc.math cimport isnan
from numpy.math cimport INFINITY

# ====================================
# Cumulative Probability Distributions
# ====================================
@cython.cdivision(True)
cdef double gauss_cdf(double x, double loc, double scale):
    '''
    Gaussian distribution.
    
    Parameters
    ----------
    x
    loc
    scale

    Returns
    -------

    '''
    cdef double arg
    arg = (x - loc) / (1.4142135623730951 * scale)
    if arg > 3.:
        return 1.
    elif arg < -3.:
        return 0.
    return (1. + erf( arg )) * 0.5

@cython.cdivision(True)
cdef double uniform_cdf(double x, double loc, double scale):
    '''
    Uniform distribution.
    
    Parameters
    ----------
    x
    loc
    scale

    Returns
    -------

    '''
    cdef double a = loc - 0.5 * scale
    cdef double b = loc + 0.5 * scale
    if x < a:
        return 0.
    elif x > b:
        return 1.
    return (x - a) / (b - a)


# ==========
# Main Class
# ==========
cdef class cColossus:

    cdef np.ndarray np_X
    cdef double     np_beta
    cdef np.ndarray np_node_indexes
    cdef np.ndarray np_value
    cdef np.ndarray np_leave_id
    cdef np.ndarray np_feature
    cdef np.ndarray np_threshold

    cdef np.ndarray np_dists
    cdef np.ndarray np_preds
    cdef np.ndarray np_bounds
    cdef np.ndarray np_y_robust

    cdef int num_samples, num_dims, num_tiles

    def __init__(self, X, beta, dists, node_indexes, value, leave_id, feature, threshold):

        self.np_X            = X
        self.np_beta         = beta
        self.np_dists        = dists
        self.np_node_indexes = node_indexes
        self.np_value        = value
        self.np_leave_id     = leave_id
        self.np_feature      = feature
        self.np_threshold    = threshold

        self.num_samples = X.shape[0]
        self.num_dims    = X.shape[1]
        # num of leaves/tiles = num of unique leave nodes associated with all input samples
        self.num_tiles   = np.unique(leave_id).shape[0]

        self.np_bounds = np.empty(shape=(self.num_tiles, self.num_dims, 2))
        self.np_preds =  np.empty(self.num_tiles)

    @cython.boundscheck(False)
    cdef void _get_bboxes(self):
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

        cdef int num_dim, num_tile, num_sample, tile_id, node_id, num_node
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
            if any(np.array_equal(node_indexes[num_sample], nodes) for nodes in node_indexes[:num_sample]):
                continue
            tile_id = tile_id + 1

            # -------------------------------------------------------
            # extract decisions made at each node leading to the leaf
            # -------------------------------------------------------
            for num_node in range(tree_depth):
                node_id = node_indexes[num_sample, num_node]

                # we assigned -1 as dummy nodes to pad the arrays to the same length, so if < 0 skip dummy node
                if node_id < 0:
                    continue

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
        print('done')


    @cython.boundscheck(False)
    cdef void _convolute(self):
        print('Convoluting...', end='')

        # -----------------------
        # Initialise memory views
        # -----------------------
        cdef double [:, :]    X      = self.np_X
        cdef double [:, :]    dists  = self.np_dists
        cdef double [:]       preds  = self.np_preds
        cdef double [:, :, :] bounds = self.np_bounds

        cdef int num_dim, num_tile, num_sample

        cdef double dist_type, dist_param
        cdef double low, high

        cdef double xi
        cdef double joint_prob

        cdef double [:] newy = np.empty(self.num_samples)

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

                # ----------------------
                # iterate over all dimensions
                # ---------------------------
                for num_dim in range(self.num_dims):

                    xi         = X[num_sample, num_dim]
                    dist_type  = dists[num_dim, 0]
                    dist_param = dists[num_dim, 1]

                    # gaussian
                    if dist_type == 0.:
                        # boundaries of the tile in this dimension
                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        if isnan(low):
                            joint_prob *= gauss_cdf(high, xi, dist_param)
                        elif isnan(high):
                            joint_prob *= 1.0 - gauss_cdf(low, xi, dist_param)
                        else:
                            joint_prob *= gauss_cdf(high, xi, dist_param) - gauss_cdf(low, xi, dist_param)

                    # uniform
                    elif dist_type == 1.:

                        low  = bounds[num_tile, num_dim, 0]
                        high = bounds[num_tile, num_dim, 1]
                        if isinf(low):
                            joint_prob *= uniform_cdf(high, xi, dist_param)
                        elif isinf(high):
                            joint_prob *= 1.0 - uniform_cdf(low, xi,dist_param)
                        else:
                            joint_prob *= uniform_cdf(high, xi, dist_param) - uniform_cdf(low, xi, dist_param)

                # do the sum already within the loop
                cache                  = joint_prob * preds[num_tile]
                yi_reweighted         += cache
                yi_reweighted_squared += cache * preds[num_tile]

            # store robust y value for the kth sample
            newy[num_sample] = yi_reweighted - self.np_beta * sqrt(yi_reweighted_squared - yi_reweighted**2)

        self.np_y_robust = np.asarray(newy)
        print('done')


# ===========================
# Functions exposed to Python
# ===========================
cpdef convolute(X, beta, dists, node_indexes, value, leave_id, feature, threshold):
    colossus = cColossus(X, beta, dists, node_indexes, value, leave_id, feature, threshold)
    colossus._get_bboxes()
    colossus._convolute()
    return colossus.np_y_robust