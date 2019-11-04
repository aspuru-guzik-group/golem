#!/usr/bin/env python

import numpy as np
import time
from scipy import stats
from sklearn.tree import DecisionTreeRegressor

import pyximport 
pyximport.install(
        setup_args = {'include_dirs': np.get_include()},
        reload_support = True)

from convolution import convolute


def timeit(method):
    """Decorator to time methods.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                      (method.__name__, (te-ts) * 1000))
        return result
    return timed


class Colossus(object):

    def __init__(self, X, y, distributions, max_depth=None, beta=0):
        '''

        Parameters
        ----------
        X : array
            2D array of shape (i,j) containing the location of the inputs. It follows the scikit-learn format used for
            features: each row i is a different sample x_i, and each column j is a feature. It can also be a pandas
            DataFrame object.
        y : array
            One-dimensional array of shape (i, 1) containing the observed responses for the inputs X.
        distributions : dict
            Dictionary with information about the uncertainty distributions to use for the inputs. A specific structure
            is expected, an example of which is provided below as an example.
        max_depth : int, optional
            The maximum depth of the regression tree. If None, nodes are expanded until all leaves are pure.
            Providing a limit to the tree depth results in faster computations but a more approximate model.
            Default is None.
        beta : int, optional
            Parameter that tunes the penalty variance, similarly to a lower confidence bound acquisition. Default is
            zero, i.e. no variance penalty. Higher values favour more reproducible results at the expense of total
            output.

        Notes
        -----
        The argument "distributions" expects a specific dict structure (example below).

        >>> distributions = {
        >>>                  0: {'distribution': stats.norm,     # 0th dimension uses a gaussian
        >>>                      'params': {'scale': 0.1}},      # with std dev 0.1
        >>>                  2: {'distribution': stats.uniform,  # 2nd dimension uses a uniform
        >>>                      'params': {'scale': 0.2}}       # with range of 0.2
        >>>                  }

        '''

        # make sure we have a np object
        self.X = np.array(X)
        self.y = np.array(y)

        self.max_depth = max_depth

        # place delta/indicator functions on the inputs with no uncertainty
        # put info in array to be passed to cython
        # TODO: see if it is better/faster to select the relevant tiles beforehand
        self.distributions = self._parse_distributions(distributions)

        self.beta = beta

        # fit regression tree to the data
        self._fit_tree_model()

        # convolute and retrieve robust values from cython code
        self.y_robust = convolute(self.X, self.beta, self.distributions, self.node_indexes, self.value, self.leave_id,
                                  self.feature, self.threshold)

        # y rescaled between 0 and 1
        self.y_robust_scaled = (self.y_robust - np.amin(self.y_robust)) / (
                    np.amax(self.y_robust) - np.amin(self.y_robust))

    def _fit_tree_model(self):
        # fit the tree regression model
        self.tree = DecisionTreeRegressor(max_depth=self.max_depth)
        self.tree.fit(self.X, self.y)

        # get info from tree model
        n_nodes = self.tree.tree_.node_count
        children_left = self.tree.tree_.children_left
        children_right = self.tree.tree_.children_right
        feature = self.tree.tree_.feature  # features split at nodes
        threshold = self.tree.tree_.threshold  # threshold used at nodes
        value = self.tree.tree_.value  # model value of leaves
        leave_id = self.tree.apply(self.X)  # identify terminal nodes
        node_indicator = self.tree.decision_path(self.X)  # get decision paths

        # get the list of nodes (paths) the samples go through
        # node_indexes = [(sample_id, indices)_0 ... (sample_id, indices)_N] with N=number of observations
        node_indexes = [node_indicator.indices[node_indicator.indptr[i]:
                                               node_indicator.indptr[i + 1]]
                        for i in range(np.shape(self.X)[0])]

        # we want the arrays in self.node_indexes to have the same length for cython
        # so pad with -1 as dummy nodes that will be skipped later on
        max_len = np.max([len(i) for i in node_indexes])
        self.node_indexes = []
        for arr in node_indexes:
            self.node_indexes.append(np.pad(arr, pad_width=(0, max_len-len(arr)), mode='constant', constant_values=-1))

        # make sure they are all np arrays
        self.node_indexes = np.array(self.node_indexes)
        self.feature = np.array(feature)
        self.threshold = np.array(threshold)
        self.value = np.array(value.flatten())  # flatten: original shape=(num_nodes, 1, 1)
        self.leave_id = np.array(leave_id)

    def _parse_distributions(self, distributions):
        # For all dimensions for which we do not have uncertainty, i.e. if they are not listed in the "distributions"
        # dict, place a very tight uniform (delta function as distribution

        delta = {'distribution': stats.uniform, 'params': {'scale': 10e-50}}
        dimensions = range(np.shape(self.X)[1])
        for d in dimensions:
            if d not in distributions:
                distributions[d] = delta

        dists = []
        for key, value in distributions.items():
            # dist shape = (num_dimensions, 2)
            # 0. = use gaussian, then the scale
            # 1. = uniform.
            # TODO: extend/rework this options and the input parsing
            dists.append([0., value['params']['scale']])
        dists = np.array(dists)

        return dists
