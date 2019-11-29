#!/usr/bin/env python

import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor

import pyximport 
pyximport.install(
        setup_args = {'include_dirs': np.get_include()},
        reload_support = True)

from .convolution import convolute


class Golem(object):

    def __init__(self, X, y, dims, distributions, scales, max_depth=None, beta=0):
        '''

        Parameters
        ----------
        X : array
            2D array of shape (i,j) containing the location of the inputs. It follows the scikit-learn format used for
            features: each row i is a different sample x_i, and each column j is a feature. It can also be a pandas
            DataFrame object.
        y : array
            One-dimensional array of shape (i, 1) containing the observed responses for the inputs X.
        dims : array
            Array indicating which input dimensions (i.e. columns) of X are to be treated probabilistically.
            The arguments in ``distributions`` and ``scales`` will be assigned to these inputs based on their order.
        distributions : array
            Array indicating which distributions to associate with the probabilistic inputs chosen in ``dims``.
            Options available are "gaussian", "uniform".
        scales : array
            Array indicating the variance of the distributions to associate with the probabilistic inputs chosen in
            ``dims``.
        max_depth : int, optional
            The maximum depth of the regression tree. If None, nodes are expanded until all leaves are pure.
            Providing a limit to the tree depth results in faster computations but a more approximate model.
            Default is None.
        beta : int, optional
            Parameter that tunes the penalty variance, similarly to a lower confidence bound acquisition. Default is
            zero, i.e. no variance penalty. Higher values favour more reproducible results at the expense of total
            output.
        '''

        # make sure we have a np object
        self.X = np.array(X)
        self.y = np.array(y)

        self.max_depth = max_depth

        # place delta/indicator functions on the inputs with no uncertainty
        # put info in array to be passed to cython
        # TODO: see if it is better/faster to select the relevant tiles beforehand
        self.distributions = self._parse_distributions(dims, distributions, scales)

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

    def _parse_distributions(self, dims, distributions, scales):

        dists_list = []
        all_dimensions = range(np.shape(self.X)[1])  # all dimensions in the input

        for dim in all_dimensions:
            if dim in dims:
                idx = dims.index(dim)
                dist = distributions[idx]
                scale = scales[idx]

                if dist == 'gaussian':
                    dists_list.append([0., scale])
                elif dist == 'uniform':
                    dists_list.append([1., scale])

            # For all dimensions for which we do not have uncertainty, i.e. if they are not listed in the dims
            # place a very tight uniform (delta function as distribution
            else:
                dists_list.append([1, 10e-50])  # tight uniform

        return np.array(dists_list)
