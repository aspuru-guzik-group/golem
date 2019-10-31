#!/usr/bin/env python

import numpy as np
import time
from scipy import stats
from sklearn.tree import DecisionTreeRegressor

import pyximport 
pyximport.install(
        setup_args = {'include_dirs': np.get_include()},
        reload_support = True)
from convolution import Convolution


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
        self.distributions = distributions

        self.beta = beta

        # place delta/indicator functions on the inputs with no uncertainty
        # TODO: see if it is better/faster to select the relevant tiles beforehand
        self._parse_distributions()

        # fit regression tree to the data
        self._fit_tree_model()

        # iterate through all paths in the tree and collect their tiles info
        self.bboxes = self.get_bboxes()

        # convolute
#        self.y_robust = self.python_convolute()
        self.y_robust = self.cython_convolute()

        # y rescaled between 0 and 1
        self.y_robust_scaled = (self.y_robust - np.amin(self.y_robust)) / (
                    np.amax(self.y_robust) - np.amin(self.y_robust))

    @timeit
    def get_bboxes(self):
        print('Computing tessellation...', end='')

        bboxes = []  # to store all tiles and their info
        done = []  # to keep track of tiles already visited

        # -----------------------------------------------
        # Iterate through the paths leading to all leaves
        # -----------------------------------------------
        for sample_id, node_index in self.node_indexes:
            # if we have duplicate paths (due to multiple samples ending up in in the same leaf)
            # skip them, otherwise we will be counting some tiles multiple times
            if any(np.array_equal(node_index, i) for i in done):
                continue
            done.append(node_index)

            # initialise a dict where to store information on the tesselation
            bbox = {}

            # -------------------------------------------------------
            # extract decisions made at each node leading to the leaf
            # -------------------------------------------------------
            for node_id in node_index:
                # if it is a terminal node, no decision is made
                # store the value assigned by the tree to this node
                if self.leave_id[sample_id] == node_id:
                    bbox['value'] = float(self.value[node_id])
                    continue

                # check if feature being evaluated is above/below node decision threshold
                # this is not needed but it is left for the moment for clarity
                if (self.X[sample_id, self.feature[node_id]] <= self.threshold[node_id]):
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                # store information on path
                # if feature is not a key in the bbox dict yet, create it...
                if self.feature[node_id] not in bbox:
                    bbox[self.feature[node_id]] = {}
                    bbox[self.feature[node_id]]['high'] = []
                    bbox[self.feature[node_id]]['low'] = []

                    if threshold_sign == "<=":
                        bbox[self.feature[node_id]]['high'].append(self.threshold[node_id])
                    else:
                        bbox[self.feature[node_id]]['low'].append(self.threshold[node_id])
                # ...otherwise just append info
                else:
                    if threshold_sign == "<=":
                        bbox[self.feature[node_id]]['high'].append(self.threshold[node_id])
                    else:
                        bbox[self.feature[node_id]]['low'].append(self.threshold[node_id])

            # parse thresholds
            bbox = self._parse_thresholds(bbox)
            # append tile to list of all tiles in the partition
            bboxes.append(bbox)

        print("done")
        return bboxes

    @timeit 
    def cython_convolute(self):

        preds = np.array([bbox['value'] for bbox in self.bboxes])
        bounds = []
        for bbox in self.bboxes:
            tile_entry = []
            for key, b in bbox.items():
                if key == 'value': continue
                param_entry = np.zeros(2) * np.nan
                if not b['low'] is None:
                    param_entry[0] = b['low']
                if not b['high'] is None:
                    param_entry[1] = b['high']
#                param_entry = [b['low'], b['high']]
                tile_entry.append(param_entry)
            bounds.append(tile_entry)
        bounds = np.array(bounds)

        dists = []
        for key, value in self.distributions.items():
            dists.append([0., value['params']['scale']])
        dists = np.array(dists)

#        print('X SHAPE', self.X.shape)
#        print('PREDS SHAPE', preds.shape)
#        print('BOUNDS', bounds)

        self.convoluter = Convolution(self.X, preds, bounds, dists, self.beta)
        return self.convoluter.convolute()


    @timeit
    def python_convolute(self):
        print('Performing convolution...', end='')

        y_reweighted = []
        y_reweighted_squared = []

        # for each observation
        for x in self.X:

            # store probability in all dimensions, which later will be multiplied together
            probabilities = np.zeros(shape=(np.shape(self.X)[1], len(self.bboxes)))
            # predicted response values for all tiles
            y_pred = np.array([bbox['value'] for bbox in self.bboxes])

            # ------------------------------
            # go through all dimensions of x
            # ------------------------------
            for i, xi in enumerate(x):
                # define distribution centered at the sample
                distribution_i = self.distributions[i]['distribution']
                params = self.distributions[i]['params']

                # fix location of uniform
                if distribution_i == stats.uniform:
                    d = distribution_i(**params, loc=xi-params['scale']/2)
                else:
                    d = distribution_i(**params, loc=xi)

                # determine probability/weight of each tile in the rectangular tesselation
                for j, bbox in enumerate(self.bboxes):

                    high = bbox[i]['high']
                    low = bbox[i]['low']

                    if low is None:
                        probability = d.cdf(high)
                    elif high is None:
                        probability = 1.0 - d.cdf(low)
                    else:
                        probability = d.cdf(high) - d.cdf(low)

                    # store
                    probabilities[i, j] = probability

            # now multiply the marginal probabilities together to get the joint
            joint_probabilities = np.prod(probabilities, axis=0)

            # check we have gone through the whole distribution (i.e. area sums up to 1)
            assert np.isclose(np.sum(joint_probabilities), 1.0)

            # multiply the probabilities by the value of the tiles to get the reweighted
            # value for y (response) corresponding to the x just parsed
            assert len(y_pred) == len(joint_probabilities)
            yi_reweighted = np.dot(joint_probabilities, y_pred)
            yi_reweighted_squared = np.dot(joint_probabilities, y_pred ** 2)

            # append reweighted y_i values to the list of all reweighted y
            y_reweighted.append(yi_reweighted)
            y_reweighted_squared.append(yi_reweighted_squared)

        # now we have all the reweighted y values - E[f(x)] and E[f(x)^2]
        # if beta > 0 we are penalising also the variance
        if self.beta > 0:
            variance = np.array(y_reweighted_squared) - np.array(y_reweighted) ** 2
            newy = np.array(y_reweighted) - self.beta * np.sqrt(variance)
        else:
            newy = np.array(y_reweighted)

        print("done")
        return newy


    def _fit_tree_model(self):
        # fit the tree regression model
        self.tree = DecisionTreeRegressor(max_depth=self.max_depth)
        self.tree.fit(self.X, self.y)

        # get info from tree model
        self.n_nodes = self.tree.tree_.node_count
        self.children_left = self.tree.tree_.children_left
        self.children_right = self.tree.tree_.children_right
        self.feature = self.tree.tree_.feature  # features split at nodes
        self.threshold = self.tree.tree_.threshold  # threshold used at nodes
        self.value = self.tree.tree_.value  # model value of leaves
        self.leave_id = self.tree.apply(self.X)  # identify terminal nodes
        self.node_indicator = self.tree.decision_path(self.X)  # get decision paths

        # get the list of nodes (paths) the samples go through
        # node_indexes = [(sample_id, indices)_0 ... (sample_id, indices)_N] with N=number of observations
        self.node_indexes = [(i, self.node_indicator.indices[self.node_indicator.indptr[i]:
                                                             self.node_indicator.indptr[i + 1]])
                             for i in range(np.shape(self.X)[0])]


    def _parse_thresholds(self, bbox):
        ''' Parse dictionary containing all decisions made through the tree and retain only
        relevant information that define the bounding box around the sample.
        '''
        for feat in bbox:
            if feat == 'value':
                continue

            # for samples at the edges, there is no high/low
            # we are at domain boundary
            if len(bbox[feat]['high']) == 0:
                bbox[feat]['high'] = [None]
            if len(bbox[feat]['low']) == 0:
                bbox[feat]['low'] = [None]

            # select bounding box
            high = np.amin(bbox[feat]['high'])
            low = np.amax(bbox[feat]['low'])
            bbox[feat]['high'] = high
            bbox[feat]['low'] = low
        return bbox

    def _parse_distributions(self):
        # For all dimensions for which we do not have uncertainty, i.e. if they are not listed in the "distributions"
        # dict, place a very tight uniform (delta function as distribution

        delta = {'distribution': stats.uniform, 'params': {'scale': 10e-50}}
        dimensions = range(np.shape(self.X)[1])
        for d in dimensions:
            if d not in self.distributions:
                self.distributions[d] = delta
