#!/usr/bin/env python

import numpy as np
import time
from scipy import stats
from sklearn.tree import DecisionTreeRegressor

import pyximport 
pyximport.install(
        setup_args = {'include_dirs': np.get_include()},
        reload_support = True)

#from convolution import Convolution
#from convolution import Boxes
from convolution import cColossus


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

        #print(self.value)
        #print(self.value.shape)
        #quit()
        # initialise cython colossus functions
        self.convoluter = cColossus(self.X, self.beta, self.node_indexes, self.value, self.leave_id, self.feature, self.threshold)

        # iterate through all paths in the tree and collect their tiles info
        #self.convoluter.get_bboxes()

        exit()
        # convolute
        self.convoluter.convolute()

        # retrieve robust values
        self.y_robust = self.convoluter.y_robust

        # y rescaled between 0 and 1
        self.y_robust_scaled = (self.y_robust - np.amin(self.y_robust)) / (
                    np.amax(self.y_robust) - np.amin(self.y_robust))

    @timeit
    def cython_get_bboxes(self):

        # we want the arrays in self.node_indexes to have the same length
        self.node_indexes = [i[1] for i in self.node_indexes]
        max_len = np.max([len(i) for i in self.node_indexes])
        new_indices = []
        for arr in self.node_indexes:
            new_indices.append(np.pad(arr, pad_width=(0,max_len-len(arr)), mode='constant', constant_values=-1))
        new_indices = np.array(new_indices)

        print(self.feature)
        quit()

        self.bboxes_getter = Boxes(self.X, new_indices, self.leave_id, self.feature, self.threshold)
        values, bounds = self.bboxes_getter.get_bboxes()
        return values, bounds

    @timeit
    def python_get_bboxes(self):
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

        self.convoluter = Convolution(self.X, preds, bounds, dists, self.beta)
        return self.convoluter.convolute()

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