#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from copy import deepcopy
import logging
logging.basicConfig(format='[%(levelname)s] [%(asctime)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')

#from .extensions import convolute, Delta
from .extensions import _get_bboxes, _convolute, Delta


class Golem(object):

    def __init__(self, goal='min', forest_type='dt', ntrees=1, random_state=None, verbose=True):
        """

        Parameters
        ----------
        forest_type : str
            Type of forest.
        ntrees : int, str
            Number of trees to use. Use 1 for a single regression tree, or more for a forest. If 1 is selected, the
            choice of `forest_type` will be discarded.
        random_state : int, optional
            Fix random seed
        verbose : bool, optional.

        Attributes
        ----------
        y_robust : array
        y_robust_scaled : array
        forest : object

        Methods
        -------
        get_tiles
        """

        # ---------
        # Init vars
        # ---------
        self.X = None
        self._X = None
        self.y = None
        self._y = None

        self.dims = None
        self.distributions = None
        self._distributions = None
        self.scales = None
        self.low_bounds = None
        self.high_bounds = None
        self.freeze_loc = None

        self.beta = None
        self._beta = None

        self._ys_robust = None
        self._bounds = None
        self._preds = None

        self._cat_cols = None

        # ---------------
        # Store arguments
        # ---------------

        # options for the tree
        self.ntrees = ntrees
        self._ntrees = self._parse_ntrees_arg(ntrees)
        self.max_depth = None
        self.random_state = random_state
        self.forest_type = forest_type

        # other options
        self.goal = goal
        self.verbose = verbose
        # True=1, False=0 for cython
        if self.verbose is True:
            self._verbose = 1
            logging.getLogger().setLevel(logging.INFO)
        elif self.verbose is False:
            self._verbose = 0
            logging.getLogger().setLevel(logging.WARNING)

        # select/initialise model
        self._init_forest_model()

    def fit(self, X, y):
        """Fit the tree-based model to partition the input space.

        Parameters
        ----------
        X : array
            2D array of shape (i,j) containing the location of the inputs. It follows the scikit-learn format used for
            features: each row i is a different sample x_i, and each column j is a feature. It can also be a pandas
            DataFrame object.
        y : array
            One-dimensional array of shape (i, 1) containing the observed responses for the inputs X.
        """
        self.X = X
        self.y = y
        self._X = self._parse_X(X)
        self._y = self._parse_y(y)
        # fit regression tree(s) to the data
        self.forest.fit(self._X, self._y)

    def predict(self, X, return_std=False):

        # make sure input dimensions match training
        assert np.shape(X)[1] == np.shape(self._X)[1]

        # convolute each tree and take the mean robust estimate
        _ys_robust = []
        _ys_robust_std = []

        for i, tree in enumerate(self.forest.estimators_):
            logging.info(f'Evaluating tree number {i}')

            # this is only for gradient boosting
            # TODO: remove gradient boosting
            if isinstance(tree, np.ndarray):
                tree = tree[0]

            #node_indexes, value, leave_id, feature, threshold = self._parse_tree(tree=tree)
            #y_robust, y_robust_std, _, _ = convolute(X, self._distributions, node_indexes,
            #                                         value, leave_id, feature, threshold, self._verbose)
            y_robust, y_robust_std = _convolute(X, self._distributions, self._preds[i], self._bounds[i])
            _ys_robust.append(y_robust)
            _ys_robust_std.append(y_robust_std)

        # take the average across all trees
        if return_std is True:
            y_robust = np.mean(_ys_robust, axis=0)  # expectation of the output
            y_robust_std = np.std(_ys_robust, axis=0)  # conditional standard deviation
            return y_robust, y_robust_std
        elif return_std is False:
            return np.mean(_ys_robust, axis=0)

    def reweight(self, distributions, dims=None):
        """Reweight the measurements to obtain robust merits that depend on the specified uncertainty.

        Parameters
        ----------
        distributions : array, dict
            Array or dictionary of distribution objects from the ``dists`` module.
        dims : array
            Array indicating which input dimensions (i.e. columns) of X are to be treated probabilistically. If passing
            dictionaries as arguments, this is not needed. If passing arrays instead, the ``distributions`` will be
            assigned to these inputs based on their order.
        """
        self.dims = dims
        self.distributions = distributions

        # parse distributions info
        # if we received a DataFrame we expect dist info to be dicts, otherwise they should be lists/arrays
        if self._df_X is None:
            self._distributions = self._parse_distributions_lists()
        else:
            self._distributions = self._parse_distributions_dicts()

        # convolute each tree and take the mean robust estimate
        self._ys_robust = []
        self._ys_robust_std = []
        self._bounds = []
        self._preds = []
        for i, tree in enumerate(self.forest.estimators_):
            logging.info(f'Evaluating tree number {i}')

            # this is only for gradient boosting
            if isinstance(tree, np.ndarray):
                tree = tree[0]

            node_indexes, value, leave_id, feature, threshold = self._parse_tree(tree=tree)
            #y_robust, y_robust_std, _bounds, _preds = convolute(self._X, self._distributions, node_indexes,
            #                                                    value, leave_id, feature, threshold, self._verbose)
            _bounds, _preds = _get_bboxes(self._X, node_indexes, value, leave_id, feature, threshold)
            y_robust, y_robust_std = _convolute(self._X, self._distributions, _preds, _bounds)

            self._ys_robust.append(y_robust)
            self._ys_robust_std.append(y_robust_std)
            self._bounds.append(_bounds)
            self._preds.append(_preds)

        # take the average across all trees
        self.y_robust = np.mean(self._ys_robust, axis=0)  # expectation of the output
        self.y_robust_std = np.mean(self._ys_robust_std, axis=0)  # variance of the output

    def get_robust_merits(self, beta=0, normalize=False):
        """Retrieve the values of the robust merits.

        Parameters
        ----------
        beta : int, optional
            Parameter that tunes the penalty variance, similarly to a lower confidence bound acquisition. Default is
            zero, i.e. no variance penalty. Higher values favour more reproducible results at the expense of total
            output.
        normalize : bool, optional
            Whether to return normalized values between 0 and 1.

        Returns
        -------
        y_robust : array
            Values of the robust merits.
        """
        self.beta = beta
        if self.goal == 'min':
            self._beta = -beta
        elif self.goal == 'max':
            self._beta = beta
        else:
            raise ValueError(f"value {self.goal} for argument `goal` not recognized. It can only be 'min' or 'max'")

        # multiply by beta
        merits = self.y_robust - self._beta * self.y_robust_std

        # return
        if normalize is True:
            return (merits - np.amin(merits)) / (np.amax(merits) - np.amin(merits))
        else:
            return merits

    def get_expect_and_std(self):
        """Return the expectation and the standard deviation of the output.

        Returns
        -------
        mean, std: (array, array)
            The mean and standard deviation of the response/measurements given the uncertainty in the inputs used to
            reweight the response/measurement values.
        """
        return self.y_robust, self.y_robust_std

    def get_tiles(self, tree_number=0):
        """Returns information about the tessellation created by the decision tree.

        Parameters
        ----------
        tree_number : int
            The index of the tree to parse. Default is 0, i.e. the first tree.

        Returns
        -------
        tiles : list
            List of tiles with information about the lower/upper boundary of the tile in all dimensions, and the
            predicted output by the decision tree model.
        """
        _bounds = self._bounds[tree_number]
        _preds = self._preds[tree_number]

        tiles = []
        for bounds, pred in zip(_bounds, _preds):
            tile = {}
            for i, bound in enumerate(bounds):
                tile[i] = {}
                tile[i]['low'] = bound[0]
                tile[i]['high'] = bound[1]
                assert tile[i]['high'] > tile[i]['low']
            tile['y_pred'] = pred
            tiles.append(tile)
        return tiles

    def _parse_X(self, X):
        self._df_X = None  # initialize to None
        if isinstance(X, pd.DataFrame):
            # encode categories as ordinal data - we do not use OneHot encoding because it increases the
            # dimensionality too much (which slows down the convolution) and because we expand trees until
            # leaves are pure anyway. Ordinal encoding is not ideal, but in this case better than OneHot.
            self._df_X = deepcopy(X)

            # identify categorical variables
            cols = self._df_X.columns
            num_cols = self._df_X._get_numeric_data().columns
            cat_cols = list(set(cols) - set(num_cols))
            self._cat_cols = cat_cols

            # encode variables as ordinal data
            for col in cat_cols:
                # note that cat vars are encoded to numbers alphabetically
                self._df_X.loc[:, col] = self._df_X.loc[:, col].astype("category").cat.codes

            return np.array(self._df_X, dtype=np.float64)
        else:
            return np.array(X)

    @staticmethod
    def _parse_y(y):
        # if 1-d vector, all good
        if len(np.shape(y)) == 1:
            return np.array(y)
        # if e.g. a 2d vector: [[1], [2], [3]], flatten
        else:
            return np.array(y).flatten()

    def _parse_ntrees_arg(self, ntrees):
        if isinstance(ntrees, int):
            return ntrees
        elif isinstance(ntrees, str):
            if ntrees == 'sqrt':
                return int(np.floor(np.sqrt(np.shape(self._X)[0])))
            elif ntrees == 'log2':
                return int(np.floor(np.log2(np.shape(self._X)[0] + 1)))
            elif ntrees == 'n*sqrt':
                return int(np.floor(np.sqrt(np.shape(self._X)[0]) * np.shape(self._X)[1]))
            elif ntrees == 'n*log2':
                return int(np.floor(np.log2(np.shape(self._X)[0] + 1) * np.shape(self._X)[1]))
        else:
            raise ValueError(f'invalid argument "{ntrees}" provided to ntrees')

    def _init_forest_model(self):
        # Multiple Regression Trees. RF with Bootstrap=False: we just build a trees where we have random splits
        # because the improvement criterion will be the same for different potential splits
        if self.forest_type == 'dt':
            self.forest = RandomForestRegressor(n_estimators=self._ntrees, bootstrap=False, max_features=None,
                                                random_state=self.random_state, max_depth=self.max_depth)
        # Random Forest
        elif self.forest_type == 'rf':
            self.forest = RandomForestRegressor(n_estimators=self._ntrees, bootstrap=True, max_features=None,
                                                random_state=self.random_state, max_depth=self.max_depth)
        # Extremely Randomized Trees
        elif self.forest_type == 'et':
            # do not bootstrap ExtraTrees
            self.forest = ExtraTreesRegressor(n_estimators=self._ntrees, bootstrap=False, max_features=None,
                                              random_state=self.random_state, max_depth=self.max_depth)
        # Gradient Boosting
        elif self.forest_type == 'gb':
            self.forest = GradientBoostingRegressor(n_estimators=self._ntrees, max_features=None,
                                                    random_state=self.random_state, max_depth=self.max_depth)
        else:
            raise NotImplementedError

    def _parse_tree(self, tree):
        # get info from tree model
        feature = tree.tree_.feature  # features split at nodes
        threshold = tree.tree_.threshold  # threshold used at nodes
        value = tree.tree_.value  # model value of leaves
        leave_id = tree.apply(self._X)  # identify terminal nodes
        node_indicator = tree.decision_path(self._X)  # get decision paths

        # get the list of nodes (paths) the samples go through
        # node_indexes = [(sample_id, indices)_0 ... (sample_id, indices)_N] with N=number of observations
        _node_indexes = [node_indicator.indices[node_indicator.indptr[i]:
                                               node_indicator.indptr[i + 1]]
                        for i in range(np.shape(self._X)[0])]

        # we want the arrays in self.node_indexes to have the same length for cython
        # so pad with -1 as dummy nodes that will be skipped later on
        max_len = np.max([len(i) for i in _node_indexes])
        node_indexes = []
        for arr in _node_indexes:
            node_indexes.append(np.pad(arr, pad_width=(0, max_len - len(arr)), mode='constant', constant_values=-1))

        # make sure they are all np arrays
        node_indexes = np.array(node_indexes)
        feature = np.array(feature)
        threshold = np.array(threshold)
        value = np.array(value.flatten())  # flatten: original shape=(num_nodes, 1, 1)
        leave_id = np.array(leave_id)

        return node_indexes, value, leave_id, feature, threshold

    def _parse_distributions_lists(self):
        # ===========================================================
        # Case 1: X passed is a np.array --> no categorical variables
        # ===========================================================

        dists_list = []  # list of distribution objects

        # we then expect dims and distributions to be lists
        _check_type(self.distributions, list, name='distributions')
        _check_type(self.dims, list, name='dims')

        all_dimensions = range(np.shape(self._X)[1])  # all dimensions in the input

        for dim in all_dimensions:
            if dim in self.dims:
                idx = self.dims.index(dim)
                dist = self.distributions[idx]
                _check_data_within_bounds(dist, self._X[:, dim])

                # append dist instance to list of dists
                dists_list.append(dist)

            # For all dimensions for which we do not have uncertainty, use Delta
            else:
                dist = Delta()
                dists_list.append(dist)

        return np.array(dists_list)

    def _parse_distributions_dicts(self):
        # ==========================================================================
        # Case 1: X passed is a pd.DataFrame --> we might have categorical variables
        # ==========================================================================

        dists_list = []  # each row: dist_type_idx, scale, lower_bound, upper_bound

        # we then expect dims, distributions, scales to be dictionaries
        _check_type(self.distributions, dict, name='distributions')

        if self.dims is not None:
            logging.info('a DataFrame was passed as `X`, `distributions` and `scales` are dictionaries. '
                         'The argument `dims` is not needed and will be discarded.')

        all_columns = list(self._df_X.columns)  # all dimensions in the _df_X dataframe

        for col in all_columns:
            if col in self.distributions.keys():
                dist = self.distributions[col]
                _check_data_within_bounds(dist, self._df_X.loc[:, col])
                _warn_if_dist_var_mismatch(col, self._cat_cols, dist)

                # append dist instance to list of dists
                dists_list.append(dist)

            # For all dimensions for which we do not have uncertainty, use Delta
            else:
                dist = Delta()
                dists_list.append(dist)

        return np.array(dists_list)


def _check_type(myobject, mytype, name=''):
    if not isinstance(myobject, mytype):
        raise TypeError(f'[ ERROR ]: `{name}` is expected to be a {mytype} but it is {myobject}\n')


def _check_data_within_bounds(dist, data):
    if hasattr(dist, 'low_bound'):
        if np.min(data) < dist.low_bound:
            raise ValueError(f'Data contains out-of-bound samples: {np.min(data)} is lower than the '
                             f'chosen lower bound ({dist.low_bound}) in {type(dist).__name__}')
    if hasattr(dist, 'high_bound'):
        if np.max(data) > dist.high_bound:
            raise ValueError(f'Data contains out-of-bound samples: {np.max(data)} is larger than the '
                             f'chosen upper bound ({dist.high_bound}) in {type(dist).__name__}')


def _warn_if_dist_var_mismatch(col, cat_cols, dist):
    if type(dist).__name__ == 'Categorical':
        if col not in cat_cols:
            logging.warning(f'Variable "{col}" was not identified by Golem as a categorical variable, but you have '
                            f'selected {type(dist).__name__} as its distribution. Verify your input.')
    else:
        if col in cat_cols:
            logging.warning(f'Variable "{col}" was identified by Golem as a categorical variable, but a distribution '
                            f'for continuous variables ("{dist}") was selected for it. Verify your input.')
