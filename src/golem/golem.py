#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from copy import deepcopy

import pyximport
pyximport.install(
        setup_args={'include_dirs': np.get_include()},
        reload_support=True)

from .convolution import convolute


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
        elif self.verbose is False:
            self._verbose = 0

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

    def reweight(self, distributions, scales, bounds=None, dims=None):
        """Reweight the measurements to obtain robust merits that depend on the specified uncertainty.

        Parameters
        ----------
        distributions : array, dict
            Array indicating which distributions to associate with the probabilistic inputs chosen in ``dims``.
            Options available are "gaussian", "uniform".
        scales : array, dict
            Array indicating the variance of the distributions to associate with the probabilistic inputs chosen in
            ``dims``.
        bounds : dict
        dims : array
            Array indicating which input dimensions (i.e. columns) of X are to be treated probabilistically.
            The arguments in ``distributions`` and ``scales`` will be assigned to these inputs based on their order.
        """
        self.dims = dims
        self.distributions = distributions
        self.scales = scales

        # parse distributions info
        self._distributions = self._parse_distributions(dims, distributions, scales)

        # convolute each tree and take the mean robust estimate
        self._ys_robust = []
        self._ys_robust_std = []
        self._bounds = []
        self._preds = []
        for i, tree in enumerate(self.forest.estimators_):
            if self.verbose is True:
                print(f'Evaluating tree number {i}')

            # this is only for gradient boosting
            if isinstance(tree, np.ndarray):
                tree = tree[0]

            node_indexes, value, leave_id, feature, threshold = self._parse_tree(tree=tree)
            y_robust, y_robust_std, _bounds, _preds = convolute(self._X, self._distributions, node_indexes,
                                                                value, leave_id, feature, threshold, self._verbose)
            self._ys_robust.append(y_robust)
            self._ys_robust_std.append(y_robust_std)
            self._bounds.append(_bounds)
            self._preds.append(_preds)

        # take the average across all trees
        self.y_robust = np.mean(self._ys_robust, axis=0)
        self.y_robust_std = np.mean(self._ys_robust_std, axis=0)

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

    def _parse_distributions(self, dims, distributions, scales):

        dists_list = []  # each row: dist_type_idx, scale

        # ===========================================================
        # Case 1: X passed is a np.array --> no categorical variables
        # ===========================================================
        if self._df_X is None:
            # we then expect dims, distributions, scales to be lists
            _check_type(dims, list, name='dims')
            _check_type(distributions, list, name='distributions')
            _check_type(scales, list, name='scales')

            all_dimensions = range(np.shape(self._X)[1])  # all dimensions in the input

            for dim in all_dimensions:
                if dim in dims:
                    idx = dims.index(dim)
                    dist = distributions[idx]
                    scale = scales[idx]

                    if dist == 'gaussian':
                        dists_list.append([0., scale])
                    elif dist == 'uniform':
                        dists_list.append([1., scale])

                # For all dimensions for which we do not have uncertainty, we tag them with -1, which
                # indicates a delta function
                else:
                    dists_list.append([-1., -1.])  # -1 = delta function in the cython file

            return np.array(dists_list)

        # =======================================================================
        # Case 2: X passed is a DataFrame --> we might have categorical variables
        # =======================================================================
        else:
            # we then expect dims, distributions, scales to be dictionaries
            _check_type(distributions, dict, name='distributions')
            _check_type(scales, dict, name='scales')
            _check_matching_keys(distributions, scales)
            if self.verbose is True and self.dims is not None:
                print('[ WARNING ]: A DataFrame was passed as `X`, `distributions` and `scales` are dictionaries. '
                      'The argument `dims` is not needed and will be discarded.')

            all_columns = list(self._df_X.columns)  # all dimensions in the _df_X dataframe

            for col in all_columns:
                if col in distributions.keys():
                    dist = distributions[col]
                    scale = scales[col]

                    if dist == 'gaussian':
                        dists_list.append([0., scale])
                        _warn_if_cat_col(col, self._cat_cols, dist)
                    elif dist == 'uniform':
                        dists_list.append([1., scale])
                        _warn_if_cat_col(col, self._cat_cols, dist)
                    # categorical distribution
                    elif dist == 'categorical':
                        assert 0 < scale < 1  # make sure scale is a fraction
                        num_categories = len(set(self._df_X.loc[:, col]))
                        scale_overloaded = num_categories + scale  # add scale to num_cats to pass both info
                        dists_list.append([-2., scale_overloaded])
                        _warn_if_real_col(col, self._cat_cols, dist)

                # For all dimensions for which we do not have uncertainty, we tag them with -1, which
                # indicates a delta function
                else:
                    dists_list.append([-1., -1.])  # -1 = delta function in the cython file

            return np.array(dists_list)


def _check_type(myobject, mytype, name=''):
    if not isinstance(myobject, mytype):
        sys.exit(f'[ ERROR ]: `{name}` is expected to be a {mytype} but it is {myobject}\n')


def _check_matching_keys(dict1, dict2):
    if dict1.keys() != dict2.keys():
        sys.exit(f'[ ERROR ]: dictionary keys mismatch:\n{dict1.keys()} vs {dict2.keys()}\n')


def _warn_if_cat_col(col, cat_cols, dist):
    if col in cat_cols:
        print(f'[ WARNING ]: variable "{col}" was identified by Golem as a categorical variable, but a distribution '
              f'for continuous variables ("{dist}") was selected for it. Please make sure there is no error in '
              f'your inputs.\n')


def _warn_if_real_col(col, cat_cols, dist):
    if col not in cat_cols:
        print(f'[ WARNING ]: variable "{col}" was not identified by Golem as a categorical variable, but you have '
              f'selected a distribution for categorical variables ("{dist}"). Please make sure there is no error in '
              f'your inputs.\n')
