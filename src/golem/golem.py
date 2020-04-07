#!/usr/bin/env python

from os import cpu_count
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from copy import deepcopy
import time
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor

from .extensions import get_bboxes, convolute, Delta
from .utils import customMutation, create_deap_toolbox, cxDummy, Logger, parse_time
from .utils import random_sampling, second_sample


class Golem(object):

    def __init__(self, forest_type='dt', ntrees=1, goal='min', nproc=None, random_state=None, verbose=True):
        """

        Parameters
        ----------
        forest_type : str
            Type of forest.
        ntrees : int, str
            Number of trees to use. Use 1 for a single regression tree, or more for a forest. If 1 is selected, the
            choice of `forest_type` will be discarded.
        nproc : int
            Number of processors to use. If not specified, all but one available processors will be used. Each processor
            will process a different tree; therefore there is no benefit in using ``nproc`` > ``ntrees``.
        goal : str
            The optimization goal, "min" for minimization and "max" for maximization. This is used only by the methods
            ``recommend`` and ``get_merit``.
        random_state : int, optional
            Fix random seed
        verbose : bool, optional.
            Whether to print information to screen. If ``False`` only warnings and errors will be displayed.

        Attributes
        ----------
        y_robust : array
            Expectation of the merits under the specified uncertainties.
        std_robust : array
            Standard deviation of the merits under the specified uncertainties.
        forest : object
            ``sklearn`` object for the chosen ensemble regressor.
        """

        # ---------
        # Init vars
        # ---------
        self.X = None
        self._X = None
        self.y = None
        self._y = None

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

        self._ys_robust = None
        self._stds_robust = None
        self.y_robust = None
        self.y_robust_std = None
        self.std_robust = None
        self.std_robust_std = None

        self.param_space = None
        self.forest = None

        # ---------------
        # Store arguments
        # ---------------

        # options for the tree
        self.ntrees = ntrees
        self._ntrees = None
        self.max_depth = None
        self.goal = goal
        self.random_state = random_state
        self.forest_type = forest_type

        # other options
        self.nproc = nproc
        if nproc is None:
            self._nproc = cpu_count() - 1  # leave 1 CPU free
        else:
            self._nproc = nproc

        self.verbose = verbose
        if self.verbose is True:
            self.logger = Logger("Golem", 2)
        elif self.verbose is False:
            self.logger = Logger("Golem", 0)

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

        # determine number of trees and select/initialise model
        self._ntrees = self._parse_ntrees_arg(self.ntrees)
        self._init_forest_model()

        # fit regression tree(s) to the data
        self.forest.fit(self._X, self._y)

        # ----------------------------
        # parse trees to extract tiles
        # ----------------------------
        self._bounds = []
        self._preds = []

        start = time.time()

        # parse all trees, use multiple processors when we have >1 trees
        with ProcessPoolExecutor(max_workers=self._nproc) as executor:
            for _bounds, _preds in executor.map(self._parse_tree, self.forest.estimators_):
                self._bounds.append(_bounds)
                self._preds.append(_preds)

        end = time.time()
        self.logger.log(f'{self._ntrees} tree(s) parsed in %.2f %s' % parse_time(start, end), 'INFO')

    def predict(self, X, distributions):
        """Reweight the measurements to obtain robust merits that depend on the specified uncertainty.

        Parameters
        ----------
        X : np.array, pd.DataFrame
            Array or DataFrame containing the input locations for which to predict their robust merit. Provide the same
            input X you passed to the ``fit`` method if you want to reweight the merit of the samples.
        distributions : array, dict
            Array or dictionary of distribution objects from the ``dists`` module.
        """
        if self.forest is None:
            message = 'Cannot make a prediction before the forest model having been trained - call the "fit" method first'
            self.logger.log(message, 'ERROR')
            return None

        # make sure input dimensions match training
        _X = self._parse_X(X)
        if np.shape(_X)[1] != np.shape(self._X)[1]:
            message = (f'Number of features of the model must match the input. Model n_features is {np.shape(self._X)[1]} '
                       f'and input n_features is {np.shape(_X)[1]}')
            self.logger.log(message, 'ERROR')
            return None

        self.distributions = distributions
        # parse distributions info
        if isinstance(distributions, dict):
            self._distributions = self._parse_distributions_dicts()
        elif isinstance(distributions, list):
            # make sure input dimensions match number of distributions
            if np.shape(_X)[1] != len(distributions):
                message = (
                    f'Number of distributions must match the number of input parameters. Number of distributions '
                    f'is {len(distributions)} and number of inputs is {np.shape(_X)[1]}')
                self.logger.log(message, 'ERROR')
                return None
            self._distributions = self._parse_distributions_lists()
        else:
            raise TypeError("Argument `distributions` needs to be either a list or a dictionary")

        # make sure size of distributions equal input dimensionality
        if len(self._distributions) != np.shape(_X)[1]:
            message = (f'Mismatch between the number of distributions provided ({len(self._distributions)}) and '
                       f'the dimensionality of the input ({np.shape(_X)[1]})')
            self.logger.log(message, 'FATAL')
            raise ValueError(message)

        # -----------------------------------------------------
        # convolute each tree and take the mean robust estimate
        # -----------------------------------------------------
        start = time.time()
        self._ys_robust = []
        self._stds_robust = []

        # define args that will go into convolute
        args = []
        for i, tree in enumerate(self.forest.estimators_):
            args_i = (_X, self._distributions, self._preds[i], self._bounds[i])
            args.append(args_i)

        # perform convolution, use multiple processors when we have >1 trees
        with ProcessPoolExecutor(max_workers=self._nproc) as executor:
            for y_robust, std_robust in executor.map(convolute, *zip(*args)):
                self._ys_robust.append(y_robust)
                self._stds_robust.append(std_robust)

        # log performance
        end = time.time()
        message = f'Convolution of {_X.shape[0]} samples performed in %.2f %s' % parse_time(start, end)
        self.logger.log(message, 'INFO')

        # take the average across all trees
        self.y_robust = np.mean(self._ys_robust, axis=0)  # expectation of the output, E[f(X)]
        self.y_robust_std = np.std(self._ys_robust, axis=0)  # Var[E[f(X)]]
        self.std_robust = np.mean(self._stds_robust, axis=0)  # variance of the output, Var[f(X)]
        self.std_robust_std = np.std(self._stds_robust, axis=0)  # Var[Var[f(X)]]

        return self.y_robust

    def get_merits(self, beta=0, normalize=False):
        """Retrieve the values of the robust merits. If ``beta`` is zero, what is returned is equivalent to the
        attribute ``y_robust``. If ``beta > 0`` then a multi-objective merit is constructed by considering both the
        expectation and standard deviation of the output.

        Parameters
        ----------
        beta : int, optional
            Parameter that tunes the penalty variance, similarly to a upper/lower confidence bound acquisition. Default is
            zero, i.e. no variance penalty. Higher values favour more reproducible results at the expense of total
            output.
        normalize : bool, optional
            Whether to return normalized values between 0 and 1.

        Returns
        -------
        merits : array
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
        merits = self.y_robust - self._beta * self.std_robust

        # return
        if normalize is True:
            return (merits - np.amin(merits)) / (np.amax(merits) - np.amin(merits))
        else:
            return merits

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

    def set_param_space(self, param_space):
        """Define the parameter space (the domain) of the optimization.

        Parameters
        ----------
        param_space : list
            List of dictionaries containing information on each input variable. Each dictionary should contain the key
            "type", which can take the value "continuous", "discrete", or "categorical". Continuous and discrete variables
            should also contain the keys "low" and "high" that set the bounds of the domain. Categorical variables should
            contain the key "categories" with a list of the categories.

        Examples
        --------
        >>> golem = Golem()
        >>> var1 = {"type": "continuous", "low": 1.5, "high": 5.5}
        >>> var2 = {"type": "discrete", "low": 0, "high": 10}
        >>> var3 = {"type": "categorical", "categories": ["red", "blue", "green"]}
        >>> param_space = [var1, var2, var3]
        >>> golem.set_param_space(param_space)
        """

        # perform quality control on input
        self._check_type(param_space, list, 'param_space')
        for param in param_space:
            if 'type' not in param.keys():
                message = f'key "type" is required for all input parameters'
                self.logger.log(message, 'FATAL')
                raise ValueError(message)

            paramtype = param['type']
            if param['type'] in ['continuous', 'discrete']:
                if 'low' not in param.keys():
                    message = f'key "low" is required for input parameter of type "{paramtype}"'
                    self.logger.log(message, 'FATAL')
                    raise ValueError(message)
                if 'high' not in param.keys():
                    message = f'key "high" is required for input parameter of type "{paramtype}"'
                    self.logger.log(message, 'FATAL')
                    raise ValueError(message)
            elif param['type'] == 'categorical':
                if 'categories' not in param.keys():
                    message = f'key "categories" is required for input parameter of type "{paramtype}"'
                    self.logger.log(message, 'FATAL')
                    raise ValueError(message)
            else:
                message = (f'parameter of type "{paramtype}" is not recognized, choose among "continuous", '
                           f'"discrete", or "categorical"')
                self.logger.log(message, 'FATAL')
                raise ValueError(message)

        # all good ==> store param space
        self.param_space = param_space

    def recommend(self, X, y, distributions, xi=0.1, pop_size=1000, ngen=10, cxpb=0.5, mutpb=0.3,
                  verbose=False):
        """Recommend next query location for the robust optimization.

        Parameters
        ----------
        X : array
            Two-dimensional array with the input parameters for all past observations.
        y : array
            Array with the measurements/outputs corresponding for all parameters in ``X``.
        distributions : list
            List of Golem distribution objects representing the uncertainty about the location of the input parameters.
        xi : float
            Trade-off parameter of Expected Improvement criterion. The larger it is the more exploration will be
            favoured.
        pop_size : int
            Population size for the Genetic Algorithm based optimization of the acquisition function.
        ngen : int
            Number of generations to use in the GA optimization of the acquisition function.
        cxpb : float
            Probability of cross-over for the GA.
        mutpb : float
            Probability of mutation for the GA.
        verbose : bool
            Whether to print information about the GA progress.

        Returns
        -------
        X_next : list
            List with suggested parameters for the next location to query.
        """

        # check we have what is needed
        if self.param_space is None:
            message = ('`param_space` has not been defined - please set it via the method `set_param_space`')
            self.logger.log(message, 'FATAL')
            raise ValueError(message)

        # if no samples, random sampling
        if len(y) == 0:
            X_next = random_sampling(self.param_space)
            return X_next
        # if one sample, place second somewhat far from first
        elif len(y) == 1:
            X_next = second_sample(X, self.param_space)
            return X_next

        # check distributions chosen against param_space
        self._check_dists_match_param_space(distributions, self.param_space)

        # import GA tools
        try:
            from deap import base, creator, tools, algorithms
        except ImportError as error:
            message = ('module "deap" is required by Golem for the optimization of the acquisition. '
                       'Install it with "pip install deap"')
            self.logger.log(message, 'FATAL')
            raise

        # fit samples
        self.fit(X, y)

        # print some info but then switch off, otherwise it'll go crazy with messages during the GA opt
        self.logger.log(f'Optimizing acquisition - running GA with population of '
                        f'{pop_size} and {ngen} generations', 'INFO')
        previous_verbosity = self.logger.verbosity
        if self.logger.verbosity > 1:
            self.logger.update_verbosity(1)

        # setup GA with DEAP
        creator.create("FitnessMax", base.Fitness, weights=[1.0])  # we maximise the acquisition
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # make toolbox
        toolbox, attrs_list = create_deap_toolbox(self.param_space)
        toolbox.register("individual", tools.initCycle, creator.Individual, attrs_list, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self._expected_improvement, distributions=distributions, xi=xi)
        toolbox.register("mutate", customMutation, attrs_list=attrs_list, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # mating type depends on how many genes we have
        if np.shape(X)[1] == 1:
            toolbox.register("mate", cxDummy)  # i.e. no crossover
        elif np.shape(X)[1] == 2:
            toolbox.register("mate", tools.cxUniform, indpb=0.5)
        else:
            toolbox.register("mate", tools.cxTwoPoint)

        # run eaSimple
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof,
                            verbose=verbose)

        # now restore logger verbosity
        self.logger.update_verbosity(previous_verbosity)

        X_next = list(hof[0])

        # DEAP cleanup
        del creator.FitnessMax
        del creator.Individual

        return X_next

    def _expected_improvement(self, X, distributions, xi=0.1):

        # make sure we have a 2-dim array
        X = np.array(X)
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)

        # compute quantities needed
        mu_sample = self.predict(self._X, distributions=distributions)
        mu = self.predict(X=X, distributions=distributions)
        sigma = self.y_robust_std

        # pick incumbent
        if self.goal == 'max':
            mu_current_best = np.max(mu_sample)
        elif self.goal == 'min':
            mu_current_best = np.min(mu_sample)
        else:
            message = (f'cannot understand goal "{self.goal}". It should be either "min" or "max". '
                       f'We will assume it is "min"')
            self.logger.log(message, 'ERROR')
            mu_current_best = np.min(mu_sample)

        # avoid zero division by removing sigmas=0
        sigma_orig = sigma  # copy of actual sigmas
        sigma[sigma == 0.0] = 1.

        # compute EI
        if self.goal == 'max':
            imp = mu - mu_current_best - xi
        else:
            imp = mu_current_best - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # if sigma was zero, then EI is also zero
        ei[sigma_orig == 0.0] = 0.0

        return ei

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
            return np.array(X).astype('double')  # cast to double, as we expect double in cython

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
        """Extract information about tree structure and eventually the bounds and predictions for all
        tiles/terminal leaves"""

        # this is only for gradient boosting
        if isinstance(tree, np.ndarray):
            tree = tree[0]

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

        # -------------------------------------------------------------
        # extract bounds of all tiles/leaves and associated predictions
        # -------------------------------------------------------------
        bounds, preds = get_bboxes(self._X, node_indexes, value, leave_id, feature, threshold)

        return bounds, preds

    def _parse_distributions_lists(self):

        dists_list = []  # list of distribution objects

        # we then expect distributions to be lists
        self._check_type(self.distributions, list, name='distributions')

        all_dimensions = range(np.shape(self._X)[1])  # all dimensions in the input

        for dim in all_dimensions:
            dist = self.distributions[dim]
            self._check_data_within_bounds(dist, self._X[:, dim])

            # append dist instance to list of dists
            dists_list.append(dist)

        return np.array(dists_list)

    def _parse_distributions_dicts(self):

        dists_list = []  # each row: dist_type_idx, scale, lower_bound, upper_bound

        # we then expect distributions to be dictionaries
        self._check_type(self.distributions, dict, name='distributions')

        all_columns = list(self._df_X.columns)  # all dimensions in the _df_X dataframe

        for col in all_columns:
            if col in self.distributions.keys():
                dist = self.distributions[col]
                self._check_data_within_bounds(dist, self._df_X.loc[:, col])
                self._warn_if_dist_var_mismatch(col, self._cat_cols, dist)

                # append dist instance to list of dists
                dists_list.append(dist)

            # For all dimensions for which we do not have uncertainty, use Delta
            else:
                dist = Delta()
                dists_list.append(dist)

        return np.array(dists_list)

    def _check_type(self, myobject, mytype, name=''):
        if not isinstance(myobject, mytype):
            message = f'`{name}` is expected to be a {mytype} but it is {myobject}\n'
            self.logger.log(message, 'ERROR')

    def _check_data_within_bounds(self, dist, data):
        if hasattr(dist, 'low_bound'):
            if np.min(data) < dist.low_bound:
                message = (f'Data contains out-of-bound samples: {np.min(data)} is lower than the '
                           f'chosen lower bound ({dist.low_bound}) in {type(dist).__name__}')
                self.logger.log(message, 'ERROR')
        if hasattr(dist, 'high_bound'):
            if np.max(data) > dist.high_bound:
                message = (f'Data contains out-of-bound samples: {np.max(data)} is larger than the '
                           f'chosen upper bound ({dist.high_bound}) in {type(dist).__name__}')
                self.logger.log(message, 'ERROR')

    def _warn_if_dist_var_mismatch(self, col, cat_cols, dist):
        if type(dist).__name__ == 'Categorical':
            if col not in cat_cols:
                message = (f'Variable "{col}" was not identified by Golem as a categorical variable, but you have '
                           f'selected {type(dist).__name__} as its distribution. Verify your input.')
                self.logger.log(message, 'WARNING')
        else:
            if col in cat_cols:
                message = (f'Variable "{col}" was identified by Golem as a categorical variable, but a distribution '
                           f'for continuous variables ("{dist}") was selected for it. Verify your input.')
                self.logger.log(message, 'WARNING')

    def _check_dists_match_param_space(self, distributions, param_space):
        for dist, param in zip(distributions, param_space):
            dist_name = type(dist).__name__
            if param['type'] == 'continuous':
                if dist_name in ['Poisson', 'DiscreteLaplace', 'Categorical', 'FrozenPoisson',
                                 'FrozenDiscreteLaplace', 'FrozenCategorical']:
                    message = f'{dist_name} distribution was chosen for a continuous variable'
                    self.logger.log(message, 'WARNING')
            elif param['type'] == 'discrete':
                if dist_name in ['Normal', 'TruncatedNormal', 'FoldedNormal', 'Uniform', 'TruncatedUniform',
                                 'BoundedUniform', 'Gamma', 'Categorical', 'FrozenNormal', 'FrozenUniform',
                                 'FrozenGamma', 'FrozenCategorical']:
                    message = f'{dist_name} distribution was chosen for a discrete variable'
                    self.logger.log(message, 'WARNING')
            elif param['type'] == 'categorical':
                if dist_name in ['Normal', 'TruncatedNormal', 'FoldedNormal', 'Uniform', 'TruncatedUniform',
                                 'BoundedUniform', 'Gamma', 'Poisson', 'DiscreteLaplace', 'FrozenNormal',
                                 'FrozenUniform', 'FrozenGamma', 'FrozenPoisson', 'FrozenDiscreteLaplace']:
                    message = f'{dist_name} distribution was chosen for a categorical variable'
                    self.logger.log(message, 'WARNING')
