#!/usr/bin/env python

import numpy as np
import sys
import traceback
import pandas as pd


def customMutation(individual, attrs_list, indpb=0.2, continuous_scale=0.1, discrete_scale=0.1):
    """Mutation

    Parameters
    ----------
    indpb : float
        Independent probability for each attribute to be mutated.
    """

    assert len(individual) == len(attrs_list)

    for i, attr in enumerate(attrs_list):

        # determine whether we are performing a mutation
        if np.random.random() < indpb:
            vartype = attr.__name__

            if "continuous" in vartype:
                # Gaussian perturbation with scale being 0.1 of domain range
                bound_low = attr.args[0]
                bound_high = attr.args[1]
                scale = (bound_high - bound_low) * continuous_scale
                individual[i] += np.random.normal(loc=0.0, scale=scale)
                individual[i] = _project_bounds(individual[i], bound_low, bound_high)
            elif "discrete" in vartype:
                # add/substract an integer by rounding Gaussian perturbation
                # scale is 0.1 of domain range
                bound_low = attr.args[0]
                bound_high = attr.args[1]
                scale = (bound_high - bound_low) * discrete_scale
                delta = np.random.normal(loc=0.0, scale=scale)
                individual[i] += np.round(delta, decimals=0)
                individual[i] = _project_bounds(individual[i], bound_low, bound_high)
            elif "categorical" in vartype:
                # resample a random category
                individual[i] = attr()
            else:
                raise ValueError()
        else:
            continue

    return individual,


def cxDummy(ind1, ind2):
    """Dummy crossover that does nothing. This is used when we have a single gene in the chromosomes, such that
    crossover would not change the population.
    """
    return ind1, ind2


def create_deap_toolbox(param_space):
    from deap import base

    toolbox = base.Toolbox()
    attrs_list = []

    for i, param in enumerate(param_space):
        vartype = param['type']

        if vartype in 'continuous':
            toolbox.register(f"x{i}_{vartype}", np.random.uniform, param['low'], param['high'])

        elif vartype in 'discrete':
            toolbox.register(f"x{i}_{vartype}", np.random.randint, param['low'], param['high'])

        elif vartype in 'categorical':
            toolbox.register(f"x{i}_{vartype}", np.random.choice, param['categories'])

        attr = getattr(toolbox, f"x{i}_{vartype}")
        attrs_list.append(attr)

    return toolbox, attrs_list


def _project_bounds(x, x_low, x_high):
    if x < x_low:
        return x_low
    elif x > x_high:
        return x_high
    else:
        return x


def random_sampling(param_space):
    X_next = []
    for param in param_space:
        vartype = param['type']
        if vartype in 'continuous':
            x = np.random.uniform(low=param['low'], high=param['high'])
        elif vartype in 'discrete':
            x = np.random.randint(low=param['low'], high=param['high'])
        elif vartype in 'categorical':
            x = np.random.choice(param['categories'])
        X_next.append(x)
    return X_next


def second_sample(X, param_space):
    """Rule to generate second sample"""

    if len(np.shape(X)) > 1:
        # remove one dimension
        if isinstance(X, list) or isinstance(X, np.ndarray):
            X = X[0]
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()[0]
        else:
            raise NotImplementedError

    X = list(X)

    X_next = []
    for xi, param in zip(X, param_space):
        vartype = param['type']
        if vartype in 'continuous':
            if (xi - param['low']) > (param['high'] - xi):
                x = xi - (xi - param['low']) / 2.
            else:
                x = xi + (param['high'] - xi) / 2.
        elif vartype in 'discrete':
            if (xi - param['low']) > (param['high'] - xi):
                x = int(xi - (xi - param['low']) / 2.)
            else:
                x = int(xi + (param['high'] - xi) / 2.)
        elif vartype in 'categorical':
            x = np.random.choice(param['categories'])
        X_next.append(x)
    return X_next


def parse_time(start, end):
    elapsed = end - start  # elapsed time in seconds
    if elapsed < 1.0:
        return elapsed * 1000., 'ms'
    else:
        return elapsed, 's'


class Logger:

    VERBOSITY_LEVELS = {0: ['ERROR', 'FATAL'],
                        1: ['WARNING', 'ERROR', 'FATAL'],
                        2: ['INFO', 'WARNING', 'ERROR', 'FATAL'],
                        3: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL']}

    WRITER = {'DEBUG': sys.stdout, 'INFO': sys.stdout,
              'WARNING': sys.stderr, 'ERROR': sys.stderr, 'FATAL': sys.stderr}

    GREY = '0;37'
    WHITE = '1;37'
    YELLOW = '1;33',
    LIGHT_RED = '1;31',
    RED = '0;31'

    COLORS = {'DEBUG': WHITE, 'INFO': GREY, 'WARNING': YELLOW, 'ERROR': LIGHT_RED, 'FATAL': RED}

    def __init__(self, template, verbosity=0):
        self.template = template
        if isinstance(verbosity, dict):
            verbosity = verbosity['default']
        self.verbosity = verbosity
        self.verbosity_levels = self.VERBOSITY_LEVELS[self.verbosity]

    def update_verbosity(self, verbosity=0):
        if isinstance(verbosity, dict):
            verbosity = verbosity['default']
        self.verbosity = verbosity
        self.verbosity_levels = self.VERBOSITY_LEVELS[self.verbosity]

    def log(self, message, message_type):

        # check if we need to log the message
        if message_type in self.verbosity_levels:
            color = self.COLORS[message_type]
            error_message = None
            if message_type in ['WARNING', 'ERROR', 'FATAL']:
                error_message = traceback.format_exc()
                if not 'NoneType: None' in error_message:
                    self.WRITER[message_type].write(error_message)
            uncolored_message = '[%s] %s ... %s ...\n' % (message_type, self.template, message)
            message = "\x1b[%sm" % (color) + uncolored_message + "\x1b[0m"
            self.WRITER[message_type].write(message)
            return error_message, uncolored_message
