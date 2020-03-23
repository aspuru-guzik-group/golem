#!/usr/bin/env python

import numpy as np


def gauss(x, loc, scale):
    pass


def truncated_gauss(x, loc, scale, low_bound, high_bound):
    pass


def folded_gauss(x, loc, scale, low_bound, high_bound):
    pass


def uniform(x, loc, scale):
    pass


def truncated_uniform(x, loc, scale, low_bound, high_bound):
    pass


def bounded_uniform(x, loc, scale, low_bound, high_bound):
    pass


def gamma(x, loc, scale, low_bound, high_bound):
    pass


def poisson(x, loc, low_bound):
    pass


def discrete_laplace(x, loc=0, sigma=1):
    p = np.exp(-1 / sigma)
    return (1 - p) / (1 + p) * (p ** np.abs(x-loc))


def categorical():
    pass