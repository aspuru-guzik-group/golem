#!/usr/bin/env python

import numpy as np
import time
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from copy import deepcopy
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from colossus import Colossus


def test_case_0(N, seed=42, noise_scale0=1, noise_scale1=1, noise_scale2=1):
    np.random.seed(seed)
    NUM_SAMPLES = N

    domain_samples_0 = np.random.normal(0, 0.5, 5 * NUM_SAMPLES)
    domain_samples_1 = np.random.normal(7, 1., 5 * NUM_SAMPLES)
    domain_samples_2 = np.random.normal(-4, 0.5, NUM_SAMPLES)
    obs_domain = np.concatenate([domain_samples_0, domain_samples_1, domain_samples_2])

    values_samples_0 = np.random.normal(5, 0.25 * noise_scale1, 5 * NUM_SAMPLES)
    values_samples_1 = np.random.normal(3, 0.01 * noise_scale2, 5 * NUM_SAMPLES)
    values_samples_2 = np.random.normal(0, 0.01 * noise_scale0, NUM_SAMPLES)
    obs_values = np.concatenate([values_samples_0, values_samples_1, values_samples_2])

    obs_domain = (obs_domain - np.amin(obs_domain)) / (np.amax(obs_domain) - np.amin(obs_domain))
    obs_values = (obs_values - np.amin(obs_values)) / (np.amax(obs_values) - np.amin(obs_values))

    return obs_domain, obs_values


def test_case_1(N, seed=42, noise=None, sparse=False):
    '''
    N : int
        choose number of observed points
    seed : int
        random seed for numpy
    noise : float, optional
        std dev of gaussian noise to be added to y (obs_values)
    sparse : bool
        if True, the lower and broader peak will have a lower density of points.
        Since a few points are dropped, N will be smaller than requested.
    '''
    if sparse is True:
        x1 = np.linspace(0, 0.5, int(round(N * 0.5, 0)), endpoint=False)
        x2 = np.linspace(0.5, 0.9, int(round(N * 0.4 / 3, 0)), endpoint=False)
        x3 = np.linspace(0.9, 1, int(round(N * 0.1 + 1, 0)))
        x = np.concatenate([x1, x2, x3])
    else:
        x = np.linspace(0, 1, N)

    y = []
    for i in x:
        if i >= 0.1 and i < 0.25:
            y.append(1)
        elif i >= 0.5 and i < 0.9:
            y.append(0.8)
        else:
            y.append(0)

    y = np.array(y)

    # add noise if needed
    if noise is not None:
        np.random.seed(seed)
        y += np.random.normal(loc=0.0, scale=noise, size=len(x))
        # rescale
        y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))

    return x, y


def test_case_2(N, seed=42, noise=None):
    '''
    N : int
        choose number of observed points
    seed : int
        random seed for numpy
    noise : float, optional
        std dev of gaussian noise to be added to y (obs_values)
    '''
    x = np.linspace(0, 1, N)

    y = []
    for i in x:
        if i >= 0.1 and i < 0.4:
            y.append(0.75)
        elif i >= 0.6 and i < 0.9:
            if i >= 0.7 and i < 0.775:
                y.append(1)
            elif i < 0.65 or i > 0.85:
                y.append(1)
            else:
                y.append(0.5)
        else:
            y.append(0)

    y = np.array(y)

    # add noise if needed
    if noise is not None:
        np.random.seed(seed)
        y += np.random.normal(loc=0.0, scale=noise, size=len(x))
        # rescale
        y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))

    return x, y

def f_poly(x,y):
    '''max at (2.8,4)
    (0.9047619,0.91836735) if scaled to unit cube
    '''
    f = -(2*(x**6) -
         12.2*(x**5) +
         21.2*(x**4) +
         6.2*x -
         6.4*(x**3) -
         4.7*(x**2) +
         y**6 -
         11*(y**5) +
         43.3*(y**4) -
         10*y -
         74.8*(y**3) +
         56.9*(y**2) -
         4.1*x*y -
         0.1*(y**2)*(x**2) +
         0.4*(y**2)*x +
         0.4*(x**2)*y)
    return f



# data
x0_bounds = [-1, 3.2]
x1_bounds = [-0.5,4.4]
x0 = np.linspace(x0_bounds[0], x0_bounds[1], 20)
x1 = np.linspace(x1_bounds[0], x1_bounds[1], 20)
X0, X1 = np.meshgrid(x0, x1)
Y = f_poly(X0, X1)

# rescale on unit cube
X0 = (X0-np.amin(X0)) / (np.amax(X0)-np.amin(X0))
X1 = (X1-np.amin(X1)) / (np.amax(X1)-np.amin(X1))
Y = (Y-np.amin(Y)) / (np.amax(Y)-np.amin(Y))

# plot
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
contours = ax1.contour(X0, X1, Y, 8, colors='k')
_ = ax1.clabel(contours, inline=True, fontsize=9, fmt='%.2f')
_ = ax1.imshow(Y, extent=[0,1,0,1], origin='lower', cmap='RdBu_r', alpha=0.5)
#_ = plt.colorbar()
_ = ax1.set_title("Max at 0.90,0.92")

# put into "sklearn" format
Xy = pd.DataFrame({'x0':X0.flatten(), 'x1':X1.flatten(), 'y':Y.flatten()})
X = Xy.iloc[:,:-1]
y = Xy.iloc[:,-1:]
Xy

#scales = [0.2, 0.2]
#t = Colossus(X=X, y=y, max_depth=100, distributions=[stats.norm, stats.norm], scales=scales, beta=0)

distributions = {
                0: {'distribution': stats.norm,
                    'params': {'scale': 0.2}},
                1: {'distribution': stats.norm,
                    'params': {'scale': 0.2}}
                }

t = Colossus(X=X, y=y, max_depth=None, distributions=distributions, beta=0)
newy = np.reshape(t.y_robust_scaled, newshape=np.shape(X0))

# plot
#fig, ax = plt.subplots(1,1)
contours = ax2.contour(X0, X1, newy, 8, colors='k')
_ = ax2.clabel(contours, inline=True, fontsize=9, fmt='%.2f')
_ = ax2.imshow(newy, extent=[0,1,0,1], origin='lower', cmap='RdBu_r', alpha=0.5)
#_ = plt.colorbar()
_ = ax2.set_title("Max at: %s"  % np.array(X)[np.argmax(t.y_robust_scaled)])

plt.show()