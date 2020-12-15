.. _golemclass:

Golem Class
===========

The main class ``Golem`` allows to estimate the robust merits of a set of parameters, as well as the
robust objective function, based on a set of observations/samples as the training set. This is achieved via an interface
similar to that of ``scikit-learn``, where the two main methods are ``fit`` and ``predict``.

First, we instantiate the ``Golem`` class::

    golem = Golem(ntrees=1, goal='min', nproc=1)

Assuming we have a set of parameters ``X`` and their corresponding objective function evaluations ``y``, we can fit the
tree-based model used by ``Golem``::

    golem.fit(X, y)

We can now use ``Golem`` to estimate the robust merits for any set of input parameters ``X_pred``, given known/assumed
probability distributions representing the uncertainty of each input variable. For instance, if we have a 2-dimensional
input space, where the first variable has normally-distributed uncertainty, and the second one has uniform uncertainty::

    golem.predict(X_pred, distributions=[Normal(0.1), Uniform(0.5)])

For a complete example on how to use the ``Golem`` class, see the `Basic Usage`_ example.

API Reference
-------------
.. currentmodule:: golem

.. autoclass:: Golem
   :noindex:
   :members: y_robust
   :exclude-members:


   .. rubric:: Methods

   .. autosummary::
      fit
      predict
      get_merits
      get_tiles
      set_param_space
      recommend


.. _Basic Usage: examples/notebooks/basic_usage.ipynb

