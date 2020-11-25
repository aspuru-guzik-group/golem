.. _distributions:

Distributions
=============

The ``predict`` method of **Golem** requires to specify the type of input uncertainty via a list of probability
distributions::

    golem.predict(X=X, distributions=[Normal(std=0.1), Uniform(urange=0.5)])

Below are the distributions implemented and available in **Golem**. Note that, in addition to the distributions below,
the :ref:`delta` function can be used to indicate variables with no uncertainty::

    golem.predict(X=X, distributions=[Delta(), Uniform(urange=0.5)])


Continuous Distributions
------------------------

.. toctree::
   :maxdepth: 1

   uniform
   truncated_uniform
   bounded_uniform
   normal
   truncated_normal
   folded_normal
   gamma



Discrete and Categorical Distributions
--------------------------------------

.. toctree::
   :maxdepth: 1

   poisson
   discrete_laplace
   categorical


Frozen Distributions
--------------------
These are distributions that do not change location depending on the input sample. Contrary to the above classes,
in the "frozen" distributions the location is fixed when instantiating the class. This may be useful when the input
uncertainty is not due to a control factor one can influence, but it is caused by an environmental factor causing
uncertainty in the conditions.

.. toctree::
   :maxdepth: 1

   frozen_uniform
   frozen_normal
   frozen_gamma
   frozen_poisson
   frozen_discrete_laplace
   frozen_categorical


Custom Distributions
--------------------
If you would like to model input uncertainty using a distribution not available in **Golem**, you can provide any
user-defined distribution as shown in the `Custom Distributions`_ example.

.. _Custom Distributions: ../examples/notebooks/custom_distribution.ipynb