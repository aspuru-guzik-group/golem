.. _distributions:

Distributions
=============

List of uncertainty distributions available in Golem and how to use them...

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
in the "frozen" distributions the location is fixed when instantiating the class.

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
How to use a user-defined distribution...