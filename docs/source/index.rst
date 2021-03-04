Golem: a probabilistic approach to robust optimization
======================================================

.. image:: https://travis-ci.com/matteoaldeghi/golem.svg?token=bMWWqBdm3xytautMLsPK&branch=master
    :target: https://travis-ci.com/matteoaldeghi/golem
.. image:: https://codecov.io/gh/matteoaldeghi/golem/branch/master/graph/badge.svg?token=JJOHSUa4zX
    :target: https://codecov.io/gh/matteoaldeghi/golem

**Golem** is a Python tool that allows to compute the expectation and variance of a black-box objective function
based on the specified uncertainty of the input variables. It can thus be used to see how different levels of input
uncertainty might affect the location of the optimum, or it can be used in conjunction with optimization algorithms
to enable robust optimization.

At the basis of the algorithm is the use of supervised tree-based models, such as regression trees and random forests.
Please refer to the publication for the details of the approach.

.. toctree::
   :maxdepth: 1
   :caption: User documentation

   install
   golem_class
   distributions/index
   examples/index


Citation
--------
If you use **Golem** in scientific publications, please cite the following paper:

* M. Aldeghi, F. Häse, R.J. Hickman, I. Tamblyn, A. Aspuru-Guzik. `Golem: An algorithm for robust experiment and process optimization <http://target>`_. *arXiv* (2021), 2010.04153

::

    @misc{golem,
      title={Golem: An algorithm for robust experiment and process optimization},
      author={Matteo Aldeghi and Florian Häse and Riley J. Hickman and Isaac Tamblyn and Alán Aspuru-Guzik},
      year={2021},
      eprint={arXiv},
      archivePrefix={},
      primaryClass={stat.ML}


License
-------
**Golem** is distributed under an MIT License.
