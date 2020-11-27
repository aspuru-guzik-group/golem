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

* M. Aldeghi, F. Häse, A. Aspuru-Guzik. `Title <http://target>`_. *Journal* **Issue** (Year), pp-pp

::

    @article{golem,
        title = {title},
        author = {authors},
        journal = {journal},
        volume = {vol},
        number = {num},
        pages = {pp--pp},
        year = {year},
        doi = {doi}
    }


License
-------
**Golem** is distributed under an MIT License.
