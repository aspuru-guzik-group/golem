# Golem: An algorithm for robust experiment and process optimization
[![Build Status](https://travis-ci.com/matteoaldeghi/golem.svg?token=bMWWqBdm3xytautMLsPK&branch=master)](https://travis-ci.com/matteoaldeghi/golem)
[![codecov](https://codecov.io/gh/matteoaldeghi/golem/branch/master/graph/badge.svg?token=JJOHSUa4zX)](https://codecov.io/gh/matteoaldeghi/golem)

``Golem`` is an algorithm for robust optimization. It can be used in conjunction with any optimization algorithms or
design of experiment strategy of choice. ``Golem`` helps identifying optimal solutions that are robust to input uncertainty, 
thus ensuring the reproducible performance of optimized experimental protocols and processes. It can be used to analyze 
the robustness of past experiments, or to guide experiment planning algorithms toward robust solutions on the fly. For 
more details on the algorithm and its behaviour please refer to [this publication](https://).

You can find more details in the [documentation](https://).

###  Installation
``Golem`` can be installed with ``pip``:

```
pip install matter-golem
```

### Dependencies
The installation requires:
* ``python >= 3.6``
* ``cython``
* ``numpy``
* ``scipy >= 1.4``
* ``pandas``
* ``scikit-learn``

###  Citation
``Golem`` is research software. If you make use of it in scientific publications, please cite the following article:

```
@misc{golem,
      title={Golem: An algorithm for robust experiment and process optimization}, 
      author={Matteo Aldeghi and Florian Häse and Riley J. Hickman and Isaac Tamblyn and Alán Aspuru-Guzik},
      year={2021},
      eprint={arXiv},
      archivePrefix={},
      primaryClass={stat.ML}
```

###  License
``Golem`` is distributed under an MIT License.
