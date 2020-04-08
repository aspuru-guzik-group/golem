# Golem
[![Build Status](https://travis-ci.com/matteoaldeghi/golem.svg?token=bMWWqBdm3xytautMLsPK&branch=master)](https://travis-ci.com/matteoaldeghi/golem)
[![codecov](https://codecov.io/gh/matteoaldeghi/golem/branch/master/graph/badge.svg?token=JJOHSUa4zX)](https://codecov.io/gh/matteoaldeghi/golem)


### TODO list
- [x] categorical variables
- [x] better dealing of input dimensions with no uncertainty
- [x] expand set of uncertainty distributions available
- [x] allow passing DataFrame and column names as input
- [x] allow "freezing" a distribution for application to rubustness against whole variable
- [x] add discrete distributions (Poisson and discrete Laplace)
- [x] add EI criterion to allow simple tree-based optimization directly via Golem?
- [x] allow passing custom cdfs

### TODO tests
- [ ] tests for frozen distributions
- [ ] tests for discrete distributions

### Known issues/useful infos
- Multiprocessing does not work on Jupyter (use nproc=1)
- To use categorical variables in method fit, withouth having defined a param_space, the argument X need to be a pd.DataFrame
