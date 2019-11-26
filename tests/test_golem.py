import pytest
from golem import Golem
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_1d_continuous_0():
    # inputs
    x = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.])
    y = np.array([0., 1., 0., 0.8, 0.8, 0.])

    # test a few input options
    t = Golem(X=x.reshape(-1, 1), y=y, dims=[0], distributions=['gaussian'], scales=[0.2], beta=0)
    expected = np.array([0.24669807, 0.43618458, 0.48359262, 0.56032173, 0.50570125, 0.24209494])
    assert_array_almost_equal(expected, t.y_robust)

    t = Golem(X=x.reshape(-1, 1), y=y, dims=[0], distributions=['uniform'], scales=[0.15], beta=0)
    expected = y
    assert_array_almost_equal(expected, t.y_robust)

    t = Golem(X=x.reshape(-1, 1), y=y, dims=[0], distributions=['uniform'], scales=[0.4], beta=0)
    expected = np.array([0.25, 0.50000001, 0.44999998, 0.59999997, 0.60000001, 0.20000001])
    assert_array_almost_equal(expected, t.y_robust)


def test_1d_continuous_1():
    # inputs
    x = np.array([0.0, 0.05263158, 0.10526316, 0.15789474, 0.21052632,
                  0.26315789, 0.31578947, 0.36842105, 0.42105263, 0.47368421,
                  0.52631579, 0.57894737, 0.63157895, 0.68421053, 0.73684211,
                  0.78947368, 0.84210526, 0.89473684, 0.94736842, 1.0])
    y = np.array([0.  , 0.  , 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.  , 0.  , 0.  ,
                  0.  , 1.  , 0.5 , 1.  , 0.5 , 0.5 , 1.  , 0.  , 0.  ])

    # test a few input options
    t = Golem(X=x.reshape(-1, 1), y=y, dims=[0], distributions=['gaussian'], scales=[0.1], beta=0)
    expected = np.array([0.16115857, 0.29692691, 0.45141613, 0.58211272, 0.65490033,
                         0.65515297, 0.58387391, 0.45953573, 0.32598349, 0.24245312,
                         0.25046356, 0.34463613, 0.47534375, 0.58269396, 0.63168726,
                         0.61834583, 0.55102012, 0.43894327, 0.30129135, 0.17149024])
    assert_array_almost_equal(expected, t.y_robust)

    t = Golem(X=x.reshape(-1, 1), y=y, dims=[0], distributions=['gaussian'], scales=[0.1], beta=1)
    expected = np.array([-0.14689474, -0.06985585,  0.08428458,  0.26949584,  0.40531932,
                          0.40574873,  0.27183403,  0.091973  , -0.05302822, -0.12763612,
                         -0.14015192, -0.08176542,  0.04979055,  0.20142471,  0.29739284,
                          0.28884459,  0.18382555,  0.03298808, -0.10270032, -0.17547864])
    assert_array_almost_equal(expected, t.y_robust)

    t = Golem(X=x.reshape(-1, 1), y=y, dims=[0], distributions=['uniform'], scales=[0.2], beta=1)
    expected = np.array([-0.15122178, -0.0854665 ,  0.11190192,  0.44088347,  0.75      ,
                          0.75      ,  0.44088347,  0.11190192, -0.0854665 , -0.15122178,
                         -0.20162907, -0.11981431,  0.07080252,  0.3735199 ,  0.44302233,
                          0.44302235,  0.3136041 ,  0.05350977, -0.11981434, -0.20162905])
    assert_array_almost_equal(expected, t.y_robust)


def test_2d_continuous():
    pass