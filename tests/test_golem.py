import pytest
from golem import Golem
from golem import Delta, Normal, TruncatedNormal, FoldedNormal
from golem import Uniform, TruncatedUniform, BoundedUniform
from golem import Gamma, DiscreteLaplace, Categorical
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal


def test_1d_continuous_0():
    # inputs
    x = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.]).reshape(-1,1)
    y = np.array([0., 1., 0., 0.8, 0.8, 0.])

    g = Golem(forest_type='dt', ntrees=1, random_state=42, verbose=True)
    g.fit(X=x, y=y)

    # -----------------------
    # Unbounded distributions
    # -----------------------
    y_robust = g.predict(X=x, distributions=[Normal(std=0.2)])
    expected = np.array([0.24669807, 0.43618458, 0.48359262, 0.56032173, 0.50570125, 0.24209494])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[Uniform(urange=0.15)])
    expected = y
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[Uniform(urange=0.4)])
    expected = np.array([0.25, 0.50000001, 0.44999998, 0.59999997, 0.60000001, 0.20000001])
    assert_array_almost_equal(expected, y_robust)

    # ---------------------
    # Bounded distributions
    # ---------------------
    y_robust = g.predict(X=x, distributions=[TruncatedNormal(std=0.2, low_bound=0, high_bound=1)])
    expected = np.array([0.49339614, 0.51845691, 0.49553503, 0.57415898, 0.60108568, 0.48418987])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[FoldedNormal(std=0.2, low_bound=0, high_bound=1)])
    expected = np.array([0.49339614, 0.49696822, 0.48975576, 0.56552209, 0.55896091, 0.48418987])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[BoundedUniform(urange=0.4, low_bound=0, high_bound=1)])
    expected = np.array([0.50000001, 0.50000001, 0.44999998, 0.59999997, 0.60000001, 0.60000001])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[TruncatedUniform(urange=0.4, low_bound=0, high_bound=1)])
    expected = np.array([0.49999999, 0.50000001, 0.44999998, 0.59999997, 0.60000001, 0.40000002])
    assert_array_almost_equal(expected, y_robust)

    # ----------------------------------------
    # Bounded distributions: only lower bounds
    # ----------------------------------------
    y_robust = g.predict(X=x, distributions=[TruncatedNormal(std=0.2, low_bound=0)])
    expected = np.array([0.49339614, 0.51843739, 0.49485054, 0.56107913, 0.50571726, 0.24209494])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[FoldedNormal(std=0.2, low_bound=0)])
    expected = np.array([0.49339614, 0.49696822, 0.48956966, 0.56055436, 0.50570125, 0.24209494])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[BoundedUniform(urange=0.4, low_bound=0)])
    expected = np.array([0.50000001, 0.50000001, 0.44999998, 0.59999997, 0.60000001, 0.20000001])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[TruncatedUniform(urange=0.4, low_bound=0)])
    expected = np.array([0.49999999, 0.50000001, 0.44999998, 0.59999997, 0.60000001, 0.20000001])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[Gamma(std=0.2, low_bound=0)])
    expected = np.array([0.4401813, 0.57280083, 0.47757399, 0.54286822, 0.48623131, 0.20307416])
    assert_array_almost_equal(expected, y_robust)

    # ----------------------------------------
    # Bounded distributions: only upper bounds
    # ----------------------------------------
    y_robust = g.predict(X=x, distributions=[TruncatedNormal(std=0.2, high_bound=1)])
    expected = np.array([0.24669807, 0.43619839, 0.48424631, 0.57336588, 0.60106306, 0.48418987])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[FoldedNormal(std=0.2, high_bound=1)])
    expected = np.array([0.24669807, 0.43618458, 0.48377873, 0.56528946, 0.55896091, 0.48418987])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[BoundedUniform(urange=0.4, high_bound=1)])
    expected = np.array([0.25,       0.50000001, 0.44999998, 0.59999997, 0.60000001, 0.60000001])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[TruncatedUniform(urange=0.4, high_bound=1)])
    expected = np.array([0.25,       0.50000001, 0.44999998, 0.59999997, 0.60000001, 0.40000002])
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=x, distributions=[Gamma(std=0.2, high_bound=1)])
    expected = np.array([0.22292533, 0.41057959, 0.43963827, 0.57149698, 0.63397455, 0.43864493])
    assert_array_almost_equal(expected, y_robust)


def test_1d_continuous_1():
    # inputs
    x = np.array([0.0, 0.05263158, 0.10526316, 0.15789474, 0.21052632,
                  0.26315789, 0.31578947, 0.36842105, 0.42105263, 0.47368421,
                  0.52631579, 0.57894737, 0.63157895, 0.68421053, 0.73684211,
                  0.78947368, 0.84210526, 0.89473684, 0.94736842, 1.0]).reshape(-1, 1)
    y = np.array([0.  , 0.  , 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.  , 0.  , 0.  ,
                  0.  , 1.  , 0.5 , 1.  , 0.5 , 0.5 , 1.  , 0.  , 0.  ])

    # test a few input options
    g = Golem(forest_type='dt', ntrees=1, goal='max', random_state=42, verbose=True)
    g.fit(X=x, y=y)

    g.predict(X=x, distributions=[Normal(std=0.1)])
    y_robust = g.get_merits(beta=0)
    expected = np.array([0.16115857, 0.29692691, 0.45141613, 0.58211272, 0.65490033,
                         0.65515297, 0.58387391, 0.45953573, 0.32598349, 0.24245312,
                         0.25046356, 0.34463613, 0.47534375, 0.58269396, 0.63168726,
                         0.61834583, 0.55102012, 0.43894327, 0.30129135, 0.17149024])
    assert_array_almost_equal(expected, y_robust)

    g.predict(X=x, distributions=[Normal(std=0.1)])
    y_robust = g.get_merits(beta=1)
    expected = np.array([-0.14689474, -0.06985585,  0.08428458,  0.26949584,  0.40531932,
                          0.40574873,  0.27183403,  0.091973  , -0.05302822, -0.12763612,
                         -0.14015192, -0.08176542,  0.04979055,  0.20142471,  0.29739284,
                          0.28884459,  0.18382555,  0.03298808, -0.10270032, -0.17547864])
    assert_array_almost_equal(expected, y_robust)

    g.predict(X=x, distributions=[Uniform(urange=0.2)])
    y_robust = g.get_merits(beta=1)
    expected = np.array([-0.15122178, -0.0854665 ,  0.11190192,  0.44088347,  0.75      ,
                          0.75      ,  0.44088347,  0.11190192, -0.0854665 , -0.15122178,
                         -0.20162907, -0.11981431,  0.07080252,  0.3735199 ,  0.44302233,
                          0.44302235,  0.3136041 ,  0.05350977, -0.11981434, -0.20162905])
    assert_array_almost_equal(expected, y_robust)


def test_2d_continuous_0():
    def bertsimas(x, y):
        f = (2 * (x ** 6) -
             12.2 * (x ** 5) +
             21.2 * (x ** 4) +
             6.2 * x -
             6.4 * (x ** 3) -
             4.7 * (x ** 2) +
             y ** 6 -
             11 * (y ** 5) +
             43.3 * (y ** 4) -
             10 * y -
             74.8 * (y ** 3) +
             56.9 * (y ** 2) -
             4.1 * x * y -
             0.1 * (y ** 2) * (x ** 2) +
             0.4 * (y ** 2) * x +
             0.4 * (x ** 2) * y)
        return f

    # bounds
    xlims = [-1, 3.2]
    ylims = [-0.5, 4.4]

    # 2D input
    x0 = np.linspace(xlims[0], xlims[1], 8)
    x1 = np.linspace(ylims[0], ylims[1], 8)
    X0, X1 = np.meshgrid(x0, x1)
    Y = bertsimas(X0, X1)

    # reshape
    X = np.concatenate([X0.flatten().reshape(-1,1), X1.flatten().reshape(-1,1)], axis=1)
    y = Y.flatten()

    # ------------------------
    # test a few input options
    # ------------------------
    g = Golem(forest_type='dt', ntrees=1, goal='max', random_state=42, verbose=True)
    g.fit(X=X, y=y)

    g.predict(X=X, distributions=[Normal(std=0.8), Normal(std=0.8)])
    y_robust = g.get_merits(beta=0)
    expected = np.array([40.70398764, 32.65853133, 28.32176438, 29.50136531, 33.78318761,
       38.89942994, 45.21695841, 52.69336383, 32.9453956 , 24.17391611,
       18.91310959, 19.1464824 , 22.56245141, 26.93182147, 32.65418315,
       39.72785478, 30.63073356, 20.96347974, 14.55201802, 13.58952519,
       15.88786976, 19.26768862, 24.17974726, 30.69221847, 34.91764764,
       24.41112272, 16.90338347, 14.7715317 , 15.93777386, 18.27322435,
       22.29519952, 28.17109307, 40.93976156, 29.74423589, 21.31150865,
       18.15248179, 18.27262877, 19.58970671, 22.70121525, 27.90238289,
       44.18203674, 32.46462821, 23.30082043, 19.28191895, 18.46755727,
       18.81715986, 21.02008144, 25.52487339, 46.27471965, 34.20280293,
       24.51113795, 19.82415779, 18.22914323, 17.72032607, 19.08138919,
       22.92300264, 50.84378107, 38.57380311, 28.56558833, 23.44719409,
       21.3150151 , 20.18723393, 20.92235044, 24.26175858])
    assert_array_almost_equal(expected, y_robust)

    g.predict(X=X, distributions=[Uniform(urange=1.5), Uniform(urange=1.5)])
    y_robust = g.get_merits(beta=0)
    expected = np.array([42.8954175 , 31.2560375 , 24.30410955, 29.80189359, 36.76967766,
       39.45322131, 46.65495774, 55.77870765, 31.24932108, 18.67162188,
       10.45930192, 14.83680595, 20.824422  , 22.66790962, 29.16970204,
       37.85953597, 28.38026605, 14.65101084,  4.8771309 ,  7.83923491,
       12.55761095, 13.27801855, 18.80289096, 26.86734492, 36.24564109,
       21.56592589, 10.47744595, 12.20054997, 15.75552599, 15.38813356,
       19.90080596, 27.28695996, 45.48840621, 30.05932699, 17.90320705,
       18.56371105, 21.06112704, 19.64121457, 23.10640694, 29.76134096,
       47.15488941, 31.1775422 , 18.20074228, 17.9750463 , 19.5207423 ,
       17.08358982, 19.46602221, 25.33681628, 46.81266511, 30.48814592,
       16.93762604, 16.00213009, 16.70194613, 13.28283368, 14.54722613,
       19.58096029, 53.30018781, 36.81492061, 22.95892072, 21.58522476,
       21.71412078, 17.59136829, 18.01940071, 22.41459488])
    assert_array_almost_equal(expected, y_robust)

    g.predict(X=X, distributions=[Uniform(urange=1.5), Normal(std=0.8)])
    y_robust = g.get_merits(beta=0)
    expected = np.array([41.81959333, 29.96985125, 22.73388651, 27.97647458, 34.71790352,
       37.20393286, 44.2369958 , 53.25421061, 34.30397883, 21.63783439,
       13.30066228, 17.55593597, 23.42394351, 25.15044444, 31.53787198,
       40.14869923, 32.28977257, 18.62228969,  8.92198751, 11.93199958,
       16.67261394, 17.38959014, 22.88536143, 30.91917672, 36.85934086,
       22.26392567, 11.27881033, 13.07541696, 16.6740336 , 16.3204198 ,
       20.81700885, 28.1799597 , 43.11507999, 27.77194562, 15.72153372,
       16.45787272, 19.00125065, 17.59742708, 21.0488353 , 27.68217119,
       46.53623873, 30.64388165, 17.77373856, 17.62887353, 19.22957461,
       16.82160133, 19.20738706, 25.07019811, 48.75240951, 32.50455606,
       19.06442229, 18.24082378, 19.05404859, 15.74985623, 17.13068011,
       22.24652079, 53.39183038, 36.957542  , 23.18984506, 21.93153518,
       22.2029004 , 18.24970023, 18.8743681 , 23.41804032])
    assert_array_almost_equal(expected, y_robust)

    g.predict(X=X, distributions=[Uniform(urange=1.5), Delta()])
    y_robust = g.get_merits(beta=0)
    expected = np.array([50.69535694, 39.41668414, 32.94702818, 38.86929222, 46.2037643 ,
       49.19620395, 56.64904438, 65.92545028, 21.44558393,  8.81425914,
        0.53608319,  4.86654722, 10.82593928, 12.66001891, 19.17121934,
       27.87516527, 28.95982493, 15.17694413,  5.3372082 ,  8.25227222,
       12.94242427, 13.65342387, 19.18770429, 27.26627026, 34.30071993,
       19.56737913,  8.4130432 , 10.08910723, 13.61585925, 13.23905883,
       17.76113925, 25.16140524, 46.93506892, 31.45236412, 19.23038821,
       19.84385223, 22.31304424, 20.88372379, 24.3583242 , 31.02737023,
       54.14443192, 38.11345912, 25.07080321, 24.79806723, 26.31553923,
       23.86897875, 26.26081914, 32.14572521, 35.14300892, 18.76486412,
        5.14848821,  4.16595223,  4.83754422,  1.40902371,  2.68282409,
        7.73067019, 59.90279992, 43.37857912, 29.43544321, 27.91950723,
       27.85105921, 23.47585866, 23.59633903, 27.75420517])
    assert_array_almost_equal(expected, y_robust)

    g.predict(X=X, distributions=[Delta(), Normal(std=0.8)])
    y_robust = g.get_merits(beta=0)
    expected = np.array([51.54574388, 19.12524226, 22.85343748, 26.18312952, 35.49063838,
       42.22236407, 32.22598659, 62.26630594, 44.39465949, 10.75905751,
       13.38604535, 15.72842301, 24.16251049, 30.13470779, 19.49269491,
       49.00127185, 42.8247528 ,  7.70815228,  8.97201003, 10.06912607,
       17.37582038, 22.33849298, 10.80482386, 39.53961301, 47.80118736,
       11.32836594, 11.30741053, 11.19112113, 17.35581774, 21.24790037,
        8.71504901, 36.52206366, 54.37802414, 16.83487721, 15.74862523,
       14.5720682 , 19.68152613, 22.523399  ,  8.94536682, 35.7122296 ,
       58.0266604 , 19.72525509, 17.81927191, 15.76151086, 19.92829194,
       21.76601514,  7.12236048, 32.76212794, 60.38120555, 21.61855234,
       19.14257848, 16.40608395, 19.78538877, 20.72689292,  5.07827642,
       29.60433925, 65.08675091, 26.10368274, 23.30014571, 20.12893981,
       22.96638504, 23.25888141,  6.8541089 , 30.51686753])
    assert_array_almost_equal(expected, y_robust)

    g.predict(X=X, distributions=[Normal(std=0.8), Normal(std=0.8)])
    y_robust = g.get_merits(beta=1)
    expected = np.array([20.98193969, 12.75933982, 10.32132122, 11.93935509, 15.35389623,
       18.89057628, 23.24143348, 30.44337903, 12.8983337 ,  3.94355047,
        1.0179131 ,  2.12497054,  4.69172624,  7.3220051 , 10.95281735,
       17.61357528, 11.62599335,  2.14737624, -0.35343126,  1.42467053,
        3.62218697,  5.202056  ,  7.77144999, 13.91753423, 15.2290975 ,
        5.06006705,  2.42149155,  4.80473107,  7.06025407,  7.78635802,
        9.41142273, 15.06060216, 20.66381366,  9.59301638,  6.13926777,
        7.88966823,  9.33714117,  9.12562961,  9.99228692, 14.90056455,
       23.34798155, 11.46863821,  7.0661326 ,  7.6896919 ,  7.91192383,
        6.77630077,  6.99633477, 11.10217231, 24.40852822, 12.05746151,
        6.95540145,  6.78705572,  6.3548293 ,  4.78383947,  4.69791703,
        8.37190267, 28.95544536, 16.3531763 , 11.01953162, 10.68062095,
        9.97610287,  8.03913705,  7.68481882, 11.13276007])
    assert_array_almost_equal(expected, y_robust)


def test_np_input_equals_pd_input():

    # =======
    # 1D test
    # =======
    x = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.]).reshape(-1, 1)
    y = np.array([0., 1., 0., 0.8, 0.8, 0.])
    X = pd.DataFrame({'x': x.flatten(), 'y': y})

    # test a few input options
    g1 = Golem(forest_type='dt', ntrees=1, goal='max', random_state=42, verbose=True)
    g1.fit(X=x, y=y)
    g1.predict(X=x, distributions=[Normal(std=0.2)])
    y_robust1 = g1.get_merits(beta=0)
    g2 = Golem(forest_type='dt', ntrees=1, goal='max', random_state=42, verbose=True)
    g2.fit(X=X[['x']], y=y)
    g2.predict(X=X[['x']], distributions={'x': Normal(std=0.2)})
    y_robust2 = g2.get_merits(beta=0)
    assert_array_almost_equal(y_robust1, y_robust2)

    g1 = Golem(forest_type='dt', ntrees=1, goal='max', random_state=42, verbose=True)
    g1.fit(X=x.reshape(-1, 1), y=y)
    g1.predict(X=x, distributions=[Uniform(urange=0.15)])
    y_robust1 = g1.get_merits(beta=0)
    g2 = Golem(forest_type='dt', ntrees=1, goal='max', random_state=42, verbose=True)
    g2.fit(X=X[['x']], y=y)
    g2.predict(X=X[['x']], distributions={'x': Uniform(urange=0.15)})
    y_robust2 = g2.get_merits(beta=0)
    assert_array_almost_equal(y_robust1, y_robust2)

    g1 = Golem(forest_type='dt', ntrees='sqrt', goal='max', random_state=42, verbose=True)
    g1.fit(X=x.reshape(-1, 1), y=y)
    g1.predict(X=x, distributions=[Uniform(urange=0.4)])
    y_robust1 = g1.get_merits(beta=1)
    g2 = Golem(forest_type='dt', ntrees='sqrt', goal='max', random_state=42, verbose=True)
    g2.fit(X=X[['x']], y=y)
    g2.predict(X=X[['x']], distributions={'x': Uniform(urange=0.4)})
    y_robust2 = g2.get_merits(beta=1)
    assert_array_almost_equal(y_robust1, y_robust2)

    # =======
    # 2D test
    # =======
    def objective(array):
        return np.sum(array**2, axis=1)

    # 2D input
    x0 = np.linspace(-1, 1, 10)
    x1 = np.linspace(-1, 1, 10)
    X = np.array([x0, x1]).T
    y = objective(X)
    dfX = pd.DataFrame({'x0': x0, 'x1': x1})

    # tests
    g1 = Golem(forest_type='dt', ntrees=1, goal='max', random_state=42, verbose=True)
    g1.fit(X=X, y=y)
    g1.predict(X=X, distributions=[Normal(std=0.2), Normal(std=0.2)])
    y_robust1 = g1.get_merits(beta=0)
    g2 = Golem(forest_type='dt', ntrees=1, goal='max', random_state=42, verbose=True)
    g2.fit(X=dfX, y=y)
    g2.predict(X=dfX, distributions={'x0': Normal(std=0.2), 'x1': Normal(std=0.2)})
    y_robust2 = g2.get_merits(beta=0)
    assert_array_almost_equal(y_robust1, y_robust2)

    g1 = Golem(forest_type='dt', ntrees=1, goal='max', random_state=42, verbose=True)
    g1.fit(X=X, y=y)
    g1.predict(X=X, distributions=[Uniform(urange=0.3), Uniform(urange=0.3)])
    y_robust1 = g1.get_merits( beta=0)
    g2 = Golem(forest_type='dt', ntrees=1, goal='max', random_state=42, verbose=True)
    g2.fit(X=dfX, y=y)
    g2.predict(X=dfX, distributions={'x0': Uniform(urange=0.3), 'x1': Uniform(urange=0.3)})
    y_robust2 = g2.get_merits(beta=0)
    assert_array_almost_equal(y_robust1, y_robust2)

    g1 = Golem(forest_type='dt', ntrees='log2', goal='max', random_state=42, verbose=True)
    g1.fit(X=X, y=y)
    g1.predict(X=X, distributions=[Uniform(urange=0.3), Uniform(urange=0.3)])
    y_robust1 = g1.get_merits(beta=1)
    g2 = Golem(forest_type='dt', ntrees='log2', goal='max', random_state=42, verbose=True)
    g2.fit(X=dfX, y=y)
    g2.predict(X=dfX, distributions={'x0': Uniform(urange=0.3), 'x1': Uniform(urange=0.3)})
    y_robust2 = g2.get_merits(beta=1)
    assert_array_almost_equal(y_robust1, y_robust2)

    g1 = Golem(forest_type='dt', ntrees=3, goal='max', random_state=42, verbose=True)
    g1.fit(X=X, y=y)
    g1.predict(X=X, distributions=[Uniform(urange=0.3), Uniform(urange=0.3)])
    y_robust1 = g1.get_merits(beta=1)
    g2 = Golem(forest_type='dt', ntrees=3, goal='max', random_state=42, verbose=True)
    g2.fit(X=dfX, y=y)
    g2.predict(X=dfX, distributions={'x0': Uniform(urange=0.3), 'x1': Uniform(urange=0.3)})
    y_robust2 = g2.get_merits(beta=1)
    assert_array_almost_equal(y_robust1, y_robust2)

    g1 = Golem(forest_type='dt', ntrees=3, goal='max', random_state=42, verbose=True)
    g1.fit(X=X, y=y)
    g1.predict(X=X, distributions=[Uniform(urange=0.3), Uniform(urange=0.3)])
    y_robust1 = g1.get_merits(beta=1)
    g2 = Golem(forest_type='dt', ntrees=3, goal='max', random_state=42, verbose=True)
    g2.fit(X=dfX, y=y)
    g2.predict(X=dfX, distributions={'x0': Uniform(urange=0.3), 'x1': Uniform(urange=0.3)})
    y_robust2 = g2.get_merits(beta=1)
    assert_array_almost_equal(y_robust1, y_robust2)


def test_1d_categorical():
    # inputs
    x0 = ['red', 'blue', 'green']
    y = [3., 1., 0.]
    Xy = pd.DataFrame({'x0': x0, 'y': y})

    # test
    g = Golem(forest_type='dt', ntrees=1, random_state=42, verbose=False)
    g.fit(X=Xy.iloc[:, :-1], y=Xy.iloc[:, -1])

    y_robust = g.predict(X=Xy.iloc[:, :-1], distributions={'x0': Categorical(categories=x0, unc=0.4)})
    expected = [2.0, 1.2, 0.8]
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=Xy.iloc[:, :-1], distributions={'x0': Categorical(categories=x0, unc=0.0000000000001)})
    expected = y
    assert_array_almost_equal(expected, y_robust)


def test_2d_continuous_categorical():
    # inputs
    x0 = [0.5, 0.5, 0.5]
    x1 = ['red', 'blue', 'green']
    y = [3, 1, 0]
    Xy = pd.DataFrame({'x0': x0, 'x1': x1, 'y': y})

    # test
    g = Golem(forest_type='dt', ntrees=1, random_state=42, verbose=False)
    g.fit(X=Xy.iloc[:, :-1], y=Xy.iloc[:, -1])

    y_robust = g.predict(Xy.iloc[:, :-1], distributions={'x1': Categorical(categories=x1, unc=0.2)})
    expected = [2.5, 1.1, 0.4]
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(Xy.iloc[:, :-1], distributions={'x1': Categorical(categories=x1, unc=0.0000000000001)})
    expected = y
    assert_array_almost_equal(expected, y_robust)


def test_2d_categorical():
    # inputs
    x0 = ['cat', 'cat', 'cat', 'dog', 'dog', 'dog']
    x1 = ['red', 'blue', 'green', 'red', 'blue', 'green']
    y = [3., 1., 0., 7., 5., 4.]
    Xy = pd.DataFrame({'x0': x0, 'x1': x1, 'y': y})

    # test
    g = Golem(forest_type='dt', ntrees=1, random_state=42, verbose=False)
    g.fit(X=Xy.iloc[:, :-1], y=Xy.iloc[:, -1])

    y_robust = g.predict(X=Xy.iloc[:, :-1], distributions={'x0': Categorical(categories=list(set(x0)), unc=0.2),
                              'x1': Categorical(categories=list(set(x1)), unc=0.2)})
    expected = [3.3, 1.9, 1.2, 5.7, 4.3, 3.6]
    assert_array_almost_equal(expected, y_robust)

    y_robust = g.predict(X=Xy.iloc[:, :-1], distributions={'x0': Categorical(categories=list(set(x0)), unc=0.0000000000001),
                              'x1': Categorical(categories=list(set(x1)), unc=0.0000000000001)})
    expected = y
    assert_array_almost_equal(expected, y_robust)


def test_1d_continuous_bounded_inf_equals_unbounded():
    # inputs
    x = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.]).reshape(-1, 1)
    y = np.array([0., 1., 0., 0.8, 0.8, 0.])

    # test a few input options
    g = Golem(forest_type='dt', ntrees=1, random_state=42, verbose=True)
    g.fit(X=x, y=y)

    # ---------
    # Gaussians
    # ---------
    y_robust_ref = g.predict(X=x, distributions=[Normal(std=0.2)])

    y_robust = g.predict(X=x, distributions=[FoldedNormal(std=0.2, low_bound=-np.inf, high_bound=np.inf)])
    assert_array_almost_equal(y_robust_ref, y_robust)

    y_robust = g.predict(X=x, distributions=[TruncatedNormal(std=0.2, low_bound=-np.inf, high_bound=np.inf)])
    assert_array_almost_equal(y_robust_ref, y_robust)

    # --------
    # Uniforms
    # --------
    y_robust_ref = g.predict(X=x, distributions=[Uniform(urange=0.2)])

    y_robust = g.predict(X=x, distributions=[BoundedUniform(urange=0.2, low_bound=-np.inf, high_bound=np.inf)])
    assert_array_almost_equal(y_robust_ref, y_robust)

    y_robust = g.predict(X=x, distributions=[TruncatedUniform(urange=0.2, low_bound=-np.inf, high_bound=np.inf)])
    assert_array_almost_equal(y_robust_ref, y_robust)


def test_multiprocessing():

    # =======
    # 1D test
    # =======
    x = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.]).reshape(-1, 1)
    y = np.array([0., 1., 0., 0.8, 0.8, 0.])

    # test a few input options
    g1 = Golem(forest_type='rf', ntrees=5, goal='max', nproc=1, random_state=42, verbose=True)
    g1.fit(X=x, y=y)
    g1.predict(X=x, distributions=[Normal(std=0.2)])
    y_robust1 = g1.get_merits(beta=0)
    g2 = Golem(forest_type='rf', ntrees=5, goal='max', nproc=None, random_state=42, verbose=True)
    g2.fit(X=x, y=y)
    g2.predict(X=x, distributions=[Normal(std=0.2)])
    y_robust2 = g2.get_merits(beta=0)
    assert_array_almost_equal(y_robust1, y_robust2)

    g1 = Golem(forest_type='dt', ntrees=5, goal='max', nproc=1, random_state=42, verbose=True)
    g1.fit(X=x.reshape(-1, 1), y=y)
    g1.predict(X=x, distributions=[Uniform(urange=0.15)])
    y_robust1 = g1.get_merits(beta=0)
    g2 = Golem(forest_type='dt', ntrees=5, goal='max', nproc=None, random_state=42, verbose=True)
    g2.fit(X=x.reshape(-1, 1), y=y)
    g2.predict(X=x, distributions=[Uniform(urange=0.15)])
    y_robust2 = g2.get_merits(beta=0)
    assert_array_almost_equal(y_robust1, y_robust2)

    # =======
    # 2D test
    # =======
    def objective(array):
        return np.sum(array**2, axis=1)

    # 2D input
    x0 = np.linspace(-1, 1, 10)
    x1 = np.linspace(-1, 1, 10)
    X = np.array([x0, x1]).T
    y = objective(X)

    # tests
    g1 = Golem(forest_type='rf', ntrees=5, goal='max', random_state=42, verbose=True)
    g1.fit(X=X, y=y)
    g1.predict(X=X, distributions=[Normal(std=0.2), Normal(std=0.2)])
    y_robust1 = g1.get_merits(beta=0)
    g2 = Golem(forest_type='rf', ntrees=5, goal='max', random_state=42, verbose=True)
    g2.fit(X=X, y=y)
    g2.predict(X=X, distributions=[Normal(std=0.2), Normal(std=0.2)])
    y_robust2 = g2.get_merits(beta=0)
    assert_array_almost_equal(y_robust1, y_robust2)

    g1 = Golem(forest_type='et', ntrees=4, goal='max', nproc=1, random_state=42, verbose=True)
    g1.fit(X=X, y=y)
    g1.predict(X=X, distributions=[Uniform(urange=0.3), Uniform(urange=0.3)])
    y_robust1 = g1.get_merits(beta=0)
    g2 = Golem(forest_type='et', ntrees=4, goal='max', nproc=None, random_state=42, verbose=True)
    g2.fit(X=X, y=y)
    g2.predict(X=X, distributions=[Uniform(urange=0.3), Uniform(urange=0.3)])
    y_robust2 = g2.get_merits(beta=0)
    assert_array_almost_equal(y_robust1, y_robust2)


def test_recommend():
    """simple check for the moment: see if it runs fine"""

    def objective(array, invert=False):
        if invert is True:
            return -np.sum(array ** 2)
        else:
            return np.sum(array**2)

    # --------------------
    # 2D input, continuous
    # --------------------
    g = Golem(forest_type='rf', ntrees=10, goal='min', nproc=1, random_state=42, verbose=True)
    param_space = [{'type': 'continuous', 'low': -1, 'high': 1},
                   {'type': 'continuous', 'low': -1, 'high': 1}]
    g.set_param_space(param_space)
    X_obs = []
    y_obs = []
    for i in range(5):
        X_next = g.recommend(X=X_obs, y=y_obs, distributions=[Uniform(0.2), Uniform(0.2)], pop_size=10, ngen=5)
        y_next = objective(np.array(X_next))
        X_obs.append(X_next)
        y_obs.append(y_next)

    # variation of the above
    g = Golem(forest_type='rf', ntrees=10, goal='max', nproc=1, random_state=42, verbose=True)
    param_space = [{'type': 'continuous', 'low': -1, 'high': 1},
                   {'type': 'continuous', 'low': -1, 'high': 1}]
    g.set_param_space(param_space)
    X_obs = []
    y_obs = []
    for i in range(5):
        X_next = g.recommend(X=X_obs, y=y_obs, distributions=[Normal(0.1), Normal(0.1)], pop_size=10, ngen=5)
        y_next = objective(np.array(X_next), invert=True)
        X_obs.append(X_next)
        y_obs.append(y_next)

    # variation of the above
    g = Golem(forest_type='rf', ntrees=10, goal='min', nproc=None, random_state=42, verbose=True)
    param_space = [{'type': 'continuous', 'low': -1, 'high': 1},
                   {'type': 'continuous', 'low': -1, 'high': 1}]
    g.set_param_space(param_space)
    X_obs = []
    y_obs = []
    for i in range(5):
        X_next = g.recommend(X=X_obs, y=y_obs, distributions=[Uniform(0.2), Uniform(0.2)], pop_size=10, ngen=5)
        y_next = objective(np.array(X_next))
        X_obs.append(X_next)
        y_obs.append(y_next)

    # -------------------------------------------
    # 3D input: continuous, discrete, categorical
    # -------------------------------------------
    def objective(x_list):
        a = x_list[0]
        b = x_list[1]
        c = x_list[2]
        return a ** 2 + b / 2. + 3 * len(c)

    cats = ['red', 'blue', 'green', 'pink', 'yellow']
    param_space = [{'type': 'continuous', 'low': 0., 'high': 1.},
                   {'type': 'discrete', 'low': 1, 'high': 100},
                   {'type': 'categorical', 'categories': cats}]

    # test optimization passing DataFrames, 1 processor
    # -------------------------------------------------
    g = Golem(forest_type='rf', ntrees=10, random_state=42, verbose=False, goal='min', nproc=1)
    g.set_param_space(param_space)
    dists = [Uniform(0.1), DiscreteLaplace(5), Categorical(cats, 0.4)]

    df = pd.DataFrame({'x0': [], 'x1': [], 'x2': [], 'obj': []})
    for i in range(5):
        X = df.loc[:, ['x0', 'x1', 'x2']]
        y = df.loc[:, 'obj']
        X_next = g.recommend(X=X, y=y, distributions=dists, pop_size=30, ngen=10, verbose=False)
        y_next = objective(X_next)
        # append to dataframe
        df_next = pd.DataFrame({'x0': [X_next[0]], 'x1': [X_next[1]], 'x2': [X_next[2]], 'obj': [y_next]})
        df = df.append(df_next)

    # test optimization passing list of lists, multiple processors
    # ------------------------------------------------------------
    g = Golem(forest_type='rf', ntrees=10, random_state=42, verbose=False, goal='min', nproc=None)
    g.set_param_space(param_space)
    dists = [Uniform(0.1), DiscreteLaplace(5), Categorical(cats, 0.4)]

    X_obs = []
    y_obs = []
    for i in range(10):
        X_next = g.recommend(X=X_obs, y=y_obs, distributions=dists, pop_size=30, ngen=10, verbose=False)
        y_next = objective(X_next)
        X_obs.append(X_next)
        y_obs.append(y_next)

def test_get_tiles():
    """just checking method runs at the moment"""

    def objective(array):
        return np.sum(array ** 2, axis=1)

    # 2D input
    x0 = np.linspace(-1, 1, 10)
    x1 = np.linspace(-1, 1, 10)
    X = np.array([x0, x1]).T
    y = objective(X)

    # tests
    g = Golem(forest_type='rf', ntrees=5, goal='max', random_state=42, verbose=True)
    g.fit(X=X, y=y)
    for i in range(5):
        _ = g.get_tiles(tree_number=i)
