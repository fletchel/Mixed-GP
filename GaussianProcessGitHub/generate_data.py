from numpy import linspace
from numpy.random import uniform, choice
from scipy.stats import norm
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

def artificial_data(interval_num=25, point_num=25, grid_size=250):
    n = interval_num + point_num
    low = 0
    high = 10

    # draw from a gaussian process prior w/ mean 0 and some random reasonable kappa (

    l = 2 + 0.5*np.random.normal()
    n_std = float(2*np.random.rand(1))

    x_ = np.linspace(low, high, grid_size).reshape(-1,1)

    distances = np.square(distance_matrix(x_, x_))

    Sigma = np.exp(-distances / (2*l**2))

    gauss_draw = np.random.multivariate_normal(np.zeros(grid_size), Sigma)

    plt.plot(x_, gauss_draw)
    plt.show()

    rand_points = np.random.choice(grid_size, n)
    mus = gauss_draw[rand_points]

    intervals = [[float(x_[rand_points[i]]), generate_interval(mu, n_std)[0:2]] for i, mu in enumerate(mus[0:interval_num])]

    for i in range(len(intervals)):

        intervals[i] = [intervals[i][0], intervals[i][1][0], intervals[i][1][1]]

    points = np.random.normal(mus[interval_num:interval_num+point_num], scale=(n_std**2)*np.ones(point_num))
    points = [[float(x_[rand_points[i+interval_num]]), p] for i, p in enumerate(points)]

    f = gauss_draw

    return intervals, points, n_std, l, f, x_



def generate_interval(mu, sig, l_bounds=None):

    # given a normal distribution, return a sampled interval

    if not l_bounds:

        l_bounds = [sig, 2*sig]

    l = uniform(l_bounds[0], l_bounds[1])

    grid = linspace(mu-3*sig, mu+3*sig-l, num=300)  # generates the a in [a,b] ([a,b] = [a,b+l])

    probs = [norm.cdf(a+l, loc=mu, scale=sig) - norm.cdf(a, loc=mu, scale=sig) for a in grid]
    probs = probs/sum(probs)

    a = choice(grid, p=probs)

    return [a,a+l]
