# this code samples 25 intervals + 25 points from a GP drawn from a flat prior and fits to these points using EP

import numpy as np
import generate_data
from GaussianProcess import GaussianProcess, viz_points

intervals, points, noise, l, f, x = generate_data.artificial_data(25, 25, 500)

interval_data = np.array(intervals)
point_data = np.array(points)

gp = GaussianProcess(dim=1,l=l, noise=noise, amplitude=1)

gp.add_train_data(point_data, "point")
gp.add_train_data(interval_data, "interval")
gp.Full_EP(10)

gp.visualise_GP(viz_variance=True)

