import matplotlib.pyplot as plt

def viz_interval(intervals):

    x = intervals[:, 0]
    a = intervals[:, 1]
    b = intervals[:, 2]

    for i in range(len(x)):

        plt.vlines(x[i], ymin=a[i], ymax=b[i])

def viz_points(points):

    x = points[:, 0]
    y = points[:, 1]


    plt.scatter(x,y, marker='o', color='red')

