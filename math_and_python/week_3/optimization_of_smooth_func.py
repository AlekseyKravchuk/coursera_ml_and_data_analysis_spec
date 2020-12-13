import numpy as np
from math import sin
from math import exp
from matplotlib import pyplot as plt
from scipy import optimize


# return 1-d numpy array containing values of function f(x) = sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2) within [1, 30]
def get_Y(x):
    return sin(float(x) / 5) * exp(float(x) / 10) + 5 * exp(float(-x) / 2)


if __name__ == '__main__':
    X = np.arange(1, 31, 0.1)
    Y_f = np.array(list(map(get_Y, X)))

    plt.plot(X, Y_f, label='function to minimize', linewidth=5)

    # naming the x axis
    plt.xlabel('x - axis')

    # naming the y axis
    plt.ylabel('y - axis')

    # plotting the points
    plt.plot()

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()

    x0 = np.array([10])
    res = optimize.minimize(get_Y, x0, options={'disp': True})
    print(f'res.x = {res.x}')
