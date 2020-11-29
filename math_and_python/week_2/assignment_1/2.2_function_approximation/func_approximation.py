import numpy as np
from math import sin
from math import exp
from matplotlib import pyplot as plt
from scipy import linalg


# return an b vector as a numpy column vector (right-hand side of equation in a vector form Ax = b)
# it also used to compute values of function f(x) = sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)
def get_vector_b(points_tup):
    def f(x):
        return sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)

    # return np.array([f(x) for x in points_tup])
    return np.array([f(x) for x in points_tup])[:, np.newaxis]


# return an A  quare matrix matrix (m x m, where m - is length of points_tup)
def get_matrix_A(points):
    return np.array([[x ** i for i, _ in enumerate(points)] for x in points])


def get_vector_x(matrix_A, vector_b):
    return linalg.solve(matrix_A, vector_b)


def get_polynom_values(X, polynom_coeffs):
    polynom_values = []
    for x in X:
        polynom_values.append(sum([a * x ** power for power, a in enumerate(polynom_coeffs)]))
    return np.array(polynom_values)


def print_results(matrices_A, vectors_b, vectors_x):
    print('matrices_A[0] =')
    print(matrices_A[0])
    print(f'matrices_A[0].shape = {matrices_A[0].shape}', end='\n\n')

    print('vectors_b[0] = ')
    print(vectors_b[0])
    print(f'vectors_b[0].shape = {vectors_b[0].shape}', end='\n\n')

    print('vectors_x[0] =')
    print(vectors_x[0])
    print(f'vectors_x[0].shape = {vectors_x[0].shape}')
    print('*********************************')

    print('matrices_A[1] =')
    print(matrices_A[1])
    print(f'matrices_A[1].shape = {matrices_A[1].shape}', end='\n\n')

    print('vectors_b[1] = ')
    print(vectors_b[1])
    print(f'vectors_b[1].shape = {vectors_b[1].shape}', end='\n\n')

    print('vectors_x[1] =')
    print(vectors_x[1])
    print(f'vectors_x[1].shape = {vectors_x[1].shape}')

    print('*********************************')
    print('matrices_A[2] =')
    print(matrices_A[2])
    print(f'matrices_A[2].shape = {matrices_A[2].shape}', end='\n\n')

    print('vectors_b[2] = ')
    print(vectors_b[2])
    print(f'vectors_b[2].shape = {vectors_b[2].shape}', end='\n\n')

    print('vectors_x[2] =')
    print(vectors_x[2])
    print(f'vectors_x[2].shape = {vectors_x[2].shape}')


def main():
    solution_file = 'solution_week2_2.txt'
    # ================== Step 1: approximation (interpolation) given function by polynomial of degree n ============
    # Solve Ax = b, A - matrix, X - vector of unknowns, b - vector, containing values of f(X) for given values of X
    pivot_points = ((1., 15.), (1., 8., 15.), (1, 4, 10, 15))
    matrices_A = [get_matrix_A(points_tup) for points_tup in pivot_points]
    vectors_b = [get_vector_b(points_tup) for points_tup in pivot_points]
    vectors_x = [get_vector_x(matrix_A, vectors_b[idx]) for idx, matrix_A in enumerate(matrices_A)]

    # ======================================== Step 2: plotting results ===========================================
    # ============================ Step 2.1: Calculating needed values of X and Y =================================
    X = np.arange(0, 16, 0.1)
    Y_f = get_vector_b(X)
    Y_polynom_degree_1 = get_polynom_values(X, vectors_x[0].flatten())
    Y_polynom_degree_2 = get_polynom_values(X, vectors_x[1].flatten())
    Y_polynom_degree_3 = get_polynom_values(X, vectors_x[2].flatten())

    # ===================================== Step 2.2: Plotting graphs ============================================
    plt.plot(X, Y_f, label='approximable function', linewidth=5)
    plt.plot(X, Y_polynom_degree_1, label='Polynomial of degree 1')
    plt.plot(X, Y_polynom_degree_2, label='Polynomial of degree 2')
    plt.plot(X, Y_polynom_degree_3, label='Polynomial of degree 3')

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

    # print elements of the equation: Ax = b
    print_results(matrices_A, vectors_b, vectors_x)

    # ================================ Step 3: Print needed results to solution file ==================================
    np.savetxt(solution_file, vectors_x[2], newline=' ', fmt='%.2f')


if __name__ == '__main__':
    main()
