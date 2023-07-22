import matplotlib.pyplot as plt
import numpy

def construct_matrix_random(num_rows):
    """
    Two columns with values from 6*random.rand() - 3
    """
    matrix_out = numpy.random.rand(num_rows, 2)
    # 6*random.rand() - 3
    matrix_out[:,0:1] *= 6
    matrix_out[:,0:1] -= 3

    return matrix_out

def construct_matrix_linear(num_rows = 50, num_rows_noise = 10):
    """
    Third column with x1+x2 > 0.5 and noise sampled from (0,1)
    """
    matrix_out = construct_matrix_random(num_rows + num_rows_noise)
    vector_noise = numpy.random.randint(low=0, high=1+1, size=num_rows_noise)

    # Vector of 0.5 < x1+x2 and noise sampled from binomial
    vector_binary = (matrix_out[0:num_rows].sum(axis=1) > 0.5)*1
    vector = numpy.append(vector_binary, vector_noise, axis = 0)
    # add as a third column
    matrix_out = numpy.append(matrix_out, vector.reshape(-1,1), axis=1)

    return matrix_out

def construct_matrix_one_circle(num_rows = 50, num_rows_noise = 10):
    """
    Third column with x1^2 + x2^2 < 2.8 and noise sampled from (0,1)
    """
    matrix_out = construct_matrix_random(num_rows + num_rows_noise)
    vector_noise = numpy.random.randint(low=0, high=1+1, size=num_rows_noise)

    # Vector of x1^2 + x2^2 < 2.8 and noise sampled from binomial
    vector_binary = (numpy.sum(numpy.square(matrix_out[0:num_rows,:]), axis=1) < 2.8)*1
    vector = numpy.append(vector_binary, vector_noise, axis = 0)
    # add as a third column
    matrix_out = numpy.append(matrix_out, vector.reshape(-1,1), axis=1)

    return matrix_out

def construct_matrix_two_circle(num_rows = 50, num_rows_noise = 10):
    """
    Third column with x1^2 + x2^2 < 2.8 and noise sampled from (0,1)
    """
    matrix_out = construct_matrix_random(num_rows + num_rows_noise)
    vector_noise = numpy.random.randint(low=0, high=1+1, size=num_rows_noise)

    # Vector of x1^2 + x2^2 < 2.8 and noise sampled from binomial
    x2_squared = numpy.square(matrix_out[0:num_rows, 1])
    mask_circle1 = ((matrix_out[0:num_rows, 0]-1)**2 + x2_squared) < 2
    mask_circle2 = ((matrix_out[0:num_rows, 0]+1)**2 + x2_squared) < 2
    vector_binary = (mask_circle1 | mask_circle2)
    vector = numpy.append(vector_binary, vector_noise, axis = 0)
    # add as a third column
    matrix_out = numpy.append(matrix_out, vector.reshape(-1,1), axis=1)

    return matrix_out

matrix_linear = construct_matrix_linear(num_rows=50, num_rows_noise=10)
matrix_circular = construct_matrix_one_circle(num_rows=100, num_rows_noise=10)
matrix_two_circles = construct_matrix_two_circle(num_rows = 200, num_rows_noise = 20)