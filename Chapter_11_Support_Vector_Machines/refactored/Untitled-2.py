
# # Calculating the similarity matrix and using it to fit an SVM
import numpy

import matplotlib.pyplot as plt

import _plotting
# import utils

data = numpy.array(
    [ [0,-1,0,0,1,-1,1], [0,0,-1,1,0,1,-1],  ]
).T
vector_labels = numpy.array([0,0,0,1,1,1,1])

_plotting.plot_scatter(data[vector_labels == 0, 0], data[vector_labels == 0, 1], marker = 's')
_plotting.plot_scatter(data[vector_labels == 1, 0], data[vector_labels == 1, 1], marker = 's')

# ### Calculating the similarity matrix

def similarity(x, y):
    return numpy.exp(-1*numpy.sum(numpy.square(numpy.subtract(x, y))))

def construct_matrix_pairwise(matrix):
    matrix_out = numpy.zeros(shape=(len(matrix), len(matrix)))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix_out[i,j] = similarity(matrix[i,:], matrix[j,:])
    
    return matrix_out

# ### Fitting a (linear) SVM to the similarity matrix


svm = tc.svm_classifier.create(data_with_similarities, target='y')


svm.coefficients


coefs = svm.coefficients['value']
coefs


# ### Plotting the classifier


def svm_rbf(p):
    similarities = [similarity(p, [row['x1'], row['x2']]) for row in data]
    return np.dot(similarities, [-1,-1,-1,1,1,1,1])


features = np.array(pd.DataFrame(data['x1','x2']))
labels = np.array(data['y'])


def plot_model(X, y, model):
    X = np.array(X)
    y = np.array(y)
    plot_step = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = np.array([model(i) for i in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,colors = 'k',linewidths = 3)
    plt.contourf(xx, yy, Z, colors=['red', 'blue'], alpha=0.2, levels=[-20,0,20])
    utils.plot_points(X, y)
    plt.show()


plot_model(features, labels, svm_rbf)








