import matplotlib.pyplot as plt
import numpy

def plot_scatter(vector_x, vector_y, x_label = "", y_label = "",  legend = None, **kwargs):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend is not None:
        plt.legend(legend)
    plt.scatter(vector_x, vector_y, **kwargs)