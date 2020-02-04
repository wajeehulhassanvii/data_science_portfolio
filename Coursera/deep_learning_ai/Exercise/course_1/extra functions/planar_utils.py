import numpy as np

def plot_decision_boundary():
    pass


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + np.exp(-x))

    return s


def load_planar_dataset():
    pass


def load_extra_datasets():
    pass