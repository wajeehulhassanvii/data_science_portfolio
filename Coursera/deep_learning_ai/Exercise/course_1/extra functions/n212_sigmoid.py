"""
用np.exp实现的sigmoid函数
可以进行广播计算，克服了basic_sigmoid的问题
"""
# GRADED FUNCTION: sigmoid

import numpy as np  # this means you can access numpy functions by writing np.function() instead of numpy.function()


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-x))
    ### END CODE HERE ###

    return s

if __name__ == "__main__":
    print(help(np))
    x = np.array([1, 2, 3])

    print(sigmoid(x))
