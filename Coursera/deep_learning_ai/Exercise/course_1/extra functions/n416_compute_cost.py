from deeplearning_ai.L1_NeuralNetworksAndDeepLearning.n410_dnn_utils import *
from deeplearning_ai.L1_NeuralNetworksAndDeepLearning.testCases_v4 import *
import numpy as np

# GRADED FUNCTION: compute_cost

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    ### END CODE HERE ###

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def compute_cost_test():
    Y, AL = compute_cost_test_case()

    print("cost = " + str(compute_cost(AL, Y)))


if __name__ == "__main__":
    compute_cost_test()
