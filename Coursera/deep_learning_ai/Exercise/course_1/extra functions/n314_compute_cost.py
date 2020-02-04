from deeplearning_ai.L1_NeuralNetworksAndDeepLearning.testCases_v2 import *
import numpy as np

# GRADED FUNCTION: compute_cost

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    ### START CODE HERE ### (≈ 2 lines of code)
    logprobs = np.multiply(np.log(A2) ,Y) + np.multiply(( 1 -Y), np.log( 1 -A2))
    #logprobs = np.multiply(np.log(A2), Y) + (1 - Y) * np.log(1 - A2)
    cost = - 1 / m *np.sum(logprobs)
    ### END CODE HERE ###

    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17  assert(isinstance(cost, float))

    return cost


def compute_cost_test():
    A2, Y_assess, parameters = compute_cost_test_case()
    print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


if __name__ == "__main__":
    compute_cost_test()

