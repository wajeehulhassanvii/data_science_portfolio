import tensorflow as tf
import numpy as np


def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)

    ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3 ,1), dtype=tf.float32 ,name = "X")
    W = tf.Variable(np.random.randn(4 ,3), dtype=tf.float32)
    b = tf.Variable(np.random.randn(4 ,1), dtype=tf.float32)
    Y = tf.add(tf.matmul(W ,X), b)
    ### END CODE HERE ###

    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate

    ### START CODE HERE ###
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    result = sess.run(Y)
    ### END CODE HERE ###

    # close the session
    sess.close()

    return result