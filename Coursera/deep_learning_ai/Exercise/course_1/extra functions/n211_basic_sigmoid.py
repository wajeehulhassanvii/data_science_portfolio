"""
用math实现的sigmoid函数
缺点：当参数是列表是无法执行“广播”计算
"""
import math

# GRADED FUNCTION: basic_sigmoid


def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + math.exp(-x))
    ### END CODE HERE ###

    return s


if __name__ == "__main__":
    print(basic_sigmoid(3))

    x = [1, 2, 3]
    print(basic_sigmoid(x))