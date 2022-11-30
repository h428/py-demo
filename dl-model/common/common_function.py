import numpy as np


def sigmoid(z):
    """
    sigmoid 激活函数，对输入 z 计算激活值

    @param z: 一个标量或者一个 numpy 多维数组
    @return 返回激活值
    """

    a = 1 / (1 + np.exp(-z))

    return a
