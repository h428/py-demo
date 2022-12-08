import numpy as np


def sigmoid(z):
    """
    sigmoid 激活函数，对输入 z 计算激活值

    @param z: 一个标量或者一个 numpy 多维数组
    @return 返回激活值
    """

    a = 1 / (1 + np.exp(-z))

    return a


def min_max_normalization(array, axis=0):
    """
    min-max 归一化
    @param array: numpy ndarray
    @param axis: 维度
    @return: 返回归一化后的数组
    """
    min_num = np.min(array, axis=axis) * 1.0
    max_num = np.max(array, axis=axis) * 1.0
    delta = (max_num - min_num)
    # if axis == 1:
    #     min_num = min_num.reshape(-1, 1)
    #     delta = delta.reshape(-1, 1)
    return (array - min_num) / delta
