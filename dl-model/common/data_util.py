import numpy as np


def flatten_data(data, data_num_in_start=True):
    idx = 0 if data_num_in_start else -1
    # 确定样本数量
    m = data.shape[idx]
    return data.reshape(m, -1) if data_num_in_start else data.reshape(-1, m)


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


def vec2row(vec):
    """
    将非标准向量转化为标准行向量
    @param vec: 形状为 (m,) 的非标准向量
    @return: 标准行向量，若无法处理返回原值
    """
    if vec.ndim != 1:
        return vec
    return vec.reshape(1, -1)


def vec2col(vec):
    """
    将非标准向量转化为标准列向量
    @param vec: 形状为 (m,) 的非标准向量
    @return: 标准列向量，若无法处理返回原值
    """
    if vec.ndim != 1:
        return vec
    return vec.reshape(-1, 1)
