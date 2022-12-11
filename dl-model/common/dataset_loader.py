import os

import h5py
import numpy as np


class DataSet:

    def __init__(self, x, y, test_x=None, test_y=None, classes_info=None):
        # 训练样本
        self.x = x
        # 训练标签
        self.y = y
        # 测试样本
        self.test_x = test_x
        # 测试标签
        self.test_y = test_y
        # 标签的文本解释
        self.classes_info = classes_info


def load_dataset_cat_and_non_cat(print_dataset_info=False, prefix=""):
    """
    读取猫咪和非猫咪数据集，主要用于二分类任务

    x: 样本输入 x 是一张图片，形状为 (64, 64, 3)，样本按行堆叠，故 x 规格为 (m, 64, 64, 3)
    y: 样本标签 y 是一个 scala，可能取值为 0/1 表示该图片是否猫咪，样本按行堆叠，故 y 规格为 (m, 1)

    数据来源：吴恩达深度学习/逻辑回归作业

    @return 返回猫咪数据集的 DataSet 实例
    """

    # 打开训练集数据文件，注意为 win 下写法，Linux 要修改
    train_dataset = h5py.File(prefix + "datasets" + os.sep + "train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    # 打开测试集数据文件，注意为 win 下写法，Linux 要修改
    test_dataset = h5py.File(prefix + "datasets" + os.sep + "test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape(-1, 1)
    test_set_y_orig = test_set_y_orig.reshape(-1, 1)

    if print_dataset_info:
        m_train = train_set_x_orig.shape[0]
        m_test = test_set_x_orig.shape[0]
        num_px = train_set_x_orig.shape[1]
        print("Number of training examples: m_train = " + str(m_train))
        print("Number of testing examples: m_test = " + str(m_test))
        print("Height/Width of each image: num_px = " + str(num_px))
        print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print("train_set_x shape: " + str(train_set_x_orig.shape))
        print("train_set_y shape: " + str(train_set_y_orig.shape))
        print("test_set_x shape: " + str(test_set_x_orig.shape))
        print("test_set_y shape: " + str(test_set_y_orig.shape))

    return DataSet(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes)


def load_planar_dataset():
    """
    加载花型数据

    x: 样本输入 x 是一个二维坐标，表示平面上的一个点，样本按行堆叠，故输入矩阵 x 的规模为 (m, 2) 矩阵，每行是一个坐标点
    y: 样本标签 y 是一个浮点数，可能取值为 0 或 1，表示红点或者蓝点，样本按行堆叠故标签矩阵的规模为 (m, 1) 矩阵，每行是一个分类

    数据来源：吴恩达深度学习/浅层神经网络编程作业

    @return: 返回花型数据的数据集
    """
    np.random.seed(1)
    m = 400  # 样本数量
    n = int(m / 2)  # 每类样本的数量，均分
    d = 2  # 坐标点的维度
    x = np.zeros((m, d))  # 初始化 x 为 (m, 2) 矩阵，每行存储一个坐标点
    y = np.zeros((m, 1), dtype='uint8')  # 每行存储对应分类，0 表示红点，1 表示蓝点
    a = 4  # 最大花瓣数

    # 对每个分类分别生成 200 个样本
    for j in range(2):
        ix = range(n * j, n * (j + 1))  # 确定第 j 类的数据集下标取值范围，红点为 [0, 200),蓝点为 [200, 400)
        t = np.linspace(j * 3.12, (j + 1) * 3.12, n) + np.random.randn(n) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(n) * 0.2  # radius
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]  # 分别确定坐标的 x1, x2 向量，然后将两个分标准向量按列堆叠形成 (m, 2) 矩阵
        y[ix] = j  # 第 j 类标签，0 为红点，1 为蓝点

    return DataSet(x, y)  # 返回数据集
