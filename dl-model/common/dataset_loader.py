import os

import h5py
import numpy as np


class DataSet:

    def __init__(self, x, y, test_x=None, test_y=None, classes_info=None):
        self.x = x
        self.y = y
        self.test_x = test_x
        self.test_y = test_y
        self.classes_info = classes_info


def load_dataset_cat_and_non_cat(print_dataset_info=False, prefix=""):
    """
    读取猫咪和非猫咪数据集，主要用于二分类任务
    数据来源：吴恩达深度学习/逻辑回归作业

    :return 返回一个元组，包含下列内容
        train_set_x_orig : 训练集图片，规模为 (样本数量, 64, 64, 3)，其中 64 * 64 为图片宽高， 3 为 RGB 通道，其他数据集类似
        train_set_y_orig : 训练集标签，规模为 (1, 训练样本数量)
        test_set_x_orig : 测试集图片，规模为 (样本数量, 64, 64, 3)
        test_set_y_orig : 测试集标签，规模为 (1, 测试样本数量)
        classes : 标签含义，是一个数组，就两个元素，['non-cat', 'cat']
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
    np.random.seed(1)
    m = 400  # number of examples
    n = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y
