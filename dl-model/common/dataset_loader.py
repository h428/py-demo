import os

import h5py
import numpy as np
import sklearn
import sklearn.datasets
import common.data_util as data_util
import common.visualize_util as visualize_util
import scipy.io


class DataSet:
    """
    DataSet 相关说明如下：
    1. 约定数据按行堆叠，即第一维度表示样本数量
    2. 所有数据预处理会影响自身数据，且返回 self 以便能够以链式方式进行调用
    """

    def __init__(self, x, y, test_x=None, test_y=None, classes_info=None):
        self.x = data_util.vec2col(x)  # 训练样本，规模为 (m, n_x)，如果只有单个特征，确保按行堆叠
        self.y = data_util.vec2col(y)  # 训练标签，规模为 (m, 1)，如果样本只有一个标签，确保按行堆叠

        m = self.x.shape[0]  # 样本数量
        n_x = np.prod(self.x.shape[1:])  # 样本特征数量
        assert m == self.y.shape[0], "训练样本 y 的数量和 x 的数量不一致"

        self.has_test = test_x is not None and test_y is not None

        if self.has_test:
            # 测试样本
            self.test_x = data_util.vec2col(test_x)
            # 测试标签
            self.test_y = data_util.vec2col(test_y)

            assert self.test_x.shape[0] == self.test_y.shape[0], "测试样本 y 的数量和 x 的数量不一致"
            assert n_x == np.prod(self.test_x.shape[1:]), "测试样本的特征数量和训练样本的特征数量不一致"

        # 标签的文本解释
        self.classes_info = classes_info

    def print_dataset_info(self):
        m_train = self.x.shape[0]
        n_x = np.prod(self.x.shape[1:])
        print("训练样本数量：" + str(m_train))
        print("单个样本形状 %s，特征数量为 %s" % (str(self.x.shape[1:]), str(n_x)))
        print("训练样本 x 形状：" + str(self.x.shape))
        print("训练样本 y 形状：" + str(self.y.shape))

        if self.has_test:
            print("测试样本数量: " + str(self.test_x.shape[0]))
            print("测试样本 x 形状：" + str(self.test_x.shape))
            print("测试样本 y 形状：" + str(self.test_y.shape))

    def flatten_x(self):
        """
        拉平 x，将第一位看做样本数量 m，后续的全看成某个样本的特征 n_x，拉平为 (m, n_x) 矩阵
        @return: 链式编程需要返回 self
        """
        self.x = data_util.flatten_data(self.x)
        if self.has_test:
            self.test_x = data_util.flatten_data(self.test_x)
        return self

    def min_max(self):
        """
        对数据进行 min_max 归一化
        @return:
        """
        self.x = data_util.min_max_normalization(self.x)
        self.test_x = data_util.min_max_normalization(self.test_x)
        return self

    def divide(self, maximum):
        self.x = self.x / maximum
        if self.has_test:
            self.test_x = self.test_x / maximum
        return self

    def scatter2d(self):
        visualize_util.scatter2d(self.x[:, 0], self.x[:, 1], self.y)


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


def load_ellipse_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    train_Y = train_Y.reshape(-1, 1)
    test_Y = test_Y.reshape(-1, 1)
    return DataSet(train_X, train_Y, test_X, test_Y)


def load_extra_datasets():
    """
    加载 sklearn 提供的 4 个常用的二维坐标形状数据集和一个完全随机的二维坐标点数据集

    其中 blobs 数据集具有多个分类，可用于测试多分类任务，若想用于二分类，可以采用 Y = Y%2 设置 Y 后用于训练；

    no_structure 数据集则为随机点，基本无法绘制一个合适的决策边界

    @return: 五个数据集组成的元组，x.shape = (m, 2), y.shape = (m. 1)
    """
    n = 200
    half = n // 2
    noisy_circles = sklearn.datasets.make_circles(n_samples=n, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=n, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=n, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=n, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(n, 2), np.r_[np.zeros(half), np.ones(n - half)]

    return (DataSet(noisy_circles[0], noisy_circles[1]), DataSet(noisy_moons[0], noisy_moons[1]),
            DataSet(blobs[0], blobs[1]), DataSet(gaussian_quantiles[0], gaussian_quantiles[1]),
            DataSet(no_structure[0], no_structure[1]))


def load_french_player_shoot_position_dataset(prefix=""):
    """
    法国足球队员射门数据，(x, y)
        - x 射门位置，(x1, x2) 是一个坐标
        - y 可取值为 0/1，0（红点）表示敌方队伍，1（蓝点）表示法国队伍

    数据来源于吴恩达深度学习/C2/改善深层神经网络课后作业

    @param prefix:
    @return:
    """
    data = scipy.io.loadmat(prefix + 'datasets/data.mat')
    train_X = data['X']
    train_Y = data['y']
    test_X = data['Xval']
    test_Y = data['yval']

    # plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);

    return DataSet(train_X, train_Y, test_X, test_Y)


if __name__ == '__main__':
    ds = load_french_player_shoot_position_dataset(prefix="../")
    # ds, _, _, _, _ = load_extra_datasets()
    # visualize_util.scatter2d(ds.x[:, 0], ds.x[:, 1], ds.y)
    ds.print_dataset_info()
    ds.scatter2d()
