import numpy as np
from common.common_function import *


class Logistic:

    def __init__(self, x, y):
        """
        @param x: 训练集数据，必须是一个 (m, n) 矩阵，每一列为一个样本
        @param y: 训练集标签，必须是一个 (m, 1) 矩阵，取值必须是 0 或 1（二分类标签）
        """

        self.x = x.T
        self.y = y.T

        # 训练样本数量
        self.m_train = self.x.shape[1]
        # 单样本特征数量
        self.n_x = self.x.shape[0]
        # 特征向量
        self.w = np.zeros((self.n_x, 1))
        # 偏置
        self.b = 0
        # 代价
        self.costs = []

    def zeros(self):
        # 特征向量
        self.w = np.zeros((self.n_x, 1)).reshape(self.n_x, 1)
        # 偏置
        self.b = 0
        # 代价
        self.costs = []

    def __propagate(self):
        a = sigmoid(np.dot(self.w.T, self.x) + self.b)
        cost = np.sum(self.y * np.log(a) + (1 - self.y) * np.log(1 - a)) / - self.m_train
        dz = a - self.y
        dw = np.dot(self.x, dz.T) / self.m_train
        db = np.sum(dz) / self.m_train

        assert (dw.shape == self.w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        return dw, db, cost

    def __optimize(self, num_iterations, learning_rate, print_cost=False):

        for i in range(num_iterations):

            dw, db, cost = self.__propagate()

            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db

            if i % 100 == 0:
                self.costs.append(cost)

            # Print the cost every 100 training examples
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost[0]))

    def train(self, num_iterations, learning_rate, print_cost=False):
        self.zeros()
        self.__optimize(num_iterations, learning_rate)

    def predict(self, test_x: np.ndarray):
        test_x = self.__make_sure_data_stack_by_column(test_x)
        m_test = test_x.shape[1]
        y_prediction = np.zeros((1, m_test))
        a = sigmoid(np.dot(self.w.T, test_x) + self.b)

        y_prediction[a > 0.5] = 1
        y_prediction[a <= 0.5] = 0

        assert (y_prediction.shape == (1, m_test))

        return y_prediction

    def predict_and_compare(self, test_x, test_y):
        y_prediction_train = self.predict(self.x)
        y_prediction = self.predict(test_x)
        test_y = self.__make_sure_data_stack_by_column(test_y)
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - self.y)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction - test_y)) * 100))

    def __make_sure_data_stack_by_column(self, array):
        """
        确保样本（包括输入 x 和标签 y）是按列堆叠的
        @param array:
        @return:
        """

        # 非标准向量则直接 reshape 为标准行向量（处理 y）
        if array.ndim == 1:
            return array.reshape(1, -1)

        # 列向量则直接转为行向量（处理 y）
        if array.shape[1] == 1:
            return array.T

        # 输入测试样本是按行堆叠的，需要转置为按列堆叠（处理 x）
        if array.shape[1] == self.n_x and array.shape[0] != self.n_x:
            return array.T

        return array
