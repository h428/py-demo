import numpy as np
from common.common_function import *
import common.dataset_util as dataset_util


class Logistic:

    def __init__(self):
        self.w = None  # 特征向量
        self.b = 0  # 偏置
        self.costs = []  # 代价

    def train(self, X, Y, num_iterations, learning_rate, print_cost=False):
        X, Y, m = dataset_util.dataset_stack_by_row_to_stack_by_column(X, Y)
        n_x = X.shape[0]
        w, b = Logistic.init_parameters(n_x)
        self.w, self.b, self.costs = Logistic.optimize(w, b, X, Y, num_iterations, learning_rate, print_cost)

    def predict(self, X, Y):
        X, Y, _ = dataset_util.dataset_stack_by_row_to_stack_by_column(X, Y)
        Logistic.static_predict(self.w, self.b, X, Y)

    @staticmethod
    def init_parameters(n_x):
        w = np.zeros((n_x, 1)).reshape(n_x, 1)  # 特征向量
        b = 0  # 偏置
        return w, b

    @staticmethod
    def propagate_and_backward(w, b, X, Y):
        m = X.shape[1]  # 样本数
        A = sigmoid(np.dot(w.T, X) + b)
        cost = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / -m
        dZ = A - Y
        dw = np.dot(X, dZ.T) / m
        db = np.sum(dZ) / m

        assert (dw.shape == w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        return dw, db, cost

    @staticmethod
    def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
        costs = []

        for i in range(num_iterations):
            dw, db, cost = Logistic.propagate_and_backward(w, b, X, Y)

            w = w - learning_rate * dw
            b = b - learning_rate * db

            if i % 100 == 0:
                costs.append(cost)

            # 每 100 步打印一下 cost 值
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        return w, b, costs

    @staticmethod
    def static_predict(w, b, X, Y):
        m = X.shape[1]  # 计算样本数
        a = sigmoid(np.dot(w.T, X) + b)  # 执行前向传播进行预测
        y_prediction = np.zeros((1, m))  # 默认预测为 0
        y_prediction[a > 0.5] = 1  # 概率大于 0.5 预测为 1

        print("accuracy: {} %".format(100 - np.mean(np.abs(y_prediction - Y)) * 100))

        assert (y_prediction.shape == (1, m))

        return y_prediction
