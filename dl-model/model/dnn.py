import numpy as np
import matplotlib.pyplot as plt
import common.common_function as common_function
import common.dataset_util as dataset_util


class Dnn:

    def __init__(self):
        # 需要持久化的模型参数
        self.parameters = {}
        self.costs = []

    def train(self, X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        """
        训练一个 L 层（输入层 X 可视作 A_0 不算在层数内）的全连接神经网络，前 L-1 层为 relu，最后一层为 sigmoid
        @param X: 训练数据，样本按行堆叠，故规格为 (m, n_x)
        @param Y: 数据标签，样本按行堆叠，故规模为 (m, 1)
        @param layers_dims: 是一个 list 用于描述各层神经元数，包括输入层，故有 L+1 个元素
            - layers_dims[0] 为样本的特征数 n_x
            - layers_dims[l] 描述第 l 层网络的神经元数量，正好和第 1~L 层网络一一对应
        @param learning_rate: 学习速率
        @param num_iterations: 迭代次数
        @param print_cost: 是否打印损失
        @return: None，需要继续包量的参数作为类的实例变量保存
        """

        # 入参校验并转化 X, Y 为按列堆叠模式
        X, Y, m = dataset_util.dataset_stack_by_row_to_stack_by_column(X, Y)

        np.random.seed(1)
        costs = []  # keep track of cost

        # 参数初始化
        parameters = Dnn.initialize_parameters_deep(layers_dims)

        for i in range(0, num_iterations):
            # 前向传播
            AL, caches = Dnn.model_forward(X, parameters)
            # 计算损失
            cost = Dnn.compute_cost(AL, Y)
            # 反向传播
            grads = Dnn.model_backward(AL, Y, caches)
            # 更新参数
            parameters = Dnn.update_parameters(parameters, grads, learning_rate)

            # 每迭代 100 次，把 cost 存到 cost 数组中，并打印一下 cost，观察收敛情况
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        # 绘制 cost 图形，观察收敛情况
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # 保存参数
        self.parameters = parameters
        self.costs = costs

    def predict(self, X, Y):
        X, Y, m = dataset_util.dataset_stack_by_row_to_stack_by_column(X, Y)
        Dnn.static_predict(X, Y, self.parameters)

    @staticmethod
    def initialize_parameters_deep(layer_dims):
        """
        参数初始化
        在当前 DNN 中，网络的层数描述包括输入层但包括输出层，但可将输入 X 看做第 0 层，故 layer_dims 的长度为 L+1
        @param layer_dims: 是一个长度为 (L+1) 的 list，描述每一层的神经元数量，加上输入层 X 看做 A_0 故有 L+1 层描述
        @return: 返回一个参数字典，key 分别是 W1, b1, W2, b2, ..., WL, bL，注意只有从第 1 层开始的参数，故字典长度为 2L
        其中 Wl.shape = (layer_dims[l], layer_dims[l-1]), bl.shape = (layer_dims[l], 1)
        """

        np.random.seed(1)
        parameters = {}
        layer_num = len(layer_dims)  # number of layers in the network

        for layer in range(1, layer_num):
            # 为了避免梯度消失问题，每层的参数矩阵 w 需要乘以 sqrt(n[l-1])
            parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) / np.sqrt(
                layer_dims[layer - 1])  # *0.01
            parameters['b' + str(layer)] = np.zeros((layer_dims[layer], 1))

            assert (parameters['W' + str(layer)].shape == (layer_dims[layer], layer_dims[layer - 1]))
            assert (parameters['b' + str(layer)].shape == (layer_dims[layer], 1))

        return parameters

    @staticmethod
    def linear_forward(A, W, b):
        """
        实现单层的前向线性计算：Z[l] = np.dot(W[l], A[l-1]) + b[l]
        @param A: 前一层的激活值，数据按列堆叠，规模为 (n[l-1], m)
        @param W: 当前层的参数矩阵，规模为 (n[l], n[l-1])
        @param b: 当前层的偏置，规模为 (n[l], 1)
        @return: 是一个元组 (Z, cache)
            - Z 为当前层的线性值，规模为 (n[l], m)
            - cache 为当前层的入参缓存，包括 (A, W, b)
        """

        Z = W.dot(A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        """
        实现单层的前向传播 LINEAR->ACTIVATION ，包括线性计算和激活函数： A[l] = g(np.dot(W, A[l-1]) + b[l])，g(z) 为激活函数
        @param A_prev: 前一层的激活值，规模为 (n[l-1], m)
        @param W: 当前层的参数矩阵，规模为 (n[l], n[l-1])
        @param b: 当前层的偏置，规模为 (n[l], 1)
        @param activation: 当前层的激活函数类型，支持 relu 和 sigmoid
        @return: 是一个元组 (A, cache)
            - A 为返回当前层的激活值输出，规模为 (n[l], m)
            - cache 为当前层的缓存，包括 (linear_cache, activation_cache)，其中 linear_cache = (A_prev, W, b)，activation_cache = Z
        """

        if activation == "sigmoid":
            # 计算 sigmoid(np.dot(W, X) + b)
            Z, linear_cache = Dnn.linear_forward(A_prev, W, b)
            A = common_function.sigmoid(Z)
        else:
            # 默认为 relu，计算 relu(np.dot(W, X) + b)
            Z, linear_cache = Dnn.linear_forward(A_prev, W, b)
            A = common_function.relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, Z)

        return A, cache

    @staticmethod
    def model_forward(X, parameters):
        """
        给定数据和参数，进行 L 层全连接神经网络的前向传播，计算步骤为：[LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
        @param X: 输入数据，样本按列堆叠，规模为 (n_x, m)
        @param parameters: L 层神经网络的参数，是一个字典，key 分别为 W1, b1, ..., WL, bL
        @return: 返回一个元组 (AL, caches)，
            - AL 为第 L 层网络的激活值输出，在当前例子中是 sigmoid 值，用于二分类
            - caches 是一个 list，为前 L 层网络的缓存（注意下标要 -1），包括 (linear_cache, Z) -> ((A_prev, W, b), Z)
        """

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        # 实现前 L-1 层的 [LINEAR -> RELU]*(L-1) 计算，并将 cache 缓存到 caches 中
        for l in range(1, L):
            A_prev = A
            A, cache = Dnn.linear_activation_forward(
                A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
            caches.append(cache)

        # 实现最后一层的 sigmoid 激活值，并将 cache 添加到 caches
        AL, cache = Dnn.linear_activation_forward(
            A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
        caches.append(cache)

        assert (AL.shape == (1, X.shape[1]))

        return AL, caches

    @staticmethod
    def compute_cost(AL, Y):
        """
        计算损失函数，由于本例是二分类任务，故改用交叉熵计算
        @param AL: 神经网络预测概率值，样本按列堆叠，规模为 (1, m)
        @param Y: 数据的标签，取值为 0 或 1，样本按列堆叠，规模为 (1, m)
        @return: 返回一个浮点数，采用交叉熵计算的 cost
        """

        m = Y.shape[1]

        # 计算交叉熵
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

        # 移除 ndarray 无用的壳，确保是一个浮点数
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        return cost

    @staticmethod
    def relu_backward(dA, cache):
        """
        实现 relu 的反向求导：一直第 l 层的偏导 dA，欲求该层的 dZ，即 dZ = dA * g'(z)
        @param dA: 当前层 A 的偏导
        @param cache: 当前层的激活缓存，就是 Z
        @return: 对当前层 Z 的偏导 dZ
        """

        Z = cache
        dZ = np.array(dA, copy=True)  # 拷贝一份 dA 作为 dZ

        # 由于 relu 函数的偏导不是 0 就是 1，根据链式法则和 dZ 相乘后，只需把 Z < 0 的元素的偏导变为 0 即可
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ

    @staticmethod
    def sigmoid_backward(dA, cache):
        """
        实现 sigmoid 的反向求导：已知第 l 层的偏导 dA，欲求该层的 dZ，即 dZ = dA * g'(z)
        @param dA: 当前层 A 的偏导
        @param cache: 当前层的激活缓存，就是 Z
        @return: 对当前层 Z 的偏导 dZ
        """

        Z = cache

        # 根据 Z 重新计算出当前层的 A，用于后续求偏导，因为 sigmoid 的偏导要用到
        s = 1 / (1 + np.exp(-Z))
        # 根据 sigmoid 的导数：s'(x) = s(x) * 1 - s(x) 以及链式法则，计算 dZ
        dZ = dA * s * (1 - s)

        assert (dZ.shape == Z.shape)

        return dZ

    @staticmethod
    def linear_backward(dZ, cache):
        """
        根据 Z = np.dot(W, A_prev) + b 和链式法则，实现对 dW, db 和 dA_prev 的偏导

        求导规律（直接死记即可），对于 Z = np.dot(W, A_prev) + b，
        其中 Z.shape = (n[l], m), W.shape = (n[l], n[l-1]), A_prev.shape = (n[l-1], m), b.shape = (n[l], 1)

        dW = np.dot(dZ, A_prev.T) / m
        dA = np.dot(W.T, dZ)
        db = np.sum(dZ, axis=1) / m


        @param dZ: 当前层的 dZ，规模为 (n[l], m)
        @param cache: 当前层的线性缓存，包括 (A_prev, W, b)
        @return: 返回 (dA_prev, dW, db)
        """

        A_prev, W, b = cache
        m = A_prev.shape[1]

        # 主要根据矩阵求导的法则，或者简单记住规律也可
        dW = 1. / m * np.dot(dZ, A_prev.T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    @staticmethod
    def linear_activation_backward(dA, cache, activation="relu"):
        """
        实现单层神经网络（包括线性计算和激活层）的反向传播
        @param dA: 当前层的激活偏导 dA
        @param cache: 当前层的缓存，包括 (linear_cache, Z) -> ((A_prev, W, b), Z)
        @param activation: 当前层激活函数的类型，relu 或 sigmoid
        @return: 返回一个元组，包括 dA_prev, dW, db
        """

        linear_cache, activation_cache = cache

        if activation == "sigmoid":
            dZ = Dnn.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = Dnn.linear_backward(dZ, linear_cache)
        else:
            # 默认为 relu
            dZ = Dnn.relu_backward(dA, activation_cache)
            dA_prev, dW, db = Dnn.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    @staticmethod
    def model_backward(AL, Y, caches):
        """
        给定预测值 AL 和样本标签 Y，进行反向传播算法
        @param AL: 神经网络的最终输出概率值（sigmoid），规模为 (1, m)
        @param Y: 样本标签，规模为 (1, m)
        @param caches: 长度为 L 的 list，存储每层网络的 cache -> (linear_cache, Z) -> ((A_prev, W, b), Z)
        @return: 返回一个字典，存储对各个参数的偏导，包括 dA, dW, db
        """

        grads = {}
        L = len(caches)  # 获取网络的层数，不计算输入层
        Y = Y.reshape(AL.shape)  # 确保 Y 的规模和 AL 规模一致

        # cost 为交叉熵，根据求导公式，计算 dAL = - (y/a - (1-y)/(1-a))，作为反向传播的起点
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        grads["dA" + str(L)] = dAL

        # 第 L 层为 sigmoid 函数，根据 dAL 分别计算出：dWL, dbL 和 dA_{L-1}
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = Dnn.linear_activation_backward(
            dAL, current_cache, activation="sigmoid")

        for l in reversed(range(L - 1)):
            # 对于下标 [0, L-2]，分别代表第 1 ~ L-1 层的网络参数求解
            current_cache = caches[l]
            # 根据 dA_l，求解出 dW_l, dW_l 和 dA_{l-1}
            dA_prev_temp, dW_temp, db_temp = Dnn.linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, activation="relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    @staticmethod
    def update_parameters(parameters, grads, learning_rate):
        """
        梯度下降更新参数
        @param parameters: 参数字典，包括 W1, b1, ..., WL, BL
        @param grads: 梯度字典，包括 dW1, db1, ..., dWL, dbL
        @param learning_rate: 学习速率
        @return: 返回梯度下降后的模型参数 parameters
        """

        # 参数字典长度除以 2 即可获取层数 L（不包括输入层）
        L = len(parameters) // 2

        # 对每一层的参数都根据梯度进行参数更新
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

        return parameters

    @staticmethod
    def static_predict(X, Y, parameters):
        """
        进行预测，并打印准确率 accuracy
        @param X: 输入样本，按列堆叠，规模为 (n, m)
        @param Y: 样本标签，按列堆叠，规模为 (1, m)
        @param parameters: 模型参数字典，包括 W1, b1, ..., WL, bL
        @return: 返回预测标签，规模为 (1, m)
        """

        m = X.shape[1]
        p = np.zeros((1, m))

        # 前向传播
        probability, caches = Dnn.model_forward(X, parameters)

        # 使用循环设置将概率转化为标签
        for i in range(0, probability.shape[1]):
            if probability[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        # 打印准确率
        # print ("predictions: " + str(p))
        # print ("true labels: " + str(y))
        print("Accuracy: " + str(np.sum((p == Y) / m)))

        return p
