import numpy as np
import matplotlib.pyplot as plt
import common.model_util as model_util
import common.dataset_util as dataset_util
import common.visualize_util as visualize_util


class Dnn:

    def __init__(self, layers_dims, learning_rate=0.0075, num_iterations=3000, initialization="he", lambd=0,
                 print_cost=False, cost_save_step=100, cost_print_step=100):
        """

        @param layers_dims: 是一个 list，layers_dims[l] 用于描述描述第 l+1 层网络的神经元数量，故有 L 个元素（不包括输入层）
        @param learning_rate: 学习速率
        @param num_iterations: 迭代次数
        @param lambd: 迭代次数
        @param print_cost: 控制标记，是否在训练完毕后打印学习曲线
        @param cost_save_step: cost 保存步长
        """
        # 超参数
        self.initialization = initialization
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambd = lambd

        # 需要持久化的模型参数
        self.parameters = {}

        # 其他控制变量
        self.costs = []
        self.print_cost = print_cost
        self.cost_save_step = cost_save_step
        self.cost_print_step = cost_print_step

    def fit(self, X, Y):

        # 保存参数
        self.parameters, self.costs = Dnn.train(X, Y, self.layers_dims, self.learning_rate, self.num_iterations,
                                                self.lambd, self.initialization, self.cost_save_step,
                                                self.cost_print_step)

        # 绘制 cost 图形，观察收敛情况
        if self.print_cost:
            plt.plot(np.squeeze(self.costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()

    def test(self, X, Y, dataset_name=""):
        X, Y, m = dataset_util.dataset_stack_by_row_to_stack_by_column(X, Y)
        Dnn.predict_and_count(X, Y, self.parameters, dataset_name)

    def plot_decision_boundary(self, X, Y):
        visualize_util.plot_decision_boundary(lambda x: Dnn.predict(x.T, self.parameters).T, X, Y)

    @staticmethod
    def initialize_parameters(layer_dims_with_nx, initialization):
        """
        参数初始化
        在当前 DNN 中，网络的层数描述包括输入层但包括输出层，但可将输入 X 看做第 0 层，故 layer_dims 的长度为 L+1
        @param layer_dims_with_nx: 是一个长度为 (L+1) 的 list，描述每一层的神经元数量，加上输入层 X 看做 A_0 故有 L+1 层描述
        @param initialization 初始化方式，rule 推荐采用 "he" 初始化，tanh 则推荐采用默认初始化
        @return: 返回一个参数字典，key 分别是 W1, b1, W2, b2, ..., WL, bL，注意只有从第 1 层开始的参数，故字典长度为 2L
        其中 Wl.shape = (layer_dims[l], layer_dims[l-1]), bl.shape = (layer_dims[l], 1)
        """

        np.random.seed(3)
        parameters = {}
        layer_num = len(layer_dims_with_nx)  # number of layers in the network

        for layer in range(1, layer_num):
            # 为了避免梯度消失问题，每层的参数矩阵 w 采用 He 初始化：需要乘以 np.sqrt(2/n[l - 1])
            w = np.random.randn(layer_dims_with_nx[layer], layer_dims_with_nx[layer - 1])

            if initialization == "he":
                w = w * np.sqrt(2 / layer_dims_with_nx[layer - 1])
            else:
                w = w * np.sqrt(1 / layer_dims_with_nx[layer - 1])
            parameters['W' + str(layer)] = w
            parameters['b' + str(layer)] = np.zeros((layer_dims_with_nx[layer], 1))

        return parameters

    @staticmethod
    def forward_propagation(X, parameters):
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
            A, cache = model_util.linear_activation_forward(
                A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
            caches.append(cache)

        # 实现最后一层的 sigmoid 激活值，并将 cache 添加到 caches
        AL, cache = model_util.linear_activation_forward(
            A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
        caches.append(cache)

        return AL, caches

    @staticmethod
    def backward_propagation(AL, Y, caches, lambd):
        """
        给定预测值 AL 和样本标签 Y，进行反向传播算法
        @param AL: 神经网络的最终输出概率值（sigmoid），规模为 (1, m)
        @param Y: 样本标签，规模为 (1, m)
        @param caches: 长度为 L 的 list，存储每层网络的 cache -> (linear_cache, Z) -> ((A_prev, W, b), Z)
        @param lambd: 正则化参数
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
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = model_util.linear_activation_backward(
            dAL, current_cache, "sigmoid", lambd)

        for l in reversed(range(L - 1)):
            # 对于下标 [0, L-2]，分别代表第 1 ~ L-1 层的网络参数求解
            current_cache = caches[l]
            # 根据 dA_l，求解出 dW_l, dW_l 和 dA_{l-1}
            dA_prev_temp, dW_temp, db_temp = model_util.linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, "relu", lambd)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    @staticmethod
    def train(X, Y, layers_dims, learning_rate, num_iterations, lambd, initialization, cost_save_step, cost_print_step):
        """
        训练一个 L 层（输入层 X 可视作 A_0 不算在层数内）的全连接神经网络，前 L-1 层为 relu，最后一层为 sigmoid

        @param X: 训练数据，样本按行堆叠，故规格为 (m, n_x)
        @param Y: 数据标签，样本按行堆叠，故规模为 (m, 1)
        @param layers_dims: 是一个 list，layers_dims[l] 用于描述描述第 l+1 层网络的神经元数量，故有 L 个元素（不包括输入层）
        @param learning_rate: 学习速率
        @param num_iterations: 迭代次数
        @param lambd: 正则化参数
        @param initialization: 模型参数初始化方式
        @param cost_save_step: 记录 cost 的步长
        @param cost_print_step: 打印 cost 的步长
        @return: None，需要继续包量的参数作为类的实例变量保存
        """

        # 入参校验并转化 X, Y 为按列堆叠模式
        X, Y, m = dataset_util.dataset_stack_by_row_to_stack_by_column(X, Y)

        costs = []  # keep track of cost
        n_x = X.shape[0]

        # 参数初始化
        parameters = Dnn.initialize_parameters(np.r_[n_x, layers_dims], initialization)

        for i in range(0, num_iterations):
            # 前向传播
            AL, caches = Dnn.forward_propagation(X, parameters)
            # 计算损失
            cost = model_util.compute_cross_entropy_cost_with_regularization(AL, Y, parameters, lambd)
            # 反向传播
            grads = Dnn.backward_propagation(AL, Y, caches, lambd)
            # 更新参数
            parameters = model_util.update_parameters(parameters, grads, learning_rate)

            # 根据两个 step，把 cost 存到 cost 数组中，并打印一下 cost，观察收敛情况
            if i % cost_print_step == 0:
                print("Cost after iteration %i: %s" % (i, str(cost)))
            if i % cost_save_step == 0:
                costs.append(cost)

        return parameters, costs

    @staticmethod
    def predict(X, parameters):
        """
        进行预测，并打印准确率 accuracy
        @param X: 输入样本，按列堆叠，规模为 (n, m)
        @param parameters: 模型参数字典，包括 W1, b1, ..., WL, bL
        @return: 返回预测标签，规模为 (1, m)
        """

        # 前向传播
        AL, caches = Dnn.forward_propagation(X, parameters)

        # 将概率转化为标签
        return AL > 0.5

    @staticmethod
    def predict_and_count(X, Y, parameters, dataset_name=""):
        """
        进行预测，并打印准确率 accuracy
        @param X: 输入样本，按列堆叠，规模为 (n, m)
        @param Y: 样本标签，按列堆叠，规模为 (1, m)
        @param parameters: 模型参数字典，包括 W1, b1, ..., WL, bL
        @param dataset_name: 数据集名称
        @return: 返回预测标签，规模为 (1, m)
        """

        m = X.shape[1]

        # 前向传播
        labels = Dnn.predict(X, parameters)

        # 打印准确率
        if dataset_name != "":
            dataset_name = dataset_name + "\'s "
        print("%sAccuracy: %s" % (dataset_name, str(np.sum((labels == Y) / m))))
