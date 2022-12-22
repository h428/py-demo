import numpy as np


def sigmoid(z):
    """
    sigmoid 激活函数，对输入 z 计算激活值

    @param z: 一个标量或者一个 numpy 多维数组
    @return 返回激活值
    """

    a = 1 / (1 + np.exp(-z))
    return a


def relu(z):
    """
    relu 函数实现
    @param z:
    @return:
    """

    a = np.maximum(0, z)
    return a


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


def linear_activation_forward(A_prev, W, b, activation, keep_prob, cache_prev):
    """
    实现单层的前向传播 LINEAR->ACTIVATION ，包括线性计算和激活函数： A[l] = g(np.dot(W, A[l-1]) + b[l])，g(z) 为激活函数
    @param A_prev: 前一层的激活值，规模为 (n[l-1], m)
    @param W: 当前层的参数矩阵，规模为 (n[l], n[l-1])
    @param b: 当前层的偏置，规模为 (n[l], 1)
    @param activation: 当前层的激活函数类型，支持 relu 和 sigmoid
    @param keep_prob: Dropout 保留概率
    @param cache_prev: 前一层的缓存，为了实现 Dropout 而引入的
    @return: 是一个元组 (A, cache)
        - A 为返回当前层的激活值输出，规模为 (n[l], m)
        - cache 为当前层的缓存，包括 (linear_cache, activation_cache)，其中 linear_cache = (A_prev, W, b)，activation_cache = Z
    """

    if activation == "sigmoid":
        # 计算 sigmoid(np.dot(W, X) + b)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
        # sigmoid 为输出层，不需要 Dropout，在反向传播时也可以基于 D 判断该层需不需要 Dropout
        D = None
    else:
        # 默认为 relu，计算 relu(np.dot(W, X) + b)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = relu(Z)
        # Dropout 计算
        D = np.random.rand(*A.shape) < keep_prob
        A = A * D / keep_prob

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    activation_cache = (Z, D)
    if cache_prev:
        (_, (_, D_prev), _) = cache_prev
    else:
        D_prev = 1
    cache = (linear_cache, activation_cache, D_prev)

    return A, cache


def compute_cross_entropy_cost(AL, Y):
    """
    计算损失函数，由于本例是二分类任务，故改用交叉熵计算
    @param AL: 神经网络预测概率值，样本按列堆叠，规模为 (1, m)
    @param Y: 数据的标签，取值为 0 或 1，样本按列堆叠，规模为 (1, m)
    @return: 返回一个浮点数，采用交叉熵计算的 cost
    """

    m = Y.shape[1]
    epsilon = 1e-30

    # 计算交叉熵
    log_prob = np.multiply(-np.log(AL + epsilon), Y) + np.multiply(-np.log(1 - AL + epsilon), 1 - Y)
    cost = 1. / m * np.nansum(log_prob)

    return cost


def compute_cross_entropy_cost_with_regularization(AL, Y, parameters, lambd):
    """
    计算带带 L2 正则化的损失函数，由于本例是二分类任务，故改用交叉熵计算
    @param AL: 神经网络预测概率值，样本按列堆叠，规模为 (1, m)
    @param Y: 数据的标签，取值为 0 或 1，样本按列堆叠，规模为 (1, m)
    @param parameters: 参数字典，键的列表为  W1, b1, ..., WL, BL
    @param lambd: 正则化系数，超参数
    @return: 返回一个浮点数，采用交叉熵计算带 L2 正则化的 cost
    """

    m = Y.shape[1]
    L = len(parameters) // 2  # 网络层数

    # 计算交叉熵
    cross_entropy_cost = compute_cross_entropy_cost(AL, Y)

    # 移除 ndarray 无用的壳，确保是一个浮点数
    L2_regularization_cost = 0
    for l in range(1, L + 1):
        L2_regularization_cost += np.sum(np.square(parameters["W" + str(l)]))

    L2_regularization_cost = 1 / m * lambd / 2 * L2_regularization_cost  # 乘以系数 lambda/(2m)
    cost = cross_entropy_cost + L2_regularization_cost

    return cost


def relu_backward(dA, cache):
    """
    实现 relu 的反向求导：一直第 l 层的偏导 dA，欲求该层的 dZ，即 dZ = dA * g'(z)
    @param dA: 当前层 A 的偏导
    @param cache: 当前层的激活缓存，就是 Z
    @return: 对当前层 Z 的偏导 dZ
    """

    Z, _ = cache
    dZ = np.array(dA, copy=True)  # 拷贝一份 dA 作为 dZ

    # 由于 relu 函数的偏导不是 0 就是 1，根据链式法则和 dZ 相乘后，只需把 Z < 0 的元素的偏导变为 0 即可
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    实现 sigmoid 的反向求导：已知第 l 层的偏导 dA，欲求该层的 dZ，即 dZ = dA * g'(z)
    @param dA: 当前层 A 的偏导
    @param cache: 当前层的激活缓存，就是 Z
    @return: 对当前层 Z 的偏导 dZ
    """

    Z, _ = cache

    # 根据 Z 重新计算出当前层的 A，用于后续求偏导，因为 sigmoid 的偏导要用到
    s = 1 / (1 + np.exp(-Z))
    # 根据 sigmoid 的导数：s'(x) = s(x) * 1 - s(x) 以及链式法则，计算 dZ
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def linear_backward(dZ, cache, lambd):
    """
    根据 Z = np.dot(W, A_prev) + b 和链式法则，实现对 dW, db 和 dA_prev 的偏导

    求导规律（直接死记即可），对于 Z = np.dot(W, A_prev) + b，
    其中 Z.shape = (n[l], m), W.shape = (n[l], n[l-1]), A_prev.shape = (n[l-1], m), b.shape = (n[l], 1)

    dW = np.dot(dZ, A_prev.T) / m + lambda/m * W
    dA = np.dot(W.T, dZ)
    db = np.sum(dZ, axis=1) / m


    @param dZ: 当前层的 dZ，规模为 (n[l], m)
    @param cache: 当前层的线性缓存，包括 (A_prev, W, b)
    @param lambd: 正则化参数
    @return: 返回 (dA_prev, dW, db)
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    # 主要根据矩阵求导的法则，或者简单记住规律也可
    dW = 1. / m * np.dot(dZ, A_prev.T) + (lambd / m * W)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lambd, keep_prob):
    """
    实现单层神经网络（包括线性计算和激活层）的反向传播
    @param dA: 当前层的激活偏导 dA
    @param cache: 当前层的缓存，包括 (linear_cache, Z) -> ((A_prev, W, b), Z)
    @param activation: 当前层激活函数的类型，relu 或 sigmoid
    @param lambd: 正则化系数
    @param keep_prob: Dropout 保留概率
    @return: 返回一个元组，包括 dA_prev, dW, db
    """

    linear_cache, activation_cache, D_prev = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
        dA_prev = np.multiply(dA_prev, D_prev)
        dA_prev = dA_prev / keep_prob
    else:
        # 默认为 relu
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
        dA_prev = np.multiply(dA_prev, D_prev)
        dA_prev = dA_prev / keep_prob

    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    """
    普通梯度下降更新参数
    @param parameters: 参数字典，键的列表为  W1, b1, ..., WL, BL
    @param grads: 梯度字典，键的列表为 dW1, db1, ..., dWL, dbL
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


def initialize_velocity(parameters):
    """
    初始化 Momentum 算法所需的 velocity 字典：
        - 键："dW1", "db1", ..., "dWL", "dbL"
        - 值：和 parameters 中各个参数规模一致的 velocity，用于保存各个参数的更新动量（本质上是 dw 的指数加权平均）
    @param parameters: 参数字典，键的列表为：W1, b1, ..., WL, bL
    @return: 返回和 parameters 等规模的字典
    """

    L = len(parameters) // 2  # 计算层数
    v = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters['W' + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters['b' + str(l + 1)])

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    使用动量梯度下降法 Momentum 进行参数更新
    @param parameters: 参数字典，键的列表为 W1, b1, ..., WL, bL
    @param grads: 梯度字典，键的列表为 W1, b1, ..., WL, bL
    @param v: Momentum 算法中的动量字典 velocity
    @param beta: 和 Momentum 相关的超参数 beta
    @param learning_rate: 学习速率
    @return:
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    for l in range(L):
        # 计算动量，本质上是指数加权平均
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
        # 参数更新
        update_parameters(parameters, v, learning_rate)
        # parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        # parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v
