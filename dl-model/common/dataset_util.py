def dataset_stack_by_row_to_stack_by_column(X, Y):
    """
    将原本按行堆叠的数据集调整为按列堆叠的格式，只能处理二维矩阵
    @param X: 训练样本，约定是一个 (m, n_x) 列的矩阵
    @param Y: 样本标签，含有 m 个元素的向量，可以是列向量或非标准向量，但不能是行向量
    @return: 返回调整后的样本格式，样本按列堆叠，
        - X 规模为 (n_x, m) 的矩阵，m 为样本数
        - Y 规模为 (1, m) 的矩阵，
        - m 样本数 m
    """

    # 入参校验
    assert X.ndim == 2, "X 必须是一个按行堆叠的矩阵，每行表示一个样本"
    m = X.shape[0]  # 获取样本数
    assert Y.shape[0] == m, "X 和 Y 样本数量不一致：在 X 中找到 %d 个样本，但在 Y 中找到 %d 个样本" % (m, Y.shape[0])

    # reshape 一下 X, Y，因为整个神经网络内部要求按列堆叠数据
    X = X.T
    Y = Y.reshape(1, m)

    return X, Y, m
