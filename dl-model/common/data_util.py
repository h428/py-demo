def flatten_data(data, data_num_in_start=True):
    idx = 0 if data_num_in_start else -1
    # 确定样本数量
    m = data.shape[idx]
    return data.reshape(m, -1) if data_num_in_start else data.reshape(-1, m)
