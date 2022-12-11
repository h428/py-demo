import common.common_function as func
import common.dataset_loader as dataset_loader
import common.visualize_util as visualize_util
import common.data_util as data_util
from model.logistic import Logistic
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


def logistic_demo():
    """
    逻辑回归 demo
    @return: 无
    """
    ds = dataset_loader.load_dataset_cat_and_non_cat(False)
    # common_util.show_img(ds.x[5])

    # 调整输入数据格式，将图片拉平为列特征向量
    x = data_util.flatten_data(ds.x)
    test_x = data_util.flatten_data(ds.test_x)
    # 取出标签
    y = ds.y
    test_y = ds.test_y

    # 数据 min-max 归一化
    x = func.min_max_normalization(x)
    test_x = func.min_max_normalization(test_x)

    # 提供数据创建模型对象，并进行训练和预测
    logistic_model = Logistic(x, y)
    logistic_model.train(2000, 0.005, True)
    logistic_model.predict_and_compare(test_x, test_y)


if __name__ == '__main__':
    ds = dataset_loader.load_planar_dataset()
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(ds.x, ds.y[:, 0])

    print(clf.predict(ds.x))

    visualize_util.plot_decision_boundary(lambda x: clf.predict(x), ds.x, ds.y[:, 0])
