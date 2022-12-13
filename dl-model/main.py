import common.common_function as func
import common.dataset_loader as dataset_loader
import common.visualize_util as visualize_util
import common.data_util as data_util
from model.logistic_new import Logistic
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from model.dnn import Dnn


def logistic_demo():
    """
    逻辑回归 demo
    @return: 无
    """
    ds = dataset_loader.load_dataset_cat_and_non_cat(False)
    # common_util.show_img(ds.x[5])

    ds.flatten_x().divide(255)

    # 提供数据创建模型对象，并进行训练和预测
    logistic_model = Logistic()
    logistic_model.train(ds.x, ds.y, 2000, 0.005, True)
    logistic_model.predict(ds.x, ds.y)
    logistic_model.predict(ds.test_x, ds.test_y)


def dnn_demo():
    ds = dataset_loader.load_dataset_cat_and_non_cat(False)
    # common_util.show_img(ds.x[5])

    ds.flatten_x().divide(255)

    dnn = Dnn()
    dnn.train(ds.x, ds.y, [12288, 20, 7, 5, 1], num_iterations=2500, print_cost=True)
    dnn.predict(ds.x, ds.y)
    dnn.predict(ds.test_x, ds.test_y)


if __name__ == '__main__':
    logistic_demo()
