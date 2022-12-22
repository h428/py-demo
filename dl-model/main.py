import common.dataset_loader as dataset_loader
import common.visualize_util as visualize_util
import common.data_util as data_util
import common.model_util as model_util
from model.logistic import Logistic
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from model.dnn import Dnn
import matplotlib.pyplot as plt


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
    logistic_model.train(ds.x, ds.y, 200, 0.005, True)
    logistic_model.test(ds.x, ds.y)
    logistic_model.test(ds.test_x, ds.test_y)


def cat_dnn_demo():
    # ds = dataset_loader.load_ellipse_dataset()
    ds = dataset_loader.load_dataset_cat_and_non_cat()
    # visualize_util.show_img(ds.x[5])

    ds.flatten_x().divide(255)

    np.random.seed(1)
    dnn = Dnn([20, 7, 5, 1], num_iterations=2400, initialization="default", print_cost=True)
    dnn.fit(ds.x, ds.y)
    dnn.test(ds.x, ds.y)
    dnn.test(ds.test_x, ds.test_y)
    # Accuracy: 0.9808612440191385
    # Accuracy: 0.8


def ellipse_dnn_demo():
    """
    椭圆数据样例，来自C2/Week1/参数初始化
    @return:
    """
    ds = dataset_loader.load_ellipse_dataset()

    np.random.seed(3)
    dnn = Dnn([10, 5, 1], num_iterations=15000, learning_rate=0.01, print_cost=True, cost_save_step=1000)
    dnn.fit(ds.x, ds.y)
    dnn.plot_decision_boundary(ds.x, ds.y)
    dnn.test(ds.x, ds.y, "train set")
    dnn.test(ds.test_x, ds.test_y, "test set")
    # Cost after iteration 14000: 0.07357895962677363
    # train set's Accuracy: 0.9933333333333335
    # test set's Accuracy: 0.96


def planar_dnn_demo():
    """
    花型数据样例
    @return:
    """
    ds = dataset_loader.load_planar_dataset()
    ds.scatter2d()


def french_player_shoot_demo():
    """
    法国足球队数据样例，来自 C2/Week1/正则化
    @return:
    """
    ds = dataset_loader.load_french_player_shoot_position_dataset()
    ds.print_dataset_info()
    # ds.scatter2d()
    np.random.seed(3)
    # learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1
    dnn = Dnn([20, 3, 1], initialization="default", learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0.7,
              cost_save_step=1000, cost_print_step=10000)
    dnn.fit(ds.x, ds.y)
    dnn.test(ds.x, ds.y, "train set")
    dnn.test(ds.test_x, ds.test_y, "test set")
    dnn.plot_decision_boundary(ds.x, ds.y)


if __name__ == '__main__':
    # ellipse_dnn_demo()
    french_player_shoot_demo()
