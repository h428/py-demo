import common.common_function as func
import common.dataset_loader as dataset_loader
import common.common_util as common_util
from model.logistic import Logistic


def logistic_demo():
    """
    逻辑回归 demo
    @return: 无
    """
    ds = dataset_loader.load_dataset_cat_and_non_cat(False)
    # common_util.show_img(ds.x[5])

    # 调整输入数据格式，将图片拉平为列特征向量
    m_train = ds.x.shape[0]
    m_test = ds.test_x.shape[0]

    # reshape and flatten
    train_set_x_flatten = ds.x.reshape(m_train, -1).T
    test_set_x_flatten = ds.test_x.reshape(m_test, -1).T
    y = ds.y.T
    test_y = ds.test_y.T

    # 数据 min-max 归一化
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    # train_set_x = func.min_max_normalization(train_set_x_flatten)
    # test_set_x = func.min_max_normalization(test_set_x_flatten)

    # 提供数据创建模型对象，并进行训练和预测
    logistic_model = Logistic(train_set_x, y)
    logistic_model.train(2000, 0.005, True)
    logistic_model.predict_and_compare(test_set_x, test_y)


if __name__ == '__main__':
    logistic_demo()
