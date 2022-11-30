import common.common_function as func
import common.common_util as util
from model.logistic import Logistic


def logistic_demo():
    """
    逻辑回归 demo
    @return: 无
    """
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = util \
        .load_dataset_cat_and_non_cat(False)
    util.show_img(train_set_x_orig[5])

    # 调整输入数据格式，将图片拉平为列特征向量
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

    # 数据 min-max 归一化
    train_set_x = func.min_max_normalization(train_set_x_flatten)
    test_set_x = func.min_max_normalization(test_set_x_flatten)

    # 提供数据创建模型对象，并进行训练和预测
    logistic_model = Logistic(train_set_x, train_set_y_orig)
    logistic_model.train(2000, 0.005, True)
    logistic_model.predict_and_compare(test_set_x, test_set_y_orig)


if __name__ == '__main__':
    logistic_demo()
