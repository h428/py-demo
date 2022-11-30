import common.common_util as common_util
from model.logistic import Logistic
import os

if __name__ == '__main__':
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = common_util\
        .load_dataset_cat_and_non_cat(False)
    common_util.show_img(train_set_x_orig[5])

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.
    #
    logistic_model = Logistic(train_set_x, train_set_y_orig)
    logistic_model.train(2000, 0.005, True)
    logistic_model.predict_and_compare(test_set_x, test_set_y_orig)
