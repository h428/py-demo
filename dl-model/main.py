import common.common_util as common_util


if __name__ == '__main__':
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = \
        common_util.load_dataset_cat_and_non_cat(True)

