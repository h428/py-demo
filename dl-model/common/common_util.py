import os

import matplotlib.pyplot as plt

import common.dataset_loader as dataset_loader


def show_img(img):
    """
    显示一张图片
    :param img: 图片数据，要求格式为 (w, h, 3)，其中 3 为 RGB 通道
    :return: 无
    """
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    dataset = dataset_loader.load_dataset_cat_and_non_cat(True, ".." + os.sep)
    plt.imshow(dataset.x[5])
    plt.show()
