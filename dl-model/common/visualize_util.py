import os
import numpy as np
import matplotlib.pyplot as plt
import common.dataset_loader as dataset_loader


def show_img(img):
    """
    显示一张图片
    @param img: 图片数据，要求格式为 (w, h, 3)，其中 w, h 为图片宽高，3 为 RGB 通道
    @return: 无
    """
    plt.imshow(img)
    plt.show()


def __axis_adjustment(axis_min, axis_max):
    length = axis_max - axis_min
    if length < 3:
        return axis_min - length / 10, axis_max + length / 10

    return axis_min - 1, axis_max + 1


def plot_decision_boundary(model, x, y):
    """
    根据样本输入 x 和样本标签 y 绘制决策边界
    @param model: 一个分类模型，对于规模为 (m, 2) 的 x 输入，得到一个规模为 (m, 1) 的 y_hat 预测输出
    @param x: 输入样本，是一个 (m, 2) 矩阵，每行是一个坐标点
    @param y: 样本标签，是一个 (m, 1) 矩阵，每行是一个分类
    @return: None
    """
    # 求出横坐标 x1 的最小值、最大值，确定 x 坐标轴范围
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    # 求出纵坐标 x2 的最小值、最大值，确定 y 坐标轴范围
    y_min, y_max = x[:, 1].min(), x[:, 1].max()

    # 调整 x, y 方向的绘图边界
    x_min, x_max = __axis_adjustment(x_min, x_max)
    y_min, y_max = __axis_adjustment(y_min, y_max)

    h = 0.01
    # 使用 np.meshgrid 交叉生成网格数据矩阵，h 为步长，xx 为样本横坐标，yy 为样本纵坐标
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 按列堆叠 xx 和 yy，组成样本数据 [xx yy]，即 (m, 2) 规格
    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)  # 调整输出的形状
    # 以 Spectral 形式，根据 Z 值绘制等高线，根据 z 值不同从而形成决策边界
    # axes = plt.subplot(111)
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.xlabel("x1")
    plt.ylabel("x2")
    # 同样以 Spectral 形式，根据 y 的分类绘制原始坐标点，
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def scatter2d(x1, x2, z):
    """
    绘制散点图
    @param x1: 横坐标向量
    @param x2: 纵坐标向量
    @param z: 分类值，不同分类颜色会不一样
    @return: None
    """
    plt.scatter(x1, x2, c=z, s=40, cmap=plt.cm.Spectral)
    plt.show()
