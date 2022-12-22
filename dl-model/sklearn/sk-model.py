import common.dataset_loader as dataset_loader
import common.visualize_util as visualize_util
import sklearn
import sklearn.linear_model
import sklearn.datasets


def logistic_demo():
    # 加载花型数据，(m, 2)
    ds = dataset_loader.load_planar_dataset()
    # 采用逻辑回归模型
    clf = sklearn.linear_model.LogisticRegressionCV()
    # 喂养数据进行训练，其中 x 为 (m, 2) 矩阵，对于 y，sklearn 要求必须是非标准向量，故使用切片处理
    clf.fit(ds.x, ds.y[:, 0])
    # 绘制决策边界
    visualize_util.plot_decision_boundary(lambda x: clf.test(x), ds.x, ds.y[:, 0])


if __name__ == '__main__':
    logistic_demo()
