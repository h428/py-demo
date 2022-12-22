import numpy as np
import common.model_util as model_util


def backward_propagation_with_regularization_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(3, 5)
    Y_assess = np.array([[1, 1, 0, 1, 0]])
    cache = (np.array([[-1.52855314, 3.32524635, 2.13994541, 2.60700654, -0.75942115],
                       [-1.98043538, 4.1600994, 0.79051021, 1.46493512, -0.45506242]]),
             np.array([[0., 3.32524635, 2.13994541, 2.60700654, 0.],
                       [0., 4.1600994, 0.79051021, 1.46493512, 0.]]),
             np.array([[-1.09989127, -0.17242821, -0.87785842],
                       [0.04221375, 0.58281521, -1.10061918]]),
             np.array([[1.14472371],
                       [0.90159072]]),
             np.array([[0.53035547, 5.94892323, 2.31780174, 3.16005701, 0.53035547],
                       [-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],
                       [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]]),
             np.array([[0.53035547, 5.94892323, 2.31780174, 3.16005701, 0.53035547],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]]),
             np.array([[0.50249434, 0.90085595],
                       [-0.68372786, -0.12289023],
                       [-0.93576943, -0.26788808]]),
             np.array([[0.53035547],
                       [-0.69166075],
                       [-0.39675353]]),
             np.array([[-0.3771104, -4.10060224, -1.60539468, -2.18416951, -0.3771104]]),
             np.array([[0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402]]),
             np.array([[-0.6871727, -0.84520564, -0.67124613]]),
             np.array([[-0.0126646]]))
    return X_assess, Y_assess, cache


if __name__ == '__main__':
    X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
    grads = model_util.backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd=0.7)
    print("dW1 = " + str(grads["dW1"]))
    print("dW2 = " + str(grads["dW2"]))
    print("dW3 = " + str(grads["dW3"]))
