# coding: utf-8
from common.np import *
'''
ニューラルネットワークに使用する様々な関数を定義。
'''


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def tanh(x):
    return np.tanh(x)


def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def sum_squared_error_range(y, t):
    less_than_min = (y[:,0] < t[:,0])
    in_range = ((t[:,0] <= y[:,0]) & (y[:,0] <= t[:,1]))
    more_than_max = (t[:,1] < y[:,0])
    _error = np.sum(np.array([
        0.5 * np.sum((y[less_than_min, 0]-t[less_than_min, 0])**2),
        0.5 * np.sum((y[more_than_max, 0]-t[more_than_max, 1])**2)
    ]))
    return _error, (less_than_min, in_range, more_than_max)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
