# coding: utf-8
from turtle import forward
from common.np import *
from math import *
import networkx as nx
import sys
from common.functions import identity_function, sum_squared_error, sum_squared_error_range, tanh, softmax, sigmoid
from common.graphpool import gpfor_iter, gpfor_iter_for_GPU, gpback_iter
from common.graphconv import gcfor, gcback
from common.config import GPU
'''
グラフ畳み込みに必要なlayerの定義
'''


class Relu:
    '''
    ReLU関数の活性化関数。
    '''

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Swish:
    '''
    Swish関数の活性化関数。
    '''

    def __init__(self, beta=1):
        self.x = None
        self.out = None
        self.beta = beta

    def forward(self, x):
        self.x = x
        out = x / (1 + e**(-1*self.beta*self.x))
        self.out = out

        return out

    def backward(self, dout):
        dx = (self.beta * self.out) + \
            (1-self.beta*self.out) / (1 + e**(-1*self.beta*self.x))

        return dx


class LeakyRelu:
    '''
    LeakyReLU関数の活性化関数。
    '''

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out = np.where(self.mask, x/2, x)

        return out

    def backward(self, dout):
        dout = np.where(self.mask, dout/2, dout)
        dx = dout

        return dx


class AbsoluteValue:
    '''
    絶対値関数の活性化関数。
    '''

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out = np.where(self.mask, -x, x)

        return out

    def backward(self, dout):
        dout = np.where(self.mask, -dout, dout)
        dx = dout

        return dx


class SmoothAbsoluteValue_botsu:
    '''
    絶対値関数の0付近を滑らかにしたかった活性化関数。（ボツ）
    '''

    def __init__(self):
        self.mask1 = None
        self.mask2 = None

    def forward(self, x):
        self.mask1 = (x <= 0)
        self.mask2 = (x <= 0.5)
        out = x.copy()
        out = np.where(self.mask1, -x, x)
        out = np.where(self.mask2, x**2, x)

        return out

    def backward(self, dout):
        dout = np.where(self.mask1, -dout, dout)
        dout = np.where(self.mask2, dout*2, dout)
        dx = dout

        return dx


class SmoothAbsoluteValue:
    '''
    絶対値関数の0付近を滑らかにした活性化関数。
    '''

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        out = np.zeros(x.shape)
        out[x <= 0] = np.log(np.exp(x) + np.exp(-x))[x <= 0]
        out[np.isinf(out)] = -self.x[np.isinf(out)]
        out[x > 0] = np.log(np.exp(x) + np.exp(-x))[x > 0]
        out[np.isinf(out)] = self.x[np.isinf(out)]

        return out

    def backward(self, dout):
        dx = dout * np.tanh(self.x)

        return dx


class Mish:
    '''
    Relu関数の後継であるMish関数。全体が滑らかな連続関数になっている。
    '''

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        out = np.zeros(x.shape)
        out[x <= 0] = x[x <= 0] * tanh(np.log(1+np.exp(x[x <= 0])))
        out[x > 0] = x[x > 0] * tanh(x[x > 0] + np.log(np.exp(-x[x > 0])+1))

        return out

    def backward(self, dout):
        x_minus = self.x[self.x <= 0]
        x_plus = self.x[self.x > 0]
        dx = np.zeros(self.x.shape)

        omega_minus = 4 * (x_minus + 1) + 4 * np.exp(2*x_minus) + \
            np.exp(3*x_minus) + (4 * x_minus + 6) * np.exp(x_minus)
        delta_minus = 2 * np.exp(x_minus) + np.exp(2 * x_minus) + 2
        dx[self.x <= 0] = dout[self.x <= 0] * \
            np.exp(x_minus) * omega_minus / delta_minus / delta_minus

        omega_plus = 4 * (x_plus + 1) * np.exp(-3 * x_plus) + 4 * \
            np.exp(-x_plus) + 1 + (4 * x_plus + 6) * np.exp(-2 * x_plus)
        delta_plus = 2 * np.exp(-x_plus) + 1 + 2 * np.exp(-2 * x_plus)
        dx[self.x > 0] = dout[self.x > 0] * \
            omega_plus / delta_plus / delta_plus

        return dx


class TanhExp:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        out = x * tanh(np.exp(x))

        return out

    def backward(self, dout):
        exp_x = np.exp(self.x)
        dx_over_dout_temp = np.zeros(self.x.shape)
        dx_over_dout_temp[~np.isinf(exp_x)] = (
            self.x * (exp_x * ((tanh(exp_x)*tanh(exp_x)) - 1)))[~np.isinf(exp_x)]
        dx_over_dout_temp[np.isinf(exp_x)] = 0
        dx_over_dout = tanh(exp_x) - dx_over_dout_temp
        dx = dout * dx_over_dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=-1, keepdims=True)
        dx -= self.out * sumdx
        return dx


# class Softmax:
#     def __init__(self):
#         self.out = out

#     def forward(self, x):
#         out = softmax(x)

#         return out

#     def backward(self, dout):
#         print(dout)
#         dx = dout / self.S
#         dx1 = -1 * np.sum(dout * np.exp(self.x), axis=-
#                           1, keepdims=True) / np.square(self.S)
#         print(dx)
#         print(dx1)
#         dx += dx1
#         print(dx)

#         return dx


class Flatten:
    def __init__(self):
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        out = x.reshape(x.shape[0], -1)

        return out

    def backward(self, dout):
        dx = dout.reshape(self.shape[0], self.shape[1], self.shape[2])

        return dx


class Merge:
    '''
    配列が(-1,1)の物を結合する層。
    '''
    def __init__(self):
        self.shape_x1 = None

    def forward(self, x1, x2):
        self.shape_x1 = x1.shape
        return np.concatenate([x1,x2],axis=-1)
    
    def backward(self, dout):
        dx = dout[:, :self.shape_x1[1]]

        return dx


class GraphConvolution:
    '''
    グラフ畳み込みを行う層。
    '''

    def __init__(self, W, U=None, b=None, freeze=False):
        self.W = W
        self.b = b
        self.U = U
        self.freeze = freeze

        # 中間データ（backward時に使用）
        self.x = None
        self.adj_plus_iden = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.dU = None
        self.db = None

    def forward(self, x, adjacency_matrix):
        if adjacency_matrix.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
            adjacency_matrix = adjacency_matrix.reshape(
                1, adjacency_matrix.shape[0], adjacency_matrix.shape[1])
        self.x = x
        self.adj_plus_iden = adjacency_matrix
        out = gcfor(x, adjacency_matrix.astype(
            np.float32), self.W, self.U, self.b)
        return out

    def backward(self, dout):
        if self.adj_plus_iden.ndim == 1:
            dout = dout.reshape(1, dout.shape[0], dout.shape[1])
        if not self.b is None:
            if self.freeze:
                self.db = 0
            else:
                self.db = dout.sum(axis=0).sum(axis=0)
        dx, self.dW = gcback(dout, self.x, self.adj_plus_iden, self.W)
        if self.freeze:
            self.dW = 0
        if not self.U is None:
            self.dU = np.sum(self.x.transpose(0, 2, 1) @ dout, axis=0)
            dx += dout @ self.U.T
            if self.freeze:
                self.dU = 0

        return dx


class GraphPooling:
    '''
    グラフ畳み込みのPooling層。
    '''

    def __init__(self):
        # 中間データ (backward時に使用)
        self.x = None
        self.adj_plus_iden = None
        self.argmax_out = None

    def forward(self, x, adjacency_matrix):
        adj_one_hot = np.zeros_like(adjacency_matrix, dtype=np.int8)
        adj_one_hot[adjacency_matrix != 0] = 1
        adj_one_hot = adj_one_hot.astype(int)
        if adj_one_hot.ndim == 3:
            if GPU:
                out, self.argmax_out = gpfor_iter_for_GPU(
                    x, adj_one_hot)
            else:
                out, self.argmax_out = gpfor_iter(
                    x, adj_one_hot)

        else:
            self.adj_plus_iden = adjacency_matrix + \
                np.identity(adjacency_matrix.shape[0])
            out = np.empty(self.x.shape, np.float64)
            argmax_out = np.empty(self.x.shape, np.int8)
            for adj_row_num in range(self.adj_plus_iden.shape[0]):
                adj_row = self.x[self.adj_plus_iden[adj_row_num].astype(
                    'bool')]
                out[adj_row_num] = adj_row.max(
                    axis=0).reshape(1, self.x.shape[1])
                argmax_out[adj_row_num] = adj_row.argmax(axis=0)
            self.argmax_out = argmax_out.astype(np.int8)

        return out

    def backward(self, dout):
        if dout.ndim == 3:
            dx = gpback_iter(dout, self.argmax_out)
        else:
            dx = np.zeros(self.x.shape)
            for adj_row_num in range(self.adj_plus_iden.shape[0]):
                temporary_array = np.zeros(
                    (int(self.adj_plus_iden[adj_row_num].sum()), self.x.shape[1]))
                argmax_part = self.argmax_out[adj_row_num]
                add_matrix = dout[adj_row_num]
                temporary_array[argmax_part, list(
                    range(self.x.shape[1]))] += add_matrix
                adj_row = self.adj_plus_iden[adj_row_num]
                dx[adj_row.astype('bool')] += temporary_array
        return dx


class GraphGathering:
    '''
    グラフを畳み込んだ最後に各ノードの特徴を集約する層。
    '''

    def __init__(self):
        # 中間データ（backward時に使用）
        self.x = None
        self.ones_matrix = None

    def forward(self, x):
        self.x = x
        if x.ndim == 3:
            self.ones_matrix = np.ones((1, x.shape[1]))
            out = np.dot(self.ones_matrix, x).reshape(x.shape[0], -1)
        else:
            self.ones_matrix = np.ones((1, x.shape[0]))
            out = np.dot(self.ones_matrix, x).reshape(1, -1)
        return out

    def backward(self, dout):
        if self.x.ndim == 3:
            dx = np.dot(self.ones_matrix.T, dout.reshape(
                self.x.shape[0], 1, self.x.shape[2])).transpose(1, 0, 2)
        else:
            dx = np.dot(self.ones_matrix.T, dout)

        return dx


class GraphGathering_norm:
    '''
    グラフを畳み込んだ最後に各ノードの特徴を集約する層。
    '''

    def __init__(self):
        # 中間データ（backward時に使用）
        self.x = None
        self.ones_matrix = None
        self.adj_sum = None

    def forward(self, x, adjacency_matrix):
        self.x = x
        if x.ndim == 3:
            self.adj_sum = adjacency_matrix.copy()
            self.adj_sum[(self.adj_sum != 0)] = 1
            self.adj_sum = self.adj_sum.sum(axis=1)
            self.adj_sum[(self.adj_sum != 0)] = 1
            self.adj_sum = self.adj_sum.sum(axis=1).reshape(
                x.shape[0], 1)  # / adjacency_matrix.shape[1]
            self.ones_matrix = np.ones((1, x.shape[1]))
            out = np.dot(self.ones_matrix, x).reshape(x.shape[0], -1)
            out = out / self.adj_sum

        else:
            self.ones_matrix = np.ones((1, x.shape[0]))
            out = np.dot(self.ones_matrix, x).reshape(1, -1)
        return out

    def backward(self, dout):
        if self.x.ndim == 3:
            dout = dout / self.adj_sum
            dx = np.dot(self.ones_matrix.T, dout.reshape(
                self.x.shape[0], 1, self.x.shape[2])).transpose(1, 0, 2)
        else:
            dx = np.dot(self.ones_matrix.T, dout)

        return dx


class GraphGathering_softmax:
    '''
    グラフを畳み込んだ最後に各ノードの特徴を集約する層。
    '''

    def __init__(self):
        # 中間データ（backward時に使用）
        self.x = None
        self.sum_x = None
        self.S = None

    def forward(self, x):
        self.x = x
        if x.ndim == 2:
            self.x = self.x.reshape(1, self.x.shape[0], self.x.shape[1])
        out = np.sum(x, axis=1)
        self.sum_x = out
        out = softmax(out)
        self.S = np.sum(np.exp(self.sum_x), axis=-1)

        return out

    def backward(self, dout):
        dx = dout.reshape(dout.shape[0], 1, dout.shape[1]) / \
            self.S.reshape(self.S.shape[0], 1, 1)
        dx1 = -1 * np.sum(dout * np.exp(self.sum_x), axis=1,
                          keepdims=True).reshape(-1, 1, 1) / (self.S ** 2).reshape(-1, 1, 1)
        dx *= dx1.reshape((dx1.shape[0], 1, dx1.shape[1]))

        return dx


class SimilalityGlobalAttention():
    def __init__(self):
        self.x = None
        self.softmax = Softmax()
        self.a = None

    def forward(self, x):
        self.x = x
        x_N, x_H, x_W = x.shape
        a = self.softmax.forward(x @ x.transpose(0, 2, 1))
        self.a = a
        a_N, a_H, a_W = a.shape
        out = np.sum(x.reshape(x_N, 1, x_H, x_W) *
                     a.reshape(a_N, a_H, a_W, 1), axis=-2)

        return out

    def backward(self, dout):
        dout_N, dout_H, dout_W = dout.shape
        a_N, a_H, a_W = self.a.shape
        x_N, x_H, x_W = self.x.shape
        dout_repeat = dout.reshape(dout_N, dout_H, 1, dout_W)
        da = self.softmax.backward(
            np.sum(dout_repeat * self.x.reshape(x_N, 1, x_H, x_W), axis=-1))
        dx = np.sum(self.a.reshape(a_N, a_H, a_W, 1) * dout_repeat, axis=-3) + (da @ self.x) + \
            (self.x.transpose(0, 2, 1) @ da).transpose(0, 2, 1)

        return dx


# class DistanceGlobalAttention:
#     def __init__(self):
#         self.x = None
#         self.adj_shortest = None
#         self.softmax = Softmax()
#         self.a = None

#     def forward(self, x, adj):
#         self.x = x
#         adj_shortest_path = np.zeros_like(adj)
#         for i in range(adj.shape[0]):
#             G = nx.from_numpy_array(adj[i])
#             for j in range(adj[i].shape[0]):
#                 shortest_dict = nx.shortest_path_length(G, source=j)
#                 for k in range(len(shortest_dict)):
#                     if j == k:
#                         continue
#                     if len(shortest_dict) == 1:
#                         break
#                     adj_shortest_path[i, j, k] = 1/shortest_dict[k]
#                     # adj_shortest_path[i,j,k] = -len(shortest_dict[k])
#         adj_shortest_path = softmax(adj_shortest_path)
#         self.adj_shortest = adj_shortest_path
#         out = adj_shortest_path @ x

#         return out

#     def backward(self, dout):
#         dx = self.adj_shortest.transpose(0, 2, 1) @ dout

#         return dx


class DistanceGlobalAttention:
    def __init__(self):
        self.x = None
        self.adj_shortest = None
        self.softmax = Softmax()
        self.a = None

    def forward(self, x, adj_shortest_path):
        self.x = x
        self.adj_shortest = adj_shortest_path
        out = adj_shortest_path @ x

        return out

    def backward(self, dout):
        dx = self.adj_shortest.transpose(0, 2, 1) @ dout

        return dx


class GlobalGraphConvolution:
    '''
    グラフ畳み込みを行う層。
    '''

    def __init__(self, W, U=None, b=None, freeze=False):
        self.W = W
        self.b = b
        self.U = U
        self.freeze = freeze

        # 中間データ（backward時に使用）
        self.x = None
        self.adj_shortest = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.dU = None
        self.db = None

    def forward(self, x, adj):
        if adj.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
            adj = adj.reshape(
                1, adj.shape[0], adj.shape[1])
        self.x = x
        adj_shortest_path = np.zeros_like(adj)
        for i in range(adj.shape[0]):
            G = nx.from_numpy_array(adj[i])
            for j in range(adj[i].shape[0]):
                shortest_dict = nx.shortest_path_length(G, source=j)
                for k in range(len(shortest_dict)):
                    if j == k:
                        continue
                    if len(shortest_dict) == 1:
                        break
                    adj_shortest_path[i, j, k] = 1/shortest_dict[k]
                    # adj_shortest_path[i,j,k] = -len(shortest_dict[k])
        adj_shortest_path = softmax(adj_shortest_path)
        self.adj_shortest = adj_shortest_path
        out = gcfor(x, adj_shortest_path.astype(
            np.float32), self.W, self.U, self.b)
        return out

    def backward(self, dout):
        if self.adj_shortest.ndim == 1:
            dout = dout.reshape(1, dout.shape[0], dout.shape[1])
        if not self.b is None:
            if self.freeze:
                self.db = 0
            else:
                self.db = dout.sum(axis=0).sum(axis=0)
        dx, self.dW = gcback(dout, self.x, self.adj_shortest, self.W)
        if self.freeze:
            self.dW = 0
        if not self.U is None:
            self.dU = np.sum(self.x.transpose(0, 2, 1) @ dout, axis=0)
            dx += dout @ self.U.T
            if self.freeze:
                self.dU = 0
        return dx


class Affine:
    def __init__(self, W, b, freeze=False):
        self.W = W
        self.b = b
        self.freeze = freeze

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        if self.freeze:
            self.dW = 0
            self.db = 0
        else:
            self.dW = np.dot(self.x.T, dout)
            self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）

        return dx


class MeanSquaredError:
    def __init__(self):
        self.loss = None
        self.y = None  # 予測値の出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t.reshape(-1, 1)
        self.y = identity_function(x)
        self.loss = sum_squared_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        dx = dout*(self.y-self.t)
        return dx


class MeanSquaredOutRangeError:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        self.range_list = None
    
    def forward(self, x, t):
        self.t = t.reshape(-1,2)
        self.y = identity_function(x)
        self.loss, self.range_list = sum_squared_error_range(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        dx = np.empty(self.y.shape)
        dx[self.range_list[0],0] = dout * (self.y[self.range_list[0],0] - self.t[self.range_list[0],0])
        dx[self.range_list[1],0] = 0
        dx[self.range_list[2],0] = dout * (self.y[self.range_list[2],0] - self.t[self.range_list[2],1])
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        elif x.ndim == 3:
            N, H, W = x.shape
            x = x.reshape(N, -1)
        elif x.ndim == 2:
            pass
        else:
            print('error')
            sys.exit()

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * \
                self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * \
                self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
        elif dout.ndim == 3:
            N, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
