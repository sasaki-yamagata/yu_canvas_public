from common.np import *


def gcfor(x, adj, W, U, b):
    out = (x.reshape(x.shape[0]*x.shape[1], x.shape[2])
           @ W).reshape(x.shape[0], x.shape[1], W.shape[1])
    out = adj @ out
    if not U is None:
        for i in range(adj.shape[1]):
            adj[:, i, i] = 0
        x_mul_U = (x.reshape(x.shape[0]*x.shape[1], x.shape[2])
                   @ U).reshape(x.shape[0], x.shape[1], U.shape[1])
        out += x_mul_U
    if not b is None:
        out += b
    return out


def gcback(dout, x, adj, W):
    dx_mid = adj.transpose(0, 2, 1) @ dout
    dx = dx_mid @ W.T
    dW = np.sum(x.transpose(0, 2, 1) @ dx_mid, axis=0)

    return dx, dW
