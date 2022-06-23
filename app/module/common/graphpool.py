from common.np import *
from common.config import GPU
if GPU:
    import cupyx


def gpfor_iter(x, adj):
    xn, xh, xw = x.shape
    argmax_out = np.zeros((xn, xh, xw), dtype=np.int16)
    argmax_out[...] = -1
    out = np.zeros((xn, xh, xw))
    deg_array = adj.sum(axis=-1)
    deg_unique = np.unique(deg_array)
    if int(deg_unique[0]) == 0:
        deg_unique = deg_unique[1:]
    for deg in deg_unique:
        adj_deg_where_0, adj_deg_where_1 = np.where(deg_array == deg)
        repeat_ = adj_deg_where_0.repeat(int(deg), axis=0)
        where_ = np.where(adj[adj_deg_where_0, adj_deg_where_1] == 1)[1]
        extract_adj_features = x[(repeat_, where_)
                                 ].reshape(-1, int(deg), xw).transpose(1, 0, 2)
        argmax_part = extract_adj_features.argmax(axis=0)
        argmax_temp = argmax_out[adj_deg_where_0, adj_deg_where_1]
        argmax_part_where = np.array(np.meshgrid(np.array(range(argmax_part.shape[0])), np.array(range(argmax_part.shape[1])))).T.reshape(-1,2)
        argmax_temp[argmax_part_where[0], argmax_part_where[1]] = where_.reshape(-1, int(
            deg))[argmax_part_where[0], argmax_part[argmax_part_where[0], argmax_part_where[1]]]
        argmax_out[adj_deg_where_0, adj_deg_where_1] = argmax_temp
    where_insert = np.concatenate(np.where(argmax_out != -1)).reshape(3, -1)
    out[where_insert[0], where_insert[1], where_insert[2]] = x[(
        where_insert[0], argmax_out[where_insert[0], where_insert[1], where_insert[2]], where_insert[2])]

    return out, argmax_out


def gpfor_iter_for_GPU(x, adj):
    xn, xh, xw = x.shape
    adj_n, adj_h, adj_w = adj.shape
    huge_mat_for_argmax = np.empty((adj_n, adj_h, adj_w, xw))
    huge_mat_for_argmax[...] = -1
    adj_one_where = np.where(adj == 1)
    huge_mat_for_argmax[adj_one_where[0], adj_one_where[1],
                        adj_one_where[2], :] = x[adj_one_where[0], adj_one_where[2], :]
    argmax_out = np.argmax(huge_mat_for_argmax, axis=-2)
    out = np.max(huge_mat_for_argmax, axis=2)
    argmax_out[out == -1] = -1
    out[out == -1] = 0

    return out, argmax_out


def gpback_iter(dout, argmax_out):
    xn, xh, xw = dout.shape
    dx = np.zeros((xn, xh, xw))
    where_insert = np.concatenate(np.where(argmax_out != -1)).reshape(3, -1)
    if GPU:
        cupyx.scatter_add(dx, (where_insert[0], argmax_out[where_insert[0], where_insert[1], where_insert[2]],
                            where_insert[2]), dout[where_insert[0], where_insert[1], where_insert[2]])
    else:
        np.add.at(dx, (where_insert[0], argmax_out[where_insert[0], where_insert[1], where_insert[2]],
                       where_insert[2]), dout[where_insert[0], where_insert[1], where_insert[2]])

    return dx
